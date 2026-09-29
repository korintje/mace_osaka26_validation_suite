#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${1:-$SUITE_DIR/work}"
MODEL_DIR="${MODEL_DIR:-$SUITE_DIR/models}"
ORIGINAL_MODEL="$MODEL_DIR/mace-osaka26-small.model"
MLIAP_MODEL="$MODEL_DIR/mace-osaka26-small.model-mliap_lammps.pt"

for file in "$ORIGINAL_MODEL" "$MLIAP_MODEL"; do
    [[ -f "$file" ]] || { echo "ERROR: missing model: $file" >&2; exit 1; }
done

TEST_DIR="$WORK_ROOT/10_d3_ase_lammps_consistency"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"
ln -sfn "$ORIGINAL_MODEL" mace-osaka26-small.model
ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt
if [[ -f "$MODEL_DIR/DTYPE.txt" ]]; then
    cp "$MODEL_DIR/DTYPE.txt" model_dtype.txt
else
    printf '%s\n' float64 > model_dtype.txt
fi

cat > generate_system.py <<'PYEOF'
#!/usr/bin/env python3
"""Create a compact hydrocarbon dimer with a measurable dispersion energy."""
from ase import Atoms
from ase.io import write
import numpy as np

# Two methane molecules at a non-equilibrium separation exercise both energy
# and force.  A large nonperiodic box avoids interactions with periodic images.
tetra = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]) / np.sqrt(3.0) * 1.09
centers = [np.array([35.0, 40.0, 40.0]), np.array([38.8, 40.3, 40.2])]
symbols = []
positions = []
for center in centers:
    symbols.append("C")
    positions.append(center)
    symbols.extend(["H"] * 4)
    positions.extend(center + tetra)
atoms = Atoms(symbols, positions=positions, cell=[80.0, 80.0, 80.0], pbc=False)
write("methane_dimer.extxyz", atoms)
write("data.methane_dimer", atoms, format="lammps-data", atom_style="atomic", specorder=["C", "H"])
print("Created methane dimer: C2H8, type 1=C, type 2=H")
PYEOF
chmod +x generate_system.py

cat > in.d3_only <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary f f f
atom_style atomic
read_data data.methane_dimer
mass 1 12.011
mass 2 1.008
pair_style dispersion/d3 bj pbe 30.0 20.0
pair_coeff * * C H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe
thermo_modify format float %24.16e
dump result all custom 1 lammps_d3_forces.dump id type fx fy fz
dump_modify result sort id
dump_modify result format float %24.16e
run 0
variable result_pe equal pe
print "LAMMPS_D3_PE = $(v_result_pe:%.16e) eV" file lammps_d3_energy.txt screen yes
LAMMPS_EOF

cat > in.mace_d3 <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary f f f
atom_style atomic
newton on
read_data data.methane_dimer
mass 1 12.011
mass 2 1.008
pair_style hybrid/overlay mliap unified mace-osaka26-small.model-mliap_lammps.pt 0 dispersion/d3 bj pbe 30.0 20.0
pair_coeff * * mliap C H
pair_coeff * * dispersion/d3 C H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe
thermo_modify format float %24.16e
dump result all custom 1 lammps_total_forces.dump id type fx fy fz
dump_modify result sort id
dump_modify result format float %24.16e
run 0
variable result_pe equal pe
print "LAMMPS_MACE_D3_PE = $(v_result_pe:%.16e) eV" file lammps_total_energy.txt screen yes
LAMMPS_EOF

cat > ase_singlepoint.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

atoms = read("methane_dimer.extxyz")
dtype_name = Path("model_dtype.txt").read_text().strip()
if dtype_name not in {"float32", "float64"}:
    raise ValueError(f"Unsupported model dtype: {dtype_name}")
device = "cuda" if torch.cuda.is_available() else "cpu"
mace = MACECalculator(
    model_paths="mace-osaka26-small.model",
    device=device,
    default_dtype=dtype_name,
)
d3 = TorchDFTD3Calculator(
    atoms=atoms,
    device="cpu",
    damping="bj",
    xc="pbe",
    cutoff=30.0,
    cnthr=20.0,
    abc=False,
    dtype=torch.float64,
)

atoms.calc = d3
d3_energy = float(atoms.get_potential_energy())
d3_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
atoms.calc = mace
mace_energy = float(atoms.get_potential_energy())
mace_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
np.savez(
    "ase_results.npz",
    d3_energy=d3_energy,
    d3_forces=d3_forces,
    total_energy=mace_energy + d3_energy,
    total_forces=mace_forces + d3_forces,
)
print(f"ASE_D3_PE = {d3_energy:.16e} eV")
print(f"ASE_MACE_D3_PE = {mace_energy + d3_energy:.16e} eV")
PYEOF
chmod +x ase_singlepoint.py

cat > compare_results.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np

def energy(path, label):
    match = re.search(label + r"\s*=\s*([-+0-9.eE]+)", Path(path).read_text())
    if match is None:
        raise ValueError(f"{label} not found in {path}")
    return float(match.group(1))

def forces(path):
    lines = Path(path).read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    columns = lines[header].split()[2:]
    index = {name: columns.index(name) for name in columns}
    rows = [line.split() for line in lines[header + 1:] if line.strip()]
    rows.sort(key=lambda row: int(row[index["id"]]))
    return np.array([[float(row[index[key]]) for key in ("fx", "fy", "fz")] for row in rows])

def metrics(reference_energy, reference_forces, candidate_energy, candidate_forces):
    delta = candidate_forces - reference_forces
    return (
        candidate_energy - reference_energy,
        float(np.sqrt(np.mean(delta**2))),
        float(np.max(np.linalg.norm(delta, axis=1))),
    )

ase = np.load("ase_results.npz")
d3 = metrics(
    float(ase["d3_energy"]), ase["d3_forces"],
    energy("lammps_d3_energy.txt", "LAMMPS_D3_PE"), forces("lammps_d3_forces.dump"),
)
total = metrics(
    float(ase["total_energy"]), ase["total_forces"],
    energy("lammps_total_energy.txt", "LAMMPS_MACE_D3_PE"), forces("lammps_total_forces.dump"),
)

# The D3 implementation comparison is tight; the overlay tolerance also allows
# the selected float32 ML-IAP conversion to be validated by this test.
passed = (
    abs(d3[0]) < 1.0e-5 and d3[1] < 1.0e-5 and d3[2] < 5.0e-5
    and abs(total[0]) / len(ase["total_forces"]) < 1.0e-4
    and total[1] < 5.0e-4 and total[2] < 2.0e-3
)
report = f"""PBE-D3(BJ) ASE-LAMMPS consistency
Atoms: {len(ase['total_forces'])}
D3 energy difference [eV]: {d3[0]:.16e}
D3 force-component RMSE [eV/A]: {d3[1]:.16e}
D3 maximum atomic-vector difference [eV/A]: {d3[2]:.16e}
MACE+D3 energy difference [eV]: {total[0]:.16e}
MACE+D3 energy difference per atom [eV/atom]: {total[0] / len(ase['total_forces']):.16e}
MACE+D3 force-component RMSE [eV/A]: {total[1]:.16e}
MACE+D3 maximum atomic-vector difference [eV/A]: {total[2]:.16e}
Verdict: {'PASS' if passed else 'FAIL'}
"""
print(report, end="")
Path("comparison_summary.txt").write_text(report)
raise SystemExit(0 if passed else 1)
PYEOF
chmod +x compare_results.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
LAMMPS_BIN="${LAMMPS_BIN:-/apps/envs/lammps-mace-current/bin/lmp}"

if [[ ! -x "$LAMMPS_BIN" ]]; then
    echo "ERROR: LAMMPS executable is missing or not executable: $LAMMPS_BIN" >&2
    exit 1
fi

python - <<'PYEOF'
import importlib.util
if importlib.util.find_spec("torch_dftd") is None:
    raise SystemExit("ERROR: torch_dftd is missing; run environment/install_d3_support.sh")
PYEOF
if ! LAMMPS_HELP="$($LAMMPS_BIN -h 2>&1)"; then
    echo "ERROR: failed to execute $LAMMPS_BIN -h" >&2
    printf '%s\n' "$LAMMPS_HELP" | sed -n '1,5p' >&2
    exit 1
fi
if [[ "$LAMMPS_HELP" != *"dispersion/d3"* ]]; then
    echo "ERROR: LAMMPS lacks dispersion/d3 (EXTRA-PAIR package)" >&2
    echo "LAMMPS_BIN=$LAMMPS_BIN" >&2
    printf '%s\n' "$LAMMPS_HELP" | sed -n '1,5p' >&2
    exit 1
fi

rm -f log.lammps lammps_d3_energy.txt lammps_d3_forces.dump lammps_total_energy.txt lammps_total_forces.dump ase_results.npz comparison_summary.txt
python generate_system.py
"$LAMMPS_BIN" -in in.d3_only | tee lammps_d3.out
"$LAMMPS_BIN" -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.mace_d3 | tee lammps_mace_d3.out
python ase_singlepoint.py | tee ase.out
python compare_results.py | tee comparison.out
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-d3-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current
export LAMMPS_BIN=/apps/envs/lammps-mace-current/bin/lmp
echo "LAMMPS executable: $(readlink -f "$LAMMPS_BIN")"
./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 10: PBE-D3(BJ) ASE-LAMMPS consistency

This test validates the D3 setup required for dispersion-corrected MACE
calculations. A nonperiodic methane dimer is evaluated in two ways:

1. D3 correction only: ASE `TorchDFTD3Calculator` versus LAMMPS `dispersion/d3`.
2. Total MACE+D3: the sum of ASE calculators versus LAMMPS `hybrid/overlay`.

Both routes use PBE-D3(BJ), a 30 A interaction cutoff, a 20 A coordination
cutoff, and no Axilrod-Teller-Muto three-body term. The primary result is
`comparison_summary.txt`. A PASS confirms that energies and forces agree and
that the overlay adds D3 rather than replacing the MACE interaction.

Prerequisites can be installed and checked from the suite root with:

```bash
./environment/install_d3_support.sh
```

Run this test with `sbatch run.slurm`.
EOF

python generate_system.py
echo "Generated $TEST_DIR"
