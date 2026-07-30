#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${1:-$SUITE_DIR/work}"
MODEL_DIR="${MODEL_DIR:-$SUITE_DIR/models}"
ORIGINAL_MODEL="$MODEL_DIR/mace-osaka26-small.model"
MLIAP_MODEL="$MODEL_DIR/mace-osaka26-small.model-mliap_lammps.pt"
for file in "$ORIGINAL_MODEL" "$MLIAP_MODEL"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: missing model: $file" >&2
        exit 1
    fi
done

TEST_DIR="$WORK_ROOT/01_ase_lammps_consistency"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"
ln -sfn "$ORIGINAL_MODEL" mace-osaka26-small.model
ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_zno.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from ase.build import bulk, make_supercell
from ase.io import write

def main():
    primitive = bulk("ZnO", crystalstructure="wurtzite", a=3.25, c=5.21, u=0.3825)
    transform = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=int)
    atoms = make_supercell(primitive, transform).repeat((3, 3, 1))
    atoms.wrap()
    if len(atoms) != 72:
        raise RuntimeError(f"Expected 72 atoms, obtained {len(atoms)}")
    symbols = atoms.get_chemical_symbols()
    if symbols.count("Zn") != 36 or symbols.count("O") != 36:
        raise RuntimeError("Unexpected Zn/O composition")
    write("data.zno", atoms, format="lammps-data", atom_style="atomic", specorder=["Zn", "O"])
    print("Created data.zno: 72 atoms, Zn36O36, type 1=Zn, type 2=O")

if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_zno.py

cat > in.singlepoint <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p p
atom_style atomic
newton on
read_data data.zno
mass 1 65.38
mass 2 15.999
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe etotal press pxx pyy pzz
thermo_modify format float %24.16e
thermo_modify lost error
dump force_dump all custom 1 lammps_forces.dump id type element x y z fx fy fz
dump_modify force_dump element Zn O
dump_modify force_dump sort id
dump_modify force_dump format float %24.16e
run 0
variable mace_pe equal pe
print "LAMMPS_MACE_PE = $(v_mace_pe:%.16e) eV" file lammps_energy.txt screen yes
write_data lammps_singlepoint.data nocoeff
LAMMPS_EOF

cat > ase_singlepoint.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
from ase.io import read, write
from mace.calculators import MACECalculator

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    atoms = read("data.zno", format="lammps-data", style="atomic", Z_of_type={1: 30, 2: 8})
    atoms.calc = MACECalculator(model_paths="mace-osaka26-small.model", device="cuda", default_dtype="float32")
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise RuntimeError("Non-finite ASE/MACE result")
    np.savez("ase_results.npz", energy=energy, forces=forces)
    atoms.info["MACE_energy_eV"] = energy
    atoms.arrays["MACE_forces_eV_per_A"] = forces
    write("ase_singlepoint.extxyz", atoms)
    print(f"ASE_MACE_PE = {energy:.16e} eV")

if __name__ == "__main__":
    main()
PYEOF
chmod +x ase_singlepoint.py

cat > compare_results.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np

def read_energy():
    match = re.search(r"LAMMPS_MACE_PE\s*=\s*([-+0-9.eE]+)", Path("lammps_energy.txt").read_text())
    if match is None:
        raise ValueError("LAMMPS energy not found")
    return float(match.group(1))

def read_forces():
    lines = Path("lammps_forces.dump").read_text().splitlines()
    start = [i for i, line in enumerate(lines) if line.startswith("ITEM: TIMESTEP")][-1]
    nh = lines.index("ITEM: NUMBER OF ATOMS", start)
    n = int(lines[nh + 1])
    ah = next(i for i in range(start, len(lines)) if lines[i].startswith("ITEM: ATOMS"))
    cols = lines[ah].split()[2:]
    idx = {name: cols.index(name) for name in cols}
    rows = lines[ah + 1:ah + 1 + n]
    ids = np.array([int(row.split()[idx["id"]]) for row in rows])
    forces = np.array([[float(row.split()[idx[k]]) for k in ("fx", "fy", "fz")] for row in rows])
    order = np.argsort(ids)
    return ids[order], forces[order]

def main():
    ase = np.load("ase_results.npz")
    ae = float(ase["energy"])
    af = np.asarray(ase["forces"])
    le = read_energy()
    ids, lf = read_forces()
    df = lf - af
    de = le - ae
    de_atom = de / len(ids)
    mae = np.mean(np.abs(df))
    rmse = np.sqrt(np.mean(df**2))
    max_atom = np.linalg.norm(df, axis=1).max()
    passed = abs(de_atom) < 1.0e-5 and rmse < 1.0e-4 and max_atom < 1.0e-3
    report = f"""ASE-LAMMPS consistency
Atoms: {len(ids)}
ASE energy [eV]: {ae:.16e}
LAMMPS energy [eV]: {le:.16e}
Energy difference [eV]: {de:.16e}
Energy difference per atom [eV/atom]: {de_atom:.16e}
Force component MAE [eV/A]: {mae:.16e}
Force component RMSE [eV/A]: {rmse:.16e}
Maximum atomic-vector difference [eV/A]: {max_atom:.16e}
Verdict: {"PASS" if passed else "FAIL"}
"""
    print(report, end="")
    Path("comparison_summary.txt").write_text(report)
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
PYEOF
chmod +x compare_results.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
rm -f log.lammps lammps_energy.txt lammps_forces.dump lammps_singlepoint.data ase_results.npz ase_singlepoint.extxyz comparison_summary.txt
python generate_zno.py
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.singlepoint | tee lammps.out
python ase_singlepoint.py | tee ase.out
python compare_results.py | tee comparison.out
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-consistency
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
./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 01: ASE-LAMMPS consistency

The same 72-atom periodic wurtzite ZnO structure is evaluated with the original
MACE model through ASE and with the converted ML-IAP model through LAMMPS.

Primary result: `comparison_summary.txt`.
EOF

python generate_zno.py
echo "Generated $TEST_DIR"
