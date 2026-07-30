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

TEST_DIR="$WORK_ROOT/02_fixed_cell_relaxation"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"
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

cat > in.relax <<'LAMMPS_EOF'
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
thermo 10
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
dump traj all custom 10 trajectory.dump id type element x y z fx fy fz
dump_modify traj element Zn O
dump_modify traj sort id
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 5000 50000
variable final_pe equal pe
variable final_fmax equal fmax
variable final_fnorm equal fnorm
variable final_press equal press
print "FINAL_PE = $(v_final_pe:%.16e) eV" file relaxation_summary.txt screen yes
print "FINAL_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append relaxation_summary.txt screen yes
print "FINAL_FNORM = $(v_final_fnorm:%.16e) eV/angstrom" append relaxation_summary.txt screen yes
print "FINAL_PRESS = $(v_final_press:%.16e) bar" append relaxation_summary.txt screen yes
write_data zno_relaxed_fixed_cell.data nocoeff
LAMMPS_EOF

cat > analyze.py <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from ase.io import read

def load(path):
    return read(path, format="lammps-data", style="atomic", Z_of_type={1: 30, 2: 8})

def metrics(atoms):
    scaled = atoms.get_scaled_positions(wrap=True)
    symbols = np.array(atoms.get_chemical_symbols())
    zn = np.where(symbols == "Zn")[0]
    oxy = np.where(symbols == "O")[0]
    distances, u_values = [], []
    for i in zn:
        candidates = []
        for j in oxy:
            distance = atoms.get_distance(i, j, mic=True)
            if distance < 2.5:
                distances.append(distance)
                delta = scaled[j] - scaled[i]
                delta -= np.round(delta)
                candidates.append((distance, min(abs(delta[2]), 1.0 - abs(delta[2]))))
        nearest = sorted(candidates, key=lambda item: item[0])[:4]
        u_values.append(max(nearest, key=lambda item: item[1])[1])
    return np.array(distances), np.array(u_values)

def main():
    lines = ["Fixed-cell ZnO relaxation analysis", "=" * 36]
    for label, path in [("Initial", "data.zno"), ("Relaxed", "zno_relaxed_fixed_cell.data")]:
        distances, u_values = metrics(load(path))
        lines.extend(["", label, "-" * len(label), f"u mean: {u_values.mean():.10f}", f"u std: {u_values.std():.3e}", f"Zn-O mean [A]: {distances.mean():.10f}", f"Zn-O min [A]: {distances.min():.10f}", f"Zn-O max [A]: {distances.max():.10f}"])
    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("structure_analysis.txt").write_text(report)

if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
rm -f log.lammps trajectory.dump relaxation_summary.txt zno_relaxed_fixed_cell.data structure_analysis.txt
python generate_zno.py
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.relax | tee lammps.out
python analyze.py | tee analysis.out
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-fixed-cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current
./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 02: fixed-cell atomic relaxation

Atomic coordinates are relaxed while all cell vectors remain fixed.

Primary results:

- `relaxation_summary.txt`
- `structure_analysis.txt`
- `zno_relaxed_fixed_cell.data`
EOF

python generate_zno.py
echo "Generated $TEST_DIR"
