#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${1:-$SUITE_DIR/work}"
MODEL_DIR="${MODEL_DIR:-$SUITE_DIR/models}"

MLIAP_MODEL="$MODEL_DIR/mace-osaka26-small.model-mliap_lammps.pt"

if [[ ! -f "$MLIAP_MODEL" ]]; then
    echo "ERROR: required model not found: $MLIAP_MODEL" >&2
    exit 1
fi

TEST_DIR="$WORK_ROOT/05_surface_slab_relaxation"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_slab.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from ase.build import bulk, surface
from ase.io import write


OUTPUT = Path("data.zno_10m10")
METADATA = Path("slab_metadata.json")
GROUP_INCLUDE = Path("groups.inc")


def cluster_layers(z_coordinates, tolerance=0.25):
    order = np.argsort(z_coordinates)
    layers = []

    for atom_index in order:
        z_value = float(z_coordinates[atom_index])

        if not layers or abs(z_value - layers[-1]["mean_z"]) > tolerance:
            layers.append(
                {
                    "indices": [int(atom_index)],
                    "mean_z": z_value,
                }
            )
        else:
            layers[-1]["indices"].append(int(atom_index))
            values = z_coordinates[layers[-1]["indices"]]
            layers[-1]["mean_z"] = float(np.mean(values))

    return layers


def main():
    # Wurtzite ZnO equilibrium geometry from the preceding bulk relaxation test.
    primitive = bulk(
        "ZnO",
        crystalstructure="wurtzite",
        a=3.2827027076,
        c=5.3172809590,
        u=0.3780562399,
    )

    # Miller indices (1, 0, 0) in the three-index hexagonal basis correspond
    # to the nonpolar wurtzite (10-10) family.
    slab = surface(
        primitive,
        indices=(1, 0, 0),
        layers=8,
        vacuum=20.0,
        periodic=True,
    )

    # Increase the in-plane area to reduce artificial coupling of local relaxations.
    slab = slab.repeat((2, 2, 1))
    slab.wrap()

    symbols = slab.get_chemical_symbols()
    if symbols.count("Zn") != symbols.count("O"):
        raise RuntimeError("The generated slab is not stoichiometric")

    z = slab.positions[:, 2]
    layers = cluster_layers(z)

    if len(layers) < 6:
        raise RuntimeError(f"Unexpectedly small number of atomic layers: {len(layers)}")

    # Fix the two lowest geometric atomic layers.
    fixed_layer_count = 2
    fixed_indices = sorted(
        atom_index
        for layer in layers[:fixed_layer_count]
        for atom_index in layer["indices"]
    )

    fixed_z_max = max(float(z[index]) for index in fixed_indices)
    next_mobile_z = min(
        float(z[index])
        for index in range(len(slab))
        if index not in set(fixed_indices)
    )
    fixed_region_upper = 0.5 * (fixed_z_max + next_mobile_z)

    write(
        OUTPUT,
        slab,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Zn", "O"],
    )

    GROUP_INCLUDE.write_text(
        "\n".join(
            [
                f"region fixed_region block INF INF INF INF INF {fixed_region_upper:.12f} units box",
                "group fixed region fixed_region",
                "group mobile subtract all fixed",
                "fix freeze fixed setforce 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    cell_lengths = slab.cell.lengths()
    slab_thickness = float(z.max() - z.min())
    vacuum_estimate = float(cell_lengths[2] - slab_thickness)

    metadata = {
        "surface": "ZnO wurtzite (10-10), generated as three-index (1,0,0)",
        "atoms": len(slab),
        "zn_atoms": symbols.count("Zn"),
        "o_atoms": symbols.count("O"),
        "layers_detected": len(layers),
        "fixed_layer_count": fixed_layer_count,
        "fixed_atom_count": len(fixed_indices),
        "fixed_region_upper_A": fixed_region_upper,
        "cell_lengths_A": [float(value) for value in cell_lengths],
        "cell_angles_deg": [float(value) for value in slab.cell.angles()],
        "slab_thickness_A": slab_thickness,
        "vacuum_estimate_A": vacuum_estimate,
        "periodic": [True, True, True],
        "lammps_boundary": "p p f",
        "type_mapping": {"1": "Zn", "2": "O"},
    }

    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Created: {OUTPUT.resolve()}")
    print(f"Atoms: {len(slab)}")
    print(f"Composition: Zn{symbols.count('Zn')} O{symbols.count('O')}")
    print(f"Detected geometric layers: {len(layers)}")
    print(f"Fixed atoms: {len(fixed_indices)}")
    print(f"Slab thickness: {slab_thickness:.6f} A")
    print(f"Estimated vacuum: {vacuum_estimate:.6f} A")
    print("LAMMPS types: 1=Zn, 2=O")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_slab.py

cat > in.singlepoint <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data data.zno_10m10
mass 1 65.38
mass 2 15.999
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
run 0
variable initial_pe equal pe
variable initial_fmax equal fmax
variable initial_press equal press
print "INITIAL_PE = $(v_initial_pe:%.16e) eV" file initial_summary.txt screen yes
print "INITIAL_FMAX = $(v_initial_fmax:%.16e) eV/angstrom" append initial_summary.txt screen yes
print "INITIAL_PRESS = $(v_initial_press:%.16e) bar" append initial_summary.txt screen yes
LAMMPS_EOF

cat > in.relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data data.zno_10m10
mass 1 65.38
mass 2 15.999
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include groups.inc
thermo 10
thermo_style custom step atoms pe pxx pyy pzz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
dump trajectory all custom 10 relaxation.dump id type element x y z fx fy fz
dump_modify trajectory element Zn O
dump_modify trajectory sort id
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 10000 100000
variable final_pe equal pe
variable final_fmax equal fmax
variable final_fnorm equal fnorm
print "FINAL_PE = $(v_final_pe:%.16e) eV" file relaxation_summary.txt screen yes
print "FINAL_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append relaxation_summary.txt screen yes
print "FINAL_FNORM = $(v_final_fnorm:%.16e) eV/angstrom" append relaxation_summary.txt screen yes
write_data zno_10m10_relaxed.data nocoeff
LAMMPS_EOF

cat > analyze_slab.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from ase.io import read


INITIAL_FILE = Path("data.zno_10m10")
RELAXED_FILE = Path("zno_10m10_relaxed.data")
METADATA_FILE = Path("slab_metadata.json")
Z_OF_TYPE = {1: 30, 2: 8}


def load(path):
    return read(
        path,
        format="lammps-data",
        style="atomic",
        Z_of_type=Z_OF_TYPE,
    )


def coordination_numbers(atoms, cutoff=2.5):
    symbols = np.array(atoms.get_chemical_symbols())
    coordination = np.zeros(len(atoms), dtype=int)

    for i in range(len(atoms)):
        target = "O" if symbols[i] == "Zn" else "Zn"

        for j in range(len(atoms)):
            if symbols[j] != target:
                continue

            if atoms.get_distance(i, j, mic=True) < cutoff:
                coordination[i] += 1

    return coordination


def minimum_interatomic_distance(atoms):
    minimum = np.inf
    pair = None

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = atoms.get_distance(i, j, mic=True)

            if distance < minimum:
                minimum = distance
                pair = (i + 1, j + 1)

    return float(minimum), pair


def main():
    for path in (INITIAL_FILE, RELAXED_FILE, METADATA_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(METADATA_FILE.read_text())
    initial = load(INITIAL_FILE)
    relaxed = load(RELAXED_FILE)

    if len(initial) != len(relaxed):
        raise RuntimeError("Atom count changed during relaxation")

    displacement = relaxed.positions - initial.positions
    displacement[:, 0:2] -= np.round(
        displacement[:, 0:2] / initial.cell.lengths()[0:2]
    ) * initial.cell.lengths()[0:2]

    displacement_norm = np.linalg.norm(displacement, axis=1)
    z_initial = initial.positions[:, 2]
    fixed_mask = z_initial <= metadata["fixed_region_upper_A"]
    mobile_mask = ~fixed_mask

    initial_coordination = coordination_numbers(initial)
    relaxed_coordination = coordination_numbers(relaxed)

    initial_min_distance, initial_pair = minimum_interatomic_distance(initial)
    relaxed_min_distance, relaxed_pair = minimum_interatomic_distance(relaxed)

    top_count = max(1, len(relaxed) // 4)
    top_indices = np.argsort(relaxed.positions[:, 2])[-top_count:]

    lines = [
        "ZnO (10-10) surface-slab relaxation analysis",
        "=" * 46,
        "",
        f"Atoms: {len(relaxed)}",
        f"Fixed atoms: {int(fixed_mask.sum())}",
        f"Mobile atoms: {int(mobile_mask.sum())}",
        f"Cell lengths [A]: {' '.join(f'{x:.10f}' for x in relaxed.cell.lengths())}",
        f"Cell angles [deg]: {' '.join(f'{x:.10f}' for x in relaxed.cell.angles())}",
        f"Initial minimum distance [A]: {initial_min_distance:.10f}",
        f"Initial minimum-distance pair IDs: {initial_pair}",
        f"Relaxed minimum distance [A]: {relaxed_min_distance:.10f}",
        f"Relaxed minimum-distance pair IDs: {relaxed_pair}",
        "",
        f"Maximum displacement, all atoms [A]: {displacement_norm.max():.10f}",
        f"Maximum displacement, mobile atoms [A]: {displacement_norm[mobile_mask].max():.10f}",
        f"Mean displacement, mobile atoms [A]: {displacement_norm[mobile_mask].mean():.10f}",
        f"Maximum displacement, fixed atoms [A]: {displacement_norm[fixed_mask].max():.10e}",
        "",
        f"Initial coordination range: {initial_coordination.min()} to {initial_coordination.max()}",
        f"Relaxed coordination range: {relaxed_coordination.min()} to {relaxed_coordination.max()}",
        f"Undercoordinated atoms after relaxation, CN<4: {int(np.sum(relaxed_coordination < 4))}",
        f"Overcoordinated atoms after relaxation, CN>4: {int(np.sum(relaxed_coordination > 4))}",
        "",
        f"Top-quarter mean z displacement [A]: {displacement[top_indices, 2].mean():.10f}",
        f"Top-quarter max |z displacement| [A]: {np.abs(displacement[top_indices, 2]).max():.10f}",
    ]

    failure_reasons = []

    if not np.all(np.isfinite(relaxed.positions)):
        failure_reasons.append("non-finite relaxed coordinates")

    if relaxed_min_distance < 1.2:
        failure_reasons.append("unphysically short interatomic distance")

    if displacement_norm[fixed_mask].max() > 1.0e-8:
        failure_reasons.append("fixed atoms moved")

    if displacement_norm[mobile_mask].max() > 5.0:
        failure_reasons.append("excessive atomic displacement")

    verdict = "PASS" if not failure_reasons else "FAIL"
    lines.extend(
        [
            "",
            f"Verdict: {verdict}",
            f"Failure reasons: {', '.join(failure_reasons) if failure_reasons else 'none'}",
        ]
    )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("slab_analysis.txt").write_text(report)

    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze_slab.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

rm -f log.lammps initial_summary.txt relaxation_summary.txt relaxation.dump zno_10m10_relaxed.data slab_analysis.txt

python generate_slab.py

echo "=== Initial slab single-point calculation ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.singlepoint | tee initial.out

echo
echo "=== Fixed-bottom surface relaxation ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.relax | tee relaxation.out

echo
echo "=== Structural analysis ==="

python analyze_slab.py | tee analysis.out

echo
echo "Primary results:"
echo "  $(pwd)/initial_summary.txt"
echo "  $(pwd)/relaxation_summary.txt"
echo "  $(pwd)/slab_analysis.txt"
echo "  $(pwd)/zno_10m10_relaxed.data"
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-slab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current

./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 05: ZnO nonpolar surface-slab relaxation

## Purpose

This is the first surface-specific validation test. It checks whether
MACE-Osaka26 can stably evaluate and relax a stoichiometric, nonpolar
wurtzite ZnO (10-10) slab without obvious numerical or structural failure.

## Surface choice

The wurtzite ZnO (10-10) surface is used first because it is nonpolar and
does not require the charge-compensation treatment associated with polar
ZnO (0001) and (000-1) surfaces.

ASE generates this surface using three-index Miller indices `(1, 0, 0)`.

## Procedure

1. Start from the MACE-Osaka26 bulk-relaxed ZnO parameters:

   ```text
   a = 3.2827027076 A
   c = 5.3172809590 A
   u = 0.3780562399
   ```

2. Generate an eight-layer stoichiometric slab.
3. Repeat the surface cell 2 x 2 in-plane.
4. Add approximately 20 A of vacuum.
5. Fix the two lowest geometric atomic layers.
6. Relax all remaining atoms at fixed cell dimensions.
7. Check:
   - finite coordinates,
   - minimum interatomic distance,
   - fixed-layer immobility,
   - maximum mobile-atom displacement,
   - Zn-O coordination changes.

## Main outputs

- `slab_metadata.json`
- `initial_summary.txt`
- `relaxation_summary.txt`
- `slab_analysis.txt`
- `zno_10m10_relaxed.data`
- `relaxation.dump`

## Scope

This test does not yet evaluate surface energy or slab-thickness convergence.
It is a construction and stability test that should pass before quantitative
surface-property calculations are attempted.

## Run

Interactive:

```bash
./run.sh
```

Slurm:

```bash
sbatch run.slurm
```
EOF

python generate_slab.py

echo "Generated $TEST_DIR"
