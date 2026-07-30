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

TEST_DIR="$WORK_ROOT/06_surface_energy_convergence"
CASE_ROOT="$TEST_DIR/cases"

mkdir -p "$TEST_DIR" "$CASE_ROOT"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_bulk_reference.py <<'PYEOF'
#!/usr/bin/env python3

import numpy as np
from ase.build import bulk, make_supercell
from ase.io import write


def main():
    primitive = bulk(
        "ZnO",
        crystalstructure="wurtzite",
        a=3.2827027076,
        c=5.3172809590,
        u=0.3780562399,
    )

    transform = np.array(
        [
            [1, -1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )

    atoms = make_supercell(primitive, transform).repeat((2, 2, 2))
    atoms.wrap()

    write(
        "bulk_reference.data",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Zn", "O"],
    )

    print(f"Created bulk_reference.data with {len(atoms)} atoms")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_bulk_reference.py

cat > in.bulk_singlepoint <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p p
atom_style atomic
newton on
read_data bulk_reference.data
mass 1 65.38
mass 2 15.999
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe press pxx pyy pzz vol
thermo_modify format float %20.12e
thermo_modify lost error
run 0
variable pe_total equal pe
variable natoms_total equal atoms
variable pe_per_atom equal v_pe_total/v_natoms_total
print "BULK_PE_TOTAL = $(v_pe_total:%.16e) eV" file bulk_reference_summary.txt screen yes
print "BULK_NATOMS = $(v_natoms_total:%.0f)" append bulk_reference_summary.txt screen yes
print "BULK_PE_PER_ATOM = $(v_pe_per_atom:%.16e) eV/atom" append bulk_reference_summary.txt screen yes
LAMMPS_EOF

cat > generate_surface_cases.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import numpy as np
from ase.build import bulk, surface
from ase.io import write


CASE_ROOT = Path("cases")

LAYER_COUNTS = [4, 6, 8, 10, 12]
VACUUM_VALUES = [10.0, 15.0, 20.0, 25.0]


def cluster_layers(z_coordinates, tolerance=0.25):
    order = np.argsort(z_coordinates)
    layers = []

    for atom_index in order:
        z_value = float(z_coordinates[atom_index])

        if not layers or abs(z_value - layers[-1]["mean_z"]) > tolerance:
            layers.append({"indices": [int(atom_index)], "mean_z": z_value})
        else:
            layers[-1]["indices"].append(int(atom_index))
            values = z_coordinates[layers[-1]["indices"]]
            layers[-1]["mean_z"] = float(np.mean(values))

    return layers


def build_case(layer_count, vacuum):
    primitive = bulk(
        "ZnO",
        crystalstructure="wurtzite",
        a=3.2827027076,
        c=5.3172809590,
        u=0.3780562399,
    )

    slab = surface(
        primitive,
        indices=(1, 0, 0),
        layers=layer_count,
        vacuum=vacuum,
        periodic=True,
    )

    slab = slab.repeat((2, 2, 1))
    slab.wrap()

    symbols = slab.get_chemical_symbols()
    if symbols.count("Zn") != symbols.count("O"):
        raise RuntimeError("Generated slab is not stoichiometric")

    z = slab.positions[:, 2]
    layers = cluster_layers(z)

    fixed_layer_count = 2
    fixed_indices = sorted(
        atom_index
        for layer in layers[:fixed_layer_count]
        for atom_index in layer["indices"]
    )

    fixed_set = set(fixed_indices)
    fixed_z_max = max(float(z[index]) for index in fixed_indices)
    next_mobile_z = min(
        float(z[index])
        for index in range(len(slab))
        if index not in fixed_set
    )
    fixed_region_upper = 0.5 * (fixed_z_max + next_mobile_z)

    return slab, layers, fixed_indices, fixed_region_upper


def main():
    CASE_ROOT.mkdir(exist_ok=True)
    rows = []

    for layer_count in LAYER_COUNTS:
        for vacuum in VACUUM_VALUES:
            case_name = f"layers_{layer_count:02d}_vacuum_{vacuum:04.1f}"
            case_dir = CASE_ROOT / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            slab, layers, fixed_indices, fixed_region_upper = build_case(
                layer_count,
                vacuum,
            )

            write(
                case_dir / "data.slab",
                slab,
                format="lammps-data",
                atom_style="atomic",
                specorder=["Zn", "O"],
            )

            (case_dir / "groups.inc").write_text(
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

            area = float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])))
            z = slab.positions[:, 2]
            slab_thickness = float(z.max() - z.min())
            effective_vacuum = float(slab.cell.lengths()[2] - slab_thickness)

            metadata = {
                "case": case_name,
                "layer_count_requested": layer_count,
                "geometric_layers_detected": len(layers),
                "vacuum_requested_A": vacuum,
                "effective_vacuum_A": effective_vacuum,
                "atoms": len(slab),
                "zn_atoms": slab.get_chemical_symbols().count("Zn"),
                "o_atoms": slab.get_chemical_symbols().count("O"),
                "fixed_atom_count": len(fixed_indices),
                "fixed_region_upper_A": fixed_region_upper,
                "surface_area_A2": area,
                "cell_lengths_A": [float(value) for value in slab.cell.lengths()],
                "cell_angles_deg": [float(value) for value in slab.cell.angles()],
                "slab_thickness_A": slab_thickness,
            }

            (case_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )

            rows.append(metadata)

    with Path("surface_cases.csv").open("w", newline="") as handle:
        fieldnames = [
            "case",
            "layer_count_requested",
            "geometric_layers_detected",
            "vacuum_requested_A",
            "effective_vacuum_A",
            "atoms",
            "zn_atoms",
            "o_atoms",
            "fixed_atom_count",
            "fixed_region_upper_A",
            "surface_area_A2",
            "slab_thickness_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"Generated {len(rows)} surface cases under {CASE_ROOT.resolve()}")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_surface_cases.py

cat > in.surface_relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data ${data_file}
mass 1 65.38
mass 2 15.999
pair_style mliap unified ${model_file} 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include ${group_file}
thermo 20
thermo_style custom step atoms pe pxx pyy pzz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 10000 100000
variable final_pe equal pe
variable final_fmax equal fmax
variable final_fnorm equal fnorm
print "FINAL_PE = $(v_final_pe:%.16e) eV" file ${summary_file} screen yes
print "FINAL_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append ${summary_file} screen yes
print "FINAL_FNORM = $(v_final_fnorm:%.16e) eV/angstrom" append ${summary_file} screen yes
write_data ${output_file} nocoeff
LAMMPS_EOF

cat > collect_surface_energies.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np


EV_PER_A2_TO_J_PER_M2 = 16.021766208


def parse_value(text, key):
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*([-+0-9.eE]+)",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not find {key}")
    return float(match.group(1))


def main():
    bulk_text = Path("bulk_reference_summary.txt").read_text()
    bulk_energy_per_atom = parse_value(bulk_text, "BULK_PE_PER_ATOM")

    with Path("surface_cases.csv").open() as handle:
        cases = list(csv.DictReader(handle))

    output_rows = []

    for case in cases:
        case_dir = Path("cases") / case["case"]
        summary = (case_dir / "relaxation_summary.txt").read_text()

        slab_energy = parse_value(summary, "FINAL_PE")
        fmax = parse_value(summary, "FINAL_FMAX")
        natoms = int(case["atoms"])
        area = float(case["surface_area_A2"])

        excess_energy = slab_energy - natoms * bulk_energy_per_atom
        surface_energy_eV_A2 = excess_energy / (2.0 * area)
        surface_energy_J_m2 = surface_energy_eV_A2 * EV_PER_A2_TO_J_PER_M2

        output_rows.append(
            {
                "case": case["case"],
                "layer_count_requested": int(case["layer_count_requested"]),
                "geometric_layers_detected": int(case["geometric_layers_detected"]),
                "vacuum_requested_A": float(case["vacuum_requested_A"]),
                "effective_vacuum_A": float(case["effective_vacuum_A"]),
                "atoms": natoms,
                "surface_area_A2": area,
                "slab_energy_eV": slab_energy,
                "bulk_energy_per_atom_eV": bulk_energy_per_atom,
                "excess_energy_eV": excess_energy,
                "surface_energy_eV_per_A2": surface_energy_eV_A2,
                "surface_energy_J_per_m2": surface_energy_J_m2,
                "fmax_eV_per_A": fmax,
            }
        )

    output_rows.sort(
        key=lambda row: (
            row["vacuum_requested_A"],
            row["layer_count_requested"],
        )
    )

    fieldnames = list(output_rows[0].keys())

    with Path("surface_energy_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Collected surface-energy results:")

    for row in output_rows:
        print(
            f"{row['case']:28s} "
            f"gamma={row['surface_energy_J_per_m2']:.8f} J/m^2 "
            f"Fmax={row['fmax_eV_per_A']:.4e} eV/A"
        )


if __name__ == "__main__":
    main()
PYEOF
chmod +x collect_surface_energies.py

cat > analyze_convergence.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np


LAYER_TOLERANCE_J_M2 = 0.02
VACUUM_TOLERANCE_J_M2 = 0.01


def main():
    with Path("surface_energy_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key in (
            "layer_count_requested",
            "vacuum_requested_A",
            "surface_energy_J_per_m2",
            "fmax_eV_per_A",
        ):
            row[key] = float(row[key])

    vacuums = sorted({row["vacuum_requested_A"] for row in rows})
    layers = sorted({row["layer_count_requested"] for row in rows})

    by_vacuum = {
        vacuum: sorted(
            [row for row in rows if row["vacuum_requested_A"] == vacuum],
            key=lambda row: row["layer_count_requested"],
        )
        for vacuum in vacuums
    }

    by_layer = {
        layer: sorted(
            [row for row in rows if row["layer_count_requested"] == layer],
            key=lambda row: row["vacuum_requested_A"],
        )
        for layer in layers
    }

    lines = [
        "ZnO (10-10) surface-energy convergence",
        "=" * 40,
        "",
        f"Layer-convergence tolerance: {LAYER_TOLERANCE_J_M2:.6f} J/m^2",
        f"Vacuum-convergence tolerance: {VACUUM_TOLERANCE_J_M2:.6f} J/m^2",
    ]

    converged_layer_candidates = []

    lines.extend(["", "Thickness convergence at each vacuum", "------------------------------------"])

    for vacuum in vacuums:
        series = by_vacuum[vacuum]
        lines.append(f"Vacuum = {vacuum:.1f} A")

        for previous, current in zip(series[:-1], series[1:]):
            difference = (
                current["surface_energy_J_per_m2"]
                - previous["surface_energy_J_per_m2"]
            )
            lines.append(
                f"  layers {int(previous['layer_count_requested']):2d} -> "
                f"{int(current['layer_count_requested']):2d}: "
                f"Delta gamma = {difference:+.8f} J/m^2"
            )

            if abs(difference) <= LAYER_TOLERANCE_J_M2:
                converged_layer_candidates.append(
                    (
                        vacuum,
                        current["layer_count_requested"],
                        abs(difference),
                    )
                )

    lines.extend(["", "Vacuum convergence at each thickness", "------------------------------------"])

    converged_vacuum_candidates = []

    for layer in layers:
        series = by_layer[layer]
        lines.append(f"Layers = {int(layer)}")

        for previous, current in zip(series[:-1], series[1:]):
            difference = (
                current["surface_energy_J_per_m2"]
                - previous["surface_energy_J_per_m2"]
            )
            lines.append(
                f"  vacuum {previous['vacuum_requested_A']:4.1f} -> "
                f"{current['vacuum_requested_A']:4.1f} A: "
                f"Delta gamma = {difference:+.8f} J/m^2"
            )

            if abs(difference) <= VACUUM_TOLERANCE_J_M2:
                converged_vacuum_candidates.append(
                    (
                        layer,
                        current["vacuum_requested_A"],
                        abs(difference),
                    )
                )

    final_rows = sorted(
        rows,
        key=lambda row: (
            row["layer_count_requested"],
            row["vacuum_requested_A"],
        ),
    )

    best = final_rows[-1]

    lines.extend(
        [
            "",
            "Largest tested system",
            "---------------------",
            f"Layers: {int(best['layer_count_requested'])}",
            f"Vacuum: {best['vacuum_requested_A']:.1f} A",
            f"Surface energy: {best['surface_energy_J_per_m2']:.10f} J/m^2",
            f"Residual Fmax: {best['fmax_eV_per_A']:.6e} eV/A",
            "",
            f"Thickness-converged transitions found: {len(converged_layer_candidates)}",
            f"Vacuum-converged transitions found: {len(converged_vacuum_candidates)}",
        ]
    )

    verdict = (
        "PASS"
        if converged_layer_candidates and converged_vacuum_candidates
        else "INCOMPLETE_CONVERGENCE"
    )

    lines.extend(["", f"Verdict: {verdict}"])

    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("surface_energy_convergence_summary.txt").write_text(report)

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()

        for vacuum in vacuums:
            series = by_vacuum[vacuum]
            axis.plot(
                [row["layer_count_requested"] for row in series],
                [row["surface_energy_J_per_m2"] for row in series],
                marker="o",
                label=f"{vacuum:.0f} A vacuum",
            )

        axis.set_xlabel("Requested slab layers")
        axis.set_ylabel("Surface energy [J/m$^2$]")
        axis.legend()
        figure.tight_layout()
        figure.savefig("surface_energy_vs_layers.png", dpi=200)
        plt.close(figure)

        figure, axis = plt.subplots()

        for layer in layers:
            series = by_layer[layer]
            axis.plot(
                [row["vacuum_requested_A"] for row in series],
                [row["surface_energy_J_per_m2"] for row in series],
                marker="o",
                label=f"{int(layer)} layers",
            )

        axis.set_xlabel("Requested vacuum [A]")
        axis.set_ylabel("Surface energy [J/m$^2$]")
        axis.legend()
        figure.tight_layout()
        figure.savefig("surface_energy_vs_vacuum.png", dpi=200)
        plt.close(figure)

    except ImportError:
        print("matplotlib is not installed; plots were not created.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze_convergence.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MODEL_FILE="$(readlink -f mace-osaka26-small.model-mliap_lammps.pt)"
SURFACE_INPUT="$(readlink -f in.surface_relax)"

rm -f bulk_reference.data bulk_reference_summary.txt bulk_reference.out surface_cases.csv surface_energy_results.csv surface_energy_convergence_summary.txt surface_energy_vs_layers.png surface_energy_vs_vacuum.png

find cases -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

python generate_bulk_reference.py

echo "=== Bulk reference single-point calculation ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.bulk_singlepoint | tee bulk_reference.out

echo
echo "=== Generate surface cases ==="

python generate_surface_cases.py

echo
echo "=== Relax all surface slabs ==="

while IFS=, read -r case layer_count geometric_layers vacuum_requested effective_vacuum atoms zn_atoms o_atoms fixed_atom_count fixed_region_upper area slab_thickness; do
    if [[ "$case" == "case" ]]; then
        continue
    fi

    case_dir="$(readlink -f "cases/$case")"
    data_file="$case_dir/data.slab"
    group_file="$case_dir/groups.inc"
    output_file="$case_dir/zno_slab_relaxed.data"
    summary_file="$case_dir/relaxation_summary.txt"

    rm -f "$case_dir/log.lammps" "$case_dir/lammps.out" "$output_file" "$summary_file"

    echo
    echo "--- $case ---"

    (
        cd "$case_dir"
        lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -var data_file "$data_file" -var model_file "$MODEL_FILE" -var group_file "$group_file" -var summary_file "$summary_file" -var output_file "$output_file" -in "$SURFACE_INPUT" | tee lammps.out
    )
done < surface_cases.csv

echo
echo "=== Collect and analyze surface energies ==="

python collect_surface_energies.py | tee collect.out
python analyze_convergence.py | tee convergence.out

echo
echo "Primary results:"
echo "  $(pwd)/surface_energy_results.csv"
echo "  $(pwd)/surface_energy_convergence_summary.txt"
echo "  $(pwd)/surface_energy_vs_layers.png"
echo "  $(pwd)/surface_energy_vs_vacuum.png"
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-surfconv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current

./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 06: ZnO (10-10) surface-energy convergence

## Purpose

This test evaluates how the calculated surface energy of stoichiometric nonpolar
wurtzite ZnO (10-10) depends on:

- slab thickness,
- vacuum thickness.

It follows the successful structural-stability test in Test 05.

## Systems

Requested slab layers:

```text
4, 6, 8, 10, 12
```

Requested vacuum values:

```text
10, 15, 20, 25 A
```

The resulting matrix contains 20 independent slab calculations.

Each slab:

- is stoichiometric,
- uses a 2 x 2 in-plane repeat,
- fixes the two lowest geometric atomic layers,
- relaxes all remaining atoms,
- keeps the cell fixed.

## Surface energy

The surface energy is evaluated as:

```text
gamma = (E_slab - N E_bulk_atom) / (2 A)
```

where:

- `E_slab` is the relaxed slab energy,
- `N` is the slab atom count,
- `E_bulk_atom` is the bulk energy per atom,
- `A` is the area of one surface,
- the factor 2 assumes two equivalent surfaces.

## Main outputs

- `bulk_reference_summary.txt`
- `surface_cases.csv`
- `surface_energy_results.csv`
- `surface_energy_convergence_summary.txt`
- `surface_energy_vs_layers.png`
- `surface_energy_vs_vacuum.png`
- `cases/*/relaxation_summary.txt`
- `cases/*/zno_slab_relaxed.data`

## Important limitation

The bottom two layers are fixed while the top surface is relaxed. Therefore the two
surfaces are not strictly equivalent after relaxation, even though the surface-energy
formula divides by two. This test is intended primarily as a practical convergence
diagnostic.

For a quantitatively rigorous surface energy, a later test should use a symmetric slab
with equivalent top and bottom surfaces and symmetric relaxation constraints.

## Run

```bash
sbatch run.slurm
```

or interactively:

```bash
./run.sh
```
EOF

python generate_bulk_reference.py
python generate_surface_cases.py

echo "Generated $TEST_DIR"
