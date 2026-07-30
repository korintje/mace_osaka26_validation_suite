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

TEST_DIR="$WORK_ROOT/04_equation_of_state"
CASE_ROOT="$TEST_DIR/cases"

mkdir -p "$TEST_DIR" "$CASE_ROOT"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_zno.py <<'PYEOF'
#!/usr/bin/env python3

import numpy as np
from ase.build import bulk, make_supercell
from ase.io import write


def main():
    primitive = bulk(
        "ZnO",
        crystalstructure="wurtzite",
        a=3.25,
        c=5.21,
        u=0.3825,
    )

    transform = np.array(
        [
            [1, -1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )

    atoms = make_supercell(primitive, transform).repeat((3, 3, 1))
    atoms.wrap()

    if len(atoms) != 72:
        raise RuntimeError(f"Expected 72 atoms, obtained {len(atoms)}")

    symbols = atoms.get_chemical_symbols()
    if symbols.count("Zn") != 36 or symbols.count("O") != 36:
        raise RuntimeError("Unexpected Zn/O composition")

    write(
        "data.zno",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Zn", "O"],
    )

    print("Created data.zno")
    print("Atoms: 72")
    print("Composition: Zn36 O36")
    print("LAMMPS types: 1=Zn, 2=O")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_zno.py

cat > in.reference_relax <<'LAMMPS_EOF'
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
thermo_style custom step atoms pe press pxx pyy pzz lx ly lz vol fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 5000 50000
fix boxrelax all box/relax aniso 0.0 vmax 0.001
min_style cg
min_modify line quadratic dmax 0.02
minimize 0.0 1.0e-6 5000 50000
minimize 0.0 1.0e-6 5000 50000
variable final_pe equal pe
variable final_press equal press
variable final_volume equal vol
variable final_fmax equal fmax
print "REFERENCE_PE = $(v_final_pe:%.16e) eV" file reference_summary.txt screen yes
print "REFERENCE_PRESS = $(v_final_press:%.16e) bar" append reference_summary.txt screen yes
print "REFERENCE_VOLUME = $(v_final_volume:%.16e) angstrom^3" append reference_summary.txt screen yes
print "REFERENCE_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append reference_summary.txt screen yes
unfix boxrelax
write_data zno_reference_relaxed.data nocoeff
LAMMPS_EOF

cat > generate_eos_cases.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
from ase.io import read, write


REFERENCE_FILE = Path("zno_reference_relaxed.data")
CASE_ROOT = Path("cases")

VOLUME_RATIOS = np.array(
    [
        0.94,
        0.96,
        0.98,
        1.00,
        1.02,
        1.04,
        1.06,
    ],
    dtype=float,
)


def main():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(REFERENCE_FILE)

    reference = read(
        REFERENCE_FILE,
        format="lammps-data",
        style="atomic",
        Z_of_type={1: 30, 2: 8},
    )

    reference_volume = reference.get_volume()
    CASE_ROOT.mkdir(exist_ok=True)

    rows = []

    for index, volume_ratio in enumerate(VOLUME_RATIOS):
        linear_scale = volume_ratio ** (1.0 / 3.0)
        atoms = reference.copy()
        atoms.set_cell(reference.cell * linear_scale, scale_atoms=True)
        atoms.wrap()

        case_name = f"v{index:02d}_ratio_{volume_ratio:.2f}"
        case_dir = CASE_ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        data_file = case_dir / "data.scaled"
        write(
            data_file,
            atoms,
            format="lammps-data",
            atom_style="atomic",
            specorder=["Zn", "O"],
        )

        rows.append(
            {
                "case": case_name,
                "volume_ratio": volume_ratio,
                "linear_scale": linear_scale,
                "initial_volume_A3": atoms.get_volume(),
                "reference_volume_A3": reference_volume,
            }
        )

        (case_dir / "README.txt").write_text(
            "\n".join(
                [
                    f"Case: {case_name}",
                    f"Target V/Vref: {volume_ratio:.8f}",
                    f"Linear scale: {linear_scale:.12f}",
                    f"Initial volume [A^3]: {atoms.get_volume():.12f}",
                    "",
                    "The cell is fixed during atomic-coordinate relaxation.",
                ]
            )
            + "\n"
        )

    with Path("eos_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reference volume: {reference_volume:.12f} A^3")
    print(f"Generated {len(rows)} EOS cases under {CASE_ROOT.resolve()}")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_eos_cases.py

cat > in.eos_relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p p
atom_style atomic
newton on
read_data ${data_file}
mass 1 65.38
mass 2 15.999
pair_style mliap unified ${model_file} 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 10
thermo_style custom step atoms pe press pxx pyy pzz vol fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 5000 50000
variable final_pe equal pe
variable final_press equal press
variable final_volume equal vol
variable final_fmax equal fmax
print "FINAL_PE = $(v_final_pe:%.16e) eV" file ${summary_file} screen yes
print "FINAL_PRESS = $(v_final_press:%.16e) bar" append ${summary_file} screen yes
print "FINAL_VOLUME = $(v_final_volume:%.16e) angstrom^3" append ${summary_file} screen yes
print "FINAL_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append ${summary_file} screen yes
write_data ${output_file} nocoeff
LAMMPS_EOF

cat > collect_eos_results.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import re


CASE_TABLE = Path("eos_cases.csv")


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
    if not CASE_TABLE.exists():
        raise FileNotFoundError(CASE_TABLE)

    with CASE_TABLE.open() as handle:
        cases = list(csv.DictReader(handle))

    output_rows = []

    for case in cases:
        case_dir = Path("cases") / case["case"]
        summary_file = case_dir / "relaxation_summary.txt"

        if not summary_file.exists():
            raise FileNotFoundError(summary_file)

        text = summary_file.read_text()

        output_rows.append(
            {
                "case": case["case"],
                "volume_ratio": float(case["volume_ratio"]),
                "linear_scale": float(case["linear_scale"]),
                "volume_A3": parse_value(text, "FINAL_VOLUME"),
                "energy_eV": parse_value(text, "FINAL_PE"),
                "pressure_bar": parse_value(text, "FINAL_PRESS"),
                "fmax_eV_per_A": parse_value(text, "FINAL_FMAX"),
            }
        )

    output_rows.sort(key=lambda row: row["volume_A3"])

    with Path("eos_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print("Collected EOS results:")
    for row in output_rows:
        print(
            f"{row['case']:18s} "
            f"V={row['volume_A3']:.8f} A^3 "
            f"E={row['energy_eV']:.12f} eV "
            f"P={row['pressure_bar']:.6e} bar "
            f"Fmax={row['fmax_eV_per_A']:.6e} eV/A"
        )


if __name__ == "__main__":
    main()
PYEOF
chmod +x collect_eos_results.py

cat > fit_eos.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
from scipy.optimize import curve_fit


EV_PER_A3_TO_GPA = 160.21766208
N_ATOMS = 72


def birch_murnaghan(volume, e0, v0, b0, bp):
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + (9.0 * v0 * b0 / 16.0) * (
        ((eta - 1.0) ** 3) * bp
        + ((eta - 1.0) ** 2) * (6.0 - 4.0 * eta)
    )


def main():
    rows = []

    with Path("eos_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    volumes = np.array([float(row["volume_A3"]) for row in rows])
    energies = np.array([float(row["energy_eV"]) for row in rows])

    minimum_index = int(np.argmin(energies))
    v_guess = volumes[minimum_index]
    e_guess = energies[minimum_index]

    polynomial = np.polyfit(volumes - v_guess, energies - e_guess, deg=2)
    b_guess = max(2.0 * polynomial[0] * v_guess, 0.01)

    initial_guess = [e_guess, v_guess, b_guess, 4.0]

    lower_bounds = [
        e_guess - 10.0,
        volumes.min() * 0.90,
        1.0e-6,
        1.0,
    ]
    upper_bounds = [
        e_guess + 10.0,
        volumes.max() * 1.10,
        10.0,
        10.0,
    ]

    parameters, covariance = curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    e0, v0, b0, bp = parameters
    fitted = birch_murnaghan(volumes, *parameters)
    residuals = energies - fitted
    rmse = np.sqrt(np.mean(residuals**2))

    volume_per_formula_unit = v0 / 36.0
    energy_per_atom = e0 / N_ATOMS
    b0_gpa = b0 * EV_PER_A3_TO_GPA

    uncertainties = np.sqrt(np.diag(covariance))

    report = "\n".join(
        [
            "Third-order Birch-Murnaghan EOS fit",
            "====================================",
            f"Number of points: {len(volumes)}",
            f"Atoms per cell: {N_ATOMS}",
            f"Formula units per cell: 36",
            "",
            f"E0 [eV/cell]: {e0:.16e}",
            f"E0 [eV/atom]: {energy_per_atom:.16e}",
            f"V0 [A^3/cell]: {v0:.16e}",
            f"V0 [A^3/ZnO]: {volume_per_formula_unit:.16e}",
            f"B0 [eV/A^3]: {b0:.16e}",
            f"B0 [GPa]: {b0_gpa:.10f}",
            f"B0 prime: {bp:.10f}",
            f"Fit RMSE [eV/cell]: {rmse:.16e}",
            "",
            "Approximate one-sigma parameter uncertainties:",
            f"sigma(E0) [eV]: {uncertainties[0]:.6e}",
            f"sigma(V0) [A^3]: {uncertainties[1]:.6e}",
            f"sigma(B0) [eV/A^3]: {uncertainties[2]:.6e}",
            f"sigma(B0 prime): {uncertainties[3]:.6e}",
        ]
    ) + "\n"

    print(report, end="")
    Path("eos_fit_summary.txt").write_text(report)

    dense_volumes = np.linspace(volumes.min(), volumes.max(), 400)
    dense_energies = birch_murnaghan(dense_volumes, *parameters)

    with Path("eos_fitted_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["volume_A3", "fitted_energy_eV"])
        writer.writerows(zip(dense_volumes, dense_energies))

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.scatter(volumes / N_ATOMS, energies / N_ATOMS, label="MACE calculations")
        axis.plot(
            dense_volumes / N_ATOMS,
            dense_energies / N_ATOMS,
            label="Birch-Murnaghan fit",
        )
        axis.set_xlabel("Volume [A$^3$/atom]")
        axis.set_ylabel("Energy [eV/atom]")
        axis.legend()
        figure.tight_layout()
        figure.savefig("eos_fit.png", dpi=200)
        plt.close(figure)
    except ImportError:
        print("matplotlib is not installed; eos_fit.png was not created.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x fit_eos.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MODEL_FILE="$(readlink -f mace-osaka26-small.model-mliap_lammps.pt)"
INPUT_FILE="$(readlink -f in.eos_relax)"

rm -f log.lammps reference_summary.txt zno_reference_relaxed.data reference_relax.out eos_cases.csv eos_results.csv eos_fit_summary.txt eos_fitted_curve.csv eos_fit.png

find cases -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

python generate_zno.py

echo "=== Reference zero-pressure relaxation ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.reference_relax | tee reference_relax.out

echo
echo "=== Generate scaled-volume cases ==="

python generate_eos_cases.py

echo
echo "=== Fixed-cell relaxation at each volume ==="

while IFS=, read -r case volume_ratio linear_scale initial_volume reference_volume; do
    if [[ "$case" == "case" ]]; then
        continue
    fi

    case_dir="$(readlink -f "cases/$case")"
    data_file="$case_dir/data.scaled"
    output_file="$case_dir/zno_relaxed.data"
    summary_file="$case_dir/relaxation_summary.txt"

    rm -f "$case_dir/log.lammps" "$case_dir/lammps.out" "$output_file" "$summary_file"

    echo
    echo "--- $case: V/Vref=$volume_ratio ---"

    (
        cd "$case_dir"
        lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -var data_file "$data_file" -var model_file "$MODEL_FILE" -var summary_file "$summary_file" -var output_file "$output_file" -in "$INPUT_FILE" | tee lammps.out
    )
done < eos_cases.csv

echo
echo "=== Collect and fit EOS ==="

python collect_eos_results.py | tee collect.out
python fit_eos.py | tee fit.out

echo
echo "Primary results:"
echo "  $(pwd)/eos_results.csv"
echo "  $(pwd)/eos_fit_summary.txt"
echo "  $(pwd)/eos_fit.png"
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-eos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current

./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 04: ZnO equation of state

## Purpose

This test calculates a volume-energy curve for periodic wurtzite ZnO and fits a
third-order Birch-Murnaghan equation of state.

## Procedure

1. Generate a 72-atom wurtzite ZnO supercell.
2. Relax the reference structure, including atomic coordinates and cell dimensions,
   to approximately zero pressure.
3. Isotropically scale the relaxed reference cell to seven volume ratios:

   ```text
   0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06
   ```

4. At each fixed cell volume, relax atomic coordinates only.
5. Collect total energy, pressure, volume, and residual maximum force.
6. Fit a third-order Birch-Murnaghan equation of state.

## Main outputs

- `reference_summary.txt`
- `eos_cases.csv`
- `eos_results.csv`
- `eos_fit_summary.txt`
- `eos_fitted_curve.csv`
- `eos_fit.png`
- `cases/*/zno_relaxed.data`
- `cases/*/relaxation_summary.txt`

## Run

Interactive:

```bash
./run.sh
```

Slurm:

```bash
sbatch run.slurm
```

## Interpretation

The fitted quantities are:

- `E0`: equilibrium total energy
- `V0`: equilibrium volume
- `B0`: bulk modulus
- `B0 prime`: pressure derivative of the bulk modulus

This test evaluates an isotropic volume path around the fully relaxed reference
structure. The cell shape and `c/a` ratio are held fixed along the EOS path, while
internal atomic coordinates are relaxed at every volume.
EOF

for ratio in 0.94 0.96 0.98 1.00 1.02 1.04 1.06; do
    mkdir -p "$CASE_ROOT/template_ratio_$ratio"
    cat > "$CASE_ROOT/template_ratio_$ratio/README.txt" <<EOF
Placeholder for EOS case V/Vref = $ratio.

The numerical data file is generated by generate_eos_cases.py after the
zero-pressure reference relaxation has completed.
EOF
done

python generate_zno.py

echo "Generated $TEST_DIR"
