#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${1:-$SUITE_DIR/work}"
MODEL_DIR="${MODEL_DIR:-$SUITE_DIR/models}"

MLIAP_MODEL="$MODEL_DIR/mace-osaka26-small.model-mliap_lammps.pt"
TEST07_DIR="$WORK_ROOT/07_water_adsorption"
SOURCE_STRUCTURE="$TEST07_DIR/cases/01_zn_top_o_down/adsorbed_relaxed.data"
SOURCE_GROUPS="$TEST07_DIR/cases/01_zn_top_o_down/groups.inc"

if [[ ! -f "$MLIAP_MODEL" ]]; then
    echo "ERROR: required model not found: $MLIAP_MODEL" >&2
    exit 1
fi

if [[ ! -f "$SOURCE_STRUCTURE" ]]; then
    echo "ERROR: Test 07 relaxed adsorption structure not found:" >&2
    echo "  $SOURCE_STRUCTURE" >&2
    echo "Run Test 07 successfully before generating Test 08." >&2
    exit 1
fi

if [[ ! -f "$SOURCE_GROUPS" ]]; then
    echo "ERROR: Test 07 fixed-layer definition not found:" >&2
    echo "  $SOURCE_GROUPS" >&2
    exit 1
fi

TEST_DIR="$WORK_ROOT/08_water_adsorption_md_300K"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt
cp "$SOURCE_STRUCTURE" initial_adsorbed_relaxed.data
cp "$SOURCE_GROUPS" groups.inc

cat > in.prepare_velocities <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on

read_data initial_adsorbed_relaxed.data

mass 1 65.38
mass 2 15.999
mass 3 1.008

include groups.inc

velocity all set 0.0 0.0 0.0
velocity mobile create 300.0 20260801 mom yes rot yes dist gaussian
velocity fixed set 0.0 0.0 0.0

write_data initial_adsorbed_300K.data
LAMMPS_EOF

cat > in.md_300K <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on

read_data initial_adsorbed_300K.data

mass 1 65.38
mass 2 15.999
mass 3 1.008

pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O H

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

include groups.inc

compute mobile_temp mobile temp
compute sys_pe all pe
compute sys_ke all ke

variable sys_etotal equal c_sys_pe+c_sys_ke

timestep 0.0005

fix thermostat mobile nvt temp 300.0 300.0 0.1
fix_modify thermostat temp mobile_temp

thermo 100
thermo_style custom step time c_mobile_temp c_sys_pe c_sys_ke v_sys_etotal
thermo_modify format float %20.12e
thermo_modify lost error
thermo_modify flush yes

fix equil_stats all ave/time 100 1 100 c_mobile_temp c_sys_pe c_sys_ke v_sys_etotal file equilibration_thermo.dat

run 20000

unfix equil_stats
write_data equilibrated_300K.data nocoeff
write_restart equilibrated_300K.restart

reset_timestep 0

dump production all custom 100 production.dump id type element x y z vx vy vz
dump_modify production element Zn O H
dump_modify production sort id
dump_modify production format float %20.12e

fix production_stats all ave/time 100 1 100 c_mobile_temp c_sys_pe c_sys_ke v_sys_etotal file production_thermo.dat

run 80000

unfix production_stats
undump production

write_data final_300K.data nocoeff
write_restart final_300K.restart

variable final_temp equal c_mobile_temp
variable final_pe equal c_sys_pe
variable final_ke equal c_sys_ke
variable final_etotal equal c_sys_pe+c_sys_ke

print "FINAL_TEMPERATURE = $(v_final_temp:%.16e) K" file md_summary.txt screen yes
print "FINAL_PE = $(v_final_pe:%.16e) eV" append md_summary.txt screen yes
print "FINAL_KE = $(v_final_ke:%.16e) eV" append md_summary.txt screen yes
print "FINAL_ETOTAL = $(v_final_etotal:%.16e) eV" append md_summary.txt screen yes
print "TIMESTEP_FS = 0.5" append md_summary.txt screen yes
print "EQUILIBRATION_PS = 10.0" append md_summary.txt screen yes
print "PRODUCTION_PS = 40.0" append md_summary.txt screen yes
LAMMPS_EOF

cat > analyze_md.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import re

import numpy as np


SLAB_ATOM_COUNT = 160
WATER_O_ID = 161
WATER_H_IDS = (162, 163)

DISSOCIATION_OH_THRESHOLD_A = 1.50
DESORPTION_MIN_CONTACT_THRESHOLD_A = 3.50
DESORPTION_O_ZN_THRESHOLD_A = 4.00


def parse_lammps_dump(path):
    with Path(path).open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())

            line = handle.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Malformed dump: NUMBER OF ATOMS missing")
            atom_count = int(handle.readline().strip())

            line = handle.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Malformed dump: BOX BOUNDS missing")

            bounds = []
            for _ in range(3):
                values = [float(value) for value in handle.readline().split()[:2]]
                bounds.append(values)

            line = handle.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError("Malformed dump: ATOMS header missing")

            columns = line.split()[2:]
            column_index = {name: index for index, name in enumerate(columns)}

            required = ("id", "type", "x", "y", "z")
            missing = [name for name in required if name not in column_index]
            if missing:
                raise ValueError(f"Missing dump columns: {missing}")

            ids = np.empty(atom_count, dtype=int)
            types = np.empty(atom_count, dtype=int)
            positions = np.empty((atom_count, 3), dtype=float)

            for row in range(atom_count):
                values = handle.readline().split()
                ids[row] = int(values[column_index["id"]])
                types[row] = int(values[column_index["type"]])
                positions[row, 0] = float(values[column_index["x"]])
                positions[row, 1] = float(values[column_index["y"]])
                positions[row, 2] = float(values[column_index["z"]])

            order = np.argsort(ids)
            yield {
                "timestep": timestep,
                "ids": ids[order],
                "types": types[order],
                "positions": positions[order],
                "bounds": np.asarray(bounds, dtype=float),
            }


def minimum_image_displacement(position_a, position_b, bounds):
    displacement = np.asarray(position_b) - np.asarray(position_a)

    for axis in (0, 1):
        length = bounds[axis, 1] - bounds[axis, 0]
        displacement[axis] -= length * np.rint(displacement[axis] / length)

    return displacement


def distance(position_a, position_b, bounds):
    return float(
        np.linalg.norm(
            minimum_image_displacement(position_a, position_b, bounds)
        )
    )


def parse_thermo(path):
    rows = []

    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            values = stripped.split()

            if len(values) < 5:
                continue

            try:
                step = int(float(values[0]))
                temperature = float(values[1])
                potential = float(values[2])
                kinetic = float(values[3])
                total = float(values[4])
            except ValueError:
                continue

            rows.append(
                {
                    "step": step,
                    "temperature_K": temperature,
                    "potential_energy_eV": potential,
                    "kinetic_energy_eV": kinetic,
                    "total_energy_eV": total,
                }
            )

    if not rows:
        raise ValueError(f"No thermo rows found in {path}")

    return rows


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def main():
    trajectory_rows = []

    for frame in parse_lammps_dump("production.dump"):
        ids = frame["ids"]
        types = frame["types"]
        positions = frame["positions"]
        bounds = frame["bounds"]

        id_to_index = {
            int(atom_id): index
            for index, atom_id in enumerate(ids)
        }

        water_o = positions[id_to_index[WATER_O_ID]]
        water_h_positions = [
            positions[id_to_index[atom_id]]
            for atom_id in WATER_H_IDS
        ]

        oh_distances = [
            distance(water_o, hydrogen, bounds)
            for hydrogen in water_h_positions
        ]

        surface_zn_indices = [
            index
            for index, (atom_id, atom_type) in enumerate(zip(ids, types))
            if atom_id <= SLAB_ATOM_COUNT and atom_type == 1
        ]
        surface_o_indices = [
            index
            for index, (atom_id, atom_type) in enumerate(zip(ids, types))
            if atom_id <= SLAB_ATOM_COUNT and atom_type == 2
        ]
        slab_indices = [
            index
            for index, atom_id in enumerate(ids)
            if atom_id <= SLAB_ATOM_COUNT
        ]

        nearest_o_zn = min(
            distance(water_o, positions[index], bounds)
            for index in surface_zn_indices
        )

        nearest_h_surface_o = min(
            distance(hydrogen, positions[index], bounds)
            for hydrogen in water_h_positions
            for index in surface_o_indices
        )

        minimum_slab_water = min(
            distance(water_position, positions[index], bounds)
            for water_position in [water_o, *water_h_positions]
            for index in slab_indices
        )

        top_surface_z = max(
            positions[index, 2]
            for index in slab_indices
        )
        water_o_height = float(water_o[2] - top_surface_z)

        trajectory_rows.append(
            {
                "step": frame["timestep"],
                "time_ps": frame["timestep"] * 0.0005,
                "water_OH1_A": oh_distances[0],
                "water_OH2_A": oh_distances[1],
                "maximum_OH_A": max(oh_distances),
                "nearest_waterO_surfaceZn_A": nearest_o_zn,
                "nearest_waterH_surfaceO_A": nearest_h_surface_o,
                "minimum_slab_water_distance_A": minimum_slab_water,
                "waterO_height_above_top_atom_A": water_o_height,
            }
        )

    if not trajectory_rows:
        raise ValueError("No frames found in production.dump")

    with Path("md_geometry_timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(trajectory_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    thermo_rows = parse_thermo("production_thermo.dat")

    with Path("md_thermo_timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(thermo_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(thermo_rows)

    temperature_mean, temperature_std = mean_std(
        [row["temperature_K"] for row in thermo_rows]
    )
    total_energy_mean, total_energy_std = mean_std(
        [row["total_energy_eV"] for row in thermo_rows]
    )

    max_oh = max(row["maximum_OH_A"] for row in trajectory_rows)
    max_o_zn = max(
        row["nearest_waterO_surfaceZn_A"]
        for row in trajectory_rows
    )
    max_min_contact = max(
        row["minimum_slab_water_distance_A"]
        for row in trajectory_rows
    )
    min_h_surface_o = min(
        row["nearest_waterH_surfaceO_A"]
        for row in trajectory_rows
    )

    dissociated_frames = sum(
        row["maximum_OH_A"] > DISSOCIATION_OH_THRESHOLD_A
        for row in trajectory_rows
    )
    desorbed_frames = sum(
        (
            row["minimum_slab_water_distance_A"]
            > DESORPTION_MIN_CONTACT_THRESHOLD_A
        )
        and (
            row["nearest_waterO_surfaceZn_A"]
            > DESORPTION_O_ZN_THRESHOLD_A
        )
        for row in trajectory_rows
    )

    failure_reasons = []

    if not math.isfinite(temperature_mean):
        failure_reasons.append("non-finite mean temperature")

    if abs(temperature_mean - 300.0) > 30.0:
        failure_reasons.append("mean production temperature differs from 300 K by more than 30 K")

    if dissociated_frames > 0:
        failure_reasons.append("O-H distance exceeded the dissociation threshold")

    if desorbed_frames > 0:
        failure_reasons.append("water satisfied the desorption criterion")

    verdict = "PASS" if not failure_reasons else "REVIEW"

    lines = [
        "300 K MD of H2O adsorbed on ZnO (10-10)",
        "=" * 43,
        "",
        f"Production frames: {len(trajectory_rows)}",
        f"Production duration [ps]: {trajectory_rows[-1]['time_ps']:.6f}",
        "",
        f"Mean temperature [K]: {temperature_mean:.8f}",
        f"Temperature standard deviation [K]: {temperature_std:.8f}",
        f"Mean total energy [eV]: {total_energy_mean:.10f}",
        f"Total-energy standard deviation [eV]: {total_energy_std:.10f}",
        "",
        f"Maximum O-H distance [A]: {max_oh:.10f}",
        f"Maximum nearest water-O/surface-Zn distance [A]: {max_o_zn:.10f}",
        f"Minimum water-H/surface-O distance [A]: {min_h_surface_o:.10f}",
        f"Maximum minimum slab-water distance [A]: {max_min_contact:.10f}",
        "",
        f"Frames exceeding O-H dissociation threshold: {dissociated_frames}",
        f"Frames satisfying desorption criterion: {desorbed_frames}",
        "",
        f"Verdict: {verdict}",
        f"Review reasons: {', '.join(failure_reasons) if failure_reasons else 'none'}",
    ]

    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("md_analysis_summary.txt").write_text(report)

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.plot(
            [row["time_ps"] for row in trajectory_rows],
            [row["water_OH1_A"] for row in trajectory_rows],
            label="O-H1",
        )
        axis.plot(
            [row["time_ps"] for row in trajectory_rows],
            [row["water_OH2_A"] for row in trajectory_rows],
            label="O-H2",
        )
        axis.axhline(
            DISSOCIATION_OH_THRESHOLD_A,
            linestyle="--",
            label="dissociation threshold",
        )
        axis.set_xlabel("Time [ps]")
        axis.set_ylabel("O-H distance [A]")
        axis.legend()
        figure.tight_layout()
        figure.savefig("md_water_OH_distances.png", dpi=200)
        plt.close(figure)

        figure, axis = plt.subplots()
        axis.plot(
            [row["time_ps"] for row in trajectory_rows],
            [
                row["nearest_waterO_surfaceZn_A"]
                for row in trajectory_rows
            ],
            label="water O - surface Zn",
        )
        axis.plot(
            [row["time_ps"] for row in trajectory_rows],
            [
                row["nearest_waterH_surfaceO_A"]
                for row in trajectory_rows
            ],
            label="water H - surface O",
        )
        axis.set_xlabel("Time [ps]")
        axis.set_ylabel("Nearest distance [A]")
        axis.legend()
        figure.tight_layout()
        figure.savefig("md_adsorption_distances.png", dpi=200)
        plt.close(figure)

        figure, axis = plt.subplots()
        axis.plot(
            [
                row["step"] * 0.0005
                for row in thermo_rows
            ],
            [row["temperature_K"] for row in thermo_rows],
        )
        axis.axhline(300.0, linestyle="--")
        axis.set_xlabel("Time [ps]")
        axis.set_ylabel("Mobile-group temperature [K]")
        figure.tight_layout()
        figure.savefig("md_temperature.png", dpi=200)
        plt.close(figure)

    except ImportError:
        print("matplotlib is not installed; plots were not created.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze_md.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

rm -f log.lammps prepare_velocities.out md.out initial_adsorbed_300K.data equilibration_thermo.dat production_thermo.dat production.dump equilibrated_300K.data equilibrated_300K.restart final_300K.data final_300K.restart md_summary.txt md_geometry_timeseries.csv md_thermo_timeseries.csv md_analysis_summary.txt md_water_OH_distances.png md_adsorption_distances.png md_temperature.png analysis.out

echo "=== CPU preprocessing: initialize 300 K velocities ==="
lmp -in in.prepare_velocities | tee prepare_velocities.out

echo
echo "=== GPU/KOKKOS production MD ==="
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.md_300K | tee md.out

python analyze_md.py | tee analysis.out
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-h2o-md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current

./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 08: 300 K MD of H2O adsorbed on ZnO (10-10)

## Initial structure

The test uses the lowest-energy adsorption structure from Test 07:

```text
../07_water_adsorption/cases/01_zn_top_o_down/adsorbed_relaxed.data
```

Test 07 must be completed before this generator is run.

## MD conditions

- ensemble: NVT
- thermostat: Nose-Hoover
- target temperature: 300 K
- timestep: 0.5 fs
- equilibration: 10 ps
- production: 40 ps
- production trajectory interval: 50 fs
- bottom two geometric slab layers: fixed
- mobile group: remaining ZnO atoms plus H2O
- velocity initialization: CPU LAMMPS preprocessing
- force evaluation and MD integration: GPU/KOKKOS
- periodicity: x and y periodic, z nonperiodic


## Velocity-initialization implementation

Velocity initialization is performed in a separate CPU-only LAMMPS preprocessing
step:

```text
in.prepare_velocities
```

This creates:

```text
initial_adsorbed_300K.data
```

The GPU/KOKKOS MD input then reads that prepared data file. This separation avoids
a confirmed CUDA illegal-memory-access failure in the current LAMMPS/KOKKOS build
during `velocity mobile create`.

No interatomic potential is needed during the CPU preprocessing step.

## Why 0.5 fs?

The system contains explicit O-H stretching modes. A 0.5 fs timestep is used as
a conservative validation setting.

## Thermodynamic time series

`fix ave/time` cannot take the bare thermo keywords `pe`, `ke`, or `etotal`
directly. Those keywords also cannot be evaluated inside an equal-style variable
unless the thermo output itself lists them (which is what initializes the
underlying thermo compute). Potential and kinetic energy are therefore provided
by dedicated `compute pe` and `compute ke` computes, total energy is formed from
them with an equal-style variable, and all three are logged with `fix ave/time`.
Referencing the computes directly in `fix ave/time` also guarantees the potential
energy is tallied on every sampling step.

Pressure is intentionally omitted because this is a slab with a large vacuum
region and a nonperiodic z direction.

## Automatic analysis

The production trajectory is analyzed for:

- mean and standard deviation of temperature,
- total-energy fluctuations,
- both intramolecular O-H distances,
- nearest water-O/surface-Zn distance,
- nearest water-H/surface-O distance,
- minimum slab-water distance,
- possible H2O dissociation,
- possible desorption.

The default review thresholds are:

```text
O-H dissociation threshold: 1.50 A
minimum slab-water desorption threshold: 3.50 A
water-O/surface-Zn desorption threshold: 4.00 A
```

These are diagnostic thresholds, not rigorous reaction definitions.

## Main outputs

- `prepare_velocities.out`
- `initial_adsorbed_300K.data`
- `production.dump`
- `production_thermo.dat`
- `equilibrated_300K.data`
- `final_300K.data`
- `md_geometry_timeseries.csv`
- `md_thermo_timeseries.csv`
- `md_analysis_summary.txt`
- `md_water_OH_distances.png`
- `md_adsorption_distances.png`
- `md_temperature.png`

## Run

```bash
sbatch run.slurm
```
EOF

echo "Generated $TEST_DIR"
