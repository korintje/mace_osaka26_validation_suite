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

TEST_DIR="$WORK_ROOT/07_water_adsorption"
CASE_ROOT="$TEST_DIR/cases"

mkdir -p "$TEST_DIR" "$CASE_ROOT"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_systems.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule, surface
from ase.io import write


CASE_ROOT = Path("cases")
SURFACE_LAYERS = 10
VACUUM_A = 20.0
IN_PLANE_REPEAT = (2, 2, 1)
FIXED_GEOMETRIC_LAYERS = 2


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


def build_clean_slab():
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
        layers=SURFACE_LAYERS,
        vacuum=VACUUM_A,
        periodic=True,
    )

    slab = slab.repeat(IN_PLANE_REPEAT)
    slab.wrap()

    symbols = slab.get_chemical_symbols()

    if symbols.count("Zn") != symbols.count("O"):
        raise RuntimeError("Generated slab is not stoichiometric")

    return slab


def fixed_region_upper(slab):
    z = slab.positions[:, 2]
    layers = cluster_layers(z)

    fixed_indices = sorted(
        atom_index
        for layer in layers[:FIXED_GEOMETRIC_LAYERS]
        for atom_index in layer["indices"]
    )

    fixed_set = set(fixed_indices)
    fixed_z_max = max(float(z[index]) for index in fixed_indices)
    next_mobile_z = min(
        float(z[index])
        for index in range(len(slab))
        if index not in fixed_set
    )

    upper = 0.5 * (fixed_z_max + next_mobile_z)

    return upper, fixed_indices, layers


def center_xy(position, cell):
    result = np.array(position, dtype=float)
    result[0] = result[0] % cell.lengths()[0]
    result[1] = result[1] % cell.lengths()[1]
    return result


def top_indices_by_symbol(slab, symbol, tolerance=0.8):
    symbols = np.array(slab.get_chemical_symbols())
    indices = np.where(symbols == symbol)[0]
    top_z = slab.positions[indices, 2].max()

    return [
        int(index)
        for index in indices
        if top_z - slab.positions[index, 2] <= tolerance
    ]


def nearest_xy_to_center(slab, indices):
    center = 0.5 * (slab.cell[0] + slab.cell[1])
    center_xy = center[:2]

    return min(
        indices,
        key=lambda index: np.linalg.norm(slab.positions[index, :2] - center_xy),
    )


def make_water_o_down(target_position, height=2.15, azimuth_deg=0.0, tilt_deg=15.0):
    water = molecule("H2O")
    oxygen_index = water.get_chemical_symbols().index("O")
    water.translate(-water.positions[oxygen_index])

    # ASE H2O initially lies in a Cartesian plane. Tilt and rotate to avoid
    # artificial alignment with the surface lattice.
    water.rotate(tilt_deg, "x", center=(0.0, 0.0, 0.0))
    water.rotate(azimuth_deg, "z", center=(0.0, 0.0, 0.0))

    target = np.array(target_position, dtype=float)
    target[2] += height
    water.translate(target)

    return water


def make_water_h_down(target_position, h_surface_distance=1.75):
    water = molecule("H2O")
    symbols = water.get_chemical_symbols()
    oxygen_index = symbols.index("O")
    hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol == "H"]

    water.translate(-water.positions[hydrogen_indices[0]])

    # Point the selected O-H bond approximately along +z so that the chosen
    # hydrogen is closest to the surface and oxygen points away.
    bond_vector = water.positions[oxygen_index] - water.positions[hydrogen_indices[0]]
    bond_vector /= np.linalg.norm(bond_vector)
    target_vector = np.array([0.0, 0.0, 1.0])

    axis = np.cross(bond_vector, target_vector)
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1.0e-12:
        axis /= axis_norm
        angle = np.degrees(np.arccos(np.clip(np.dot(bond_vector, target_vector), -1.0, 1.0)))
        water.rotate(angle, axis, center=(0.0, 0.0, 0.0))

    target = np.array(target_position, dtype=float)
    target[2] += h_surface_distance
    water.translate(target)

    return water


def write_slab_groups(path, fixed_upper):
    path.write_text(
        "\n".join(
            [
                f"region fixed_region block INF INF INF INF INF {fixed_upper:.12f} units box",
                "group fixed region fixed_region",
                "group mobile subtract all fixed",
                "fix freeze fixed setforce 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )


def minimum_distance_between_sets(atoms, first_indices, second_indices):
    minimum = np.inf

    for i in first_indices:
        for j in second_indices:
            distance = atoms.get_distance(i, j, mic=True)
            minimum = min(minimum, distance)

    return float(minimum)


def main():
    CASE_ROOT.mkdir(exist_ok=True)

    slab = build_clean_slab()
    fixed_upper, fixed_indices, layers = fixed_region_upper(slab)

    write(
        "clean_slab.data",
        slab,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Zn", "O"],
    )
    write_slab_groups(Path("clean_slab_groups.inc"), fixed_upper)

    gas_water = molecule("H2O")
    gas_water.set_cell([20.0, 20.0, 20.0])
    gas_water.center()
    gas_water.set_pbc([False, False, False])

    write(
        "gas_water.data",
        gas_water,
        format="lammps-data",
        atom_style="atomic",
        specorder=["O", "H"],
    )

    top_zn_indices = top_indices_by_symbol(slab, "Zn")
    top_o_indices = top_indices_by_symbol(slab, "O")

    target_zn = nearest_xy_to_center(slab, top_zn_indices)
    target_o = nearest_xy_to_center(slab, top_o_indices)

    top_zn_position = slab.positions[target_zn].copy()
    top_o_position = slab.positions[target_o].copy()

    nearest_top_o_to_zn = min(
        top_o_indices,
        key=lambda index: slab.get_distance(target_zn, index, mic=True),
    )
    bridge_position = 0.5 * (
        slab.positions[target_zn] + slab.positions[nearest_top_o_to_zn]
    )

    configurations = [
        {
            "case": "01_zn_top_o_down",
            "description": "Water oxygen above a surface Zn site",
            "water": make_water_o_down(top_zn_position, height=2.15, azimuth_deg=20.0, tilt_deg=15.0),
            "target_surface_atom_id": target_zn + 1,
            "target_surface_element": "Zn",
        },
        {
            "case": "02_zn_o_bridge",
            "description": "Water oxygen above the midpoint of a top Zn-O pair",
            "water": make_water_o_down(bridge_position, height=2.30, azimuth_deg=75.0, tilt_deg=25.0),
            "target_surface_atom_id": target_zn + 1,
            "target_surface_element": "Zn-O bridge",
        },
        {
            "case": "03_surface_o_h_down",
            "description": "One water hydrogen directed toward a surface O site",
            "water": make_water_h_down(top_o_position, h_surface_distance=1.75),
            "target_surface_atom_id": target_o + 1,
            "target_surface_element": "O",
        },
    ]

    rows = []

    for configuration in configurations:
        case_dir = CASE_ROOT / configuration["case"]
        case_dir.mkdir(parents=True, exist_ok=True)

        system = slab.copy()
        system += configuration["water"]
        system.wrap()

        write(
            case_dir / "data.adsorbed",
            system,
            format="lammps-data",
            atom_style="atomic",
            specorder=["Zn", "O", "H"],
        )
        write_slab_groups(case_dir / "groups.inc", fixed_upper)

        slab_indices = list(range(len(slab)))
        water_indices = list(range(len(slab), len(system)))
        initial_minimum_distance = minimum_distance_between_sets(
            system,
            slab_indices,
            water_indices,
        )

        metadata = {
            "case": configuration["case"],
            "description": configuration["description"],
            "slab_atoms": len(slab),
            "water_atoms": len(configuration["water"]),
            "total_atoms": len(system),
            "target_surface_atom_id": configuration["target_surface_atom_id"],
            "target_surface_element": configuration["target_surface_element"],
            "fixed_atom_count": len(fixed_indices),
            "fixed_region_upper_A": fixed_upper,
            "initial_minimum_slab_water_distance_A": initial_minimum_distance,
            "surface_area_A2": float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))),
        }

        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        rows.append(metadata)

    with Path("adsorption_cases.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    slab_metadata = {
        "surface": "ZnO wurtzite (10-10), ASE indices (1,0,0)",
        "requested_layers": SURFACE_LAYERS,
        "requested_vacuum_A": VACUUM_A,
        "in_plane_repeat": list(IN_PLANE_REPEAT),
        "atoms": len(slab),
        "fixed_geometric_layers": FIXED_GEOMETRIC_LAYERS,
        "fixed_atom_count": len(fixed_indices),
        "fixed_region_upper_A": fixed_upper,
        "geometric_layers_detected": len(layers),
        "cell_lengths_A": [float(value) for value in slab.cell.lengths()],
        "cell_angles_deg": [float(value) for value in slab.cell.angles()],
    }

    Path("slab_metadata.json").write_text(
        json.dumps(slab_metadata, indent=2) + "\n"
    )

    print(f"Generated clean slab with {len(slab)} atoms")
    print(f"Generated {len(configurations)} water adsorption configurations")


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_systems.py

cat > in.clean_slab_relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data clean_slab.data
mass 1 65.38
mass 2 15.999
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include clean_slab_groups.inc
thermo 20
thermo_style custom step atoms pe pxx pyy pzz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-6 10000 100000
variable final_pe equal pe
variable final_fmax equal fmax
print "CLEAN_SLAB_PE = $(v_final_pe:%.16e) eV" file clean_slab_summary.txt screen yes
print "CLEAN_SLAB_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append clean_slab_summary.txt screen yes
write_data clean_slab_relaxed.data nocoeff
LAMMPS_EOF

cat > in.gas_water_relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary f f f
atom_style atomic
newton on
read_data gas_water.data
mass 1 15.999
mass 2 1.008
pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * O H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step atoms pe fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
dump trajectory all custom 1 gas_water_relaxation.dump id type element x y z fx fy fz
dump_modify trajectory element O H
dump_modify trajectory sort id
min_style cg
min_modify line quadratic dmax 0.05
minimize 0.0 1.0e-7 5000 50000
variable final_pe equal pe
variable final_fmax equal fmax
variable final_fnorm equal fnorm
print "GAS_WATER_PE = $(v_final_pe:%.16e) eV" file gas_water_summary.txt screen yes
print "GAS_WATER_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append gas_water_summary.txt screen yes
print "GAS_WATER_FNORM = $(v_final_fnorm:%.16e) eV/angstrom" append gas_water_summary.txt screen yes
write_data gas_water_relaxed.data nocoeff
LAMMPS_EOF

cat > in.adsorption_fire_prerelax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data ${data_file}
mass 1 65.38
mass 2 15.999
mass 3 1.008
pair_style mliap unified ${model_file} 0
pair_coeff * * Zn O H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include ${group_file}
thermo 20
thermo_style custom step atoms pe pxx pyy pzz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
dump trajectory all custom 20 ${dump_file} id type element x y z fx fy fz
dump_modify trajectory element Zn O H
dump_modify trajectory sort id
min_style fire
min_modify dmax 0.05
minimize 0.0 1.0e-4 1000 10000
variable stage_pe equal pe
variable stage_fmax equal fmax
variable stage_fnorm equal fnorm
print "FIRE_PRERELAX_PE = $(v_stage_pe:%.16e) eV" file ${stage_summary_file} screen yes
print "FIRE_PRERELAX_FMAX = $(v_stage_fmax:%.16e) eV/angstrom" append ${stage_summary_file} screen yes
print "FIRE_PRERELAX_FNORM = $(v_stage_fnorm:%.16e) eV/angstrom" append ${stage_summary_file} screen yes
write_data ${intermediate_file} nocoeff
LAMMPS_EOF

cat > in.adsorption_cg_final <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on
read_data ${data_file}
mass 1 65.38
mass 2 15.999
mass 3 1.008
pair_style mliap unified ${model_file} 0
pair_coeff * * Zn O H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include ${group_file}
thermo 20
thermo_style custom step atoms pe pxx pyy pzz fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
dump trajectory all custom 20 ${dump_file} id type element x y z fx fy fz
dump_modify trajectory element Zn O H
dump_modify trajectory sort id
min_style cg
min_modify line quadratic dmax 0.05
minimize 0.0 1.0e-6 20000 200000
variable final_pe equal pe
variable final_fmax equal fmax
variable final_fnorm equal fnorm
print "ADSORBED_PE = $(v_final_pe:%.16e) eV" file ${summary_file} screen yes
print "ADSORBED_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append ${summary_file} screen yes
print "ADSORBED_FNORM = $(v_final_fnorm:%.16e) eV/angstrom" append ${summary_file} screen yes
print "MINIMIZATION_FLOW = FIRE_1000_THEN_CG" append ${summary_file} screen yes
write_data ${output_file} nocoeff
LAMMPS_EOF

cat > analyze_adsorption.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import re

import numpy as np
from ase.io import read


SLAB_ATOM_COUNT = 160
EV_TO_KJ_PER_MOL = 96.4853321233


def parse_value(path, key):
    text = Path(path).read_text()
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*([-+0-9.eE]+)",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise ValueError(f"Could not find {key} in {path}")

    return float(match.group(1))


def parse_text_value(path, key, default="UNKNOWN"):
    text = Path(path).read_text()
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*(\S+)",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else default


def read_adsorbed(path):
    return read(
        path,
        format="lammps-data",
        style="atomic",
        Z_of_type={1: 30, 2: 8, 3: 1},
    )


def minimum_slab_water_distance(atoms):
    slab_indices = range(SLAB_ATOM_COUNT)
    water_indices = range(SLAB_ATOM_COUNT, len(atoms))

    minimum = np.inf
    pair = None

    for i in slab_indices:
        for j in water_indices:
            distance = atoms.get_distance(i, j, mic=True)

            if distance < minimum:
                minimum = distance
                pair = (i + 1, j + 1)

    return float(minimum), pair


def water_geometry(atoms):
    water = atoms[SLAB_ATOM_COUNT:]
    symbols = water.get_chemical_symbols()
    oxygen = symbols.index("O")
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol == "H"]

    oh1 = water.get_distance(oxygen, hydrogens[0], mic=False)
    oh2 = water.get_distance(oxygen, hydrogens[1], mic=False)

    vector1 = water.positions[hydrogens[0]] - water.positions[oxygen]
    vector2 = water.positions[hydrogens[1]] - water.positions[oxygen]
    cosine = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return float(oh1), float(oh2), float(angle)


def coordination_to_surface(atoms, cutoff=2.5):
    water_indices = range(SLAB_ATOM_COUNT, len(atoms))
    slab_indices = range(SLAB_ATOM_COUNT)
    symbols = atoms.get_chemical_symbols()

    contacts = []

    for i in slab_indices:
        for j in water_indices:
            distance = atoms.get_distance(i, j, mic=True)

            if distance < cutoff:
                contacts.append(
                    {
                        "slab_atom_id": i + 1,
                        "slab_element": symbols[i],
                        "water_atom_id": j + 1,
                        "water_element": symbols[j],
                        "distance_A": float(distance),
                    }
                )

    return contacts


def main():
    clean_slab_energy = parse_value(
        "clean_slab_summary.txt",
        "CLEAN_SLAB_PE",
    )
    gas_water_energy = parse_value(
        "gas_water_summary.txt",
        "GAS_WATER_PE",
    )

    with Path("adsorption_cases.csv").open() as handle:
        cases = list(csv.DictReader(handle))

    rows = []

    for case in cases:
        case_dir = Path("cases") / case["case"]
        adsorbed_energy = parse_value(
            case_dir / "relaxation_summary.txt",
            "ADSORBED_PE",
        )
        fmax = parse_value(
            case_dir / "relaxation_summary.txt",
            "ADSORBED_FMAX",
        )
        minimization_flow = parse_text_value(
            case_dir / "relaxation_summary.txt",
            "MINIMIZATION_FLOW",
        )

        adsorption_energy = (
            adsorbed_energy
            - clean_slab_energy
            - gas_water_energy
        )

        atoms = read_adsorbed(case_dir / "adsorbed_relaxed.data")
        minimum_distance, minimum_pair = minimum_slab_water_distance(atoms)
        oh1, oh2, hoh_angle = water_geometry(atoms)
        contacts = coordination_to_surface(atoms)

        result = {
            "case": case["case"],
            "description": case["description"],
            "adsorbed_energy_eV": adsorbed_energy,
            "clean_slab_energy_eV": clean_slab_energy,
            "gas_water_energy_eV": gas_water_energy,
            "adsorption_energy_eV": adsorption_energy,
            "adsorption_energy_kJ_per_mol": adsorption_energy * EV_TO_KJ_PER_MOL,
            "residual_fmax_eV_per_A": fmax,
            "minimization_flow": minimization_flow,
            "minimum_slab_water_distance_A": minimum_distance,
            "minimum_distance_pair_ids": str(minimum_pair),
            "water_OH1_A": oh1,
            "water_OH2_A": oh2,
            "water_HOH_angle_deg": hoh_angle,
            "surface_contact_count_below_2p5A": len(contacts),
        }
        rows.append(result)

        (case_dir / "contacts.json").write_text(
            json.dumps(contacts, indent=2) + "\n"
        )

    rows.sort(key=lambda row: row["adsorption_energy_eV"])

    with Path("adsorption_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]

    lines = [
        "H2O adsorption on ZnO (10-10)",
        "=" * 32,
        "",
        f"Clean slab energy [eV]: {clean_slab_energy:.16e}",
        f"Gas-phase H2O energy [eV]: {gas_water_energy:.16e}",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                row["case"],
                "-" * len(row["case"]),
                f"Description: {row['description']}",
                f"Adsorption energy [eV]: {row['adsorption_energy_eV']:.10f}",
                f"Adsorption energy [kJ/mol]: {row['adsorption_energy_kJ_per_mol']:.6f}",
                f"Residual Fmax [eV/A]: {row['residual_fmax_eV_per_A']:.6e}",
                f"Minimization flow: {row['minimization_flow']}",
                f"Minimum slab-water distance [A]: {row['minimum_slab_water_distance_A']:.10f}",
                f"Water O-H distances [A]: {row['water_OH1_A']:.10f}, {row['water_OH2_A']:.10f}",
                f"Water H-O-H angle [deg]: {row['water_HOH_angle_deg']:.10f}",
                f"Surface contacts below 2.5 A: {row['surface_contact_count_below_2p5A']}",
                "",
            ]
        )

    failure_reasons = []

    if not np.isfinite(best["adsorption_energy_eV"]):
        failure_reasons.append("non-finite adsorption energy")

    if best["minimum_slab_water_distance_A"] < 0.8:
        failure_reasons.append("unphysically short slab-water distance")

    if max(best["water_OH1_A"], best["water_OH2_A"]) > 1.5:
        failure_reasons.append("water dissociated or O-H bond became excessively long")

    if best["residual_fmax_eV_per_A"] > 1.0e-4:
        failure_reasons.append("insufficient force convergence")

    verdict = "PASS" if not failure_reasons else "REVIEW"

    lines.extend(
        [
            "Lowest-energy configuration",
            "---------------------------",
            f"Case: {best['case']}",
            f"Adsorption energy [eV]: {best['adsorption_energy_eV']:.10f}",
            f"Adsorption energy [kJ/mol]: {best['adsorption_energy_kJ_per_mol']:.6f}",
            "",
            f"Verdict: {verdict}",
            f"Review reasons: {', '.join(failure_reasons) if failure_reasons else 'none'}",
        ]
    )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("adsorption_summary.txt").write_text(report)

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.bar(
            [row["case"] for row in rows],
            [row["adsorption_energy_eV"] for row in rows],
        )
        axis.axhline(0.0, linewidth=1.0)
        axis.set_ylabel("Adsorption energy [eV]")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig("adsorption_energies.png", dpi=200)
        plt.close(figure)

    except ImportError:
        print("matplotlib is not installed; adsorption_energies.png was not created.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze_adsorption.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MODEL_FILE="$(readlink -f mace-osaka26-small.model-mliap_lammps.pt)"
ADSORPTION_FIRE_INPUT="$(readlink -f in.adsorption_fire_prerelax)"
ADSORPTION_CG_INPUT="$(readlink -f in.adsorption_cg_final)"

rm -f clean_slab_summary.txt clean_slab_relaxed.data clean_slab.out gas_water_summary.txt gas_water_relaxed.data gas_water_relaxation.dump gas_water.out adsorption_results.csv adsorption_summary.txt adsorption_energies.png

find cases -mindepth 1 -maxdepth 1 -type d -exec rm -f {}/relaxation_summary.txt {}/adsorbed_relaxed.data {}/adsorbed_fire_prerelaxed.data {}/fire_prerelax_summary.txt {}/fire_prerelax.dump {}/cg_final.dump {}/lammps_fire_prerelax.out {}/lammps_cg_final.out {}/lammps.out {}/contacts.json \;

python generate_systems.py

echo "=== Clean-slab reference relaxation ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.clean_slab_relax | tee clean_slab.out

echo
echo "=== Gas-phase water relaxation with LAMMPS GPU/CG ==="

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.gas_water_relax | tee gas_water.out

echo
echo "=== Adsorption-configuration relaxations ==="

while IFS=, read -r case description slab_atoms water_atoms total_atoms target_id target_element fixed_count fixed_upper initial_distance area; do
    if [[ "$case" == "case" ]]; then
        continue
    fi

    case_dir="$(readlink -f "cases/$case")"
    data_file="$case_dir/data.adsorbed"
    group_file="$case_dir/groups.inc"
    summary_file="$case_dir/relaxation_summary.txt"
    output_file="$case_dir/adsorbed_relaxed.data"
    dump_file="$case_dir/relaxation.dump"

    echo
    echo "--- $case ---"

    (
        cd "$case_dir"

        intermediate_file="$case_dir/adsorbed_fire_prerelaxed.data"
        fire_summary_file="$case_dir/fire_prerelax_summary.txt"
        fire_dump_file="$case_dir/fire_prerelax.dump"
        cg_dump_file="$case_dir/cg_final.dump"

        rm -f "$intermediate_file" "$fire_summary_file" "$fire_dump_file" "$cg_dump_file" "$summary_file" "$output_file"

        echo "Stage 1/2: FIRE pre-relaxation, maximum 1000 iterations"

        lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -var data_file "$data_file" -var model_file "$MODEL_FILE" -var group_file "$group_file" -var stage_summary_file "$fire_summary_file" -var intermediate_file "$intermediate_file" -var dump_file "$fire_dump_file" -in "$ADSORPTION_FIRE_INPUT" 2>&1 | tee lammps_fire_prerelax.out

        if [[ ! -s "$intermediate_file" ]]; then
            echo "ERROR: FIRE pre-relaxation did not produce an intermediate structure." >&2
            exit 1
        fi

        if grep -Eqi "non-numeric|nan|simulation unstable" lammps_fire_prerelax.out; then
            echo "ERROR: FIRE pre-relaxation produced non-numeric output." >&2
            exit 1
        fi

        echo "Stage 2/2: CG final relaxation from FIRE-pre-relaxed structure"

        lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -var data_file "$intermediate_file" -var model_file "$MODEL_FILE" -var group_file "$group_file" -var summary_file "$summary_file" -var output_file "$output_file" -var dump_file "$cg_dump_file" -in "$ADSORPTION_CG_INPUT" 2>&1 | tee lammps_cg_final.out

        cp lammps_cg_final.out lammps.out
    )
done < adsorption_cases.csv

echo
echo "=== Adsorption-energy and geometry analysis ==="

python analyze_adsorption.py | tee analysis.out

echo
echo "Primary results:"
echo "  $(pwd)/adsorption_results.csv"
echo "  $(pwd)/adsorption_summary.txt"
echo "  $(pwd)/adsorption_energies.png"
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-h2o
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
# Test 07: H2O adsorption on ZnO (10-10)

## Purpose

This test introduces a typical small-molecule adsorption calculation after the
clean-surface validation tests.

The adsorbate is one water molecule on a stoichiometric nonpolar wurtzite ZnO
(10-10) slab.

## Surface model

- 10 requested slab layers
- 2 x 2 in-plane repeat
- approximately 20 A vacuum
- two lowest geometric atomic layers fixed
- all other slab atoms and all H2O atoms relaxed
- fixed simulation cell

## Initial adsorption configurations

1. `01_zn_top_o_down`
   - water O above a surface Zn atom

2. `02_zn_o_bridge`
   - water O above a surface Zn-O bridge region

3. `03_surface_o_h_down`
   - one water H directed toward a surface O atom

The three configurations are relaxed independently.


## Minimization strategy

The test uses a fixed two-stage efficiency-oriented workflow:

- clean ZnO slab: FIRE,
- isolated gas-phase H2O: CG,
- each ZnO + H2O adsorption system:
  1. FIRE pre-relaxation for at most 1000 iterations,
  2. write an intermediate structure,
  3. CG final relaxation starting from that intermediate structure.

FIRE is used only to remove large initial forces efficiently. CG is always used
for the final convergence.

The following intermediate files are retained for inspection:

- `adsorbed_fire_prerelaxed.data`
- `fire_prerelax_summary.txt`
- `fire_prerelax.dump`
- `cg_final.dump`

The final summary records:

```text
MINIMIZATION_FLOW = FIRE_1000_THEN_CG
```

## Adsorption energy

```text
E_ads = E_slab+H2O - E_clean_slab - E_gas_H2O
```

Negative values indicate exothermic adsorption.

The clean slab and adsorbed systems use the same slab geometry and fixed-layer
definition. Gas-phase H2O is relaxed independently with LAMMPS ML-IAP using CG in a nonperiodic 20 A box.

## Main outputs

- `clean_slab_summary.txt`
- `gas_water_summary.txt`
- `adsorption_cases.csv`
- `adsorption_results.csv`
- `adsorption_summary.txt`
- `adsorption_energies.png`
- `cases/*/adsorbed_fire_prerelaxed.data`
- `cases/*/adsorbed_relaxed.data`
- `cases/*/contacts.json`

## Validation checks

The analysis reports:

- adsorption energy,
- residual maximum force,
- shortest slab-water distance,
- relaxed O-H distances,
- relaxed H-O-H angle,
- number of slab-water contacts below 2.5 A.

The test flags configurations with non-finite energies, severe atom overlap,
poor force convergence, or strong O-H bond cleavage.

## Scope and limitations

This test is intended to confirm that a standard molecular adsorption workflow
runs stably with MACE-Osaka26.

It does not establish quantitative accuracy against DFT. The next scientific
validation should compare adsorption structures and energies with PBE-D3(BJ)
calculations for selected configurations.
EOF

python generate_systems.py

echo "Generated $TEST_DIR"
