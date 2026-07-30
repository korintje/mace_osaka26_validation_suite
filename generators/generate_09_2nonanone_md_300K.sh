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

TEST_DIR="$WORK_ROOT/09_2nonanone_md_300K"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

ln -sfn "$MLIAP_MODEL" mace-osaka26-small.model-mliap_lammps.pt

cat > generate_system.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.io import write


SURFACE_LAYERS = 10
VACUUM_A = 24.0
IN_PLANE_REPEAT = (4, 4, 1)
FIXED_GEOMETRIC_LAYERS = 2
CARBONYL_O_SURFACE_ZN_DISTANCE_A = 2.25


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


def build_slab():
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


def place_atom(a, b, c, bond_length, angle_deg, dihedral_deg):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    axis = a - b
    axis /= np.linalg.norm(axis)
    normal = np.cross(b - c, axis)
    normal /= np.linalg.norm(normal)
    in_plane = np.cross(normal, axis)
    theta = np.radians(angle_deg)
    phi = np.radians(dihedral_deg)
    direction = (
        -np.cos(theta) * axis
        + np.sin(theta) * (np.cos(phi) * in_plane + np.sin(phi) * normal)
    )
    return a + bond_length * direction


def perpendicular_basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, trial)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def terminal_hydrogen_directions(neighbor_vector):
    neighbor = np.asarray(neighbor_vector, dtype=float)
    neighbor /= np.linalg.norm(neighbor)
    first, second = perpendicular_basis(neighbor)
    axial = -1.0 / 3.0
    radial = np.sqrt(1.0 - axial * axial)
    directions = []
    for azimuth_deg in (0.0, 120.0, 240.0):
        azimuth = np.radians(azimuth_deg)
        directions.append(
            axial * neighbor
            + radial * (np.cos(azimuth) * first + np.sin(azimuth) * second)
        )
    return directions


def methylene_hydrogen_directions(first_neighbor, second_neighbor):
    first = np.asarray(first_neighbor, dtype=float)
    second = np.asarray(second_neighbor, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.dot(first, second))
    coefficient = (-1.0 / 3.0) / (1.0 + dot)
    in_plane = coefficient * (first + second)
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    normal_weight = np.sqrt(max(0.0, 1.0 - np.dot(in_plane, in_plane)))
    return [in_plane + normal_weight * normal, in_plane - normal_weight * normal]


def build_2nonanone():
    # Heavy-atom order: C1-C2(=O)-C3-C4-C5-C6-C7-C8-C9, then O.
    carbons = [
        np.array([-1.30, -0.75, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.30, -0.75, 0.0]),
    ]
    for index in range(3, 9):
        dihedral = 180.0
        carbons.append(
            place_atom(
                carbons[index - 1],
                carbons[index - 2],
                carbons[index - 3],
                1.53,
                112.0,
                dihedral,
            )
        )
    oxygen = np.array([0.0, 1.23, 0.0])

    symbols = ["C"] * 9 + ["O"]
    positions = [position.copy() for position in carbons] + [oxygen]
    carbon_neighbors = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 8],
        8: [7],
    }

    for carbon_index in range(9):
        center = carbons[carbon_index]
        neighbors = carbon_neighbors[carbon_index]
        if carbon_index == 1:
            continue
        if len(neighbors) == 1:
            directions = terminal_hydrogen_directions(
                carbons[neighbors[0]] - center
            )
            bond_length = 1.09
        else:
            directions = methylene_hydrogen_directions(
                carbons[neighbors[0]] - center,
                carbons[neighbors[1]] - center,
            )
            bond_length = 1.09
        for direction in directions:
            symbols.append("H")
            positions.append(center + bond_length * direction)

    molecule = Atoms(symbols=symbols, positions=positions)
    counts = {
        symbol: molecule.get_chemical_symbols().count(symbol)
        for symbol in ("C", "H", "O")
    }
    if counts != {"C": 9, "H": 18, "O": 1}:
        raise RuntimeError(f"Unexpected 2-nonanone composition: {counts}")
    return molecule


def rotation_matrix_from_vectors(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1.0e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


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
        float(z[index]) for index in range(len(slab)) if index not in fixed_set
    )
    return 0.5 * (fixed_z_max + next_mobile_z), fixed_indices, layers


def nearest_top_zn_to_center(slab, tolerance=0.8):
    symbols = np.asarray(slab.get_chemical_symbols())
    zn_indices = np.where(symbols == "Zn")[0]
    top_z = slab.positions[zn_indices, 2].max()
    top_indices = [
        int(index)
        for index in zn_indices
        if top_z - slab.positions[index, 2] <= tolerance
    ]
    center_xy = 0.5 * (slab.cell[0] + slab.cell[1])[:2]
    return min(
        top_indices,
        key=lambda index: np.linalg.norm(slab.positions[index, :2] - center_xy),
    )


def minimum_slab_molecule_distance(system, slab_atom_count):
    minimum = np.inf
    for slab_index in range(slab_atom_count):
        for molecule_index in range(slab_atom_count, len(system)):
            minimum = min(
                minimum,
                system.get_distance(slab_index, molecule_index, mic=True),
            )
    return float(minimum)


def main():
    slab = build_slab()
    molecule = build_2nonanone()
    slab_atom_count = len(slab)
    fixed_upper, fixed_indices, layers = fixed_region_upper(slab)
    target_zn = nearest_top_zn_to_center(slab)
    target = slab.positions[target_zn].copy()

    # Put the carbonyl oxygen down over a surface Zn. The chain is tilted
    # slightly away from the surface and aligned with the longer cell vector.
    carbonyl_vector = molecule.positions[9] - molecule.positions[1]
    target_vector = np.array([0.22, 0.0, -1.0])
    rotation = rotation_matrix_from_vectors(carbonyl_vector, target_vector)
    molecule.positions = (rotation @ molecule.positions.T).T
    molecule.rotate(18.0, "z", center=(0.0, 0.0, 0.0))
    oxygen_position = molecule.positions[9].copy()
    adsorption_site = target.copy()
    adsorption_site[2] += CARBONYL_O_SURFACE_ZN_DISTANCE_A
    molecule.translate(adsorption_site - oxygen_position)

    system = slab.copy()
    system += molecule
    system.wrap()
    write(
        "initial_adsorbed.data",
        system,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Zn", "O", "C", "H"],
    )
    Path("groups.inc").write_text(
        "\n".join(
            [
                f"region fixed_region block INF INF INF INF INF {fixed_upper:.12f} units box",
                "group fixed region fixed_region",
                "group mobile subtract all fixed",
                f"group molecule id {slab_atom_count + 1}:{len(system)}",
                "fix freeze fixed setforce 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    molecule_ids = {
        f"C{index + 1}": slab_atom_count + index + 1
        for index in range(9)
    }
    molecule_ids["O_carbonyl"] = slab_atom_count + 10
    metadata = {
        "molecule": "2-nonanone",
        "formula": "C9H18O",
        "surface": "ZnO wurtzite (10-10), ASE indices (1,0,0)",
        "requested_layers": SURFACE_LAYERS,
        "requested_vacuum_A": VACUUM_A,
        "in_plane_repeat": list(IN_PLANE_REPEAT),
        "slab_atoms": slab_atom_count,
        "molecule_atoms": len(molecule),
        "total_atoms": len(system),
        "fixed_geometric_layers": FIXED_GEOMETRIC_LAYERS,
        "fixed_atom_count": len(fixed_indices),
        "fixed_region_upper_A": fixed_upper,
        "geometric_layers_detected": len(layers),
        "surface_area_A2": float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))),
        "cell_lengths_A": [float(value) for value in slab.cell.lengths()],
        "target_surface_zn_id": target_zn + 1,
        "initial_carbonyl_O_surface_Zn_distance_A": CARBONYL_O_SURFACE_ZN_DISTANCE_A,
        "initial_minimum_slab_molecule_distance_A": minimum_slab_molecule_distance(
            system, slab_atom_count
        ),
        "molecule_atom_ids": molecule_ids,
        "backbone_dihedrals": [
            ["C1", "C2", "C3", "C4"],
            ["C2", "C3", "C4", "C5"],
            ["C3", "C4", "C5", "C6"],
            ["C4", "C5", "C6", "C7"],
            ["C5", "C6", "C7", "C8"],
            ["C6", "C7", "C8", "C9"],
        ],
    }
    Path("system_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write(
        "initial_adsorbed.extxyz",
        system,
        format="extxyz",
    )
    print(
        f"Generated {len(system)} atoms: {slab_atom_count}-atom 4x4 slab "
        f"plus {len(molecule)}-atom 2-nonanone"
    )


if __name__ == "__main__":
    main()
PYEOF
chmod +x generate_system.py

cat > in.relax <<'LAMMPS_EOF'
clear
units metal
dimension 3
boundary p p f
atom_style atomic
newton on

read_data initial_adsorbed.data

mass 1 65.38
mass 2 15.999
mass 3 12.011
mass 4 1.008

pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O C H

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
include groups.inc

thermo 20
thermo_style custom step atoms pe fmax fnorm
thermo_modify format float %20.12e
thermo_modify lost error
thermo_modify flush yes

dump relaxation all custom 100 relaxation.dump id type element x y z fx fy fz
dump_modify relaxation element Zn O C H
dump_modify relaxation sort id
dump_modify relaxation format float %20.12e

min_style cg
min_modify line quadratic dmax 0.02
minimize 0.0 1.0e-4 5000 50000

min_modify line quadratic dmax 0.05
minimize 0.0 1.0e-5 10000 100000

variable final_pe equal pe
variable final_fmax equal fmax
print "RELAXED_PE = $(v_final_pe:%.16e) eV" file relaxation_summary.txt screen yes
print "RELAXED_FMAX = $(v_final_fmax:%.16e) eV/angstrom" append relaxation_summary.txt screen yes
write_data initial_adsorbed_relaxed.data nocoeff
LAMMPS_EOF

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
mass 3 12.011
mass 4 1.008

include groups.inc

velocity all set 0.0 0.0 0.0
velocity mobile create 300.0 20260902 mom yes rot yes dist gaussian
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
mass 3 12.011
mass 4 1.008

pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0
pair_coeff * * Zn O C H

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

dump production all custom 100 production.dump id type element xu yu zu vx vy vz
dump_modify production element Zn O C H
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

cat > analyze_conformations.py <<'PYEOF'
#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np


TIMESTEP_PS = 0.0005
DESORPTION_CONTACT_A = 4.0


def parse_dump(path):
    with Path(path).open() as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Malformed dump: NUMBER OF ATOMS missing")
            atom_count = int(handle.readline().strip())
            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Malformed dump: BOX BOUNDS missing")
            bounds = np.asarray(
                [
                    [float(value) for value in handle.readline().split()[:2]]
                    for _ in range(3)
                ]
            )
            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError("Malformed dump: ATOMS header missing")
            columns = atoms_header.split()[2:]
            column_index = {name: index for index, name in enumerate(columns)}
            coordinate_names = (
                ("xu", "yu", "zu")
                if all(name in column_index for name in ("xu", "yu", "zu"))
                else ("x", "y", "z")
            )
            required = ("id", "type", *coordinate_names)
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
                positions[row] = [
                    float(values[column_index[name]]) for name in coordinate_names
                ]
            order = np.argsort(ids)
            yield {
                "timestep": timestep,
                "ids": ids[order],
                "types": types[order],
                "positions": positions[order],
                "bounds": bounds,
            }


def minimum_image(displacement, bounds):
    result = np.asarray(displacement, dtype=float).copy()
    for axis in (0, 1):
        length = bounds[axis, 1] - bounds[axis, 0]
        result[axis] -= length * np.rint(result[axis] / length)
    return result


def distance(first, second, bounds, periodic=True):
    displacement = np.asarray(second) - np.asarray(first)
    if periodic:
        displacement = minimum_image(displacement, bounds)
    return float(np.linalg.norm(displacement))


def dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def radius_of_gyration(positions, masses):
    center = np.average(positions, axis=0, weights=masses)
    squared = np.sum((positions - center) ** 2, axis=1)
    return float(np.sqrt(np.average(squared, weights=masses)))


def parse_thermo(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            values = line.split()
            if not values or line.lstrip().startswith("#") or len(values) < 5:
                continue
            try:
                rows.append(
                    {
                        "step": int(float(values[0])),
                        "temperature_K": float(values[1]),
                        "potential_energy_eV": float(values[2]),
                        "kinetic_energy_eV": float(values[3]),
                        "total_energy_eV": float(values[4]),
                    }
                )
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No thermo rows found in {path}")
    return rows


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def circular_statistics_deg(values):
    radians = np.radians(values)
    mean_angle = np.degrees(
        np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    )
    resultant = np.hypot(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    resultant = np.clip(resultant, 1.0e-15, 1.0)
    circular_std = np.degrees(np.sqrt(-2.0 * np.log(resultant)))
    return float(mean_angle), float(circular_std)


def conformation_class(angle):
    if abs(angle) >= 120.0:
        return "trans"
    return "gauche_plus" if angle >= 0.0 else "gauche_minus"


def main():
    metadata = json.loads(Path("system_metadata.json").read_text())
    slab_count = int(metadata["slab_atoms"])
    atom_ids = metadata["molecule_atom_ids"]
    dihedral_labels = [
        "-".join(names) for names in metadata["backbone_dihedrals"]
    ]
    masses_by_type = {2: 15.999, 3: 12.011, 4: 1.008}
    rows = []

    for frame in parse_dump("production.dump"):
        ids = frame["ids"]
        types = frame["types"]
        positions = frame["positions"]
        bounds = frame["bounds"]
        index_by_id = {int(atom_id): index for index, atom_id in enumerate(ids)}
        molecule_indices = [
            index for index, atom_id in enumerate(ids) if atom_id > slab_count
        ]
        slab_indices = [
            index for index, atom_id in enumerate(ids) if atom_id <= slab_count
        ]
        molecule_positions = positions[molecule_indices]
        molecule_masses = np.asarray(
            [masses_by_type[int(types[index])] for index in molecule_indices]
        )
        carbon_positions = {
            name: positions[index_by_id[int(atom_id)]]
            for name, atom_id in atom_ids.items()
            if name.startswith("C")
        }
        carbonyl_o = positions[index_by_id[int(atom_ids["O_carbonyl"])]]
        carbonyl_c = carbon_positions["C2"]
        carbonyl_vector = carbonyl_o - carbonyl_c
        carbonyl_tilt = np.degrees(
            np.arccos(
                np.clip(
                    abs(carbonyl_vector[2]) / np.linalg.norm(carbonyl_vector),
                    0.0,
                    1.0,
                )
            )
        )
        dihedral_values = {}
        for names, label in zip(metadata["backbone_dihedrals"], dihedral_labels):
            dihedral_values[label] = dihedral(
                *(carbon_positions[name] for name in names)
            )
        minimum_contact = min(
            distance(
                positions[molecule_index],
                positions[slab_index],
                bounds,
                periodic=True,
            )
            for molecule_index in molecule_indices
            for slab_index in slab_indices
        )
        surface_top_z = max(positions[index, 2] for index in slab_indices)
        row = {
            "step": frame["timestep"],
            "time_ps": frame["timestep"] * TIMESTEP_PS,
            "end_to_end_C1_C9_A": distance(
                carbon_positions["C1"],
                carbon_positions["C9"],
                bounds,
                periodic=False,
            ),
            "molecular_radius_of_gyration_A": radius_of_gyration(
                molecule_positions, molecule_masses
            ),
            "carbonyl_tilt_from_surface_normal_deg": float(carbonyl_tilt),
            "carbonyl_O_height_above_top_atom_A": float(
                carbonyl_o[2] - surface_top_z
            ),
            "minimum_slab_molecule_distance_A": minimum_contact,
        }
        row.update(
            {
                f"dihedral_{label}_deg": value
                for label, value in dihedral_values.items()
            }
        )
        rows.append(row)

    if not rows:
        raise ValueError("No frames found in production.dump")

    with Path("conformation_timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    thermo_rows = parse_thermo("production_thermo.dat")
    with Path("md_thermo_timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(thermo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(thermo_rows)

    temperature_mean, temperature_std = mean_std(
        [row["temperature_K"] for row in thermo_rows]
    )
    energy_mean, energy_std = mean_std(
        [row["total_energy_eV"] for row in thermo_rows]
    )
    end_mean, end_std = mean_std([row["end_to_end_C1_C9_A"] for row in rows])
    rg_mean, rg_std = mean_std(
        [row["molecular_radius_of_gyration_A"] for row in rows]
    )
    tilt_mean, tilt_std = mean_std(
        [row["carbonyl_tilt_from_surface_normal_deg"] for row in rows]
    )
    desorbed = sum(
        row["minimum_slab_molecule_distance_A"] > DESORPTION_CONTACT_A
        for row in rows
    )

    dihedral_statistics = []
    for label in dihedral_labels:
        values = [row[f"dihedral_{label}_deg"] for row in rows]
        circular_mean, circular_std = circular_statistics_deg(values)
        populations = {
            state: sum(conformation_class(value) == state for value in values)
            / len(values)
            for state in ("trans", "gauche_plus", "gauche_minus")
        }
        dihedral_statistics.append(
            {
                "dihedral": label,
                "circular_mean_deg": circular_mean,
                "circular_std_deg": circular_std,
                **{f"{key}_fraction": value for key, value in populations.items()},
            }
        )

    with Path("dihedral_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dihedral_statistics[0].keys())
        )
        writer.writeheader()
        writer.writerows(dihedral_statistics)

    review_reasons = []
    if not math.isfinite(temperature_mean):
        review_reasons.append("non-finite mean temperature")
    elif abs(temperature_mean - 300.0) > 30.0:
        review_reasons.append("mean temperature differs from 300 K by more than 30 K")
    if desorbed:
        review_reasons.append("one or more frames satisfy the desorption criterion")
    verdict = "PASS" if not review_reasons else "REVIEW"

    lines = [
        "300 K MD of 2-nonanone adsorbed on ZnO (10-10)",
        "=" * 51,
        "",
        f"Production frames: {len(rows)}",
        f"Production duration [ps]: {rows[-1]['time_ps']:.6f}",
        f"Mean temperature [K]: {temperature_mean:.8f}",
        f"Temperature standard deviation [K]: {temperature_std:.8f}",
        f"Mean total energy [eV]: {energy_mean:.10f}",
        f"Total-energy standard deviation [eV]: {energy_std:.10f}",
        "",
        f"Mean C1-C9 end-to-end distance [A]: {end_mean:.8f}",
        f"C1-C9 standard deviation [A]: {end_std:.8f}",
        f"Mean molecular radius of gyration [A]: {rg_mean:.8f}",
        f"Radius-of-gyration standard deviation [A]: {rg_std:.8f}",
        f"Mean carbonyl tilt from surface normal [deg]: {tilt_mean:.8f}",
        f"Carbonyl-tilt standard deviation [deg]: {tilt_std:.8f}",
        f"Desorbed frames (> {DESORPTION_CONTACT_A:.1f} A contact): {desorbed}",
        f"Desorbed-frame fraction: {desorbed / len(rows):.8f}",
        "",
        "Backbone dihedral populations:",
    ]
    for stats in dihedral_statistics:
        lines.append(
            "  {dihedral}: mean={circular_mean_deg:.3f} deg, "
            "circ_std={circular_std_deg:.3f} deg, "
            "trans={trans_fraction:.4f}, gauche+={gauche_plus_fraction:.4f}, "
            "gauche-={gauche_minus_fraction:.4f}".format(**stats)
        )
    lines.extend(
        [
            "",
            f"Verdict: {verdict}",
            f"Review reasons: {', '.join(review_reasons) if review_reasons else 'none'}",
        ]
    )
    report = "\n".join(lines) + "\n"
    print(report, end="")
    Path("conformation_analysis_summary.txt").write_text(report)

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        for label in dihedral_labels:
            axis.plot(
                [row["time_ps"] for row in rows],
                [row[f"dihedral_{label}_deg"] for row in rows],
                label=label,
                linewidth=0.8,
            )
        axis.set_xlabel("Time [ps]")
        axis.set_ylabel("Backbone dihedral [deg]")
        axis.set_ylim(-180.0, 180.0)
        axis.legend(fontsize="small", ncol=2)
        figure.tight_layout()
        figure.savefig("backbone_dihedrals.png", dpi=200)
        plt.close(figure)

        figure, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        bins = np.linspace(-180.0, 180.0, 37)
        for axis, label in zip(axes.flat, dihedral_labels):
            axis.hist(
                [row[f"dihedral_{label}_deg"] for row in rows],
                bins=bins,
                density=True,
            )
            axis.set_title(label, fontsize="small")
            axis.set_xlim(-180.0, 180.0)
        figure.supxlabel("Dihedral [deg]")
        figure.supylabel("Probability density")
        figure.tight_layout()
        figure.savefig("dihedral_distributions.png", dpi=200)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, sharex=True)
        time = [row["time_ps"] for row in rows]
        axes[0].plot(time, [row["end_to_end_C1_C9_A"] for row in rows])
        axes[0].set_ylabel("C1-C9 [A]")
        axes[1].plot(
            time, [row["molecular_radius_of_gyration_A"] for row in rows]
        )
        axes[1].set_ylabel("Rg [A]")
        axes[2].plot(
            time,
            [row["carbonyl_tilt_from_surface_normal_deg"] for row in rows],
        )
        axes[2].set_ylabel("C=O tilt [deg]")
        axes[2].set_xlabel("Time [ps]")
        figure.tight_layout()
        figure.savefig("conformation_metrics.png", dpi=200)
        plt.close(figure)
    except ImportError:
        print("matplotlib is not installed; plots were not created.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x analyze_conformations.py

cat > run.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

rm -f log.lammps relaxation.out relaxation.dump relaxation_summary.txt initial_adsorbed_relaxed.data prepare_velocities.out initial_adsorbed_300K.data md.out equilibration_thermo.dat production_thermo.dat production.dump equilibrated_300K.data equilibrated_300K.restart final_300K.data final_300K.restart md_summary.txt conformation_timeseries.csv md_thermo_timeseries.csv dihedral_statistics.csv conformation_analysis_summary.txt backbone_dihedrals.png dihedral_distributions.png conformation_metrics.png analysis.out

echo "=== GPU/KOKKOS CG adsorption-system relaxation ==="
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.relax | tee relaxation.out

echo
echo "=== CPU preprocessing: initialize 300 K velocities ==="
lmp -in in.prepare_velocities | tee prepare_velocities.out

echo
echo "=== GPU/KOKKOS production MD ==="
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.md_300K | tee md.out

python analyze_conformations.py | tee analysis.out
SHEOF
chmod +x run.sh

cat > run.slurm <<'SLEOF'
#!/usr/bin/env bash
#SBATCH --job-name=osaka26-zno-2nonanone-md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate /apps/envs/lammps-mace-current
./run.sh
SLEOF
chmod +x run.slurm

cat > README.md <<'EOF'
# Test 09: 300 K MD of 2-nonanone adsorbed on ZnO (10-10)

This test places one 2-nonanone molecule (`C9H18O`) carbonyl-oxygen-down over a
surface Zn site, relaxes the complete adsorption system, and performs 300 K NVT
MD.

## System

- ZnO (10-10) slab: 10 requested layers
- in-plane repeat: 4 x 4
- vacuum: 24 A
- bottom two geometric layers fixed
- adsorbate: one all-trans 2-nonanone molecule
- initial carbonyl-O/surface-Zn separation: 2.25 A
- periodicity: x and y periodic, z nonperiodic

The 4 x 4 surface has four times the area of the 2 x 2 H2O adsorption slab and
reduces interactions between periodic images of the extended hydrocarbon chain.

## Calculation

1. Two-stage CG relaxation of the adsorbed system
2. CPU initialization of 300 K velocities
3. 10 ps NVT equilibration
4. 40 ps NVT production

The timestep is 0.5 fs and production frames are written every 50 fs.
Relaxation uses the KOKKOS-enabled CG minimizer throughout. The first stage
limits the maximum atomic displacement to 0.02 A and uses a moderate force
tolerance; the second stage permits 0.05 A moves and tightens convergence. The
KOKKOS FIRE minimizer and mixed GPU-pair/host-minimizer mode are intentionally
avoided because both are unstable in the target LAMMPS-MACE build.

## Conformational analysis

`analyze_conformations.py` calculates:

- all six C1-C2-C3-C4 through C6-C7-C8-C9 backbone dihedrals,
- trans, gauche+, and gauche- populations for each dihedral,
- circular mean and circular standard deviation of each dihedral,
- C1-C9 end-to-end distance,
- mass-weighted molecular radius of gyration,
- carbonyl-axis tilt from the surface normal,
- carbonyl-O height and minimum slab-molecule contact,
- desorbed-frame fraction,
- temperature and total-energy statistics.

The trans/gauche classification is descriptive: `|dihedral| >= 120 deg` is
trans, while the remaining positive and negative angles are gauche+ and gauche-.

## Main outputs

- `system_metadata.json`
- `initial_adsorbed.data`
- `initial_adsorbed_relaxed.data`
- `production.dump`
- `production_thermo.dat`
- `conformation_timeseries.csv`
- `dihedral_statistics.csv`
- `conformation_analysis_summary.txt`
- `backbone_dihedrals.png`
- `dihedral_distributions.png`
- `conformation_metrics.png`

## Run

```bash
sbatch run.slurm
```
EOF

python generate_system.py

echo "Generated $TEST_DIR"
