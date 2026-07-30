# MACE-Osaka26 ZnO validation suite

This package regenerates the pre-calculation directories and files used for validation tests of `mace-osaka26-small.model` with either a `float64` or `float32` LAMMPS ML-IAP model. 

## Tests

1. `01_ase_lammps_consistency`: compares ASE/MACE and LAMMPS ML-IAP energies and forces.
2. `02_fixed_cell_relaxation`: relaxes ZnO atomic coordinates at fixed cell dimensions.
3. `03_full_cell_relaxation`: performs fixed-cell pre-relaxation and zero-pressure cell relaxation, then analyzes the structure.
4. `04_equation_of_state`: computes a seven-point ZnO volume-energy curve and fits a third-order Birch-Murnaghan equation of state.
5. `05_surface_slab_relaxation`: generates and relaxes a stoichiometric nonpolar ZnO (10-10) slab and checks structural stability.
6. `06_surface_energy_convergence`: evaluates ZnO (10-10) surface-energy convergence against slab and vacuum thickness.
7. `07_water_adsorption`: relaxes representative molecular H2O adsorption configurations on ZnO (10-10) and calculates adsorption energies.
8. `08_water_adsorption_md_300K`: runs 300 K NVT molecular dynamics for the lowest-energy ZnO/H2O adsorption structure and analyzes adsorption stability.
9. `09_2nonanone_md_300K`: relaxes 2-nonanone on a wider ZnO (10-10) slab, runs 300 K NVT molecular dynamics, and analyzes molecular conformations.

Each test directory contains its own input-generation record, README, run script, and `Slurm` script. Tests 01–08 retain their existing system sizes; Test 09 uses a wider 4 x 4 ZnO (10-10) surface to accommodate the extended hydrocarbon chain. Test 08 requires the relaxed structure produced by running Test 07. If it is not available, the generator creates the other tests and prints instructions to rerun the same command after Test 07 completes.

## Generate tests

Activate the existing LAMMPS-MACE environment and enter a GPU allocation, then run:

```bash
./generate_all_tests.sh
```

Generate the float32 variant with:

```bash
./generate_all_tests.sh --dtype float32
```

Outputs are kept separate by default:

| Precision | Model directory | Work directory |
| --- | --- | --- |
| float64 | `models_float64/` | `work_float64/` |
| float32 | `models_float32/` | `work_float32/` |

Override the work directory with either syntax:

```bash
./generate_all_tests.sh --dtype float32 --work-root /path/to/work
./generate_all_tests.sh --dtype float32 /path/to/work
```

Run `./generate_all_tests.sh --help` for the complete CLI. `MACE_DTYPE` remains
available for compatibility, although `--dtype` takes precedence.

The script downloads the official [mace-osaka26-small.model](https://github.com/qiqb-osaka/mace-osaka26/releases/download/v0.0.1/mace-osaka26-small.model) once when needed, reuses the original weights across precisions, records checksums and environment metadata, and converts a separate ML-IAP model for each dtype.

## Individual generators

Set the precision-specific model directory explicitly:

```bash
export MODEL_DIR="$(pwd)/models_float32"
./generators/generate_01_consistency.sh ./work_float32
```

Each generated test directory contains its own README, `run.sh`, and
`run.slurm`. The scripts assume the existing LAMMPS-MACE environment described
inside those generated files.

## Run tests

```bash
cd work_float64/01_ase_lammps_consistency
sbatch run.slurm
```

Repeat for the other test directories.

## Assumptions

- Conda initialization: `$HOME/miniforge3/etc/profile.d/conda.sh`
- Environment: `/apps/envs/lammps-mace-current`
- Slurm GPU resource: `--gres=gpu:rtx3080:1`
- LAMMPS executable: `lmp`

Edit the generated `run.slurm` files when the local configuration differs.

## LAMMPS formatting rule

Every LAMMPS command occupies one physical line. No LAMMPS input command uses a
backslash for line continuation.
