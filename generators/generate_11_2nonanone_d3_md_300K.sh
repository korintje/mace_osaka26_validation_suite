#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${1:-$SUITE_DIR/work}"
SOURCE_DIR="$WORK_ROOT/09_2nonanone_md_300K"
TEST_DIR="$WORK_ROOT/11_2nonanone_d3_md_300K"

# Test 11 is deliberately derived from Test 09 so that the potential is the
# only scientific variable. Regenerate Test 09 first to avoid stale inputs.
"$SCRIPT_DIR/generate_09_2nonanone_md_300K.sh" "$WORK_ROOT"
mkdir -p "$TEST_DIR"
cp -a "$SOURCE_DIR/." "$TEST_DIR/"

TEST_DIR="$TEST_DIR" python - <<'PYEOF'
from pathlib import Path
import json
import os

test_dir = Path(os.environ["TEST_DIR"])
for pattern in (
    "*.out", "*.dump", "*.restart", "*.png", "slurm-*.out",
    "relaxation_summary.txt", "md_summary.txt",
    "conformation_analysis_summary.txt", "conformation_timeseries.csv",
    "md_thermo_timeseries.csv", "dihedral_statistics.csv",
    "equilibration_thermo.dat", "production_thermo.dat",
    "initial_adsorbed_relaxed.data", "initial_adsorbed_300K.data",
    "equilibrated_300K.data", "final_300K.data", "log.lammps",
):
    for path in test_dir.glob(pattern):
        if path.is_file() or path.is_symlink():
            path.unlink()
plain_style = "pair_style mliap unified mace-osaka26-small.model-mliap_lammps.pt 0"
plain_coeff = "pair_coeff * * Zn O C H"
d3_style = (
    "pair_style hybrid/overlay mliap unified "
    "mace-osaka26-small.model-mliap_lammps.pt 0 "
    "dispersion/d3 bj pbe 30.0 20.0"
)
d3_coeff = (
    "pair_coeff * * mliap Zn O C H\n"
    "pair_coeff * * dispersion/d3 Zn O C H"
)

for filename in ("in.relax", "in.md_300K"):
    path = test_dir / filename
    text = path.read_text()
    if text.count(plain_style) != 1 or text.count(plain_coeff) != 1:
        raise RuntimeError(f"Unexpected Test 09 pair definition in {path}")
    path.write_text(text.replace(plain_style, d3_style).replace(plain_coeff, d3_coeff))

metadata_path = test_dir / "system_metadata.json"
metadata = json.loads(metadata_path.read_text())
metadata["potential"] = {
    "base": "MACE-Osaka26 small via LAMMPS ML-IAP",
    "dispersion": "two-body PBE-D3(BJ)",
    "damping": "bj",
    "xc": "pbe",
    "cutoff_A": 30.0,
    "coordination_cutoff_A": 20.0,
    "atm_three_body": False,
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

run_path = test_dir / "run.sh"
run_text = run_path.read_text()
marker = 'cd "$(dirname "$(readlink -f "$0")")"\n'
preflight = '''
if ! lmp -h 2>&1 | grep -q 'dispersion/d3'; then
    echo "ERROR: LAMMPS lacks dispersion/d3 (EXTRA-PAIR package)" >&2
    exit 1
fi
'''
if marker not in run_text:
    raise RuntimeError("Could not locate run.sh insertion point")
run_path.write_text(run_text.replace(marker, marker + preflight, 1))

slurm_path = test_dir / "run.slurm"
slurm_path.write_text(
    slurm_path.read_text().replace(
        "#SBATCH --job-name=osaka26-zno-2nonanone-md",
        "#SBATCH --job-name=osaka26-zno-2nonanone-d3-md",
    )
)

analysis_path = test_dir / "analyze_conformations.py"
analysis_path.write_text(
    analysis_path.read_text().replace(
        '"300 K MD of 2-nonanone adsorbed on ZnO (10-10)",',
        '"300 K MACE+PBE-D3(BJ) MD of 2-nonanone adsorbed on ZnO (10-10)",',
    )
)
PYEOF

cat > "$TEST_DIR/README.md" <<'EOF'
# Test 11: PBE-D3(BJ) 300 K MD of 2-nonanone on ZnO (10-10)

This is the dispersion-corrected counterpart of Test 09. It uses the identical
4 x 4 ZnO slab, initial 2-nonanone geometry, fixed atoms, two-stage relaxation,
temperature protocol, timestep, trajectory length, and conformational analysis.
The only scientific change is the potential used for relaxation and MD:

```text
MACE-Osaka26 + two-body PBE-D3(BJ)
interaction cutoff = 30.0 A
coordination cutoff = 20.0 A
ATM three-body term = disabled
```

LAMMPS applies the correction with `pair_style hybrid/overlay`. Test 10 must pass
before this production-scale test is interpreted. Direct comparison of Test 09
and Test 11 isolates the effect of the D3 correction on adsorption stability,
carbonyl tilt, molecular radius of gyration, end-to-end distance, and backbone
dihedral populations.

Run with:

```bash
sbatch run.slurm
```

The primary result is `conformation_analysis_summary.txt`; detailed outputs
match those documented for Test 09. D3 settings are recorded in
`system_metadata.json`.
EOF

echo "Generated $TEST_DIR from Test 09 with PBE-D3(BJ) overlay"
