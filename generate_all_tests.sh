#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DTYPE="float64"
WORK_ROOT=""

usage() {
    cat <<'EOF'
Usage: ./generate_all_tests.sh [OPTIONS] [WORK_ROOT]

Generate the MACE-Osaka26 validation suite with the selected model precision.

Options:
  --dtype {float64|float32}  ML-IAP model precision (default: float64)
  --work-root PATH          Output directory (default: work_<dtype>)
  -h, --help                Show this help

Examples:
  ./generate_all_tests.sh --dtype float64
  ./generate_all_tests.sh --dtype float32
  ./generate_all_tests.sh --dtype float32 --work-root /scratch/work_fp32

The MACE_DTYPE environment variable is accepted for backward compatibility,
but --dtype takes precedence.
EOF
}

if [[ -n "${MACE_DTYPE:-}" ]]; then
    DTYPE="$MACE_DTYPE"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)
            [[ $# -ge 2 ]] || { echo "ERROR: --dtype requires a value" >&2; exit 2; }
            DTYPE="$2"
            shift 2
            ;;
        --dtype=*)
            DTYPE="${1#*=}"
            shift
            ;;
        --work-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --work-root requires a path" >&2; exit 2; }
            WORK_ROOT="$2"
            shift 2
            ;;
        --work-root=*)
            WORK_ROOT="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$WORK_ROOT" ]]; then
                echo "ERROR: multiple work directories specified" >&2
                exit 2
            fi
            WORK_ROOT="$1"
            shift
            ;;
    esac
done

case "$DTYPE" in
    float64|float32) ;;
    *)
        echo "ERROR: --dtype must be float64 or float32 (got: $DTYPE)" >&2
        exit 2
        ;;
esac

WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/work_${DTYPE}}"
MODEL_DIR="$SCRIPT_DIR/models_${DTYPE}"
MODEL_URL="https://github.com/qiqb-osaka/mace-osaka26/releases/download/v0.0.1/mace-osaka26-small.model"
ORIGINAL_MODEL="$MODEL_DIR/mace-osaka26-small.model"
MLIAP_MODEL="$MODEL_DIR/mace-osaka26-small.model-mliap_lammps.pt"

mkdir -p "$MODEL_DIR" "$WORK_ROOT"

for command in python mace_create_lammps_model sha256sum; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command" >&2
        exit 1
    fi
done

if [[ ! -f "$ORIGINAL_MODEL" ]]; then
    REUSABLE_MODEL=""
    for candidate in \
        "$SCRIPT_DIR/models_float64/mace-osaka26-small.model" \
        "$SCRIPT_DIR/models_float32/mace-osaka26-small.model" \
        "$SCRIPT_DIR/models/mace-osaka26-small.model"; do
        if [[ -f "$candidate" ]]; then
            REUSABLE_MODEL="$candidate"
            break
        fi
    done

    if [[ -n "$REUSABLE_MODEL" ]]; then
        echo "Reusing original model: $REUSABLE_MODEL"
        cp "$REUSABLE_MODEL" "$ORIGINAL_MODEL"
    else
        echo "Downloading MACE-Osaka26 small..."
        if command -v curl >/dev/null 2>&1; then
            curl --fail --location --retry 3 --output "$ORIGINAL_MODEL.part" "$MODEL_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget --tries=3 --output-document="$ORIGINAL_MODEL.part" "$MODEL_URL"
        else
            echo "ERROR: curl or wget is required" >&2
            exit 1
        fi
        mv "$ORIGINAL_MODEL.part" "$ORIGINAL_MODEL"
    fi
fi

sha256sum "$ORIGINAL_MODEL" > "$MODEL_DIR/SHA256SUMS"

if [[ ! -f "$MLIAP_MODEL" ]]; then
    echo "Converting model to ML-IAP (dtype=$DTYPE)..."
    (
        cd "$MODEL_DIR"
        mace_create_lammps_model mace-osaka26-small.model --format=mliap --dtype="$DTYPE"
    )
fi

if [[ ! -f "$MLIAP_MODEL" ]]; then
    echo "ERROR: ML-IAP model was not created: $MLIAP_MODEL" >&2
    exit 1
fi

printf '%s\n' "$DTYPE" > "$MODEL_DIR/DTYPE.txt"
sha256sum "$MLIAP_MODEL" >> "$MODEL_DIR/SHA256SUMS"

{
    echo "Generated: $(date --iso-8601=seconds)"
    echo "Conversion dtype: $DTYPE"
    echo "Host: $(hostname)"
    echo "Python: $(command -v python)"
    echo "LAMMPS: $(command -v lmp || true)"
    python - <<'PYEOF'
import sys
import importlib.metadata as metadata
print(f"Python version: {sys.version.split()[0]}")
for package in ["torch", "ase", "mace-torch", "cuequivariance", "cuequivariance-torch"]:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PYEOF
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    fi
} > "$MODEL_DIR/environment.txt"

export MODEL_DIR
for test_number in 01 02 03 04 05 06 07; do
    generator=("$SCRIPT_DIR"/generators/generate_"$test_number"_*.sh)
    "${generator[0]}" "$WORK_ROOT"
done

# Test 08 consumes the relaxed structure produced by running Test 07.
TEST07_RELAXED="$WORK_ROOT/07_water_adsorption/cases/01_zn_top_o_down/adsorbed_relaxed.data"
if [[ -f "$TEST07_RELAXED" ]]; then
    "$SCRIPT_DIR/generators/generate_08_water_adsorption_md_300K.sh" "$WORK_ROOT"
else
    echo
    echo "NOTE: Test 08 was not generated yet."
    echo "  It needs the relaxed structure from a completed Test 07 run:"
    echo "    $TEST07_RELAXED"
    echo "  Run Tests 01-07 first, then run this command again with the same options."
fi

"$SCRIPT_DIR/generators/generate_09_2nonanone_md_300K.sh" "$WORK_ROOT"

cat > "$WORK_ROOT/README.md" <<EOF
# Generated calculation directories (model dtype: $DTYPE)

Model directory: \`$MODEL_DIR\`

Execution order:

1. \`01_ase_lammps_consistency\`
2. \`02_fixed_cell_relaxation\`
3. \`03_full_cell_relaxation\`
4. \`04_equation_of_state\`
5. \`05_surface_slab_relaxation\`
6. \`06_surface_energy_convergence\`
7. \`07_water_adsorption\`
8. \`08_water_adsorption_md_300K\`
9. \`09_2nonanone_md_300K\`

Each directory contains a README, \`run.sh\`, and \`run.slurm\`.
EOF

echo "Generated tests (dtype=$DTYPE) under: $WORK_ROOT"
