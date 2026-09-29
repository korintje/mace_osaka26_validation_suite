#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/apps/envs/lammps-mace-current"
USE_SUDO=0

usage() {
    cat <<'EOF'
Usage: ./environment/install_d3_support.sh [--sudo] [ENV_PREFIX]

Install torch-dftd and verify the Python and LAMMPS D3 implementations.

Options:
  --sudo     Use sudo only for the pip installation step
  -h, --help Show this help

Default ENV_PREFIX: /apps/envs/lammps-mace-current
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sudo) USE_SUDO=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            ENV_PREFIX="$1"
            shift
            [[ $# -eq 0 ]] || { echo "ERROR: too many arguments" >&2; exit 2; }
            ;;
    esac
done

[[ -d "$ENV_PREFIX" ]] || {
    echo "ERROR: target conda environment does not exist: $ENV_PREFIX" >&2
    exit 1
}

PYTHON_BIN="$ENV_PREFIX/bin/python"
LAMMPS_BIN="$ENV_PREFIX/bin/lmp"
for executable in "$PYTHON_BIN" "$LAMMPS_BIN"; do
    [[ -x "$executable" ]] || {
        echo "ERROR: required executable is missing or not executable: $executable" >&2
        exit 1
    }
done

echo "Target conda environment: $ENV_PREFIX"
if [[ -w "$ENV_PREFIX/lib" ]]; then
    "$PYTHON_BIN" -m pip install "torch-dftd==0.5.3"
elif [[ "$USE_SUDO" -eq 1 ]]; then
    command -v sudo >/dev/null 2>&1 || {
        echo "ERROR: sudo was requested but is not available" >&2
        exit 1
    }
    # Elevate only installation. Validation remains an ordinary-user check.
    sudo "$PYTHON_BIN" -m pip install "torch-dftd==0.5.3"
else
    cat >&2 <<EOF
ERROR: target environment is not writable: $ENV_PREFIX
If you administer this shared environment, rerun with:
  ./environment/install_d3_support.sh --sudo "$ENV_PREFIX"
Otherwise ask its administrator to run that command.
EOF
    exit 1
fi

"$PYTHON_BIN" - <<'PYEOF'
import importlib.metadata as metadata
import torch
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

print(f"torch: {torch.__version__}")
print(f"torch-dftd: {metadata.version('torch-dftd')}")
print(f"TorchDFTD3Calculator: {TorchDFTD3Calculator.__module__}")
PYEOF

if ! LAMMPS_HELP="$($LAMMPS_BIN -h 2>&1)"; then
    echo "ERROR: failed to execute $LAMMPS_BIN -h" >&2
    printf '%s\n' "$LAMMPS_HELP" | sed -n '1,5p' >&2
    exit 1
fi
if [[ "$LAMMPS_HELP" != *"dispersion/d3"* ]]; then
    cat >&2 <<EOF
ERROR: $LAMMPS_BIN does not provide pair_style dispersion/d3.
Rebuild or replace LAMMPS (version 4Feb2025 or newer) with the EXTRA-PAIR
package enabled, while retaining ML-IAP, PYTHON and KOKKOS support.
EOF
    exit 1
fi

echo "D3 support is ready: torch-dftd 0.5.3 and LAMMPS dispersion/d3 found."
