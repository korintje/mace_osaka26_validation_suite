#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-/apps/envs/lammps-mace-current}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available. Initialize conda first." >&2
    exit 1
fi
if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "ERROR: target conda environment does not exist: $ENV_PREFIX" >&2
    exit 1
fi

echo "Installing torch-dftd into $ENV_PREFIX"
# --no-deps protects the working CUDA/PyTorch stack. torch-dftd's runtime
# dependencies (torch, ASE and NumPy) are already provided by the MACE env.
conda run -p "$ENV_PREFIX" python -m pip install --no-deps "torch-dftd==0.5.3"

conda run -p "$ENV_PREFIX" python - <<'PYEOF'
import importlib.metadata as metadata
import torch
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

print(f"torch: {torch.__version__}")
print(f"torch-dftd: {metadata.version('torch-dftd')}")
print(f"TorchDFTD3Calculator: {TorchDFTD3Calculator.__module__}")
PYEOF

LAMMPS_BIN="$(conda run -p "$ENV_PREFIX" bash -lc 'command -v lmp' | tail -n 1)"
if [[ -z "$LAMMPS_BIN" ]]; then
    echo "ERROR: lmp is not available in $ENV_PREFIX" >&2
    exit 1
fi
if ! conda run -p "$ENV_PREFIX" lmp -h 2>&1 | grep -q 'dispersion/d3'; then
    cat >&2 <<EOF
ERROR: $LAMMPS_BIN does not provide pair_style dispersion/d3.
Rebuild or replace LAMMPS (version 4Feb2025 or newer) with the EXTRA-PAIR
package enabled, while retaining ML-IAP, PYTHON and KOKKOS support.
EOF
    exit 1
fi

echo "D3 support is ready: torch-dftd 0.5.3 and LAMMPS dispersion/d3 found."
