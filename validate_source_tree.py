#!/usr/bin/env python3
from pathlib import Path
import os
import re
import subprocess

root = Path(__file__).resolve().parent
shell_files = [root / "generate_all_tests.sh", *sorted((root / "generators").glob("*.sh"))]
bash = os.environ.get("BASH", "bash")
for path in shell_files:
    subprocess.run([bash, "-n", str(path)], check=True)

for path in shell_files:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"cat > .*? <<'LAMMPS_EOF'\n(.*?)\nLAMMPS_EOF", text, flags=re.S)
    for block in blocks:
        bad = [line for line in block.splitlines() if line.rstrip().endswith("\\")]
        if bad:
            raise RuntimeError(f"LAMMPS line continuation found in {path}: {bad}")

manifest_entries = {
    line.strip()
    for line in (root / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
actual_entries = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
    and not any(part.startswith(".") for part in path.relative_to(root).parts)
    and "__pycache__" not in path.parts
}
if manifest_entries != actual_entries:
    missing = sorted(manifest_entries - actual_entries)
    unlisted = sorted(actual_entries - manifest_entries)
    raise RuntimeError(
        f"MANIFEST mismatch; missing={missing}, unlisted={unlisted}"
    )

print(
    "Validation passed: shell syntax, LAMMPS formatting, and MANIFEST are valid."
)
