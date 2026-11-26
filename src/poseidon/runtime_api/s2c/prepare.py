"""Prepare the pinned, private GPU implementation without changing the source checkout."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

archive, local_patch, native_patch, output = map(Path, sys.argv[1:])
with tempfile.TemporaryDirectory(dir=output.parent) as temp:
    subprocess.run(["tar", "xzf", str(archive), "-C", temp], check=True)
    for patch in (local_patch, native_patch):
        with patch.open("rb") as source:
            subprocess.run(["patch", "--silent", "--batch", "-p1"], cwd=temp, stdin=source, check=True)
    gpu = Path(temp) / "src/poseidon/gpu"
    for source in gpu.rglob("*"):
        if source.suffix not in {".cpp", ".cu", ".h", ".cuh"}:
            continue
        data = source.read_text().replace("poseidon/gpu/", "poseidon/s2c_gpu/")
        data = re.sub(r"\bnamespace gpu\b", "namespace s2c_gpu", data)
        data = re.sub(r"\bgpu::", "s2c_gpu::", data)
        data = re.sub(r"poseidon::gpu\b", "poseidon::s2c_gpu", data)
        dest = output / "poseidon/s2c_gpu" / source.relative_to(gpu)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_text() != data:
            dest.write_text(data)
