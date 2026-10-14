"""Compile and run one real CUDA kernel using the approved native prefix."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from provision_cuda import ROOT, WORK, digest


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Requires source root")
    os.umask(0o077)
    folder = Path(tempfile.mkdtemp(prefix="cuda-toolchain-", dir=WORK / "results"))
    build = Path(tempfile.mkdtemp(prefix="cuda-toolchain-", dir=WORK / "build-poseidon"))
    source = ROOT / "scripts/baseline/cuda_toolchain_smoke.cu"
    cuda = WORK / "deps/cuda-12.5.1"
    binary = build / "cuda_toolchain_smoke"
    commands = [("compile", [str(cuda / "bin/nvcc"), "-ccbin", "/usr/bin/g++", "-std=c++17",
                  "-arch=sm_89", "-O2", "--cudart=shared", "-Xlinker", "-rpath",
                  "-Xlinker", str(cuda / "lib"), str(source), "-o", str(binary)], 90),
                ("execute", [str(binary)], 30)]
    report = dict(status="running", source_sha256=digest(source), gpu_kernel_executed=False,
                  fhe_executed=False, stages={})
    print(f"CUDA toolchain evidence: {folder}", flush=True)
    for stage, command, seconds in commands:
        try:
            proc = subprocess.run(command, capture_output=True, timeout=seconds)
            (folder / (stage+".stdout")).write_bytes(proc.stdout)
            (folder / (stage+".stderr")).write_bytes(proc.stderr)
            report["stages"][stage] = dict(command=command, exit_code=proc.returncode)
            if proc.returncode:
                raise ValueError(stage+" failed; inspect saved stderr")
            if stage == "execute":
                report["result"] = json.loads(proc.stdout)
                if report["result"]["status"] != "passed":
                    raise ValueError("Kernel did not pass")
                report.update(status="passed", gpu_kernel_executed=True, binary_sha256=digest(binary))
        except Exception as error:
            report.update(status="failed", failure_layer=stage, diagnostic=str(error))
            break
    (folder / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
