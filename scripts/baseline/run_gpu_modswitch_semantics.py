"""Real native Poseidon GPU primitive evidence, not HEVM execution or an API run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from build_poseidon_gpu import BUILD, CUDA, GMP
from hecate_python_env import ROOT, WORK, digest

SOURCES = (
    "scripts/baseline/build_poseidon_gpu.py",
    "scripts/baseline/run_gpu_modswitch_semantics.py",
    "src/poseidon/mgpu/CMakeLists.txt",
    "src/poseidon/tests/frontends/dacapo/gpu_ckks_modswitch_parity_test.cpp",
    "src/poseidon/gpu/gpu_evaluator.cpp",
    "src/poseidon/gpu/gpu_modswitch_handler.cpp",
    "src/poseidon/gpu/gpu_uploader.cpp",
    "src/poseidon/gpu/gpu_parameter.cpp",
    "src/poseidon/parameters_literal.cpp",
    "src/poseidon/crt_context.cpp",
)


def run(*, binary_name="poseidon_gpu_ckks_modswitch_parity_tests", prefix="poseidon-gpu-modswitch-",
        sources=SOURCES, backend="poseidon_gpu_primitives"):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from source root")
    os.umask(0o077)
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=WORK / "results"))
    binary = BUILD / "bin" / binary_name
    library = BUILD / "libposeidon_shared.so"
    report = dict(status="running", gpu_executed=False, fhe_executed=False,
                  hevm_executed=False, agent_calls=0, source_hashes={}, checks={})
    print(f"GPU primitive evidence: {folder}", flush=True)
    # No API credentials or Nix wrapper environment reaches the executable.
    env = dict(PATH="/usr/bin:/bin", OMP_NUM_THREADS="2",
               LD_LIBRARY_PATH=str(GMP / "lib")+":"+str(CUDA / "lib"))
    try:
        for name in sources:
            target = folder / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            report["source_hashes"][str(target.relative_to(folder))] = digest(target)
            if digest(target) != digest(ROOT / name):
                raise ValueError("Source changed during snapshot")
        report["binary_sha256"] = digest(binary)
        report["library_sha256"] = digest(library)
        for name, command in (
            ("binary_dynamic", ["readelf", "-d", str(binary)]),
            ("library_dynamic", ["readelf", "-d", str(library)]),
            ("loader", ["ldd", str(binary)]),
            ("primitive", [str(binary)]),
        ):
            report["failure_layer"] = name
            result = subprocess.run(command, env=env, capture_output=True,
                                    timeout=90 if name == "primitive" else 15)
            for suffix, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
                (folder / (name+"."+suffix)).write_bytes(raw)
            report["checks"][name] = dict(command=command, exit_code=result.returncode,
                stdout_sha256=digest(folder / (name+".stdout")),
                stderr_sha256=digest(folder / (name+".stderr")))
            if result.returncode:
                raise ValueError(name+" failed; see captured logs")
            if name != "primitive" and (b"/nix/store/" in result.stdout or b"not found" in result.stdout):
                raise ValueError("Native GPU binary has mixed or unresolved runtime libraries")
            if name == "primitive":
                data = json.loads(result.stdout)
                report["result"] = data
                if (data.get("status") != "passed" or not data.get("gpu_executed") or data.get("hevm_executed")
                        or data.get("backend") != backend):
                    raise ValueError("Invalid GPU primitive execution marker")
        if digest(binary) != report["binary_sha256"] or digest(library) != report["library_sha256"]:
            raise ValueError("Executable changed during test")
        report.pop("failure_layer", None)
        report.update(status="passed", gpu_executed=True, fhe_executed=True)
    except Exception as error:
        report.update(status="failed", diagnostic=str(error))
    (folder / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
