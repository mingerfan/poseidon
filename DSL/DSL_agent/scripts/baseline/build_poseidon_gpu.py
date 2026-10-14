"""Offline Poseidon GPU build using existing Nix CMake + native GCC/CUDA/GMP.

Does not install packages. Tests run only with --run-parity. Generated files stay in the
separate native WSL build tree; no Dacapo build settings are changed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest

BUILD = WORK / "build-poseidon/agent-dsl-gpu-native"
CUDA = WORK / "deps/cuda-12.5.1"
GMP = WORK / "deps/gmp-6.3.0"
NIX_ROOT = WORK / "deps/nix-portable-v012/.nix-portable/nix"
PINNED_CMAKE = "/nix/store/mbw6qprvn80p8k7z3lq44hnz6b332xf6-cmake-3.28.3/bin/cmake"


def native_environment(environ):
    """Use Nix only for CMake itself, never its compiler/linker wrappers."""
    env = {k: v for k, v in environ.items() if not k.startswith("NIX_")}
    for name in ("LD", "AR", "AS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
                 "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH",
                 "GCC_EXEC_PREFIX", "LD_PRELOAD", "CMAKE_PREFIX_PATH", "CMAKE_LIBRARY_PATH",
                 "CMAKE_INCLUDE_PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR"):
        env.pop(name, None)
    env.update(PATH=str(CUDA / "bin")+":/usr/bin:/bin", CPLUS_INCLUDE_PATH=str(GMP / "include"),
               LIBRARY_PATH=str(GMP / "lib"), LD_LIBRARY_PATH=str(GMP / "lib")+":"+str(CUDA / "lib"),
               OMP_NUM_THREADS="2")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--native-tools", action="store_true",
                        help="Use cached read-only CMake namespace without Nix launcher/database access")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true", help="Build GPU runtime and tests; do not execute")
    action.add_argument("--run-parity", action="store_true", help="Run the tc128 CPU/GPU primitive test, not HEVM")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Requires source root")
    if not args.inside:
        if args.native_tools:
            if not (NIX_ROOT / PINNED_CMAKE.removeprefix("/nix/")).is_file():
                raise ValueError("Pinned cached CMake is missing; no automatic install")
            # Trusted build only. Independent read-only store, no launcher tmpbin
            # race with a live Agent batch; no Nix evaluator or network access.
            command = ["/usr/bin/bwrap", "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
                       "--ro-bind", "/etc", "/etc", "--ro-bind", "/sys", "/sys",
                       "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
                       "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                       "--ro-bind", str(ROOT), str(ROOT), "--bind", str(WORK), str(WORK),
                       "--ro-bind", str(NIX_ROOT), "/nix", "--tmpfs", "/tmp", "--chdir", str(ROOT),
                       "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--unshare-net", "--die-with-parent",
                       "--", "/usr/bin/env", "-i", "PATH="+str(Path(PINNED_CMAKE).parent)+":/usr/bin:/bin",
                       "PYTHONDONTWRITEBYTECODE=1", "/usr/bin/python3", str(Path(__file__).resolve()),
                       "--inside", "--native-tools",
                       *(["--build"] if args.build else ["--run-parity"] if args.run_parity else [])]
            return subprocess.run(command, timeout=1800 if args.build else 180).returncode
        command = shlex.join([str(VENV / "bin/python"), str(Path(__file__).resolve()), "--inside",
                             *(["--build"] if args.build else ["--run-parity"] if args.run_parity else [])])
        return enter_nix(command, seconds=1800 if args.build else 180)
    if not os.environ.get("IN_NIX_SHELL") and not args.native_tools:
        raise ValueError("Requires existing pinned Nix CMake")
    os.umask(0o077)
    cmake = shutil.which("cmake")
    if not cmake:
        raise ValueError("Pinned CMake is missing")
    if args.native_tools and cmake != PINNED_CMAKE:
        raise ValueError("Unexpected cached CMake path")
    folder = Path(tempfile.mkdtemp(prefix="poseidon-gpu-build-", dir=WORK / "results"))
    if args.build:
        command = [cmake, "--build", str(BUILD), "--target", "poseidon_mgpu_gpu_runtime_smoke_tests",
                   "poseidon_gpu_ckks_modswitch_parity_tests", "poseidon_gpu_drop_modulus_schedule_tests",
                   "poseidon_mgpu_drop_modulus_tests", "poseidon_mgpu_ir_tests",
                   "poseidon_mgpu_schedule_json_tests", "poseidon_mgpu_runtime_tests",
                   "poseidon_mgpu_static_pipeline_tests", "poseidon_mgpu_static_placement_tests",
                   "poseidon_mgpu_copy_insertion_tests", "poseidon_mgpu_static_scheduler_tests",
                   "poseidon_mgpu_topology_tests", "poseidon_mgpu_poseidon_gpu_preflight_tests",
                   "poseidon_mgpu_gpu_execution_preflight_tests", "poseidon_mgpu_dacapo_adapter_tests", "-j2"]
    elif args.run_parity:
        command = [str(BUILD / "bin/poseidon_gpu_ckks_modswitch_parity_tests")]
    else:
        command = [cmake, "-S", str(ROOT), "-B", str(BUILD), "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=/usr/bin/gcc", "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
            "-DCMAKE_MAKE_PROGRAM=/usr/bin/make", "-DCMAKE_LINKER=/usr/bin/ld",
            "-DZLIB_INCLUDE_DIR=/usr/include", "-DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so",
            "-DCMAKE_CUDA_COMPILER="+str(CUDA / "bin/nvcc"), "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++",
            "-DCUDAToolkit_ROOT="+str(CUDA), "-DCMAKE_CUDA_ARCHITECTURES=89",
            "-DPOSEIDON_BUILD_DEPS=OFF", "-DPOSEIDON_BUILD_EXAMPLES=OFF", "-DPOSEIDON_USE_HARDWARE=OFF",
            "-DPOSEIDON_USE_MSGSL=OFF", "-DPOSEIDON_USE_ZSTD=OFF", "-DPOSEIDON_USE_INTEL_HEXL=OFF",
            "-DPOSEIDON_BUILD_MGPU=ON", "-DPOSEIDON_BUILD_MGPU_TESTS=ON", "-DPOSEIDON_BUILD_MGPU_TOOLS=ON",
            "-DPOSEIDON_BUILD_MGPU_GPU_RUNTIME=ON", "-DPOSEIDON_BUILD_MGPU_CUDA_COMM=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
            "-DCMAKE_CXX_FLAGS=-I"+str(GMP / "include"),
            "-DCMAKE_SHARED_LINKER_FLAGS=-L"+str(GMP / "lib")+" -Wl,-rpath,"+str(GMP / "lib"),
            "-DCMAKE_EXE_LINKER_FLAGS=-L"+str(GMP / "lib")+" -Wl,-rpath,"+str(GMP / "lib")]
    report = dict(status="running", command=command, source_sha256=digest(Path(__file__)),
                  gpu_executed=False, fhe_executed=False, build_dir=str(BUILD))
    print(f"Poseidon GPU build evidence: {folder}", flush=True)
    env = native_environment(os.environ)
    try:
        with (folder / "output.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    timeout=1750 if args.build else 150)
        report.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
        if args.run_parity and result.returncode == 0:
            report["result"] = json.loads((folder / "output.log").read_text())
            if report["result"]["status"] != "passed":
                raise ValueError("GPU primitive did not pass")
            report.update(gpu_executed=True, fhe_executed=True, hevm_executed=False,
                          binary_sha256=digest(Path(command[0])))
    except Exception as error:
        report.update(status="failed", diagnostic=str(error))
    (folder / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    print((folder / "output.log").read_text()[-10000:], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
