"""Fail-closed bubblewrap capability boundary, using already installed tools.

No broad host-root/source/results mounts; never fall back to unsandboxed execution.
Not a VM/kernel security proof. Caller must use only trusted worker commands.
"""
import os
from pathlib import Path
import resource
import subprocess

from hecate_python_env import ROOT, VENV, WORK
from python_compiler_smoke import BUILD
from seal_cpu_golden import KEY_BUILD, PROFILE
from seal_artifact_gate import require
from workspace_paths import WORK_BASE
from platform_config import configuration

MODULES = ("hevm_abi.py", "platform_config.py", "platform-profiles.json", "workspace_paths.py", "candidate_trace.py", "candidate_contract.py", "candidate_worker.py", "hecate_contract.py", "public_construction.py", "function_construction.py", "lexical_scope.py", "construction_calls.py",
           "packed_native_exercises.py", "packed-native-exercises-v1.json",
           "public_numeric.py", "public_strings.py", "public_polynomial.py", "object_arrays.py", "construction_exercises.py", "construction-exercises-v1.json", "seal_artifact_gate.py", "seal_cpu_golden.py", "python_compiler_smoke.py",
           "hecate_python_env.py", "continue_dacapo_cpp.py", "spatial_ops.py", "cipher_abi.py", "chunked_input_abi.py", "packed_input_abi.py",
           "object_unary_exercises.py", "object-unary-exercises-v1.json",
           "scalar_conversion.py", "scalar_conversion_exercises.py", "scalar-conversion-exercises-v1.json",
           "object_arithmetic_exercises.py", "object-arithmetic-exercises-v1.json",
           "decorated_functions.py", "native_function_rules.py", "native_array_core.py", "native_public_loops.py", "native_array_alias.py",
           "native_function_exercises.py", "native-function-exercises-v1.json", "native_array_exercises.py", "compiler_configuration.py", "native_star_exercises.py")


def command(payload, output, argv, keys=None):
    require(output.resolve().is_relative_to(WORK / "results") and output.is_dir(), "Invalid sandbox output")
    require(payload.resolve().is_relative_to(WORK / "results") and payload.is_file(), "Invalid sandbox payload")
    library = os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    require(all(Path(p).is_relative_to("/nix/store") for p in library.split(":")), "Non-Nix library search path")
    cmd = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
           "--clearenv", "--ro-bind", "/nix/store", "/nix/store", "--ro-bind", str(VENV), str(VENV),
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/app",
           "--ro-bind", str(payload), "/payload.json", "--bind", str(output), "/out",
           "--ro-bind", str(ROOT / "third_party/dacapo/python/hecate"), "/frontend",
           "--ro-bind", str(BUILD / "lib"), "/hecate/build/lib",
           "--ro-bind", str(BUILD / "lib"), str(BUILD / "lib"),
           "--ro-bind", str(BUILD / "bin/hecate-opt"), "/hecate-opt",
           "--ro-bind", str(PROFILE), "/profile.json",
           "--ro-bind", str(BUILD / "hevm-abi.json"), str(BUILD / "hevm-abi.json"),
           "--ro-bind", str(KEY_BUILD / "libseal_golden_metadata.so"), str(KEY_BUILD / "libseal_golden_metadata.so")]
    for name in MODULES:
        cmd += ["--ro-bind", str(ROOT / "scripts/baseline" / name), "/app/" + name]
    # New packing verification is opt-in; legacy installations do not require it.
    packed_observer=KEY_BUILD/'libseal_packed_metadata.so'
    if packed_observer.is_file():
        cmd += ['--ro-bind',str(packed_observer),str(packed_observer)]
    if keys is not None:
        require(keys.resolve().is_relative_to(WORK / "results") and keys.is_dir(), "Invalid keys directory")
        cmd += ["--ro-bind", str(keys), "/keys"]
    variables = dict(PATH="/nonexistent", LD_LIBRARY_PATH=library, HECATE="/hecate",
                     POSEIDON_WORK_ROOT=str(WORK_BASE), POSEIDON_PLATFORM=configuration()['id'], HOME="/nonexistent",
                     PYTHONPATH="/app:/frontend", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                     OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", LANG="C.UTF-8")
    for name, value in variables.items():
        cmd += ["--setenv", name, value]
    return cmd + ["--chdir", "/out", "--", *argv]


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (150, 155))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def run(payload, output, argv, log, seconds=60, keys=None):
    cmd = command(payload, output, argv, keys)
    with log.open("w") as stream:
        # Strip host loader overrides so the existing Ubuntu bwrap uses its own libc.
        result = subprocess.run(["/usr/bin/timeout", "-k", "3s", str(seconds), *cmd],
                                env={"PATH": "/usr/bin:/bin", "POSEIDON_PROBE_SECRET": "must-not-survive"},
                                stdout=stream, stderr=subprocess.STDOUT, timeout=seconds+10,
                                close_fds=True, preexec_fn=limits)
    return result.returncode
