"""Real Hecate -> existing Dacapo SEAL CPU execution; no Poseidon GPU claim.

Only trusted manual goldens. No installs/downloads, no bootstrap, no new HEVM
backend. The venv/Nix shell and subprocess isolation are NOT an Agent sandbox.
"""
from platform_config import identity, require_python_packages, loaded_libraries
from hevm_abi import check_elf, require_native_abi
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile
import time

from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix
from python_compiler_smoke import BUILD, CASES as BASE_CASES, SOURCE, logged, prepare_reference
from seal_artifact_gate import inspect_artifacts, require

PROFILE = ROOT / "third_party/dacapo/profiled_SEAL_CPU.json"
SCRIPT = ROOT / "scripts/baseline/seal_cpu_golden.py"
WATERLINE = 40
EXTENDED_CASES = ("rotate1", "rotate2", "square", "quartic", "mlp4x4x2")
CASES = BASE_CASES + EXTENDED_CASES
SCALAR_OUTPUTS = ("linear4x2", "mlp4x4x2")
KEY_BUILD = WORK / "build-dacapo/seal-golden-keys"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(actual, reference, atol, rtol):
    import numpy as np
    actual, reference = np.asarray(actual), np.asarray(reference)
    require(actual.shape == reference.shape and actual.ndim == 2 and actual.size > 0, "Output shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "Non-finite numerical result")
    error = np.abs(actual - reference)
    nonzero = reference != 0
    relative = np.full(reference.shape, np.nan)
    relative[nonzero] = error[nonzero] / np.abs(reference[nonzero])
    norm = float(np.linalg.norm(actual) * np.linalg.norm(reference))
    return dict(passed=bool(np.all(error <= atol + rtol * np.abs(reference))), atol=atol, rtol=rtol,
                compared_values=int(actual.size), mae=float(error.mean()), max_absolute_error=float(error.max()),
                max_nonzero_reference_relative_error=float(relative[nonzero].max()) if nonzero.any() else None,
                nonzero_reference_count=int(nonzero.sum()),
                cosine_similarity=float(np.vdot(actual.ravel(), reference.ravel()) / norm) if norm else None,
                actual=actual.tolist(), reference=reference.tolist(), absolute_error=error.tolist(),
                relative_error=[[float(x) if np.isfinite(x) else None for x in row] for row in relative],
                elementwise_pass=(error <= atol + rtol * np.abs(reference)).tolist())


def sealed_snapshot(name, data):
    """Stock C API opens paths; supply sealed bytes we actually validated."""
    import fcntl
    fd = os.memfd_create(name, os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    with os.fdopen(os.dup(fd), "wb") as stream:
        stream.write(data)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK |
                fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
    return fd, f"/proc/self/fd/{fd}".encode()


def worker(case, directory, keys):
    selectors = [[i, 0] for i in range(2)] if case in SCALAR_OUTPUTS else [[0, i] for i in range(4)]
    return execute_artifact(directory, keys, selectors)


def execute_artifact(directory, keys, selectors, *, rotation_steps=(1, 2), expected_inputs=1,
                     logical_inputs=None, encrypted_zero_input=False, execution_abi=None, input_period=4):
    """Shared real-runtime driver; selectors index result ciphertexts then slots."""
    # Bound native failures; process exit frees a VM with no upstream destroy API.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    import numpy as np
    raw = (directory / "lowered._hecate_golden.hevm").read_bytes()
    cst = (directory / "_hecate_golden.cst").read_bytes()
    gate = inspect_artifacts(raw, cst, rotation_steps=rotation_steps, expected_inputs=expected_inputs,
                             execution_abi=execution_abi,input_period=input_period)
    packed=execution_abi is not None
    # Validate independent input arrays BEFORE loading/initializing the native VM.
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        inputs = arrays["inputs"].copy()
    logical_inputs = expected_inputs if logical_inputs is None else logical_inputs
    require(type(logical_inputs) is int and 1 <= logical_inputs <= 4 and type(encrypted_zero_input) is bool and
            expected_inputs == logical_inputs + int(encrypted_zero_input), 'Logical/auxiliary ciphertext count mismatch')
    input_shape = (4, input_period) if logical_inputs == 1 else (4, logical_inputs, input_period)
    require(inputs.shape == input_shape and np.isfinite(inputs).all(), "Unexpected golden input")
    batches = inputs[:, None, :] if logical_inputs == 1 else inputs
    if encrypted_zero_input:
        # This is ordinary client-side public-key encryption below, not a runtime
        # oracle or a replacement for bootstrap. User input/reference files stay immutable.
        batches = np.concatenate((batches, np.zeros((4, 1, input_period), dtype=np.float64)), axis=1)
    require(type(selectors) is list and 1 <= len(selectors) <= (256 if packed else 4) and
            all(type(pair) is list and len(pair) == 2 and
                all(type(i) is int for i in pair) and 0 <= pair[0] < len(gate["res_dst"]) and
                0 <= pair[1] < 16384 for pair in selectors), "Invalid output slot selectors")
    require(set(pair[0] for pair in selectors) == set(range(len(gate["res_dst"]))),
            "Every result ciphertext must have a declared output binding")
    hfd, hpath = sealed_snapshot("validated-hevm", raw)
    cfd, cpath = sealed_snapshot("validated-cst", cst)
    require_native_abi(BUILD)
    check_elf(BUILD / "lib/libSEAL_HEVM.so")
    check_elf(KEY_BUILD / "libseal_golden_metadata.so")
    lib = ctypes.CDLL(str(BUILD / "lib/libSEAL_HEVM.so"))
    ptr, chars, i64, doubles = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)
    # Match the actual C++ C ABI (stock runner.py incorrectly adds an init bool).
    signatures = {"initFullVM": ([chars], ptr), "load": ([ptr, chars, chars], None),
                  "preprocess": ([ptr], None), "run": ([ptr], None),
                  "encrypt": ([ptr, i64, doubles, ctypes.c_int], None),
                  "decrypt_result": ([ptr, i64, doubles], None),
                  "getArgLen": ([ptr], i64), "getResLen": ([ptr], i64), "getCtxt": ([ptr, i64], ptr)}
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    observer = ctypes.CDLL(str(KEY_BUILD / "libseal_golden_metadata.so"))
    observer.verify_galois_file.argtypes = [chars, chars, ctypes.POINTER(i64), ctypes.c_uint64]
    observer.verify_galois_file.restype = ctypes.c_int
    required_steps = (i64 * len(gate["rotation_steps"]))(*gate["rotation_steps"])
    if packed:
        check_elf(KEY_BUILD/'libseal_packed_metadata.so')
        packed_observer=ctypes.CDLL(str(KEY_BUILD/'libseal_packed_metadata.so'))
        packed_observer.verify_packed_galois_file.argtypes=[chars,chars,ctypes.POINTER(i64),ctypes.c_uint64,ctypes.c_uint64]
        packed_observer.verify_packed_galois_file.restype=ctypes.c_int
        key_check=packed_observer.verify_packed_galois_file(os.fsencode(keys/'parm.seal'),os.fsencode(keys/'gal.seal'),
                                                          required_steps,len(gate['rotation_steps']),input_period)
    else:
        key_check = observer.verify_galois_file(os.fsencode(keys / "parm.seal"), os.fsencode(keys / "gal.seal"),
                                                required_steps, len(gate["rotation_steps"]))
    require(key_check == 0, "Actual Galois key preflight failed (missing key or invalid key file)")
    observer.describe_cipher.argtypes = [ptr, ctypes.POINTER(ctypes.c_uint64), doubles,
                                         ctypes.POINTER(ctypes.c_uint64)]
    observer.describe_cipher.restype = ctypes.c_int
    if encrypted_zero_input:
        observer.inspect_cipher_integrity.argtypes = [ptr, ctypes.POINTER(ctypes.c_uint64)]
        observer.inspect_cipher_integrity.restype = ctypes.c_int
    zero_fingerprints = []

    def metadata(vm, register, expected_level, expected_scale):
        level, scale, polys = ctypes.c_uint64(), ctypes.c_double(), ctypes.c_uint64()
        require(observer.describe_cipher(lib.getCtxt(vm, register), ctypes.byref(level),
                    ctypes.byref(scale), ctypes.byref(polys)) == 0, "Metadata observation failed")
        require(math.isfinite(scale.value) and scale.value > 0, "Invalid runtime scale")
        log_scale = math.log2(scale.value)
        require(level.value == expected_level and abs(log_scale - expected_scale) <= 1e-6,
                "Runtime level/scale differs from compiler metadata")
        require(polys.value == 2, "Expected compact two-polynomial ciphertext (including relinearization)")
        return dict(data_modulus_count=level.value, log2_scale=log_scale, polynomials=polys.value)
    begin = time.monotonic()
    vm = lib.initFullVM(os.fsencode(keys))
    require(bool(vm), "Native VM initialization returned null")
    lib.load(vm, cpath, hpath)
    os.close(hfd)
    os.close(cfd)
    require(lib.getArgLen(vm) == expected_inputs and lib.getResLen(vm) == len(gate["res_dst"]), "C ABI I/O mismatch")
    lib.preprocess(vm)
    output, observations = [], []
    for batch in batches:
        input_metadata = []
        for index, vector in enumerate(batch):
            vector = np.ascontiguousarray(vector, dtype=np.float64)
            lib.encrypt(vm, index, vector.ctypes.data_as(doubles), vector.size)
            input_metadata.append(metadata(vm, index, gate["arg_level"][index], gate["arg_scale"][index]))
            if encrypted_zero_input and index == logical_inputs:
                fingerprint = ctypes.c_uint64()
                require(observer.inspect_cipher_integrity(lib.getCtxt(vm, index), ctypes.byref(fingerprint)) == 0,
                        'Auxiliary zero must be a nontransparent ciphertext')
                zero_fingerprints.append(f'{fingerprint.value:016x}')
        observation = dict(outputs=[])
        if expected_inputs == 1:
            observation["input"] = input_metadata[0]
        else:
            observation["inputs"] = input_metadata
        lib.run(vm)
        slots = np.zeros((len(gate["res_dst"]), 16384), dtype=np.float64)
        for index in range(len(slots)):
            observation["outputs"].append(metadata(vm, gate["res_dst"][index], gate["res_level"][index],
                                                     gate["res_scale"][index]))
            lib.decrypt_result(vm, index, slots[index].ctypes.data_as(doubles))
        output.append(np.array([slots[i, j] for i, j in selectors], dtype=np.float64))
        observations.append(observation)
    np.save(directory / "decrypted.npy", np.stack(output), allow_pickle=False)
    evidence = dict(platform_identity=identity(), mapped_libraries=loaded_libraries(),
        backend="upstream_SEAL_HEVM_CPU", encrypted_execution=True,
        rotation_key_check={"actual_key_file_verified": True, "required_steps": gate["rotation_steps"]},
        bootstrap_executed=False, input_batches=len(inputs), ciphertext_metadata=observations,
        encrypted_input_count=expected_inputs,
        wall_seconds=time.monotonic() - begin,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if encrypted_zero_input:
        require(len(set(zero_fingerprints)) == len(batches), 'Zero ciphertext was not freshly randomized per batch')
        evidence['auxiliary_encrypted_zero'] = dict(logical_input_count=logical_inputs, index=logical_inputs,
            fresh_per_batch=True, nontransparent=True, binding='trusted_client_public_key_encryption',
            ciphertext_fingerprints=zero_fingerprints)
    if packed:
        evidence.update(execution_abi=execution_abi,input_slot_period=input_period)
    dump(directory / 'execution.json', evidence)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--keys", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=("base", "extended", "all"), default="all")
    selection.add_argument("--case", choices=CASES, help="Run one existing golden without the rest of its suite")
    return parser.parse_args(argv)


def selected_cases(args):
    if args.case:
        return (args.case,)
    return BASE_CASES if args.suite == "base" else EXTENDED_CASES if args.suite == "extended" else CASES


def main():
    args = parse_args()
    require(Path.cwd().resolve() == ROOT, f"Requires cwd {ROOT}")
    if not args.inside:
        require(not args.worker, "Worker requires pinned environment")
        selection = f"--case {args.case}" if args.case else f"--suite {args.suite}"
        command = (f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
                   f'{shlex.quote(str(VENV / "bin/python"))} {shlex.quote(str(SCRIPT))} --inside {selection}')
        return enter_nix(command, seconds=900)
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Requires pinned Nix/venv")
    if args.worker:
        require(args.directory is not None and args.keys is not None, "Missing worker paths")
        require(args.directory.resolve().is_relative_to(WORK / "results") and
                args.keys.resolve().is_relative_to(WORK / "results"), "Requires native result paths")
        return worker(args.worker, args.directory, args.keys)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require_python_packages(torch, np)
    os.umask(0o077)
    result = Path(tempfile.mkdtemp(prefix="seal-cpu-golden-", dir=WORK / "results"))
    print(f"SEAL CPU golden evidence: {result}", flush=True)
    env = dict(os.environ, HECATE=str(WORK / "build-dacapo/hecate-python-root"),
        PYTHONPATH=str(ROOT / "third_party/dacapo/python/hecate"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    require((Path(env["HECATE"]) / "build").resolve() == BUILD, "Missing or unexpected HECATE compatibility link")
    selected = selected_cases(args)
    report = dict(platform_identity=identity(), status="running", backend="upstream_SEAL_HEVM_CPU", cases=[], suite="single" if args.case else args.suite,
        selected_cases=list(selected), report_schema=2,
        poseidon_gpu_execution_validated=False, encrypted_execution_validated=False,
        compiler_profile=str(PROFILE), profile_sha256=digest(PROFILE), waterline=WATERLINE,
        runtime_sha256=digest(BUILD / "lib/libSEAL_HEVM.so"),
        source_hashes={str(path.relative_to(ROOT)): digest(path) for path in
            [SCRIPT, ROOT / "scripts/baseline/python_compiler_smoke.py", ROOT / "scripts/baseline/seal_artifact_gate.py",
             ROOT / "scripts/baseline/seal_keys/main.cpp", ROOT / "scripts/baseline/seal_keys/metadata.cpp",
             ROOT / "scripts/baseline/seal_keys/CMakeLists.txt",
             ROOT / "third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp", *[p for p in SOURCE.rglob("*") if p.is_file()]]})
    try:
        for case in selected:
            directory = result / case
            directory.mkdir()
            item = dict(case=case, status="reference_prepared", **prepare_reference(case, directory))
            report["cases"].append(item)
            item["layout"]["constant_policy"] = "Stock SEAL_HEVM repeats src[i % src.size()]"
            trace = [str(VENV / "bin/python"), str(SOURCE / "trace_golden.py"), case, "--output", str(directory)]
            item["trace_command"], item["trace_exit_code"] = trace, logged(trace, directory / "trace.log", env=env)
            require(item["trace_exit_code"] == 0, f"{case}: frontend failed; see trace.log")
            command = [str(BUILD / "bin/hecate-opt"), str(directory / "trace_golden.mlir"), "--eva",
                f"--ckks-config={PROFILE}", f"--waterline={WATERLINE}", "--enable-debug-printer",
                "--mlir-disable-threading", "--verify-each", "--dump-pass-pipeline", "-o", str(directory / "lowered.mlir")]
            item["compile_command"], item["compiler_exit_code"] = command, logged(command, directory / "compile.log", env=env)
            require(item["compiler_exit_code"] == 0, f"{case}: compiler failed; see compile.log")
            item["gate"] = inspect_artifacts((directory / "lowered._hecate_golden.hevm").read_bytes(),
                                             (directory / "_hecate_golden.cst").read_bytes())
            required_ops = {"rotate1": (1,), "rotate2": (1,), "square": (8,), "quartic": (3, 8),
                            "mlp4x4x2": (1, 3, 6, 7, 8, 9)}.get(case, ())
            require(all(item["gate"]["opcode_counts"].get(str(opcode), 0) > 0 for opcode in required_ops),
                    f"{case}: compiler artifact does not exercise required opcode coverage")
            item["status"] = "compiled"
            item["input_artifact_hashes"] = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
            print(f"{case}: trace/compile/gate passed; {item['gate']['opcode_counts']}", flush=True)
            dump(result / "report.json", report)
        key_build = KEY_BUILD
        configure = ["cmake", "-S", str(ROOT / "scripts/baseline/seal_keys"), "-B", str(key_build), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", "-DSEAL_DIR=" + os.environ["SEAL_DIR"]]
        report["key_configure_command"] = configure
        require(logged(configure, result / "key-configure.log", env=env) == 0, "Key helper configure failed")
        require(logged(["cmake", "--build", str(key_build), "-j2",
                        "--target", "seal_golden_keys", "seal_golden_metadata"], result / "key-build.log", env=env) == 0,
                "Key helper build failed")
        report["metadata_observer_sha256"] = digest(key_build / "libseal_golden_metadata.so")
        keys = result / "private-keys"
        keys.mkdir(mode=0o700)
        print("Creating stock-parameter tc128 keys (rotations 1,2 only)", flush=True)
        require(logged([str(key_build / "seal_golden_keys"), str(keys)], result / "parameters.json", seconds=120, env=env) == 0,
                "SEAL parameter validation/key setup failed")
        report["parameters"] = json.loads((result / "parameters.json").read_text())
        report["private_key_directory"] = str(keys)
        for item in report["cases"]:
            case, directory = item["case"], result / item["case"]
            command = [str(VENV / "bin/python"), str(SCRIPT), "--inside", "--worker", case,
                       "--directory", str(directory), "--keys", str(keys)]
            item["execution_command"] = command
            item["execution_exit_code"] = logged(command, directory / "execution.log", seconds=150, env=env)
            for name, expected in item["input_artifact_hashes"].items():
                require(digest(directory / name) == expected, f"Immutable test artifact changed: {name}")
            if item["execution_exit_code"] != 0:
                item["status"] = "runtime_failed"
            else:
                item["execution"] = json.loads((directory / "execution.json").read_text())
                with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
                    reference = arrays["reference"].copy()
                fixed = json.loads((SOURCE / "fixtures.json").read_text())
                item["comparison"] = compare(np.load(directory / "decrypted.npy", allow_pickle=False), reference,
                                               fixed["atol"], fixed["rtol"])
                item["status"] = "passed" if item["comparison"]["passed"] else "numerical_failed"
                item["layout"]["backend_layout_validated"] = item["comparison"]["passed"]
            print(f"{case}: {item['status']}", flush=True)
            dump(result / "report.json", report)
        report["encrypted_execution_validated"] = all(c["execution_exit_code"] == 0 for c in report["cases"])
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        print(f"Stopped: {error}", flush=True)
    finally:
        dump(result / "report.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
