"""Native Hecate C-API -> Earth -> CKKS -> HEVM/CST diagnostic only.

This deliberately does NOT claim Hecate Python/Torch tracing or FHE execution.
ABI signatures and frontend opcodes come from the pinned tools/frontend.cpp.
"""
from platform_config import nix_platform_options
from workspace_paths import nix_environment_options
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from continue_dacapo_cpp import DEPS, ROOT, WORK, WRAPPER


def emit_case(library, case, directory):
    ptr, index, text = ct.c_void_p, ct.c_size_t, ct.c_char_p
    signatures = {
        "init": (ptr, []), "finalize": (None, [ptr]),
        "createFunc": (index, [ptr, text, ct.POINTER(ct.c_int), index, text, index]),
        "initFunc": (None, [ptr, index, ct.POINTER(index), index]),
        "createConstant": (index, [ptr, ct.POINTER(ct.c_double), ct.c_int64, text, index]),
        "createBinary": (index, [ptr, index, index, index, text, index]),
        "setOutput": (None, [ptr, index, ct.POINTER(index), index]),
        "save": (text, [ptr, text, text]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(library, name)
        function.restype, function.argtypes = result, args
    context = library.init()
    if not context:
        raise RuntimeError("Hecate context initialization returned null")
    source = str(Path(__file__).resolve()).encode()
    input_ir = directory / "input.mlir"
    try:
        function = library.createFunc(context, case.encode(), (ct.c_int * 1)(1), 1, source, 1)
        args = (index * 1)()
        library.initFunc(context, function, args, 1)
        if case == "add":
            output = library.createBinary(context, 6, args[0], args[0], source, 2)
        elif case == "mul_plain":
            weights = (ct.c_double * 4)(1.5, -2.0, 0.25, 3.0)
            plain = library.createConstant(context, weights, 4, source, 2)
            output = library.createBinary(context, 8, args[0], plain, source, 3)
        else:
            raise ValueError(case)
        library.setOutput(context, function, (index * 1)(output), 1)
        library.save(context, str(directory).encode(), str(input_ir).encode())
    finally:
        library.finalize(context)
    return input_ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"Requires cwd {ROOT}")
    if not args.inside:
        metadata = subprocess.run(["timeout", "-k", "3s", "90s", *WRAPPER,
            "nix", "eval", "--offline", "--json", *nix_platform_options(), "--file", DEPS, "metadata"],
            capture_output=True, text=True, check=True, timeout=100)
        bash = json.loads(metadata.stdout)["shell_bash"]
        command = ["timeout", "-k", "10s", "5m", "env", f"NIX_BUILD_SHELL={bash}",
            *WRAPPER, "nix-shell", "--pure", *nix_environment_options(), "--option", "substitute", "false",
            "--max-jobs", "0", "--option", "builders", "", "--option",
            "allow-import-from-derivation", "false", *nix_platform_options(), "src/poseidon/tools/dacapo/dacapo-shell.nix",
            "--run", "python3 scripts/baseline/native_compiler_smoke.py --inside"]
        raise SystemExit(subprocess.run(command, timeout=315).returncode)
    if not os.environ.get("IN_NIX_SHELL"):
        raise SystemExit("--inside requires the pinned Nix shell")
    build = WORK / "build-dacapo/hecate-18.1.2-nix"
    result = Path(tempfile.mkdtemp(prefix="native-compiler-", dir=WORK / "results"))
    report = {"scope": "native_c_api_compiler_diagnostic", "status": "running", "cases": [],
              "python_dsl_tracing_validated": False, "encrypted_execution_validated": False,
              "compiler_config": str(ROOT / "third_party/dacapo/config.json"),
              "parameter_note": "Unchanged upstream compiler profile; not approved for FHE execution"}
    report_path = result / "report.json"
    print(f"Native compiler diagnostic: {report_path}", flush=True)
    try:
        library = ct.CDLL(str(build / "lib/libHecateFrontend.so"))
        for case in ("add", "mul_plain"):
            directory = result / case
            directory.mkdir()
            item = {"case": case, "status": "running"}
            report["cases"].append(item)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            earth = emit_case(library, case, directory)
            command = [str(build / "bin/hecate-opt"), str(earth), "--eva",
                "--ckks-config=" + report["compiler_config"], "--enable-debug-printer",
                "--mlir-disable-threading", "--verify-each", "--dump-pass-pipeline",
                "-o", str(directory / "lowered.mlir")]
            item["command"] = command
            with (directory / "compile.log").open("w") as log:
                compiled = subprocess.run(["timeout", "-k", "3s", "60s", *command],
                    stdout=log, stderr=subprocess.STDOUT, timeout=70)
            item["compiler_exit_code"] = compiled.returncode
            if compiled.returncode:
                raise RuntimeError(f"{case}: compiler failed; see {directory / 'compile.log'}")
            names = ["input.mlir", "lowered.earth.mlir", "lowered.ckks.mlir", "lowered.mlir",
                     f"_hecate_{case}.cst", f"lowered._hecate_{case}.hevm"]
            item["artifacts"] = {}
            for name in names:
                data = (directory / name).read_bytes()
                if not data:
                    raise RuntimeError(f"Empty artifact: {name}")
                item["artifacts"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            item["status"] = "compiled"
        report["status"] = "compiled"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Both native graphs compiled; no Python DSL or encrypted-execution claim.", flush=True)


if __name__ == "__main__":
    main()
