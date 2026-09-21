"""Real Hecate Python -> compiler -> Poseidon adapter gate with PyTorch references.

Stops before encrypted execution: neither the SEAL_HEVM runtime nor a simulated
backend is used to manufacture decrypted outputs. Run the installed venv only
inside the pinned, already realized Nix shell.
"""
from platform_config import identity, require_python_packages
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix

CASES = ("add", "mul_plain", "linear4x2")
SOURCE = ROOT / "scripts/baseline/golden_cases"
BUILD = WORK / "build-dacapo/hecate-18.1.2-nix"
DUMP = WORK / "build-poseidon/agent-dsl-adapter/bin/poseidon_mgpu_dacapo_hevm_dump"


def prepare_reference(case, directory):
    import numpy as np
    import torch

    fixed = json.loads((SOURCE / "fixtures.json").read_text())
    inputs = np.array([[0.0] * 4, fixed["signed_input"],
        np.random.default_rng(fixed["seed"]).uniform(-1, 1, 4), fixed["boundary_input"]], dtype=np.float64)
    spec = importlib.util.spec_from_file_location(f"reference_{case}", SOURCE / case / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model()
    with torch.no_grad():
        reference = np.stack([model(torch.from_numpy(x.copy())).numpy() for x in inputs])
    scalar_outputs = case in ("linear4x2", "mlp4x4x2")
    weight = np.array(fixed["linear_weight"] if scalar_outputs else fixed["mul_weights"], dtype=np.float64)
    bias = np.array(fixed["linear_bias"], dtype=np.float64)
    extra = {name: np.array(fixed[name], dtype=np.float64) for name in
             ("mlp_hidden_weight", "mlp_hidden_bias")} if case == "mlp4x4x2" else {}
    np.savez(directory / "arrays.npz", inputs=inputs, reference=reference, weight=weight, bias=bias, **extra)
    (directory / "reference-fx.txt").write_text(str(torch.fx.symbolic_trace(model).graph) + "\n")
    return dict(input_shape=[4], output_shape=[2] if scalar_outputs else [4],
        dtype="float64", reference_device="cpu", input_names=["zero", "signed", "seed42", "boundary"],
        inputs=inputs.tolist(), reference=reference.tolist(),
        privacy={"inputs": "encrypted", "weights_and_bias": "public"},
        comparison={"status": "not_run", "reason": "No decrypted Poseidon outputs", "atol": fixed["atol"],
                    "rtol": fixed["rtol"], "mae": None, "max_absolute_error": None,
                    "nonzero_reference_relative_error": None, "cosine_similarity": None},
        layout={"input_ciphertexts": 1, "input_slot_pattern": "repeat [x0,x1,x2,x3] across all slots",
                "output_ciphertexts": 2 if scalar_outputs else 1,
                "output_slots": "slot 0 of each ciphertext" if scalar_outputs else "slots 0..3",
                "constant_policy": "HEVM adapter repeats CST vector via src[i % size]; direct encoder zero padding unchanged",
                "rotation_steps": [1, 2] if scalar_outputs else [int(case[-1])] if case.startswith("rotate") else [],
                "backend_layout_validated": False})


def logged(command, log_path, seconds=60, env=None):
    with log_path.open("w") as log:
        return subprocess.run(["timeout", "-k", "3s", str(seconds), *command],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds + 10).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"Requires cwd {ROOT}")
    if not args.inside:
        command = f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 {shlex.quote(str(VENV / "bin/python"))} scripts/baseline/python_compiler_smoke.py --inside'
        raise SystemExit(enter_nix(command))
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix) != VENV:
        raise SystemExit("Requires approved venv inside pinned Nix shell")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require_python_packages(torch, np)
    # Stock expr.py hardcodes $HECATE/build/lib. This derived symlink keeps the
    # upstream submodule and source tree untouched, including paths with spaces.
    compatibility = WORK / "build-dacapo/hecate-python-root"
    compatibility.mkdir(exist_ok=True)
    link = compatibility / "build"
    if link.is_symlink():
        if link.resolve() != BUILD:
            raise RuntimeError("Preserving unexpected existing HECATE link")
    elif link.exists():
        raise RuntimeError("Preserving existing non-symlink HECATE build directory")
    else:
        link.symlink_to(BUILD, target_is_directory=True)
    env = dict(os.environ, HECATE=str(compatibility), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT / "third_party/dacapo/python/hecate"),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    result = Path(tempfile.mkdtemp(prefix="python-compiler-", dir=WORK / "results"))
    report = dict(platform_identity=identity(), scope="real_hecate_python_trace_compile_adapter_gate", status="running", cases=[],
        python_dsl_tracing_validated=False, encrypted_execution_validated=False,
        compiler_config=str(ROOT / "third_party/dacapo/config.json"),
        parameter_note="Unchanged upstream compiler profile; NOT approved for GPU execution",
        versions={"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in SOURCE.rglob("*") if p.is_file()})
    print(f"Python compiler evidence: {result}", flush=True)
    try:
        environment = [str(VENV / "bin/python"), "scripts/baseline/check_dacapo_environment.py",
                       "--llvm-prefix", os.environ["LLVM_ROOT"], "--seal-prefix", os.environ["SEAL_ROOT"]]
        report["environment_gate_exit_code"] = logged(environment, result / "environment-gate.json")
        if report["environment_gate_exit_code"]:
            raise RuntimeError("Prerequisite gate failed; see environment-gate.json")
        for case in CASES:
            directory = result / case
            directory.mkdir()
            item = dict(case=case, status="reference_prepared", **prepare_reference(case, directory))
            report["cases"].append(item)
            trace = [str(VENV / "bin/python"), str(SOURCE / "trace_golden.py"), case, "--output", str(directory)]
            item["trace_command"] = trace
            item["trace_exit_code"] = logged(trace, directory / "trace.log", env=env)
            if item["trace_exit_code"]:
                raise RuntimeError(f"{case}: Hecate Python tracing failed; see trace.log")
            command = [str(BUILD / "bin/hecate-opt"), str(directory / "trace_golden.mlir"), "--eva",
                "--ckks-config=" + report["compiler_config"], "--enable-debug-printer",
                "--mlir-disable-threading", "--verify-each", "--dump-pass-pipeline",
                "-o", str(directory / "lowered.mlir")]
            item["compile_command"] = command
            item["compiler_exit_code"] = logged(command, directory / "compile.log")
            if item["compiler_exit_code"]:
                raise RuntimeError(f"{case}: compiler failed; see compile.log")
            item["status"] = "compiled"
            artifacts = ["arrays.npz", "trace_golden.mlir", "lowered.earth.mlir", "lowered.ckks.mlir",
                         "lowered.mlir", "_hecate_golden.cst", "lowered._hecate_golden.hevm"]
            item["artifacts"] = {name: dict(bytes=(directory / name).stat().st_size,
                                           sha256=digest(directory / name)) for name in artifacts}
            adapter = [str(DUMP), "--hevm", str(directory / "lowered._hecate_golden.hevm"),
                "--constants", str(directory / "_hecate_golden.cst"), "--devices", "1", "--opcode-summary",
                "--communication-plan", "--communication-execution-preflight", "--poseidon-gpu-preflight",
                "--require-ready", "--write-summary-json", str(directory / "poseidon-preflight.json"), "--no-schedule"]
            item["adapter_command"] = adapter
            host_env = dict(os.environ)
            host_env.pop("LD_LIBRARY_PATH", None)
            if case == "linear4x2":
                # Independent constant-payload diagnostic, NOT execution of the
                # artifact's level/scale profile and NOT a CPU HEVM backend.
                constant_env = dict(host_env, POSEIDON_HEVM_LINEAR_CST=str(directory / "_hecate_golden.cst"))
                constant_command = [str(DUMP.with_name("poseidon_mgpu_hevm_plaintext_encoding_tests"))]
                code = logged(constant_command, directory / "constant-encoding.log", env=constant_env)
                item["constant_payload_encoding"] = {
                    "exit_code": code, "scope": "encode_decode_only_at_existing_CPU_tc128_defaults",
                    "uses_artifact_execution_parameters": False,
                    "N": 16384, "slots": 8192, "data_level": 7, "log2_scale": 48,
                    "encrypted_execution_validated": False,
                }
                if code:
                    raise RuntimeError("Real Linear CST encode/decode regression failed")
            item["adapter_exit_code"] = logged(adapter, directory / "adapter.log", env=host_env)
            gate_report = json.loads((directory / "poseidon-preflight.json").read_text())
            item["execution_gate"] = gate_report["execution_gate"]
            item["status"] = "adapter_ready_execution_not_run" if gate_report["execution_gate"]["ok"] else "adapter_blocked"
            print(f"{case}: trace=0 compile=0 adapter={item['adapter_exit_code']}", flush=True)
            (result / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["python_dsl_tracing_validated"] = True
        report["status"] = "adapter_blocked" if any(c["status"] == "adapter_blocked" for c in report["cases"]) else "compiled_execution_not_run"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (result / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Real Python tracing/compilation completed; encrypted differential comparison NOT run.", flush=True)
    return 2 if report["status"] == "adapter_blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
