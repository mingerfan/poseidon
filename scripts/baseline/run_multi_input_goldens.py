"""Bounded 2..4-input golden ABI validation using the existing SEAL CPU runtime.

No provider calls, new backend, bootstrap or installations. Public Agent input
schema remains unchanged until this prerequisite and its integration tests pass.
"""
from platform_config import identity, require_python_packages
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix
from hecate_contract import validate_function
from multi_input_fixtures import CASES, NAMES, CONSTANTS, fixture_inputs, reference, torch_model
from python_compiler_smoke import BUILD, logged
from seal_artifact_gate import inspect_artifacts, require

SCRIPT = ROOT / "scripts/baseline/run_multi_input_goldens.py"
SOURCES = ROOT / "scripts/baseline/golden_cases/multi_input"
CONTRACT = "hecate-function-v3"
PROFILE = ROOT / "third_party/dacapo/profiled_SEAL_CPU.json"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def trace(directory):
    import numpy as np
    import hecate as hc
    from candidate_trace import evaluate_tree
    spec = json.loads((directory / "manifest.json").read_text())
    source = (directory / "golden.py").read_text()
    validate_function(source, spec["constants"], spec["outputs"],
                      contract=CONTRACT, input_names=spec["input_names"])
    constants = {k: np.asarray(v if isinstance(v, list) else [v], dtype=np.float64)
                 for k, v in spec["constants"].items()}

    @hc.func(",".join(["c"] * len(spec["input_names"])))
    def golden(*inputs):
        return evaluate_tree(source, constants, encrypted_inputs=dict(zip(spec["input_names"], inputs)))

    hc.save(str(directory), str(directory))
    dump(directory / "trace-evidence.json", dict(frontend="real_Hecate",
        candidate_python_executed=False, input_names=spec["input_names"], contract=CONTRACT))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--keys", type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Requires source cwd")
    if not args.inside:
        require(not args.trace and not args.worker and not args.keys, "Worker requires pinned environment")
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
                   f'{shlex.quote(str(VENV / "bin/python"))} {shlex.quote(str(SCRIPT))} --inside')
        if args.case:
            command += " --case " + args.case
        return enter_nix(command, seconds=1800)
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Requires pinned Nix/venv")
    for path in (args.trace, args.worker, args.keys):
        if path:
            require(path.resolve().is_relative_to(WORK / "results"), "Worker path outside native results")
    # Child compiler/runtime processes never inherit provider secrets.
    for name in ("DEEPSEEK_API_KEY",):
        os.environ.pop(name, None)
    if args.trace:
        return trace(args.trace)
    if args.worker:
        from seal_cpu_golden import execute_artifact
        require(args.keys is not None, "Missing keys")
        manifest = json.loads((args.worker / "manifest.json").read_text())
        return execute_artifact(args.worker, args.keys, manifest["selectors"],
                                expected_inputs=len(manifest["input_names"]))
    import numpy as np
    import torch
    from seal_cpu_golden import compare, KEY_BUILD
    torch.set_num_threads(2)
    require_python_packages(torch, np)
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix="multi-input-golden-", dir=WORK / "results"))
    print(f"Multi-input evidence: {root}", flush=True)
    env = dict(os.environ, HECATE=str(WORK / "build-dacapo/hecate-python-root"),
               PYTHONPATH=str(ROOT / "third_party/dacapo/python/hecate"), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    selected = [(name, False) for name in ([args.case] if args.case else CASES)]
    if args.case is None:
        selected.append(("ordered_subtract", True))
    report = dict(platform_identity=identity(), status="running", generator="manual_golden", agent_calls=0,
        backend="upstream_SEAL_HEVM_CPU", poseidon_gpu_validated=False, contract=CONTRACT,
        runtime_sha256=digest(BUILD / "lib/libSEAL_HEVM.so"), profile_sha256=digest(PROFILE),
        frontend_sha256=digest(BUILD / "lib/libHecateFrontend.so"), cases=[],
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
            [SCRIPT, ROOT / "scripts/baseline/multi_input_fixtures.py",
             ROOT / "scripts/baseline/candidate_trace.py", ROOT / "scripts/baseline/hecate_contract.py",
             ROOT / "scripts/baseline/seal_artifact_gate.py", ROOT / "scripts/baseline/seal_cpu_golden.py"]})
    try:
        keys = root / "private-keys"
        keys.mkdir(mode=0o700)
        require(logged([str(KEY_BUILD / "seal_golden_keys"), str(keys)], root / "parameters.json",
                       seconds=120, env=env) == 0, "Existing key helper failed")
        report["parameters"] = json.loads((root / "parameters.json").read_text())
        for index, (case, wrong) in enumerate(selected):
            directory = root / f"case-{index:03d}"
            directory.mkdir()
            arity, logical_outputs = CASES[case]
            nresults = 2 if case == "dual_outputs" else 1
            selectors = [[0, i] for i in range(logical_outputs)] if nresults == 1 else [[0, 0], [1, 0]]
            manifest = dict(case=case, input_names=list(NAMES[:arity]), constants=CONSTANTS,
                            outputs=nresults, selectors=selectors)
            source = (SOURCES / ("wrong_order.py" if wrong else case+".py")).read_text()
            item = dict(case=case, directory=directory.name, counterexample=wrong,
                        expected_numeric_pass=not wrong, failure_layer=None, status="running")
            report["cases"].append(item)
            layer = "reference"
            try:
                inputs = np.asarray(fixture_inputs(arity), dtype=np.float64)
                refs = np.asarray([reference(case, batch.tolist(), CONSTANTS) for batch in inputs])
                model = torch_model(case, CONSTANTS)
                with torch.no_grad():
                    torch_refs = np.stack([model(*[torch.from_numpy(v.copy()) for v in batch]).numpy()
                                           for batch in inputs])
                require(np.allclose(refs, torch_refs, atol=1e-12, rtol=1e-12), "Independent reference/Torch mismatch")
                np.savez(directory / "arrays.npz", inputs=inputs, reference=refs)
                dump(directory / "manifest.json", manifest)
                (directory / "golden.py").write_text(source)
                layer = "static_check"
                item["static_check"] = validate_function(source, CONSTANTS, nresults,
                                                          contract=CONTRACT, input_names=NAMES[:arity])
                immutable = [directory / n for n in ("arrays.npz", "manifest.json", "golden.py")]
                initial_hashes = {p.name: digest(p) for p in immutable}
                layer = "frontend"
                command = [str(VENV / "bin/python"), str(SCRIPT), "--inside", "--trace", str(directory)]
                item["trace_command"] = command
                require(logged(command, directory / "trace.log", env=env) == 0, "Multi-input trace failed")
                layer = "compiler"
                command = [str(BUILD / "bin/hecate-opt"), str(directory / "run_multi_input_goldens.mlir"),
                    "--eva", f"--ckks-config={PROFILE}", "--waterline=40", "--enable-debug-printer",
                    "--mlir-disable-threading", "--verify-each", "-o", str(directory / "lowered.mlir")]
                item["compile_command"] = command
                require(logged(command, directory / "compile.log", env=env) == 0, "Multi-input compiler failed")
                layer = "artifact_gate"
                hp, cp = directory / "lowered._hecate_golden.hevm", directory / "_hecate_golden.cst"
                item["artifact_gate"] = inspect_artifacts(hp.read_bytes(), cp.read_bytes(), expected_inputs=arity)
                immutable.extend((hp, cp))
                initial_hashes.update({p.name: digest(p) for p in (hp, cp)})
                item["immutable_hashes"] = initial_hashes
                layer = "seal_runtime"
                command = [str(VENV / "bin/python"), str(SCRIPT), "--inside", "--worker", str(directory), "--keys", str(keys)]
                item["execution_command"] = command
                require(logged(command, directory / "execution.log", seconds=150, env=env) == 0,
                        "Multi-input encrypted execution failed")
                item["execution"] = json.loads((directory / "execution.json").read_text())
                layer = "integrity"
                require(all(digest(p) == initial_hashes[p.name] for p in immutable), "Frozen artifact changed")
                item["immutable_artifacts_verified"] = True
                layer = "numerical_comparison"
                item["comparison"] = compare(np.load(directory / "decrypted.npy", allow_pickle=False), refs, 1e-5, 1e-4)
                item["status"] = "passed" if item["comparison"]["passed"] else "numerical_failed"
                item["matched_expected"] = item["comparison"]["passed"] == (not wrong)
                if not item["comparison"]["passed"]:
                    item["failure_layer"] = layer
            except Exception as error:
                item.update(status="failed", failure_layer=layer, matched_expected=False, error=str(error))
            print(f"{index+1}/{len(selected)} {case} wrong={wrong}: {item['status']}", flush=True)
            dump(root / "report.json", report)
        report["status"] = "passed" if all(c["matched_expected"] for c in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=str(error))
    finally:
        dump(root / "report.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
