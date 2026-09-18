"""Data-only model description -> PyTorch/FX -> rule-generated Hecate -> SEAL.

First stage reference baseline, NOT Agent inference. No network/install/model API.
All case failures remain in report denominators and have a diagnostic layer.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix
from python_compiler_smoke import BUILD, SOURCE, logged
from seal_cpu_golden import KEY_BUILD, PROFILE, WATERLINE, compare, dump, execute_artifact
from seal_artifact_gate import inspect_artifacts, require
from hecate_contract import CONTRACT_ROTATIONS
from cipher_abi import physical_input_names, execution_options

SCRIPT = ROOT / "scripts/baseline/run_model_batch.py"
LAYERS = ("model_description", "model_reference", "fx_translation", "dsl_trace", "compiler",
          "artifact_gate", "seal_runtime", "numerical_comparison", "integrity")


def summarize(items, planned=None):
    denominator = len(items) if planned is None else planned
    require(type(denominator) is int and denominator >= len(items), "Invalid planned case denominator")
    metrics = {}
    for field in ("translated", "traced", "compiled", "executed", "numerically_correct"):
        count = sum(c.get(field, False) is True for c in items)
        metrics[field] = dict(numerator=count, denominator=denominator, rate=count / denominator if denominator else None)
    failures = [c for c in items if c.get("status") != "passed"]
    return dict(generator="deterministic-fx-v0", agent_calls=0, metrics=metrics,
                failure_layers=dict(Counter(c.get("failure_layer") for c in failures)),
                failure_categories=dict(Counter(c.get("failure_category", "unclassified") for c in failures)))


def frozen_manifest_valid(path, expected_sha256):
    try:
        return path.is_file() and not path.is_symlink() and digest(path) == expected_sha256
    except OSError:
        return False


def prepare_case(descriptor, folder):
    import numpy as np
    import torch
    from model_catalog import build_model, test_inputs
    from fx_to_hecate import state_arrays
    model, shape = build_model(descriptor)
    multiple = descriptor["schema"] == 3
    if multiple:
        from multi_input_fixtures import fixture_inputs
        original = np.asarray(fixture_inputs(len(shape["inputs"])), dtype=np.float64)
        logical = [original[:, i, :].reshape(4, *spec["shape"]) for i, spec in enumerate(shape["inputs"])]
    elif descriptor['schema'] == 5:
        from packed_model import test_inputs as packed_inputs
        original = packed_inputs(shape)
    elif descriptor['schema'] == 4:
        from chunked_model import test_inputs as chunked_inputs
        original = chunked_inputs(shape)
    else:
        original = test_inputs(shape)
    before = state_arrays(model,max_elements=4096 if descriptor['schema']==5 else 128)
    with torch.no_grad():
        if multiple:
            reference = np.stack([model(*[torch.from_numpy(x[i].copy()) for x in logical]).numpy() for i in range(4)])
        else:
            reference = np.stack([model(torch.from_numpy(x.copy())).numpy() for x in original])
    logical_output_shape=None
    if descriptor['schema']==5:
        from packed_model import validate as validate_packed
        logical_output_shape=validate_packed(descriptor)['output_shape']
        require(reference.shape==(4,*logical_output_shape),'PyTorch logical output shape differs from model contract')
        reference=reference.reshape(4,-1)
    if descriptor["schema"] in (2, 3, 4, 5):
        from model_graph import evaluate_reference
        supplied = ([{s["name"]: x[i].tolist() for s, x in zip(shape["inputs"], logical)} for i in range(4)]
                    if multiple else [x.tolist() for x in original])
        independent = np.asarray([evaluate_reference(descriptor, x) for x in supplied], dtype=np.float64)
        require(independent.shape == reference.shape and
                np.allclose(independent, reference, atol=1e-12, rtol=1e-12),
                "Independent graph reference disagrees with PyTorch")
        # Use independent arithmetic as the frozen target, never translated DSL.
        reference = independent
    require(reference.ndim == 2 and 1 <= reference.shape[1] <= (256 if descriptor['schema']==5 else 4) and np.isfinite(reference).all(), "Invalid plaintext reference")
    after = state_arrays(model,max_elements=4096 if descriptor['schema']==5 else 128)
    require(before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before),
            "Reference evaluation mutated model state")
    np.savez(folder / "weights.npz", **before)
    if descriptor['schema']==5:
        from packed_input_abi import ABI,pack_inputs
        extra=({'reference_logical':reference.reshape(4,*logical_output_shape)} if len(logical_output_shape)>1 else {})
        np.savez(folder/'arrays.npz',inputs=pack_inputs(original,shape),logical_inputs=original,reference=reference,**extra)
        return model,dict(execution_abi=ABI,input_shape=shape)
    if descriptor['schema'] == 4:
        from chunked_model import lower
        from chunked_input_abi import pack_inputs
        from model_graph import build_graph_model
        np.savez(folder / 'arrays.npz', inputs=pack_inputs(original, shape),
                 logical_inputs=original, reference=reference)
        physical, plan = lower(descriptor)
        lowered_model, lowered_shape = build_graph_model(physical)
        lowered_shape['model_input_binding'] = plan
        return lowered_model, lowered_shape
    # Runtime reads exactly four flattened input elements; original logical shape
    # is separately preserved, and the PyTorch reference never reads decrypted data.
    np.savez(folder / "arrays.npz", inputs=original if multiple else original.reshape(4, 4),
             logical_inputs=original, reference=reference)
    return model, shape


def run_inside(args):
    import numpy as np
    import torch
    from model_catalog import descriptors, validate_descriptor
    from fx_to_hecate import translate
    require(torch.__version__ == "2.0.1+cpu" and np.__version__ == "1.25.2", "Unexpected dependencies")
    torch.set_num_threads(2)
    os.umask(0o077)
    result = Path(tempfile.mkdtemp(prefix="fx-batch-", dir=WORK / "results"))
    print(f"Batch evidence: {result}", flush=True)
    benchmark = None
    try:
        files = getattr(args, "case_files", None) or ([args.case] if args.case else [])
        require(len(files) <= 96 and all(p.is_file() and p.stat().st_size <= 65536 for p in files), "Model descriptor size/count limit")
        custom_manifest = None
        if getattr(args, 'case_manifest', None):
            from custom_batch_manifest import load_manifest
            rows, custom_manifest, manifest_digest = load_manifest(args.case_manifest, getattr(args, 'manifest_sha256', None))
            candidates = [row['descriptor'] for row in rows]
        else:
            candidates = [json.loads(p.read_text()) for p in files] if files else descriptors()
        if getattr(args, "extended", False):
            from expanded_model_suite import manifest
            benchmark = manifest()
            selected = benchmark["entries"]
            if args.smoke:
                selected = [e for e in selected if e["configuration"] == 0]
            candidates = [e["descriptor"] for e in selected]
        require(all(d.get('schema')!=5 for d in candidates),
                'Schema5 requires the per-case packed key ABI: use run_packed_model_goldens.py or run_candidate.py; this legacy batch shares period4 keys')
    except Exception as error:
        item = dict(status="failed", failure_layer="model_description", failure_category="input",
                    diagnostic=str(error))
        dump(result / "report.json", dict(status="failed", cases=[item], summary=summarize([item])))
        return 1
    if args.smoke and not args.case and not getattr(args, "case_files", None) and custom_manifest is None and benchmark is None:
        candidates = [c for c in candidates if c["id"] in ("affine-0", "linear-1", "mlp2-1", "mlp3-1", "flatten_linear-1")]
    env = dict(os.environ, HECATE=str(WORK / "build-dacapo/hecate-python-root"),
               PYTHONPATH=str(ROOT / "third_party/dacapo/python/hecate"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    report = dict(status="running", generator="deterministic-fx-v0", agent_calls=0,
                  backend="upstream_SEAL_HEVM_CPU", cases=[], selected_descriptors=candidates,
                  profile_sha256=digest(PROFILE), waterline=WATERLINE,
                  runtime_sha256=digest(BUILD / "lib/libSEAL_HEVM.so"), poseidon_gpu_validated=False,
                  source_hashes={str(p.relative_to(ROOT)): digest(p) for p in (SCRIPT,
                      ROOT / "scripts/baseline/fx_to_hecate.py", ROOT / "scripts/baseline/model_catalog.py",
                      ROOT / "scripts/baseline/model_graph.py",
                      ROOT / "scripts/baseline/custom_batch_manifest.py",
                      ROOT / "scripts/baseline/spatial_ops.py",
                      ROOT / "scripts/baseline/expanded_model_suite.py",
                      ROOT / "scripts/baseline/trace_translated.py", ROOT / "scripts/baseline/hecate_contract.py",
                      ROOT / "scripts/baseline/seal_artifact_gate.py", ROOT / "scripts/baseline/seal_cpu_golden.py")})
    fixed = json.loads((SOURCE / "fixtures.json").read_text())
    family_map = {e["descriptor"]["id"]: e["family"] for e in benchmark["entries"]} if benchmark else {}
    if benchmark:
        report["benchmark"] = {k: v for k, v in benchmark.items() if k != "entries"}
        dump(result / "benchmark-manifest.json", benchmark)
    report["tolerance"] = {k: fixed[k] for k in ("atol", "rtol")}
    if custom_manifest is not None:
        snapshot = result/'custom-manifest.json'
        dump(snapshot, custom_manifest)
        report['custom_manifest'] = dict(file=snapshot.name, sha256=digest(snapshot), source_sha256=manifest_digest)
    keys = result / "private-keys"
    try:
        # Local compilation only; no FetchContent, install, network or system writes.
        command = ["cmake", "-S", str(ROOT / "scripts/baseline/seal_keys"), "-B", str(KEY_BUILD), "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DSEAL_DIR=" + os.environ["SEAL_DIR"]]
        require(logged(command, result / "key-configure.log", env=env) == 0, "Key helper configure failed")
        require(logged(["cmake", "--build", str(KEY_BUILD), "-j2"], result / "key-build.log", env=env) == 0,
                "Key helper build failed")
        keys.mkdir(mode=0o700)
        expanded_keys = any(type(d) is dict and type(d.get("nodes")) is list and
                            any(type(n) is dict and n.get("op") == "rotate" for n in d["nodes"]) for d in candidates)
        key_steps = CONTRACT_ROTATIONS["hecate-function-v2"] if expanded_keys else (1, 2)
        require(logged([str(KEY_BUILD / "seal_golden_keys"), str(keys), *[str(s) for s in key_steps]],
                       result / "parameters.json", 120, env) == 0,
                "Key setup failed")
        report["parameters"] = json.loads((result / "parameters.json").read_text())
        for index, descriptor in enumerate(candidates):
            folder = result / f"case-{index:03d}"
            folder.mkdir()
            item = dict(descriptor=descriptor, folder=folder.name, status="running", stages=[],
                        translated=False, traced=False, compiled=False, executed=False, numerically_correct=False)
            if benchmark:
                item["benchmark_family"] = family_map[descriptor["id"]]
            report["cases"].append(item)
            layer = "model_description"
            frozen = {}
            try:
                family, configuration = validate_descriptor(descriptor)
                item.update(case=descriptor["id"], family=family, holdout_for_agent=family in ("fanout", "residual"))
                dump(folder / "model.json", descriptor)
                item["stages"].append(layer)
                layer = "model_reference"
                model, shape = prepare_case(descriptor, folder)
                frozen = {name: digest(folder / name) for name in ("model.json", "weights.npz", "arrays.npz")}
                item["stages"].append(layer)
                layer = "fx_translation"
                payload = translate(model, shape)
                dump(folder / "translation.json", payload)
                (folder / "generated.py").write_text(payload["hecate_source"])
                (folder / "reference-fx.txt").write_text(payload["fx_graph"] + "\n")
                item.update(translated=True, layout=payload["layout"], static_check=payload["static_check"])
                item["stages"].append(layer)
                layer = "dsl_trace"
                command = [str(VENV / "bin/python"), str(ROOT / "scripts/baseline/trace_translated.py"), str(folder)]
                item["trace_command"] = command
                item["trace_exit_code"] = logged(command, folder / "trace.log", env=env)
                require(item["trace_exit_code"] == 0, "Hecate tracing failed; see trace.log")
                item["traced"] = True
                item["stages"].append(layer)
                layer = "compiler"
                command = [str(BUILD / "bin/hecate-opt"), str(folder / "trace_translated.mlir"), "--eva",
                           f"--ckks-config={PROFILE}", f"--waterline={WATERLINE}", "--enable-debug-printer",
                           "--mlir-disable-threading", "--verify-each", "-o", str(folder / "lowered.mlir")]
                item["compile_command"] = command
                item["compile_exit_code"] = logged(command, folder / "compile.log", env=env)
                require(item["compile_exit_code"] == 0, "Compiler failed; see compile.log")
                item["compiled"] = True
                item["stages"].append(layer)
                layer = "artifact_gate"
                item["artifact_gate"] = inspect_artifacts((folder / "lowered._hecate_golden.hevm").read_bytes(),
                                                          (folder / "_hecate_golden.cst").read_bytes(),
                                                          rotation_steps=CONTRACT_ROTATIONS[payload["static_check"]["contract"]],
                                                          expected_inputs=len(physical_input_names(payload['layout'])))
                for p in folder.iterdir():
                    if p.is_file():
                        frozen.setdefault(p.name, digest(p))  # Never re-freeze a changed reference.
                item["stages"].append(layer)
                layer = "seal_runtime"
                command = [str(VENV / "bin/python"), str(SCRIPT), "--inside", "--worker", str(folder), "--keys", str(keys)]
                item["execution_command"] = command
                item["execution_exit_code"] = logged(command, folder / "execution.log", 150, env)
                require(item["execution_exit_code"] == 0, "Native SEAL execution failed; see execution.log")
                item["execution"] = json.loads((folder / "execution.json").read_text())
                item["executed"] = True
                item["stages"].append(layer)
                layer = "numerical_comparison"
                with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
                    reference = arrays["reference"].copy()
                item["comparison"] = compare(np.load(folder / "decrypted.npy", allow_pickle=False), reference,
                                               fixed["atol"], fixed["rtol"])
                require(item["comparison"]["passed"], "Frozen elementwise numerical tolerance failed")
                item["numerically_correct"] = True
                item["stages"].append(layer)
                item["status"] = "passed"
            except Exception as error:
                category = "input" if layer in ("model_description", "model_reference", "fx_translation") else "pipeline"
                item.update(status="failed", failure_layer=layer, failure_category=category, diagnostic=str(error))
            finally:
                item["frozen_hashes"] = frozen
                mismatches = [name for name, expected in frozen.items() if not (folder / name).is_file() or digest(folder / name) != expected]
                if mismatches:
                    item.update(status="failed", failure_layer="integrity", failure_category="integrity",
                                diagnostic=str(mismatches), numerically_correct=False)
                dump(folder / "case-report.json", item)
                report["summary"] = summarize(report["cases"], len(candidates))
                dump(result / "report.json", report)
                print(f"{index+1}/{len(candidates)} {item.get('case', 'invalid')}: {item['status']} {item.get('failure_layer', '')}", flush=True)
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="infrastructure_failed", failure_layer="environment_or_key_setup", diagnostic=str(error))
        # Preserve every planned case even when shared infrastructure prevents attempts.
        for index in range(len(report["cases"]), len(candidates)):
            report["cases"].append(dict(descriptor=candidates[index], status="not_run",
                                         failure_layer="environment_or_key_setup", failure_category="infrastructure"))
            if benchmark:
                report["cases"][-1]["benchmark_family"] = family_map[candidates[index]["id"]]
    finally:
        if custom_manifest is not None and not frozen_manifest_valid(result/'custom-manifest.json', report['custom_manifest']['sha256']):
            report.update(status='failed', failure_layer='integrity', diagnostic='Frozen custom manifest changed')
        report["summary"] = summarize(report["cases"], len(candidates))
        dump(result / "report.json", report)
        if report['status'] in ('passed', 'failed', 'infrastructure_failed'):
            from result_retention import cleanup_run
            try:
                cleaned = cleanup_run(result, WORK / 'results')
                print(f"Key cleanup: {cleaned['bytes']} bytes", flush=True)
            except (OSError, ValueError) as error:
                print(f"Key cleanup deferred: {type(error).__name__}", flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", type=Path, help="Data-only model JSON; arbitrary Python files are not imported")
    selection.add_argument("--case-files", type=Path, nargs="+", help="Explicit descriptor files sharing one runtime key set")
    selection.add_argument('--case-manifest', type=Path, help='Same self-contained user-graph manifest accepted by the Agent batch')
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    selection.add_argument("--extended", action="store_true", help="Versioned 16-family x 6-configuration suite")
    parser.add_argument("--smoke", action="store_true", help="Five representative cases before full 48-case suite")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--unit-tests", action="store_true", help="Run FX/model tests in pinned CPU Torch environment")
    args = parser.parse_args()
    if args.manifest_sha256 and (not args.inside or not args.case_manifest):
        parser.error('--manifest-sha256 is reserved for the isolated manifest entry')
    if args.case_manifest and args.smoke:
        parser.error('--case-manifest is exact; do not silently shrink it with --smoke')
    if args.case_manifest and (args.unit_tests or args.worker or args.keys):
        parser.error('--case-manifest selects a batch, not unit tests or a runtime worker')
    require(Path.cwd().resolve() == ROOT, f"Requires cwd {ROOT}")
    if not args.inside:
        require(not args.worker and not args.keys, "Worker paths only inside pinned environment")
        extra = (" --case " + shlex.quote(str(args.case.resolve()))) if args.case else ""
        if args.case_files:
            extra += " --case-files " + " ".join(shlex.quote(str(p.resolve())) for p in args.case_files)
        if args.case_manifest:
            from custom_batch_manifest import load_manifest
            _, _, manifest_digest = load_manifest(args.case_manifest)
            extra += ' --case-manifest ' + shlex.quote(str(args.case_manifest.resolve())) + ' --manifest-sha256 ' + manifest_digest
        extra += " --extended" if args.extended else ""
        extra += " --smoke" if args.smoke else ""
        extra += " --unit-tests" if args.unit_tests else ""
        command = (f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
                   f'{shlex.quote(str(VENV / "bin/python"))} {shlex.quote(str(SCRIPT))} --inside{extra}')
        return enter_nix(command, seconds=3600 if args.extended or args.case_manifest else 900)
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Requires pinned Nix/venv")
    if args.unit_tests:
        import unittest
        tests = unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT / "scripts/baseline"), pattern=p)
                                   for p in ("test_fx_to_hecate.py", "test_model_graph.py", "test_multi_input.py", "test_schema3_graph.py", "test_spatial.py", "test_expanded_suite.py", "test_wide_linear.py", "test_grouped_spatial.py"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    if args.worker:
        require(args.keys is not None and args.worker.resolve().is_relative_to(WORK / "results") and
                args.keys.resolve().is_relative_to(WORK / "results"), "Invalid worker paths")
        manifest = json.loads((args.worker / "translation.json").read_text())
        return execute_artifact(args.worker, args.keys, manifest["layout"]["output_selectors"],
                                rotation_steps=CONTRACT_ROTATIONS[manifest["static_check"]["contract"]],
                                **execution_options(manifest['layout']))
    return run_inside(args)


if __name__ == "__main__":
    raise SystemExit(main())
