"""Offline trusted-source and mathematical audit for the bounded v16 MHA draft.

Run only through the existing coordinated pure Nix Python. No SDK, compiler,
FHE, Provider, HTTP, installer or production source-capture entry is invoked.
Private arrays are saved only inside the fresh local evidence directory.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import unittest
from pathlib import Path
from output_contract import SOURCE_FORMAT, flatten
from source_plan import BASE, ROOT, digest, require, source_identity, capture, verify, source_reference
import fixtures

WORK = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
VENV = WORK / "venvs/hecate-2.0.1-cpu"
SECONDS = 180
ARTIFACT_BYTES = 32 * 1024**2

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def samples(signature, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    for index in range(16):
        row = {}
        for item in signature:
            shape = item["shape"]
            size = math.prod(shape)
            if index == 0:
                value = np.zeros(shape, dtype=np.float64)
            elif index == 1:
                value = np.linspace(-.15, .15, size, dtype=np.float64).reshape(shape)
            elif index == 3:
                value = np.asarray([(-1)**i*.15 for i in range(size)], dtype=np.float64).reshape(shape)
            else:
                value = rng.uniform(-.15, .15, shape).astype(np.float64)
            row[item["name"]] = value
        yield index, row

def tensor_outputs(value, plan, torch):
    leaves, record = flatten(value, lambda x: isinstance(x, torch.Tensor),
                             plan["source_request"]["outputs"], source_format=SOURCE_FORMAT)
    require(record == plan["output_record"], "native_output_tree")
    return {name: leaf.detach().numpy().copy()
            for name, leaf in zip(record["tensor_names"], leaves)}

def public_state(module, torch):
    result = {}
    for name, value in module.state_dict().items():
        raw = value.detach().numpy()
        result[name] = dict(dtype=raw.dtype.str, shape=list(raw.shape),
                            bytes_sha256=hashlib.sha256(raw.tobytes(order="C")).hexdigest())
    return result

def files(output):
    result = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "artifact_symlink")
        if path.is_file():
            result[str(path.relative_to(output))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    require(sum(x["bytes"] for x in result.values()) <= ARTIFACT_BYTES, "artifact_budget")
    return result

def run(output):
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV,
            "existing_pinned_pure_python")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    import numpy as np
    import torch
    from benchmark_math import evaluate as core_math
    from benchmark_torch import evaluate as core_torch
    from platform_config import require_python_packages, require_platform
    require_platform("aarch64-linux")
    require_python_packages(torch, np)
    require(output.is_absolute() and output.parent.is_dir() and
            output.parent == output.parent.resolve() and
            output.parent.is_relative_to(WORK/"results") and
            not output.exists() and not output.is_symlink(), "fresh_local_evidence")
    frozen = source_identity()
    runtime_paths = (Path(torch.nn.functional.__file__).resolve(),
                     Path(sys.modules[torch.nn.MultiheadAttention.__module__].__file__).resolve())
    runtime_sources = {str(p): sha(p) for p in runtime_paths}
    start = time.monotonic()
    output.mkdir(mode=0o700)
    def observe():
        require(time.monotonic()-start <= SECONDS, "bounded_plaintext_deadline")
        require(source_identity() == frozen and
                all(sha(Path(p)) == h for p,h in runtime_sources.items()), "frozen_source_identity")
        files(output)
    freeze = dict(format="poseidon-static-mha-v16-draft-freeze-v1",
                  created_unix=time.time(), seconds=SECONDS, artifact_bytes=ARTIFACT_BYTES,
                  source_hashes=frozen, runtime_source_hashes=runtime_sources,
                  api=0, compiler=0, encrypted_execution=0, production_activation=0)
    freeze["binding"] = digest(freeze)
    dump(output/"freeze.json", freeze)
    snapshot = output/"source-snapshot"; snapshot.mkdir()
    for relative, expected in frozen.items():
        source = ROOT/relative
        value = source.read_bytes()
        require(hashlib.sha256(value).hexdigest() == expected, "snapshot_source_binding")
        target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream: stream.write(value)
    runtime_snapshot = output/"runtime-source-snapshot"; runtime_snapshot.mkdir()
    for index, (path, expected) in enumerate(runtime_sources.items()):
        value = Path(path).read_bytes()
        require(hashlib.sha256(value).hexdigest() == expected, "snapshot_runtime_binding")
        with (runtime_snapshot/(str(index)+"-"+Path(path).name)).open("xb") as stream: stream.write(value)
    rows = []
    try:
        import test_controls, test_math
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                                   for m in (test_controls, test_math))
        with (output/"units.log").open("x") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        units = dict(run=result.testsRun, failed=len(result.failures), errors=len(result.errors),
                     skipped=len(result.skipped))
        dump(output/"units-result.json", units)
        require(result.wasSuccessful() and not result.skipped, "source_binding_and_admission_controls")
        observe()
        for index, case in enumerate(fixtures.CASES):
            observe()
            module, inputs, outputs = fixtures.build(case, index)
            plan, context = capture(module, inputs, outputs,
                                    approximation=fixtures.APPROX, trusted=True)
            anchored_request = plan["source_request_sha256"]
            verify(plan, expected_source_request_sha256=anchored_request)
            case_dir = output/case["id"]; case_dir.mkdir()
            dump(case_dir/"plan.json", plan)
            (case_dir/"source-fx.py").write_text(context["source_fx"].code)
            (case_dir/"source-fx.txt").write_text(str(context["source_fx"].graph))
            state_dir = case_dir/"original-state"; state_dir.mkdir()
            state_manifest = {}
            for ordinal, (name, tensor) in enumerate(context["original_module"].state_dict().items()):
                value = tensor.detach().numpy().copy()
                state_path = state_dir/(str(ordinal)+".npy")
                with state_path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                source_state = plan["source_request"]["public_state"][name]
                require(value.dtype.str == source_state["dtype"] and
                        list(value.shape) == source_state["shape"] and
                        hashlib.sha256(value.tobytes(order="C")).hexdigest() == source_state["bytes_sha256"],
                        "original_state_file_binding")
                state_manifest[name] = dict(file=str(state_path.relative_to(case_dir)),
                                            file_sha256=sha(state_path),
                                            dtype=value.dtype.str, shape=list(value.shape),
                                            bytes_sha256=source_state["bytes_sha256"])
            dump(case_dir/"original-state.json", state_manifest)
            before = public_state(context["original_module"], torch)
            fx_before = public_state(context["source_fx"], torch)
            rng = torch.get_rng_state().clone()
            observations = []; count = 0
            errors = dict(original_reference=0., source_fx=0., lowered_numpy=0.,
                          lowered_torch=0., declared_approximation_vs_original=0.)
            for probe, values in samples(inputs, 31411+index):
                observe()
                args = [torch.from_numpy(values[item["name"]].copy()) for item in inputs]
                with torch.no_grad():
                    native = tensor_outputs(context["original_module"](*args), plan, torch)
                    traced = tensor_outputs(context["source_fx"](*[x.clone() for x in args]), plan, torch)
                original = source_reference(plan, values, approximate=False)
                expected = source_reference(plan, values, approximate=True)
                actual = core_math(plan["lowered_graph"], values, resource_profile="general-tensors-v1")
                tensor = core_torch(plan["lowered_graph"], values, resource_profile="general-tensors-v1")
                require(list(native) == list(traced) == list(original) == list(expected) ==
                        list(actual) == list(tensor), "named_output_identity")
                for name, truth in native.items():
                    comparisons = (traced[name], original[name], expected[name], actual[name], tensor[name])
                    require(all(v.dtype == np.float64 and v.shape == truth.shape and
                                np.isfinite(v).all() for v in (truth,)+comparisons), "finite_output_signature")
                    checks = (("original_reference", truth, original[name]),
                              ("source_fx", truth, traced[name]),
                              ("lowered_numpy", actual[name], expected[name]),
                              ("lowered_torch", tensor[name], expected[name]))
                    for label, left, right in checks:
                        require(np.allclose(left, right, rtol=1e-12, atol=1e-12), label)
                        errors[label] = max(errors[label], float(np.max(np.abs(left-right))))
                    errors["declared_approximation_vs_original"] = max(
                        errors["declared_approximation_vs_original"],
                        float(np.max(np.abs(expected[name]-original[name]))))
                    count += int(truth.size)
                require(public_state(context["original_module"],torch) == before and
                        public_state(context["source_fx"],torch) == fx_before and
                        torch.equal(rng,torch.get_rng_state()), "reference_source_purity")
                observations.append(dict(probe=probe, inputs={k:v.tolist() for k,v in values.items()},
                    native_source={k:v.tolist() for k,v in native.items()},
                    traced_source={k:v.tolist() for k,v in traced.items()},
                    independent_original={k:v.tolist() for k,v in original.items()},
                    independent_approximation={k:v.tolist() for k,v in expected.items()},
                    lowered_numpy={k:v.tolist() for k,v in actual.items()},
                    lowered_torch={k:v.tolist() for k,v in tensor.items()},
                    static_none_count=plan["output_record"]["static_none_count"]))
            dump(case_dir/"references.json", dict(format="poseidon-static-mha-v16-local-references-v1",
                 source_request_sha256=anchored_request, plan_binding=plan["binding"],
                 original_state_manifest_sha256=sha(case_dir/"original-state.json"), rows=observations))
            verify(plan, expected_source_request_sha256=anchored_request)
            row = dict(id=case["id"], status="source_math_passed", configuration=case,
                source_request_sha256=anchored_request, plan_binding=plan["binding"],
                plan_file_sha256=sha(case_dir/"plan.json"), source_fx_sha256=sha(case_dir/"source-fx.py"),
                original_state_manifest_sha256=sha(case_dir/"original-state.json"),
                references_sha256=sha(case_dir/"references.json"),
                source_calls=len(plan["source_request"]["calls"]), reference_groups=len(observations),
                output_values=count, named_tensor_outputs=len(outputs),
                static_none_count=plan["output_record"]["static_none_count"],
                projection_origins=plan["normalization"]["calls"], maximum_errors=errors,
                original_nodes=len(plan["mathematical_graph"]["nodes"]),
                lowered_nodes=len(plan["lowered_graph"]["nodes"]),
                production_intake_integrated=False, original_function_accuracy_certified=False)
            rows.append(row)
            dump(case_dir/"summary.json", row)
            print(json.dumps({k:row[k] for k in ("id","status","reference_groups","output_values",
                                                "static_none_count","maximum_errors")}), flush=True)
        observe()
        report = dict(format="poseidon-static-mha-v16-draft-audit-v1",
            status="draft_source_math_validated", source_format=SOURCE_FORMAT,
            source_hashes=frozen, runtime_source_hashes=runtime_sources,
            freeze_sha256=sha(output/"freeze.json"), units=units, rows=rows,
            math_passed=len(rows), failed=0, skipped=0,
            trusted_local_source_fx_captures=len(rows), source_calls=sum(r["source_calls"] for r in rows),
            reference_groups=sum(r["reference_groups"] for r in rows),
            output_values=sum(r["output_values"] for r in rows),
            artifacts=files(output), runtime=dict(python=list(sys.version_info[:3]),
                numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
            seconds=time.monotonic()-start,
            production_source_capture=0, production_intake_integrated=False, production_activation=0,
            compiler_calls=0, encrypted_execution=0, decryption=0,
            agent_generation=0, paid_calls=0, http_attempts=0,
            installation=0, download=0, parameters_changed=False, existing_reference_changed=False,
            original_function_accuracy_certified=False, new_unique_benchmark_models_claimed=0,
            plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),
            scope="Four bounded trusted local native-MHA sources and independent mathematical references; draft only")
        report["binding"] = digest(report)
        dump(output/"report.json", report)
        print(json.dumps({k:report[k] for k in ("binding","units","math_passed","reference_groups",
                                              "output_values","seconds")}), flush=True)
        return 0
    except BaseException as error:
        failure = dict(format="poseidon-static-mha-v16-draft-failure-v1",
                       status="incomplete", failure_type=type(error).__name__,
                       message=str(error), completed_cases=len(rows),
                       freeze_sha256=sha(output/"freeze.json"), seconds=time.monotonic()-start,
                       api=0, compiler=0, encrypted_execution=0, production_activation=0)
        dump(output/"failure.json", failure)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
