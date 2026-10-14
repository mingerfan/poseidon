"""Prepare isolated high-operator regressions without changing the frozen benchmark."""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT / "scripts/baseline"))


def models():
    import numpy as np
    from model_decomposition import FORMAT
    rs = dict(method="newton", domain=[.1, 4.], iterations=2)
    cases = []
    def normalization(op, index, shape, norm=None, groups=None):
        channels = shape[-1] if op == "layer_norm" else shape[1]
        weight_shape = norm if op == "layer_norm" else [channels]
        gain = np.linspace(.7, 1.1, int(np.prod(weight_shape))).reshape(weight_shape).tolist()
        bias = np.linspace(-.05, .05, int(np.prod(weight_shape))).reshape(weight_shape).tolist()
        attrs = dict(eps=.2, approximation=rs)
        if norm is not None:
            attrs["normalized_shape"] = norm
        if groups is not None:
            attrs["groups"] = groups
        return dict(format=FORMAT, id=op+"_"+str(index),
                    inputs=[dict(name="x", shape=shape)],
                    constants=dict(gain=gain, bias=bias),
                    nodes=[dict(id="normalization", op=op, inputs=["x", "gain", "bias"],
                                attrs=attrs, outputs=["y"])],
                    outputs=[dict(name="result", value="y")])
    for index, (shape, norm) in enumerate([([4], [4]), ([2, 3], [3]), ([1, 2, 3], [2, 3])]):
        cases.append(normalization("layer_norm", index, shape, norm=norm))
    for index, (shape, groups) in enumerate([([1, 2, 3], 1), ([1, 4, 2], 2), ([1, 2, 2, 2], 2)]):
        cases.append(normalization("group_norm", index, shape, groups=groups))
    for index, shape in enumerate([[1, 2, 3], [2, 2, 2], [1, 2, 2, 2]]):
        cases.append(normalization("instance_norm", index, shape))
    for index, (lhs, rhs) in enumerate([([2, 2, 3], [2, 3, 2]),
                                       ([1, 2, 3], [2, 3, 2]),
                                       ([2, 3], [2, 3, 2])]):
        cases.append(dict(format=FORMAT, id="batched_matmul_"+str(index),
                          inputs=[dict(name="x", shape=lhs), dict(name="rhs", shape=rhs)],
                          constants={}, nodes=[dict(id="product", op="matmul",
                          inputs=["x", "rhs"], attrs={}, outputs=["y"])],
                          outputs=[dict(name="result", value="y")]))
    return cases


def prepare(output):
    import numpy as np
    from benchmark_graph import canonical, digest, samples, require
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    from operator_reference import evaluate as high_ref
    from model_decomposition import decompose
    from semantic_benchmark_execution import runtime_sources
    case_root = ROOT / "scripts/baseline/cases/operator-extension-v5"
    require(not case_root.exists() and not output.exists(), "Fresh fixtures and evidence required")
    output.mkdir()
    case_root.mkdir()
    frozen = runtime_sources()
    report = dict(format="poseidon-normalization-matmul-preparation-v1",
                  source_hashes=frozen, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  planned=12, rows=[], agent_generated=False, paid_calls=0,
                  actual_compile=False, encrypted_execution=False,
                  approximation_note="Two declared Newton iterations; quality relative to the original function is reported separately")
    def save():
        (output/"report.json").write_bytes(canonical(report))
    save()
    entries = []
    for model in models():
        require(runtime_sources() == frozen, "Source changed during preparation")
        lowered, binding = decompose(model)
        require(decompose(model) == (lowered, binding), "Decomposition determinism")
        error = approximation = 0.
        for values in samples(lowered, 16):
            expected = high_ref(model, values)["result"]
            ideal = high_ref(model, values, ideal=True)["result"]
            approximation = max(approximation, float(np.max(np.abs(expected-ideal))))
            for evaluate in (math_ref, torch_ref):
                actual = evaluate(lowered, values)["result"]
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                error = max(error, float(np.max(np.abs(actual-expected))))
        identifier = model["id"]
        files = {"source": model, "model": lowered, "binding": binding}
        for kind, content in files.items():
            (case_root/(identifier+"-"+kind+".json")).write_bytes(canonical(content))
        entries.append(dict(id=identifier, operator=model["nodes"][0]["op"],
                            model=identifier+"-model.json", source=identifier+"-source.json",
                            binding=identifier+"-binding.json",
                            model_sha256=digest(lowered), source_sha256=digest(model)))
        report["rows"].append(dict(id=identifier, input_shapes=[i["shape"] for i in model["inputs"]],
                                  core_nodes=len(lowered["nodes"]), status="reference_passed",
                                  reference_groups=16, reference_max_absolute_error=error,
                                  approximation_max_absolute_error=approximation,
                                  source_sha256=digest(model), model_sha256=digest(lowered),
                                  decomposition_binding=binding["binding_sha256"]))
        save()
        print(identifier, "dual_reference_passed", flush=True)
    manifest = case_root/"manifest.json"
    manifest.write_bytes(canonical(dict(format="poseidon-operator-regression-v1", cases=entries)))
    report["manifest"] = str(manifest)
    report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    save()
    print(json.dumps(dict(models=len(entries), references=2,
                          groups=sum(r["reference_groups"] for r in report["rows"]),
                          manifest=str(manifest))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from hecate_python_env import VENV, enter_nix
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()), "--inside", "--output", str(args.output)])
        return enter_nix(command, seconds=300, environment="poseidon-cpu")
    prepare(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
