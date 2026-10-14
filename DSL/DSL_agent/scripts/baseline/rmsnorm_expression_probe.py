"""Bounded developer probe for the frozen RMSNorm failure; never an app path."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from benchmark_graph import canonical, digest, require, samples
from compiler_configuration import configuration
from unified_graph_contract import prepare, validate_candidate

MODEL_SHA256 = "54a051003a1ef5ccd6356c54ce09a3c2095c7e5ee4b7b2f22e47bd84d23f3d50"


def variants(request):
    require(digest(request["model"]) == MODEL_SHA256, "Probe is bound to the original RMSNorm model")
    require(request["layout"]["input_slot_period"] == 4, "Original RMSNorm packing")
    constants = request["public_constants"]
    def scalar(value):
        return next(k for k, v in constants.items() if type(v) in (float, int) and v == value)
    quarter, half, negative, initial, one_half = map(scalar, (.25, .5, -.5, .8944271909999159, 1.5))
    require(all(constants.get("mask"+str(i)) == [float(j == i) for j in range(4)]
                for i in range(4)), "Original output selectors")
    rows = {}
    for name, factored, public_first, early_gain in (
        ("packed-original", False, False, False),
        ("packed-factored", True, False, False),
        ("packed-early-gain", True, False, True),
        ("packed-public-first", True, True, True),
    ):
        lines = ['@hc.func("c,c")', "def golden(x, zero_ct):",
                 "    u = x * x", "    t = u + u.rotate(1)",
                 "    s = (t + t.rotate(2)) * "+quarter+" + "+quarter]
        def gain(value, result):
            lines.extend(["    a = "+value+" * mask0", "    b = ("+value+" * "+half+") * mask1",
                          "    c = ("+value+" * "+negative+") * mask2", "    d = "+value+" * mask3",
                          "    "+result+" = ((a + b) + c) + d"])
        if early_gain:
            gain("x", "gx")
        if factored:
            lines.append("    factor = s * "+negative)
        if public_first:
            lines.append("    h0 = ((factor * "+initial+") * "+initial+" + "+one_half+") * "+initial)
            start = 1
        else:
            lines.append("    h0 = zero_ct + "+initial)
            start = 0
        for i in range(start, 3):
            previous = "h"+str(i if not public_first else i-1)
            result = "h"+str(i+1 if not public_first else i)
            squared = "q"+str(i)
            lines.append("    "+squared+" = "+previous+" * "+previous)
            if factored:
                lines.append("    "+result+" = "+previous+" * ("+squared+" * factor + "+one_half+")")
            else:
                lines.append("    "+result+" = "+previous+" * (("+squared+" * s) * "+negative+" + "+one_half+")")
        h = "h2" if public_first else "h3"
        lines.append("    normalized = "+("gx" if early_gain else "x")+" * "+h)
        if early_gain:
            lines.append("    return [normalized]")
        else:
            gain("normalized", "result")
            lines.append("    return [result]")
        rows[name] = "\n".join(lines)+"\n"
    return rows


def plaintext_check(source, request):
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as torch_reference
    from test_packed_prefix_lowering import public_execute
    largest = 0.
    for inputs in samples(request["model"], 16):
        actual = public_execute(source, request, inputs)
        for evaluate in (mathematical, torch_reference):
            expected = evaluate(request["model"], inputs)
            for index, output in enumerate(request["model"]["outputs"]):
                target = expected[output["name"]].reshape(-1)
                value = actual[index][:len(target)]
                np.testing.assert_allclose(value, target, rtol=1e-12, atol=1e-12)
                largest = max(largest, float(np.max(np.abs(value-target))))
    return dict(probes=16, references=2, max_absolute_error=largest,
                kind="plaintext candidate equivalence; not encrypted evidence")


def main(*, expected_model_sha256=MODEL_SHA256, make_variants=variants, probe_kind="rmsnorm"):
    from hecate_python_env import ROOT, WORK, digest as file_digest
    from check_poseidon_helpers import snapshot
    p = argparse.ArgumentParser()
    p.add_argument("--original-evidence", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--variant", action="append", default=[], help="Run only these named expressions; repeat to select multiple")
    a = p.parse_args()
    original = a.original_evidence.resolve()
    require(original.is_relative_to(WORK/"results"), "Original evidence location")
    model = json.loads((original/"model.json").read_text())
    require(digest(model) == expected_model_sha256, "Original model identity")
    prepared = []
    for waterline in (45, 40):
        config = configuration("poseidon-cpu-eva-w"+str(waterline)+"-v1")
        request = prepare(model, config["ckks_config_sha256"], config)
        expressions = make_variants(request)
        require(len(a.variant)==len(set(a.variant)) and all(n in expressions for n in a.variant), "Unknown or repeated expression selection")
        for name, source in expressions.items():
            if a.variant and name not in a.variant: continue
            candidate = dict(schema=1, request_id=request["request_id"], hecate_source=source)
            validate_candidate(candidate, request)
            checked = plaintext_check(source, request)
            prepared.append(dict(name=name, waterline=waterline, request=request,
                                 candidate=candidate, plaintext=checked))
    frozen = snapshot()
    plan = dict(format="poseidon-expression-probe-v1", probe_kind=probe_kind, backend="poseidon_cpu",
                model_sha256=expected_model_sha256,
                original_evidence=str(original),
                original_hashes={n:file_digest(original/n) for n in ("model.json","request.json","arrays.npz")},
                sources=frozen, max_wall_seconds=600, max_artifact_bytes=1024**3,
                max_native_tasks=1, max_variants=len(prepared), paid_calls=0, agent_generated=False,
                cases=[dict(name=r["name"], waterline=r["waterline"], request_id=r["request"]["request_id"],
                            candidate_sha256=digest(r["candidate"]), plaintext=r["plaintext"])
                       for r in prepared])
    if a.plan:
        print(json.dumps(plan, indent=2))
        return 0
    require(a.output.resolve().is_relative_to(WORK/"results") and not a.output.exists(), "Fresh evidence")
    a.output.mkdir()
    (a.output/"plan.json").write_bytes(canonical(plan))
    from component_backend import qualify
    start = time.monotonic()
    records = []
    for row in prepared:
        require(time.monotonic()-start < 600 and snapshot() == frozen, "Time/source boundary")
        require(all(file_digest(original/n) == h for n,h in plan["original_hashes"].items()),
                "Historical evidence changed")
        folder = a.output/(row["name"]+"-w"+str(row["waterline"]))
        folder.mkdir()
        (folder/"request.json").write_bytes(canonical(row["request"]))
        (folder/"candidate.py").write_text(row["candidate"]["hecate_source"])
        result = qualify(row["request"], row["candidate"])
        record = dict(name=row["name"], waterline=row["waterline"], plaintext=row["plaintext"], **result)
        records.append(record)
        paths = [a.output]+[Path(r["evidence"]) for r in records if r.get("evidence")]
        used = sum(f.stat().st_size for path in paths for f in path.rglob("*") if f.is_file())
        summary = dict(plan_sha256=file_digest(a.output/"plan.json"), rows=records, planned=len(prepared),
                       passed=sum(r["status"]=="passed" for r in records),
                       failed=sum(r["status"]!="passed" for r in records),
                       not_run=len(prepared)-len(records), seconds=time.monotonic()-start, artifact_bytes=used,
                       paid_calls=0, agent_generated=False)
        (a.output/"report.json").write_text(json.dumps(summary, indent=2))
        print(row["name"], row["waterline"], result["status"], result.get("failure"), flush=True)
        require(used <= plan["max_artifact_bytes"], "Artifact boundary")
        require(result.get("failure",{}).get("layer") not in ("integrity","environment","sandbox"),
                "Environment/integrity fault")
    return int(any(r["status"]!="passed" for r in records))

if __name__ == "__main__":
    raise SystemExit(main())
