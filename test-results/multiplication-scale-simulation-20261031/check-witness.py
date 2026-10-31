"""Replay a saved schedule without running its search; check the next real conv."""

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scale_simulation", here / "simulate.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sim = module.Simulation(here / "scale-trace.json")
data = json.loads(args.results.read_text())
assert data["trace_sha256"] == sim.trace_hash
conv = json.loads((here / "convolution-dag.json").read_text())


def read_plan(p):
    return module.Plan(**{**p, "children": tuple(read_plan(x) for x in p["children"])})


def find(p, oid):
    if p.id == oid:
        return p
    for c in p.children:
        result = find(c, oid)
        if result is not None:
            return result
    return None


def conv_schedule(source):
    """Keep all linear reductions high, rescale after the final reduction.
    Both serial CP layers keep their original coefficient values.  First-layer
    encodings are s40; second-layer encodings are derived from the fixed s40 exit.
    This is a concrete schedule for the actual 257-node traced linear DAG.
    """
    depth = conv["serial_cp_depth"]
    assert depth == 2
    target = 40.0
    for q in range(source.q, 0, -1):
        total_plain = target - source.s + sim.removed(source.q, q)
        last_plain = total_plain - 40 * (depth - 1)
        if last_plain >= 40:
            break
    else:
        raise AssertionError("No convolution schedule")
    states = {"input": (source.q, source.s, 0)}
    for n in conv["nodes"]:
        inputs = [states[x] for x in n["args"]]
        assert len({x[0] for x in inputs}) == 1
        raw_q, scale, old_depth = inputs[0]
        if n["kind"] == "mul_cp":
            scale += 40 if old_depth == 0 else last_plain
        else:
            assert all(abs(x[1] - scale) < 1e-7 for x in inputs)
        assert sim.legal(raw_q, scale)
        states[n["id"]] = raw_q, scale, n["cp_depth"]
    out_q, scale, _ = states[conv["output"]]
    for r in range(out_q, q, -1):
        scale -= sim.logq[r - 1]
        assert sim.legal(r - 1, scale)
    assert abs(scale - target) < 1e-7
    return {"source": asdict(source), "output_q": q, "output_s": scale,
            "consumed_q": source.q - q, "first_plain": 40,
            "second_plain": last_plain, "serial_cp_depth": depth,
            "ordinary_add_unit_muls": 0,
            "rescale_placement": "Common RS after final reduction, before the fixed-scale consumer.",
            "dag_nodes": len(conv["nodes"])}


def boot_compatible(q, scale):
    # These selected witnesses are already at fixed s40, so only ModSwitch is
    # required. The baseline Q6/s44.845 cannot reach Q6/s40 with any RS remaining.
    if q == 6 and abs(scale - 40) > 1e-8:
        return False
    if abs(scale - 40) <= 1e-8:
        return q >= 6
    for drops in range(1, q - 5):
        plain = 40 - scale + sim.removed(6 + drops, 6)
        if plain >= 30 and scale + plain < sim.capacity[6 + drops]:
            return True
    return False


incoming = sim.input
stages = []
counts = Counter()
coefficient_scales = []
for record in data["coefficient_backward"]["stages"]:
    stage = record["stage"]
    if stage == "P1(d15)":
        external, basis_stage = {"0": incoming}, stage
    elif stage == "P2(d15)":
        external, basis_stage = {sim.roots[0]: incoming}, stage
    else:
        external = {sim.roots[1]: incoming, "0": sim.input}
        basis_stage = "P3(d27)"
    fixed, basis, _ = sim.basis(basis_stage, external,
                               record.get("retain_one_basis_rs"))
    assert basis == record["basis"] or json.dumps(basis, sort_keys=True) == json.dumps(record["basis"], sort_keys=True)
    p = read_plan(record["plan"])
    incoming = sim.verify(p, fixed)
    assert abs(incoming.s - record["s"]) < 1e-7 and incoming.q == record["q"]
    for b in basis:
        counts[sim.nodes[b["node"]].kind] += 1
        counts["basis_unit_muls"] += len(b["units"])
        counts["rs_prime_divisions"] += b["drops"]

    def visit(p):
        if p.id in fixed:
            return
        n = sim.nodes[p.id]
        counts[n.kind] += 1
        counts["rs_prime_divisions"] += p.drops
        if n.kind == "mul_cp":
            coefficient_scales.append({"node": p.id, "scalar": n.scalar,
                                       "encoding_s": p.plain})
        for c in p.children:
            visit(c)

    visit(p)
    if stage.startswith("P3"):
        root = find(p, sim.roots[2])
        assert root is not None
        stages.append({"stage": "P3", "q": root.q, "s": root.s,
                       "consumed_q": 31 - root.q})
        stage = "tail"
    stages.append({"stage": stage, "q": incoming.q, "s": incoming.s,
                   "consumed_q": 31 - incoming.q})
assert counts["mul_cc"] == 27 and counts["mul_cp"] == 30, counts
assert len(coefficient_scales) == 30
baseline = data["baseline"]["stages"][-1]
old_state = module.State(**baseline)
new_conv = conv_schedule(incoming)
old_conv = conv_schedule(old_state)
out = {"same_core_multiplications": {"cc": 27, "coefficient_cp": 30},
       "coefficient_values_unchanged": True,
       "ordinary_polynomial_add_unit_muls": 0,
       "ordinary_polynomial_add_rescales": 0,
       "actual_prime_replay_passed": True,
       "counts": dict(counts), "stages": stages,
       "coefficient_scales": coefficient_scales,
       "encoding_min_s": min(x["encoding_s"] for x in coefficient_scales),
       "encoding_max_s": max(x["encoding_s"] for x in coefficient_scales),
       "baseline_next_conv": old_conv, "planned_next_conv": new_conv,
       "baseline_relu_boot_compatible": boot_compatible(old_state.q, old_state.s),
       "planned_relu_boot_compatible": boot_compatible(incoming.q, incoming.s),
       "planned_relu_then_conv_boot_compatible": boot_compatible(new_conv["output_q"], new_conv["output_s"]),
       "baseline_relu_then_conv_total_q": 31 - old_conv["output_q"],
       "planned_relu_then_conv_total_q": 31 - new_conv["output_q"],
       "scope": "Scale/level feasibility only. No ciphertext accuracy, full-network Boot count, latency, or million-node complexity claim."}
path = args.results.with_suffix(".witness-check.json")
path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
