"""Fast scale/level prototype. No compiler, ciphertext, or GPU execution.

The threshold polynomial DAG is reconstructed from the saved trace. Linear
regions use a declared joint-planning contract; the reference ReLU uses its
measured Q31/s40 -> Q9/s40 contract. This is a controlled model, not DaCapo.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
import hashlib
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
TRACE = HERE.parent / "resnet20-scale-trace-20261027/scale-trace.json"
FIXTURE = (
    HERE.parents[1] / "third_party/ckks-runtime/tests/fixtures/resnet20-manual-q50.json"
)
DATA = json.loads(TRACE.read_text())
Q = json.loads(FIXTURE.read_text())["rns_moduli"]
LOGQ = [math.log2(q) for q in Q]
CAPACITY = [sum(LOGQ[:i]) for i in range(len(Q) + 1)]


@dataclass(frozen=True, slots=True)
class State:
    q: int
    s: float
    nominal: int


class Infeasible(Exception):
    pass


def normalized(q, s, nominal):
    while q > 1 and s - LOGQ[q - 1] >= 40:
        s -= LOGQ[q - 1]
        q -= 1
        nominal = math.ceil(s)
    if q < 6 or s >= CAPACITY[q]:
        raise Infeasible
    return State(q, s, nominal)


def add(a, b):
    nominal = max(a.nominal, b.nominal)
    s = max(a.s + nominal - a.nominal, b.s + nominal - b.nominal)
    return State(min(a.q, b.q), s, nominal)


def arithmetic(a, b, kind):
    if kind == "mul_cc":
        return normalized(min(a.q, b.q), a.s + b.s, a.nominal + b.nominal)
    if kind == "mul_cp":
        return normalized(a.q, a.s + 40, a.nominal + 40)
    if kind == "add_cc":
        return add(a, b)
    if kind in ("add_cp", "negate"):
        return a
    raise ValueError(kind)


def polynomial_regions():
    source = DATA["compiler"]
    alias = {"0": "0"}
    groups = {stage: [] for stage in ("P1(d15)", "P2(d15)", "P3(d27)", "尾部")}
    for op in source["operations"]:
        oid, kind = op["id"], op["kind"]
        if not source["values"][oid]["cipher"]:
            continue
        if kind == "input":
            continue
        cipher_args = [a for a in op["inputs"] if source["values"][a]["cipher"]]
        args = [alias[a] for a in cipher_args]
        if kind in ("transfer", "rescale", "mod_switch", "relinearize") or (
            kind == "mul_cp" and op["scalar"] == 1
        ):
            alias[oid] = args[0]
        else:
            alias[oid] = oid
            groups[op["stage"]].append((oid, kind, tuple(args)))
    roots = dict(zip(groups, ("103", "211", "388", "395")))
    regions = {}
    for stage, ops in groups.items():
        internal = {o[0] for o in ops}
        external = tuple(
            sorted({a for _, _, args in ops for a in args if a not in internal})
        )
        regions[stage] = (tuple(ops), external, alias[roots[stage]])
    assert [len(v[1]) for v in regions.values()] == [1, 1, 1, 2]
    return regions


REGIONS = polynomial_regions()


@lru_cache(maxsize=100000)
def poly(stage, incoming):
    ops, external, output = REGIONS[stage]
    states = dict(zip(external, incoming))
    try:
        for oid, kind, args in ops:
            a = states[args[0]]
            b = states[args[1]] if len(args) > 1 else a
            states[oid] = arithmetic(a, b, kind)
        return states[output]
    except Infeasible:
        return None


@lru_cache(maxsize=100000)
def alignment(state, floor):
    if state.q < 6:
        return None
    if floor == "ideal_exit":
        # Counterfactual oracle: the producer supplies s40 at no level cost.
        # This is not an executable CKKS alignment or a relaxed real Boot API.
        return (0, 0.0)
    if abs(state.s - 40) <= 1e-8:
        return (0, 0.0)
    plain = 40 - state.s
    for drops in range(1, state.q - 5):
        plain += LOGQ[5 + drops]
        if plain >= floor and state.s + plain < CAPACITY[6 + drops]:
            return (drops, plain)
    return None


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    kind: str
    inputs: tuple
    output: str
    stage: str = ""
    cp_depth: int = 0
    terminal: bool = False


def network(policy):
    nodes = []

    def linear(name, x, depth=2, terminal=False):
        nodes.append(
            Node(name, "linear", (x,), name, cp_depth=depth, terminal=terminal)
        )
        return name

    def relu(name, x):
        if policy == "manual22":
            nodes.append(Node(name, "manual22", (x,), name))
            return name
        mapping = {"0": x}
        for stage, (_, external, output) in REGIONS.items():
            key = name + "/" + stage
            nodes.append(
                Node(key, "poly", tuple(mapping[a] for a in external), key, stage=stage)
            )
            mapping[output] = key
        return key

    x = relu("relu0", linear("stem", "input"))
    for block in range(9):
        shortcut = x
        if block in (3, 6):
            shortcut = linear(f"block{block}/projection", x, depth=1)
        h = relu(f"relu{2 * block + 1}", linear(f"block{block}/conv1", x))
        h = linear(f"block{block}/conv2", h)
        key = f"block{block}/sum"
        nodes.append(Node(key, "add", (h, shortcut), key))
        x = relu(f"relu{2 * block + 2}", key)
    linear("pool-fc", x, depth=3, terminal=True)
    return nodes


@lru_cache(maxsize=100000)
def evaluate(node, inputs):
    if node.kind == "poly":
        return poly(node.stage, inputs)
    if node.kind == "manual22":
        a = inputs[0]
        # A measured region contract, used only at its verified entry state.
        return State(9, 40.0, 40) if a.q >= 31 and abs(a.s - 40) <= 1e-8 else None
    if node.kind == "add":
        return add(*inputs)
    if node.kind == "linear":
        a = inputs[0]
        removed = 0.0
        minimum_q = 1 if node.terminal else 6
        for drops in range(a.q - minimum_q + 1):
            # Joint encoding / delayed rescaling lower bound for a linear DAG.
            # Every serial plaintext multiplication gets >=40 encoding bits.
            encoding_total = 40 - a.s + removed
            if encoding_total >= 40 * node.cp_depth - 1e-8:
                return State(a.q - drops, 40.0, 40)
            removed += LOGQ[a.q - drops - 1]
        return None
    raise ValueError(node.kind)


@dataclass(slots=True)
class Plan:
    boots: int
    states: dict
    events: tuple


def simulate(policy, floor, width=32):
    nodes = network(policy)
    last_use = {"pool-fc": len(nodes)}
    for i, node in enumerate(nodes):
        for v in node.inputs:
            last_use[v] = i
    plans = [Plan(0, {"input": State(34, 40.0, 40)}, ())]
    maximum_frontier = 1
    for i, node in enumerate(nodes):
        candidates = {}
        ids = tuple(dict.fromkeys(node.inputs))
        for plan in plans:
            eligible = [
                key
                for key in ids
                if not (
                    plan.states[key].q >= 31 and abs(plan.states[key].s - 40) <= 1e-8
                )
                and alignment(plan.states[key], floor) is not None
            ]
            for choices in product((False, True), repeat=len(eligible)):
                states = plan.states.copy()
                events = list(plan.events)
                for key, refresh in zip(eligible, choices):
                    if refresh:
                        before = states[key]
                        drops, plain = alignment(before, floor)
                        events.append(
                            dict(
                                before_node=node.name,
                                value=key,
                                q=before.q,
                                scale=before.s,
                                alignment_q=drops,
                                plain_scale=plain,
                            )
                        )
                        states[key] = State(31, 40.0, 40)
                output = evaluate(node, tuple(states[key] for key in node.inputs))
                if output is None or (not node.terminal and output.q < 6):
                    continue
                states[node.output] = output
                states = {
                    key: s for key, s in states.items() if last_use.get(key, -1) > i
                }
                signature = tuple(
                    (key, s.q, round(s.s, 9), s.nominal)
                    for key, s in sorted(states.items())
                )
                candidate = Plan(len(events), states, tuple(events))
                previous = candidates.get(signature)
                if previous is None or candidate.boots < previous.boots:
                    candidates[signature] = candidate
        if not candidates:
            return dict(
                policy=policy,
                floor=floor,
                beam=width,
                feasible=False,
                failed_at=node.name,
            )
        maximum_frontier = max(maximum_frontier, len(candidates))
        plans = sorted(
            candidates.values(),
            key=lambda p: (
                p.boots,
                -min(s.q for s in p.states.values()),
                -sum(s.q for s in p.states.values()),
                sum(abs(s.s - 40) for s in p.states.values()),
            ),
        )[:width]
    result = min(plans, key=lambda p: p.boots)
    history = replay(policy, result.events)
    return dict(
        policy=policy,
        floor=floor,
        beam=width,
        feasible=True,
        boots=result.boots,
        alignment_q=sum(e["alignment_q"] for e in result.events),
        alignment_distribution=dict(Counter(e["alignment_q"] for e in result.events)),
        boot_events=result.events,
        output=dict(q=result.states["pool-fc"].q, s=result.states["pool-fc"].s),
        level_accounting=history,
        maximum_candidates_at_step=maximum_frontier,
        nodes=len(nodes),
    )


def replay(policy, events):
    """Account for levels on each region's principal path, separately from entry drops."""
    states = {"input": State(34, 40.0, 40)}
    rows = []
    by_node = {}
    for event in events:
        by_node.setdefault(event["before_node"], []).append(event)
    for node in network(policy):
        for event in by_node.get(node.name, []):
            before = states[event["value"]]
            assert before.q == event["q"] and abs(before.s - event["scale"]) < 1e-8
            states[event["value"]] = State(31, 40.0, 40)
        incoming = tuple(states[key] for key in node.inputs)
        output = evaluate(node, incoming)
        assert output is not None
        states[node.output] = output
        primary = incoming[-1] if node.stage == "尾部" else incoming[0]
        category = (
            "relu"
            if node.kind in ("poly", "manual22")
            else "projection"
            if "projection" in node.name
            else "pool_fc"
            if node.terminal
            else "residual_merge"
            if node.kind == "add"
            else "convolution"
        )
        rows.append(
            dict(
                node=node.name,
                category=category,
                input_q=[s.q for s in incoming],
                input_scales=[s.s for s in incoming],
                output_q=output.q,
                output_scale=output.s,
                principal_path_level_use=primary.q - output.q,
            )
        )
    totals = Counter()
    for row in rows:
        totals[row["category"]] += row["principal_path_level_use"]
    return dict(
        region_principal_path_levels=dict(totals),
        boot_entry_rescale_levels=sum(e["alignment_q"] for e in events),
        boot_entry_modswitch_levels=sum(e["q"] - 6 - e["alignment_q"] for e in events),
        boot_entry_total_levels=sum(e["q"] - 6 for e in events),
        per_region=rows,
    )


def calibrate():
    states = {"0": State(31, 40.0, 40)}
    observed = []
    for stage, (_, external, output) in REGIONS.items():
        value = poly(stage, tuple(states[k] for k in external))
        assert value is not None
        states[output] = value
        observed.append(dict(stage=stage, q=value.q, scale=value.s))
    for row, reference_id in zip(observed, ("103", "211", "388", "395")):
        reference = DATA["compiler"]["values"][reference_id]
        assert row["q"] == reference["q"]
        assert abs(row["scale"] - reference["scale"]) < 1e-8
    assert [r["q"] for r in observed] == [25, 17, 8, 6], observed
    assert abs(observed[-1]["scale"] - 44.845572038296936) < 1e-8, observed
    sample = State(7, 56.018905617583094, 57)
    assert alignment(sample, 30) is None
    assert alignment(sample, 15)[0] == 1
    return observed


if __name__ == "__main__":
    started = time.perf_counter()
    calibration = calibrate()
    rows = []
    for width in [8, 32]:
        for policy in ["manual22", "threshold"]:
            for floor in [30, 24, 20, 15, 8, 0, "ideal_exit"]:
                row = simulate(policy, floor, width)
                rows.append(row)
                print(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("boot_events", "level_accounting")
                    },
                    flush=True,
                )
    result = dict(
        model="Python region DAG; joint linear-region lower bound held fixed; bounded beam over Boot choices",
        calibration=calibration,
        polynomial_arithmetic_nodes=sum(len(v[0]) for v in REGIONS.values()),
        source_trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        source_fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        rows=rows,
        seconds=time.perf_counter() - started,
        limitations=[
            "Manual22 is the verified fresh-input contract, not a new generic polynomial optimizer.",
            "Linear regions use their CP-depth encoding lower bound rather than every convolution operation.",
            "Beam search and region-only cuts differ from production DaCapo; counts are model results.",
            "No ciphertext precision/noise simulation.",
            "ideal_exit is a zero-level s40 producer oracle, not an executable alignment operation.",
        ],
    )
    (HERE / "simulation-results.json").write_text(json.dumps(result, indent=2) + "\n")
    from dataclasses import asdict

    (HERE / "network-nodes.json").write_text(
        json.dumps(
            {
                policy: [asdict(n) for n in network(policy)]
                for policy in ("manual22", "threshold")
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print("seconds", result["seconds"])
