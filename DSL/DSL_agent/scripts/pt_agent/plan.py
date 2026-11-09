"""E5a: agent-side bootstrap placement expressed by call number.

A refresh plan is computed on the same cost DAG the depth-dp estimate uses, while a ``record`` RefreshBackend
numbers every ciphertext-producing call of the lowering. The depth-dp simulation names, for each bootstrapped
node, the users that read its refreshed value; the plan translates those (user, node) pairs into
(call number, operand position) substitutions. The ``plan`` policy replays the same lowering and refreshes
exactly those operands, so the cost model and an exported package insert the same bootstraps.
"""

import importlib
import json
from pathlib import Path
import sys
import time

from .backends import CountBackend, DagBackend, RefreshBackend
from .backends.cost_dag import CostDag


PLAN_FORMAT = "pt-agent-refresh-plan-v1"


class PlanError(Exception):
    pass


def levels(profile="qwen48") -> dict:
    from .cost import PROFILES
    config = PROFILES[profile]
    return {key: config[key] for key in ("input_level", "boot_in_level", "boot_out_level")}


def record_dag(run, slots, *, frontend_normalize, profile="qwen48"):
    """Run one lowering, ``run(active)``, over a recording wrapper; returns the DAG, the recorder and seconds."""
    inner = CostDag(slots) if frontend_normalize else DagBackend(slots)
    recorder = RefreshBackend(inner, policy="record", **levels(profile))
    started = time.monotonic()
    run(recorder)
    if frontend_normalize:
        inner.finalize()
    return inner, recorder, time.monotonic()-started


def _ranges(values):
    ranges = []
    for value in values:
        if ranges and ranges[-1][1] == value:
            ranges[-1][1] = value+1
        else:
            ranges.append([value, value+1])
    return ranges


def express(inner, recorder, placement, *, profile="qwen48") -> dict:
    """Translate a depth-dp placement on ``inner`` into call-number substitutions of ``recorder``."""
    remap = getattr(inner, "remap", None)

    def node(value):
        return value if remap is None else remap.get(value)

    refreshed = {value: set(users) for value, users in placement.refreshed_uses.items()}
    ret = len(inner.kind)
    substitutions, dead, negations = [], [], {}
    for seq, (operands, result, extra) in enumerate(recorder.calls):
        target = node(result)
        if target is None:
            dead.append(seq)
            continue
        if (not any(value is not None and node(value) in refreshed for value in operands)
                and not any(node(item) in refreshed for item in extra)):
            continue
        touched = {target}
        touched.update(item for item in map(node, extra) if item is not None)
        positions = {}
        for position, value in enumerate(operands):
            if value is None:
                continue
            users = refreshed.get(node(value))
            if users and not users.isdisjoint(touched):
                positions[position] = node(value)
        # A frontend subtraction is neg+add, and value numbering can share that neg among several calls.
        # A refreshed neg is level-preserving over its operand, so the call refreshes that operand instead,
        # keyed by the neg node: one bootstrap per refreshed neg, as in the DAG.
        for item in extra:
            middle = node(item)
            users = refreshed.get(middle) if middle is not None else None
            if not users or inner.kind[middle] != "neg" or users.isdisjoint(touched):
                continue
            source = inner.args[middle][0]
            matches = [position for position, value in enumerate(operands)
                       if value is not None and node(value) == source]
            if len(matches) != 1 or matches[0] in positions:
                raise PlanError(f"Call {seq} refreshes an internal negation ambiguously")
            positions[matches[0]] = middle
            negations[middle] = source
        substitutions.extend([seq, position, canonical] for position, canonical in sorted(positions.items()))
    outputs = [[position, node(value)] for position, value in enumerate(recorder.output_nodes)
               if ret in refreshed.get(node(value), ())]
    covered = {item[-1] for item in substitutions} | {item[-1] for item in outputs}
    if covered != set(placement.boot_nodes):
        raise PlanError(f"Plan covers {len(covered)} bootstrapped nodes, the placement has {placement.boots}")
    # A refreshed negation is emitted as bootstrap(source); when the source itself is refreshed too, both are
    # the same bootstrap of the same ciphertext, which the compiler's CSE keeps once. Share one key for them.
    merged = {middle: source for middle, source in negations.items() if source in covered}
    if merged:
        substitutions = [[seq, position, merged.get(key, key)] for seq, position, key in substitutions]
    boots = len(covered)-len(merged)
    return dict(format=PLAN_FORMAT, placement="depth-dp", calls=recorder.call_count, boots=boots,
                merged=sorted(merged),
                barriers=placement.barriers, cuts=list(placement.cuts), latency_us=placement.latency_us,
                substitutions=substitutions, outputs=outputs, dead=_ranges(dead), levels=levels(profile))


def _merge(placement, plan, profile):
    """The placement with the bootstraps the plan shares (see express) counted once."""
    if not plan["merged"]:
        return placement
    from dataclasses import replace
    from .cost import PROFILES, load_latency
    boot = load_latency(profile)["bootstrap_single"][PROFILES[profile]["boot_in_level"]]
    merged = set(plan["merged"])
    node_latency = dict(placement.node_latency)
    for value in merged:
        node_latency[value] -= boot
    result = replace(placement, boots=placement.boots-len(merged), latency_us=placement.latency_us-len(merged)*boot,
                     boot_nodes=[value for value in placement.boot_nodes if value not in merged],
                     node_latency=node_latency)
    plan["latency_us"] = result.latency_us
    return result


def _placement(inner, profile):
    from .cost import depth_dp_estimate
    return depth_dp_estimate(inner, profile)


PLANNED = ("depth-dp-plan", "program-plan")


def plan_program(program, weights, *, profile="qwen48", frontend_normalize=None, keep_recorder=False):
    """Plan from pt_agent's own lowering; returns (plan, finalized DAG, placement[, recorder]).

    ``depth-dp-plan`` keeps the depth-dp placement (E5a); ``program-plan`` (E5b) keeps the lowest predicted
    latency among depth-dp, depth-dp + pruning, stage-cut and stage-cut + pruning."""
    from .lower import lower
    from .transformer_lower import uses_transformer
    if any(op.kind == "refresh" for op in program.ops):
        raise PlanError("depth-dp-plan programs must not contain refresh statements")
    if frontend_normalize is None:
        frontend_normalize = uses_transformer(program)
    inner, recorder, lower_s = record_dag(lambda active: lower(program, weights, active), program.header.slots,
                                          frontend_normalize=frontend_normalize, profile=profile)
    started = time.monotonic()
    placement = _placement(inner, profile)
    dp_s = time.monotonic()-started
    variant, summary, candidates_s = "depth-dp", None, 0.0
    if program.header.placement == "program-plan":
        from .stage_plan import candidates, model_for, select
        started = time.monotonic()
        found = candidates(inner, recorder, placement, model_for(profile))
        candidates_s = time.monotonic()-started
        variant = select(found)
        placement = found[variant]["placement"]
        summary = {name: {key: value for key, value in entry.items() if key != "placement"}
                   for name, entry in found.items() if not name.startswith("_")}
    started = time.monotonic()
    plan = express(inner, recorder, placement, profile=profile)
    placement = _merge(placement, plan, profile)
    plan.update(placement=variant, candidates=summary)
    plan["timing"] = dict(lower_s=lower_s, dp_s=dp_s, candidates_s=candidates_s, map_s=time.monotonic()-started)
    plan["source"] = "pt_agent.lower"
    if keep_recorder:
        return plan, inner, placement, recorder
    return plan, inner, placement


def verify(run, slots, plan, *, profile="qwen48") -> dict:
    """Replay one lowering under the plan policy on a CountBackend; returns the replayed counts."""
    from .cost import PROFILES
    inner = CountBackend(slots, boot_out_level=PROFILES[profile]["boot_out_level"])
    active = RefreshBackend(inner, policy="plan", plan=plan, **levels(profile))
    started = time.monotonic()
    run(active)
    stats = active.stats()
    result = dict(boots=stats["boots"], calls=stats["calls"], min_level=stats["min_level"],
                  apply_s=time.monotonic()-started, inner_boots=inner.stats()["counts"]["boot"])
    if result["boots"] != plan["boots"] or result["inner_boots"] != plan["boots"]:
        raise PlanError(f"Plan replay inserted {result['boots']} bootstraps, the plan has {plan['boots']}")
    return result


def verify_program(program, weights, plan, *, profile="qwen48") -> dict:
    from .lower import lower
    return verify(lambda active: lower(program, weights, active), program.header.slots, plan, profile=profile)


class _Package:
    """Import an exported package in this process for one planning pass, then forget its modules."""

    def __init__(self, package):
        self.package = Path(package)
        self.name = json.loads((self.package/"config/model.json").read_text())["name"]

    def __enter__(self):
        sys.path.insert(0, str(self.package))
        self.runtime = importlib.import_module(self.name+".runtime")
        self.weights_module = importlib.import_module(self.name+".weights")
        return self

    def __exit__(self, *exc):
        sys.path.remove(str(self.package))
        for key in [key for key in sys.modules if key == self.name or key.startswith(self.name+".")]:
            del sys.modules[key]


def plan_package(package, weights_path, program, *, profile="qwen48", weights=None) -> dict:
    """The exported package's plan, verified by replaying the package's own forward.

    depth-dp-plan plans on the package's forward directly (E5a). program-plan needs statement labels, which
    pt_agent's lowering supplies: it plans there and requires the package to make the identical call sequence."""
    from .transformer_lower import uses_transformer
    if any(op.kind == "refresh" for op in program.ops):
        raise PlanError("depth-dp-plan programs must not contain refresh statements")
    if program.header.placement == "program-plan":
        plan, inner, _, recorder = plan_program(program, weights, profile=profile, keep_recorder=True)
        calls, outputs = recorder.calls, recorder.output_nodes
        del inner, recorder
        with _Package(package) as loaded:
            package_weights = loaded.weights_module.load(weights_path)

            def run(active):
                loaded.runtime.invoke(active.inner, package_weights, active=active)

            _, package_recorder, _ = record_dag(run, program.header.slots,
                                                frontend_normalize=uses_transformer(program), profile=profile)
            if package_recorder.calls != calls or package_recorder.output_nodes != outputs:
                raise PlanError("The package's call sequence differs from pt_agent's lowering")
            del package_recorder, calls, outputs
            plan["source"] = "pt_agent.lower; package call sequence identical"
            plan["replay"] = verify(run, program.header.slots, plan, profile=profile)
        return plan
    with _Package(package) as loaded:
        weights = loaded.weights_module.load(weights_path)

        def run(active):
            loaded.runtime.invoke(active.inner, weights, active=active)

        inner, recorder, lower_s = record_dag(run, program.header.slots,
                                              frontend_normalize=uses_transformer(program), profile=profile)
        started = time.monotonic()
        placement = _placement(inner, profile)
        dp_s = time.monotonic()-started
        started = time.monotonic()
        plan = express(inner, recorder, placement, profile=profile)
        _merge(placement, plan, profile)
        plan["timing"] = dict(lower_s=lower_s, dp_s=dp_s, map_s=time.monotonic()-started)
        plan["source"] = "package.runtime.invoke"
        del inner, recorder
        plan["replay"] = verify(run, program.header.slots, plan, profile=profile)
    return plan
