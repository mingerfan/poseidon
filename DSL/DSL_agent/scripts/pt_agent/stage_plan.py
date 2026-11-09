"""E5b: bootstrap placement on PT-DSL statement and kernel-stage boundaries.

Every DAG node carries a label (statement order, stage order within the statement, multiplicative depth within
the stage). A cut lies between two consecutive labels. At a cut, every ciphertext produced before it, read after
it and already multiplied since its last refresh is bootstrapped; values that were not multiplied pass the cut
(DaCapo's bypass). After a cut every live ciphertext is then at boot_out_level, so the cheapest placement up to a
cut depends only on the previous cut: a dynamic program over cuts with exact level and latency simulation.
A generic pruning pass then drops single bootstraps that a fixed plan does not need; it applies to any plan.
"""

from dataclasses import dataclass
import heapq
import math
import time

from .placement import Placement


MUL = ("mul_cc", "mul_cp")


class StageOrderError(Exception):
    pass


@dataclass
class Model:
    """Latency and level rules shared with placement.depth_dp (absolute levels index the latency table)."""
    table: dict
    input_level: int
    boot_in_level: int
    boot_out_level: int

    @property
    def boot_us(self):
        return self.table["bootstrap_single"][self.boot_in_level]

    def op_latency(self, kind, ciphers, level):
        if kind in MUL:
            key = "mul_double" if kind == "mul_cc" else "mul_single"
            return self.table[key][level]+self.table["rescale_single"][level]
        key = ("rotate_single" if kind == "rot" else "negate_single" if kind == "neg"
               else "add_double" if ciphers == 2 else "add_single")
        return self.table[key][level]


def model_for(profile="qwen48"):
    from .cost import PROFILES, load_latency
    config = PROFILES[profile]
    if config.get("boot_normalization_levels", 1) != 0:
        raise StageOrderError("stage-cut placement models a bootstrap window without normalization levels")
    return Model(load_latency(profile), config["input_level"], config["boot_in_level"], config["boot_out_level"])


def _users(dag):
    n = len(dag.kind)
    users = [[] for _ in range(n)]
    for nid, args in enumerate(dag.args):
        for arg in set(args):
            users[arg].append(nid)
    for value in dag._outputs or []:
        users[value.nid].append(n)
    return users


def labels(dag, recorder):
    """(statement, stage, depth within the stage) per finalized node, from the recorder's creation stages."""
    remap = getattr(dag, "remap", None)
    n = len(dag.kind)
    stages = [None]*n
    for old, stage in enumerate(recorder.node_stage):
        new = old if remap is None else remap.get(old)
        if new is not None:
            stages[new] = stage
    if any(stage is None for stage in stages):
        raise StageOrderError("A DAG node has no recorded stage")
    local = [0]*n
    for nid, (kind, args) in enumerate(zip(dag.kind, dag.args)):
        depth = max((local[arg] for arg in args if stages[arg] == stages[nid]), default=0)
        local[nid] = depth+(kind in MUL)
    return [(stage[0], stage[1], depth) for stage, depth in zip(stages, local)]


def simulate(dag, refreshed, model):
    """Exact levels and latency of a fixed plan: refreshed[v] is the set of users reading v's bootstrap."""
    n = len(dag.kind)
    level, latency = [0]*n, [0.0]*n
    for nid, (kind, args) in enumerate(zip(dag.kind, dag.args)):
        if kind == "input":
            level[nid] = model.input_level
            continue
        current = min(model.boot_out_level if nid in refreshed.get(arg, ()) else level[arg] for arg in args)
        if kind in MUL:
            if current-1 < model.boot_in_level:
                return None
            level[nid] = current-1
        else:
            level[nid] = current
        latency[nid] = model.op_latency(kind, len(args), current)
    return level, latency


def as_placement(dag, refreshed, model, *, cuts, max_depth):
    result = simulate(dag, refreshed, model)
    if result is None:
        raise StageOrderError("Plan is infeasible")
    _, latency = result
    node_latency = {nid: value for nid, value in enumerate(latency) if value}
    for value in refreshed:
        node_latency[value] = node_latency.get(value, 0)+model.boot_us
    total = sum(latency)+len(refreshed)*model.boot_us
    return Placement(list(cuts), len(cuts), len(refreshed), total, sorted(refreshed), node_latency, max_depth,
                     {value: sorted(users) for value, users in refreshed.items()})


def stage_cut(dag, node_labels, model, *, max_depth=0, time_limit=600.0):
    """Cheapest refresh-all-touched placement over label cuts; returns (Placement, stats)."""
    if model.input_level != model.boot_out_level:
        raise StageOrderError("stage-cut assumes inputs enter at boot_out_level")
    started = time.monotonic()
    n = len(dag.kind)
    kind, args = dag.kind, dag.args
    distinct = sorted({node_labels[nid] for nid in range(n) if kind[nid] != "input"})
    index = {label: i for i, label in enumerate(distinct)}
    m = len(distinct)
    li = [-1 if kind[nid] == "input" else index[node_labels[nid]] for nid in range(n)]
    users = _users(dag)
    for nid in range(n):
        for arg in args[nid]:
            if li[arg] > li[nid]:
                raise StageOrderError(f"Node {nid} reads node {arg} from a later stage label")
    last_use = [max((m if user == n else li[user] for user in users[nid]), default=-2) for nid in range(n)]
    groups = [[] for _ in range(m)]
    for nid in range(n):
        if li[nid] >= 0:
            groups[li[nid]].append(nid)
    mul = [k in MUL for k in kind]
    tables = []
    for nid in range(n):
        k = kind[nid]
        if k == "input":
            tables.append(None)
        elif k in MUL:
            first = model.table["mul_double" if k == "mul_cc" else "mul_single"]
            tables.append((first, model.table["rescale_single"]))
        else:
            tables.append((model.table["rotate_single" if k == "rot" else "negate_single" if k == "neg"
                                       else "add_double" if len(args[nid]) == 2 else "add_single"], None))
    top, low, boot = model.boot_out_level, model.boot_in_level, model.boot_us
    best, prev = [math.inf]*m, [None]*m
    best_end, prev_end = math.inf, None
    level = [0]*n
    work = 0

    def scan(a, base, record):
        """Advance from cut a (after label a; -1 is the start); report each reachable boundary."""
        nonlocal work
        accumulated, alive, expire = 0.0, 0, {}
        entry = model.input_level if a < 0 else top
        for label in range(a+1, m):
            for nid in groups[label]:
                current = min(entry if li[arg] <= a else level[arg] for arg in args[nid])
                first, second = tables[nid]
                if mul[nid]:
                    if current-1 < low:
                        return
                    accumulated += first[current]+second[current]
                    current -= 1
                else:
                    accumulated += first[current]
                level[nid] = current
                if current < top and last_use[nid] > label:
                    alive += 1
                    expire[last_use[nid]] = expire.get(last_use[nid], 0)+1
            work += len(groups[label])
            alive -= expire.pop(label, 0)
            record(label, base+accumulated, alive)

    for a in range(-1, m-1):
        base = 0.0 if a < 0 else best[a]
        if base == math.inf:
            continue

        def record(label, cost, alive, a=a):
            nonlocal best_end, prev_end
            if label == m-1:
                if cost < best_end:
                    best_end, prev_end = cost, a
            elif cost+alive*boot < best[label]:
                best[label], prev[label] = cost+alive*boot, a

        scan(a, base, record)
        if time.monotonic()-started > time_limit:
            raise TimeoutError(f"stage-cut exceeded {time_limit} s at cut {a} of {m}")
    if prev_end is None:
        raise StageOrderError("No feasible stage-cut placement")
    cuts, a = [], prev_end
    while a is not None and a >= 0:
        cuts.append(a)
        a = prev[a]
    cuts.reverse()
    dp_s = time.monotonic()-started
    # Refreshed values of the chosen segments, from the same simulation.
    refreshed, bounds = {}, [-1]+cuts
    for start, cut in zip(bounds, cuts):
        produced = []
        entry = model.input_level if start < 0 else top
        for label in range(start+1, cut+1):
            for nid in groups[label]:
                current = min(entry if li[arg] <= start else level[arg] for arg in args[nid])
                level[nid] = current-1 if mul[nid] else current
                produced.append(nid)
        for nid in produced:
            if level[nid] < top and last_use[nid] > cut:
                refreshed[nid] = {user for user in users[nid] if (m if user == n else li[user]) > cut}
    placement = as_placement(dag, refreshed, model, cuts=[distinct[c] for c in cuts], max_depth=max_depth)
    if not math.isclose(placement.latency_us, best_end, rel_tol=1e-9, abs_tol=1e-3):
        raise StageOrderError(f"stage-cut objective {best_end} differs from its plan simulation {placement.latency_us}")
    return placement, dict(labels=m, cuts=len(cuts), dp_s=dp_s, total_s=time.monotonic()-started, node_steps=work)


def prune(dag, placement, model, *, passes=3, region_limit=50000):
    """Drop single bootstraps of a fixed plan while it stays feasible and its latency falls."""
    started = time.monotonic()
    n = len(dag.kind)
    refreshed = {value: set(users) for value, users in placement.refreshed_uses.items()}
    result = simulate(dag, refreshed, model)
    if result is None:
        raise StageOrderError("Plan to prune is infeasible")
    level, latency = result
    users = _users(dag)
    removed = 0
    for _ in range(passes):
        changed = False
        for value in sorted(refreshed, key=lambda v: (len(refreshed[v]), v)):
            readers = refreshed[value]
            heap = sorted(user for user in readers if user != n)
            heapq.heapify(heap)
            new_level, new_latency, seen, feasible = {}, {}, set(), True
            while heap:
                nid = heapq.heappop(heap)
                if nid in seen:
                    continue
                seen.add(nid)
                if len(seen) > region_limit:
                    feasible = False
                    break
                current = min(model.boot_out_level if (nid in refreshed.get(arg, ()) and arg != value)
                              else new_level.get(arg, level[arg]) for arg in dag.args[nid])
                kind = dag.kind[nid]
                if kind in MUL and current-1 < model.boot_in_level:
                    feasible = False
                    break
                result_level = current-1 if kind in MUL else current
                cost = model.op_latency(kind, len(dag.args[nid]), current)
                if cost != latency[nid]:
                    new_latency[nid] = cost
                if result_level != level[nid]:
                    new_level[nid] = result_level
                    for user in users[nid]:
                        if user != n and user not in refreshed.get(nid, ()):
                            heapq.heappush(heap, user)
            if not feasible:
                continue
            delta = sum(new_latency[nid]-latency[nid] for nid in new_latency)-model.boot_us
            if delta < 0:
                for nid, value_level in new_level.items():
                    level[nid] = value_level
                for nid, cost in new_latency.items():
                    latency[nid] = cost
                del refreshed[value]
                removed += 1
                changed = True
        if not changed:
            break
    pruned = as_placement(dag, refreshed, model, cuts=placement.cuts, max_depth=placement.max_depth)
    return pruned, dict(removed=removed, prune_s=time.monotonic()-started)


def candidates(dag, recorder, depth_placement, model, *, time_limit=600.0):
    """The four E5b candidates; each entry has boots, latency_us, seconds and its Placement (or an error)."""
    result = {}
    started = time.monotonic()
    base = as_placement(dag, {v: set(u) for v, u in depth_placement.refreshed_uses.items()}, model,
                        cuts=depth_placement.cuts, max_depth=depth_placement.max_depth)
    if base.boots != depth_placement.boots or not math.isclose(base.latency_us, depth_placement.latency_us,
                                                               rel_tol=1e-9, abs_tol=1e-3):
        raise StageOrderError("Plan simulation differs from depth-dp on its own placement")
    result["depth-dp"] = dict(placement=base, seconds=0.0)
    pruned, stats = prune(dag, base, model)
    result["depth-dp+prune"] = dict(placement=pruned, seconds=stats["prune_s"], **stats)
    try:
        cut, stats = stage_cut(dag, labels(dag, recorder), model, max_depth=depth_placement.max_depth,
                               time_limit=time_limit)
        result["stage-cut"] = dict(placement=cut, seconds=stats["total_s"], **stats)
        pruned, stats = prune(dag, cut, model)
        result["stage-cut+prune"] = dict(placement=pruned, seconds=stats["prune_s"], **stats)
    except (StageOrderError, TimeoutError) as exc:
        result["stage-cut"] = dict(error=f"{type(exc).__name__}: {exc}")
        result["stage-cut+prune"] = dict(error="stage-cut unavailable")
    for entry in result.values():
        if "placement" in entry:
            entry.update(boots=entry["placement"].boots, latency_us=entry["placement"].latency_us)
    result["_seconds"] = time.monotonic()-started
    return result


def select(result):
    names = [name for name, entry in result.items() if not name.startswith("_") and "placement" in entry]
    return min(names, key=lambda name: (result[name]["latency_us"], names.index(name)))
