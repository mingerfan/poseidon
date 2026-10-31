"""Standalone real-prime scale simulation; does not modify or invoke a compiler.

The arithmetic DAG comes from the compiler trace, never from the manual witness.
Coefficient values and mathematical dependencies are unchanged.  The backward
experiment adjusts existing coefficients, does not insert unit/RS at polynomial
additions, and treats shared basis values as fixed once produced.  It is a small
region experiment, not a scalable production search or an error bound.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time


EPS = 1e-8
FLOOR = 40.0


class InfeasibleBasis(Exception):
    """This candidate needs a correction encoding below the declared floor."""


@dataclass(frozen=True)
class State:
    q: int
    s: float
    nominal: int


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    args: tuple
    label: str
    stage: str
    scalar: float | None


@dataclass(frozen=True)
class Plan:
    id: str
    q: int
    s: float
    mul_q: int
    plain: float
    drops: int
    rs_count: int
    children: tuple


class Simulation:
    def __init__(self, trace):
        self.data = json.loads(trace.read_text())
        self.trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
        self.logq = [math.log2(x["value"]) for x in self.data["moduli"]]
        self.capacity = [0.0]
        for q in self.logq:
            self.capacity.append(self.capacity[-1] + q)
        source = self.data["compiler"]
        self.alias = {}
        self.nodes = {}
        self.users = defaultdict(set)
        for op in source["operations"]:
            if not source["values"][op["id"]]["cipher"]:
                continue
            args = tuple(self.alias[x] for x in op["inputs"]
                         if source["values"][x]["cipher"])
            if op["kind"] in ("transfer", "rescale", "mod_switch", "relinearize") or (
                op["kind"] == "mul_cp" and op["scalar"] == 1
            ):
                self.alias[op["id"]] = args[0]
                continue
            self.alias[op["id"]] = op["id"]
            node = Node(op["id"], op["kind"], args, op["label"], op["stage"],
                        op["scalar"])
            self.nodes[node.id] = node
            for arg in args:
                self.users[arg].add(node.id)
        self.roots = [self.alias[x] for x in ("103", "211", "388", "395")]
        assert Counter(n.kind for n in self.nodes.values())["mul_cp"] == 30
        assert Counter(n.kind for n in self.nodes.values())["mul_cc"] == 27
        self.input = State(31, 40.0, 40)
        self.request_count = 0

    def removed(self, high, low):
        assert 1 <= low <= high <= len(self.logq)
        return self.capacity[high] - self.capacity[low]

    def legal(self, q, s):
        return 1 <= q <= 31 and FLOOR - EPS <= s < self.capacity[q] - 1

    def normalize(self, q, s, nominal):
        while q > 1 and s - self.logq[q - 1] >= FLOOR:
            s -= self.logq[q - 1]
            q -= 1
            nominal = math.ceil(s)
        assert self.legal(q, s), (q, s)
        return State(q, s, nominal)

    def legacy_add(self, a, b):
        nominal = max(a.nominal, b.nominal)
        return State(min(a.q, b.q),
                     max(a.s + nominal - a.nominal, b.s + nominal - b.nominal),
                     nominal)

    def before_mul_options(self, a):
        result = [(a, 0)]
        for target in (40.0, 50.0, 60.0):
            if target >= a.s - EPS:
                continue
            for drops in range(1, a.q):
                q = a.q - drops
                plain = target - a.s + self.removed(a.q, q)
                if plain >= 30 and self.legal(a.q, a.s + plain):
                    result.append((State(q, target, math.ceil(target)), 1))
                    break
        return result

    def forward(self, before_mul=False):
        states = {"0": self.input}
        adjustments = []
        for n in self.nodes.values():
            if n.kind == "input":
                continue
            a = states[n.args[0]]
            if n.kind == "mul_cc":
                b = states[n.args[1]]
                choices = [((a, 0), (b, 0))]
                if before_mul:
                    choices = list(product(self.before_mul_options(a),
                                           self.before_mul_options(b)))
                    if n.args[0] == n.args[1]:
                        choices = [(x, x) for x in self.before_mul_options(a)]
                candidates = []
                for (aa, ua), (bb, ub) in choices:
                    out = self.normalize(min(aa.q, bb.q), aa.s + bb.s,
                                         aa.nominal + bb.nominal)
                    candidates.append(((-out.q, out.s, ua + ub), out, aa, bb,
                                       ua + ub))
                _, out, aa, bb, units = min(candidates, key=lambda x: x[0])
                if units:
                    adjustments.append({"node": n.id, "input_q": [a.q, b.q],
                                        "adjusted_q": [aa.q, bb.q],
                                        "output": asdict(out)})
            elif n.kind == "mul_cp":
                out = self.normalize(a.q, a.s + FLOOR, a.nominal + 40)
            elif n.kind == "add_cc":
                out = self.legacy_add(a, states[n.args[1]])
            elif n.kind in ("add_cp", "negate"):
                out = a
            else:
                raise ValueError(n)
            states[n.id] = out
        return {"stages": [asdict(states[x]) for x in self.roots],
                "consumed_q": 31 - states[self.roots[-1]].q,
                "new_before_mul_adjustments": adjustments,
                "precision_scope": "Keeps legacy Add alignment, including its low-scale units; diagnostic only."}

    def basis(self, stage, incoming, retain=None):
        """Produce shared coefficient-free basis; keep its mathematical correction
        before rescaling the multiplication.  Retain omits one otherwise legal RS.
        The encoded-one multiplications here are solely basis correction branches.
        """
        states = dict(incoming)
        pure = set(incoming)
        records = []
        exposed = []
        stage_nodes = [n for n in self.nodes.values() if n.stage == stage]
        for n in stage_nodes:
            if n.kind == "mul_cp" or not all(a in pure for a in n.args):
                continue
            pure.add(n.id)
        for n in stage_nodes:
            if n.id not in pure:
                continue
            a = states[n.args[0]]
            q, s = a.q, a.s
            units = []
            if n.kind == "mul_cc":
                b = states[n.args[1]]
                q, s = min(a.q, b.q), a.s + b.s
            elif n.kind == "add_cc":
                b = states[n.args[1]]
                q, s = min(a.q, b.q), max(a.s, b.s)
                for arg, value in zip(n.args, (a, b)):
                    delta = s - value.s
                    if delta > EPS:
                        if delta < FLOOR - EPS:
                            raise InfeasibleBasis((n.id, delta))
                        assert "基底校正" in n.label, n
                        units.append({"input": arg, "plain": delta})
            elif n.kind not in ("add_cp", "negate"):
                raise ValueError(n)
            assert self.legal(q, s), (n, q, s)
            raw_q, raw_s = q, s
            boundary = (len(self.users[n.id]) > 1 or
                        any(u not in pure for u in self.users[n.id]))
            # Negation and doubling are linear views of an existing basis,
            # not new multiplication results requiring normalization.
            boundary = boundary and n.kind != "negate" and not (
                n.kind == "add_cc" and n.args[0] == n.args[1]
            )
            if boundary:
                normal = self.normalize(q, s, math.ceil(s))
                drops = q - normal.q
                if drops:
                    exposed.append(n.id)
                if n.id == retain:
                    assert drops, n
                    drops -= 1
                q -= drops
                s -= self.removed(raw_q, q)
            states[n.id] = State(q, s, math.ceil(s))
            records.append({"node": n.id, "kind": n.kind, "args": n.args,
                            "raw_q": raw_q, "raw_s": raw_s,
                            "q": q, "s": s, "drops": raw_q - q,
                            "units": units})
        return states, records, exposed

    def planner(self, fixed):
        @lru_cache(maxsize=200000)
        def available(oid):
            if oid in fixed:
                return fixed[oid].q
            return min(available(x) for x in self.nodes[oid].args)

        def solve(oid, q, s):
            return cached(oid, q, round(s, 9))

        @lru_cache(maxsize=200000)
        def cached(oid, q, s):
            self.request_count += 1
            if not self.legal(q, s) or q > available(oid):
                return None
            if oid in fixed:
                a = fixed[oid]
                if abs(a.s - s) <= EPS:
                    return Plan(oid, q, s, q, 0, 0, 0, ())
                return None
            n = self.nodes[oid]
            if n.kind in ("add_cc", "add_cp", "negate"):
                children = tuple(solve(a, q, s) for a in n.args)
                if any(x is None for x in children):
                    return None
                return Plan(oid, q, s, q, 0, 0,
                            sum(x.rs_count for x in children), children)
            if n.kind == "mul_cp":
                assert n.args[0] in fixed, n
                a = fixed[n.args[0]]
                for mul_q in range(q, a.q + 1):
                    plain = s - a.s + self.removed(mul_q, q)
                    if plain < FLOOR - EPS:
                        continue
                    if not self.legal(mul_q, a.s + plain):
                        continue
                    child = solve(n.args[0], mul_q, a.s)
                    assert child is not None
                    return Plan(oid, q, s, mul_q, plain, mul_q - q,
                                mul_q - q, (child,))
                return None
            assert n.kind == "mul_cc", n
            sides = [i for i, a in enumerate(n.args) if a in fixed]
            assert len(sides) == 1, (n, sides)
            side = sides[0]
            a = fixed[n.args[side]]
            best = None
            for mul_q in range(q, available(oid) + 1):
                raw = s + self.removed(mul_q, q)
                if not self.legal(mul_q, raw):
                    continue
                other = solve(n.args[1 - side], mul_q, raw - a.s)
                if other is None:
                    continue
                known = solve(n.args[side], mul_q, a.s)
                assert known is not None
                children = (known, other) if side == 0 else (other, known)
                plan = Plan(oid, q, s, mul_q, 0, mul_q - q,
                            other.rs_count + mul_q - q, children)
                if best is None or plan.rs_count < best.rs_count:
                    best = plan
            return best

        return solve, available

    def best(self, fixed, root, target):
        solve, available = self.planner(fixed)
        for q in range(available(root), 0, -1):
            plan = solve(root, q, target)
            if plan is not None:
                return plan
        return None

    def verify(self, plan, fixed):
        """Independent replay of the selected witness, including actual primes."""
        if plan.id in fixed:
            state = fixed[plan.id]
            assert plan.q <= state.q and abs(plan.s - state.s) < 2e-7
            return State(plan.q, state.s, math.ceil(state.s))
        n = self.nodes[plan.id]
        args = [self.verify(x, fixed) for x in plan.children]
        assert len(args) == len(n.args)
        assert [x.id for x in plan.children] == list(n.args)
        if n.kind in ("add_cc", "add_cp", "negate"):
            assert plan.drops == 0 and plan.plain == 0
            assert all(a.q == plan.q and abs(a.s - plan.s) < 2e-7 for a in args)
            raw = args[0].s
        elif n.kind == "mul_cp":
            assert plan.plain >= FLOOR - EPS
            assert args[0].q == plan.mul_q
            raw = args[0].s + plan.plain
        else:
            assert n.kind == "mul_cc" and plan.plain == 0
            assert all(a.q == plan.mul_q for a in args)
            raw = sum(a.s for a in args)
        assert self.legal(plan.mul_q, raw)
        replay = raw
        for q in range(plan.mul_q, plan.q, -1):
            replay -= self.logq[q - 1]
            assert self.legal(q - 1, replay)
        assert abs(replay - plan.s) < 2e-7, (n.id, replay, plan.s)
        return State(plan.q, replay, math.ceil(replay))

    def backward(self, step):
        candidates = [{"state": self.input, "stages": []}]
        targets = [40 + EPS + i * step for i in range(round(30 / step) + 1)]
        for stage, root in zip(("P1(d15)", "P2(d15)"), self.roots[:2]):
            outputs = {}
            for incoming in candidates:
                previous = "0" if stage == "P1(d15)" else self.roots[0]
                try:
                    fixed, records, _ = self.basis(stage, {previous: incoming["state"]})
                except InfeasibleBasis:
                    continue
                solve, available = self.planner(fixed)
                for target in targets:
                    plan = None
                    for q in range(available(root), 0, -1):
                        plan = solve(root, q, target)
                        if plan is not None:
                            break
                    if plan is None:
                        continue
                    old = outputs.get(target)
                    if old is not None and (old["state"].q, -old["cost"]) >= (
                        plan.q, -plan.rs_count
                    ):
                        continue
                    state = self.verify(plan, fixed)
                    record = {"stage": stage, "q": state.q, "s": state.s,
                              "basis": records, "plan": asdict(plan)}
                    outputs[target] = {"state": state,
                                       "stages": incoming["stages"] + [record],
                                       "cost": plan.rs_count}
            assert outputs, stage
            candidates = list(outputs.values())
            print(json.dumps({"stage": stage, "exit_states": len(candidates),
                              "best_q": max(x["state"].q for x in candidates),
                              "requests": self.request_count}), flush=True)
        best = None
        all_results = []
        for incoming in candidates:
            external = {self.roots[1]: incoming["state"], "0": self.input}
            try:
                _, _, exposed = self.basis("P3(d27)", external)
            except InfeasibleBasis:
                continue
            for retain in (None, *exposed):
                try:
                    fixed, records, _ = self.basis("P3(d27)", external, retain)
                except InfeasibleBasis:
                    continue
                plan = self.best(fixed, self.roots[-1], 40.0)
                if plan is None:
                    continue
                state = self.verify(plan, fixed)
                record = {"stage": "P3 + tail multiplication", "q": state.q,
                          "s": state.s, "retain_one_basis_rs": retain,
                          "basis": records, "plan": asdict(plan)}
                result = {"state": state, "stages": incoming["stages"] + [record],
                          "cost": plan.rs_count + sum(r["drops"] for r in records)}
                all_results.append({"p2_s": incoming["state"].s,
                                    "p2_q": incoming["state"].q,
                                    "retain_one_basis_rs": retain,
                                    "output_q": state.q})
                if best is None or (state.q, -result["cost"]) > (
                    best["state"].q, -best["cost"]
                ):
                    best = result
        assert best is not None
        best["state"] = asdict(best["state"])
        best["consumed_q"] = 31 - best["state"]["q"]
        best["candidates"] = all_results
        best["requests"] = self.request_count
        best["target_grid_step"] = step
        best["constraints"] = {
            "coefficient_encoding_floor": 40,
            "basis_correction_unit_floor": 40,
            "ordinary_polynomial_add_unit_muls": 0,
            "ordinary_polynomial_add_rescales": 0,
            "shared_basis": "fixed; compare minimum RS with retaining one RS at one exposed basis",
            "tail_output_scale": 40,
            "boot_search": False,
            "global_optimality_claim": False,
            "precision_verified": "Encoding floors and capacity only; no ciphertext error bound or GPU run.",
        }
        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()
    start = time.perf_counter()
    sim = Simulation(args.trace)
    baseline = sim.forward()
    assert baseline["consumed_q"] == 25, baseline
    assert abs(baseline["stages"][-1]["s"] - 44.845572038296936) < EPS
    result = {"host": __import__("socket").gethostname(),
              "scope": "Same Q31/s40 ReLU arithmetic DAG; actual nonuniform Q; no Boot search.",
              "trace_sha256": sim.trace_hash,
              "baseline": baseline,
              "before_mul_only": sim.forward(before_mul=True),
              "coefficient_backward": sim.backward(args.step),
              "seconds": time.perf_counter() - start}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline_q": baseline["consumed_q"],
                      "before_mul_only_q": result["before_mul_only"]["consumed_q"],
                      "backward_q": result["coefficient_backward"]["consumed_q"],
                      "output": result["coefficient_backward"]["state"],
                      "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
