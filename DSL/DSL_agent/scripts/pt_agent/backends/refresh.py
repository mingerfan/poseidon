"""Refresh placement around the existing arithmetic backends.

Three policies share one call numbering (E5a): ``lazy`` refreshes a multiplication operand whose level is
exhausted (the original behaviour); ``record`` numbers every ciphertext-producing call over a DAG backend
without tracking levels, so a placement can be computed on the recorded DAG; ``plan`` replays the same calls
and refreshes exactly the operands a recorded placement names, by call number and operand position.
"""

from collections import Counter
import operator

from .base import BackendError, _Plain


POLICIES = ("lazy", "record", "plan")


class RValue(_Plain):
    # Plain ancestry gives reflected operations priority over an inner Plain on the left.
    def __init__(self, backend, value, level):
        self.backend, self.value, self.level = backend, value, level
        if level is not None:
            if level < backend.boot_in_level:
                raise BackendError("Refresh result level is below boot_in_level")
            backend._min_level = min(backend._min_level, level)

    def __add__(self, other):
        return self.backend._binary("add", self, other)

    def __radd__(self, other):
        return self.backend._binary("add", other, self)

    def __sub__(self, other):
        return self.backend._binary("sub", self, other)

    def __rsub__(self, other):
        return self.backend._binary("sub", other, self)

    def __mul__(self, other):
        return self.backend._binary("mul", self, other)

    def __rmul__(self, other):
        return self.backend._binary("mul", other, self)

    def __neg__(self):
        if self.backend.policy != "lazy":
            return self.backend._unary("neg", self)
        value = -self.value
        self.backend._record("negate_single", self.level)
        return RValue(self.backend, value, self.level)

    def rotate(self, k):
        if self.backend.policy != "lazy":
            return self.backend._unary("rot", self, k)
        value = self.value.rotate(k)
        if value is self.value:
            return self
        self.backend._record("rotate_single", self.level)
        return RValue(self.backend, value, self.level)

    def bootstrap(self):
        if self.backend.policy != "lazy":
            raise BackendError("Explicit refresh is unsupported under a recorded placement")
        if self.level < self.backend.boot_in_level:
            raise BackendError("bootstrap input level is below boot_in_level")
        value = self.value.bootstrap()
        self.backend._boots += 1
        self.backend._record("bootstrap_single", self.backend.boot_in_level)
        return RValue(self.backend, value, self.backend.boot_out_level)


class _Plan:
    """Read-only view of a refresh plan: call -> operand positions, each with its canonical DAG node."""

    def __init__(self, plan):
        self.calls = int(plan["calls"])
        self.substitutions = {}
        for seq, position, node in plan["substitutions"]:
            self.substitutions.setdefault(seq, {})[position] = node
        self.outputs = {position: node for position, node in plan["outputs"]}
        # Release a refreshed value after the last call that reads it, unless an output reads it.
        last = {}
        for seq, _, node in plan["substitutions"]:
            last[node] = max(seq, last.get(node, seq))
        kept = set(self.outputs.values())
        self.release = {}
        for node, seq in last.items():
            if node not in kept:
                self.release.setdefault(seq, []).append(node)
        self.dead = set()
        for start, end in plan["dead"]:
            self.dead.update(range(start, end))


class RefreshBackend:
    def __init__(self, inner, *, input_level: int, boot_in_level: int, boot_out_level: int,
                 latency_table: dict | None = None, policy: str = "lazy", plan: dict | None = None):
        if policy not in POLICIES:
            raise BackendError("Refresh policy must be lazy, record or plan")
        if (policy == "plan") != (plan is not None):
            raise BackendError("The plan policy, and only it, requires a refresh plan")
        self.inner, self.slots = inner, inner.slots
        self.input_level, self.boot_in_level, self.boot_out_level = input_level, boot_in_level, boot_out_level
        self.latency_table = latency_table
        self.policy = policy
        self._boots = 0
        self._min_level = input_level
        self._latency_us = None if latency_table is None else 0
        self._level_hist = Counter()
        self._cache = {}
        self._seq = 0
        self._plan = _Plan(plan) if plan is not None else None
        self._planned_cache = {}
        if policy == "record":
            # Per call: operand DAG nodes by position (None for plaintexts), the result node and any other
            # DAG nodes the inner backend created or reused during the call (a frontend sub is neg+add).
            self.calls, self.output_nodes, self._touched = [], None, None
            # E5b: (statement order, stage order within the statement) of every DAG node, by node id.
            self.node_stage, self._statements, self._stages = [], {}, {}
            self._statement, self._stage = -1, 0
            original = inner._apply

            def logged(kind, left, right=None, *, step=None):
                value = original(kind, left, right, step=step)
                if self._touched is not None:
                    self._touched.append(value.nid)
                if value.nid == len(self.node_stage):
                    self.node_stage.append((self._statement, self._stage))
                return value

            inner._apply = logged

    @property
    def op_context(self):
        return getattr(self.inner, "op_context", None)

    @op_context.setter
    def op_context(self, value):
        if hasattr(self.inner, "op_context"):
            self.inner.op_context = value
        if self.policy == "record" and value is not None:
            self._statement = self._statements.setdefault(value, len(self._statements))
            self._stage = 0

    def stage(self, name):
        """Stage names number in first-appearance order within a statement (record policy only)."""
        if self.policy == "record":
            names = self._stages.setdefault(self._statement, {})
            self._stage = names.setdefault(name, len(names)+1)

    @property
    def call_count(self) -> int:
        return self._seq

    def input(self, i):
        value = self.inner.input(i)
        if self.policy == "record":
            self.calls.append(((), value.nid, ()))
            if value.nid == len(self.node_stage):
                self.node_stage.append((-1, 0))
        self._seq += 1
        return RValue(self, value, None if self.policy == "record" else self.input_level)

    def const(self, vec):
        return self.inner.const(vec)

    def output(self, values):
        values = list(values)
        if self.policy == "plan":
            for position, node in self._plan.outputs.items():
                values[position] = self._planned(values[position], node)
            if self._seq != self._plan.calls:
                raise BackendError(f"Refresh plan covers {self._plan.calls} calls, this run made {self._seq}")
        if self.policy == "record":
            self.output_nodes = [value.value.nid for value in values]
        self.inner.output([value.value for value in values])

    def apply_exact(self, x, fn):
        self._seq += 1
        return RValue(self, self.inner.apply_exact(x.value, fn), x.level)

    def _record(self, key, level, *, rescale=False):
        self._level_hist[level] += 1
        if self.latency_table is not None:
            self._latency_us += self.latency_table[key][level]
            if rescale:
                self._latency_us += self.latency_table["rescale_single"][level]

    def _refreshed(self, x):
        key = id(x)
        if key not in self._cache:
            self._cache[key] = (x, x.bootstrap())
        return self._cache[key][1]

    def _planned(self, x, node):
        """The refreshed value of DAG node ``node``; duplicates of one node share a single bootstrap."""
        if self._planned_cache.get(node, False) is None:
            raise BackendError(f"Refresh plan reads node {node} after its last planned use")
        if node not in self._planned_cache:
            if not isinstance(x, RValue) or x.backend is not self:
                raise BackendError("A planned refresh must name a ciphertext of this backend")
            if x.level is None or x.level < self.boot_in_level:
                raise BackendError("Planned bootstrap input level is below boot_in_level")
            value = x.value.bootstrap()
            self._boots += 1
            self._record("bootstrap_single", self.boot_in_level)
            self._planned_cache[node] = RValue(self, value, self.boot_out_level)
        return self._planned_cache[node]

    def _substituted(self, operands):
        seq = self._seq
        dead = seq in self._plan.dead
        for position, node in self._plan.substitutions.get(seq, {}).items():
            operands[position] = self._planned(operands[position], node)
        for node in self._plan.release.get(seq, ()):
            # The operands list still holds this call's reference; later calls never read the node.
            self._planned_cache[node] = None
        return dead

    def _levels(self, ciphertexts, dead, *, multiply):
        """Operand and result levels under the plan policy (None for calls outside the output DAG)."""
        if dead:
            return None, None
        if any(value.level is None for value in ciphertexts):
            raise BackendError(f"Call {self._seq} reads a value outside the output DAG")
        level = min(value.level for value in ciphertexts)
        result_level = level-multiply
        if result_level < self.boot_in_level:
            raise BackendError(f"Planned placement leaves call {self._seq} below boot_in_level")
        return level, result_level

    def _binary(self, kind, left, right):
        operands = [left, right]
        for value in operands:
            if isinstance(value, RValue) and value.backend is not self:
                raise BackendError("Values belong to different refresh backends")
        if self.policy == "lazy":
            for index, value in enumerate(operands):
                if isinstance(value, RValue) and kind == "mul" and value.level <= self.boot_in_level:
                    operands[index] = self._refreshed(value)
        dead = self._substituted(operands) if self.policy == "plan" else False
        ciphertexts = [value for value in operands if isinstance(value, RValue)]
        raw = [value.value if isinstance(value, RValue) else value for value in operands]
        if self.policy == "record":
            self._touched = []
            result = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}[kind](*raw)
            nodes = tuple(value.value.nid if isinstance(value, RValue) else None for value in operands)
            extra = tuple(node for node in self._touched if node != result.nid)
            self._touched = None
            self.calls.append((nodes, result.nid, extra))
            self._seq += 1
            return RValue(self, result, None)
        if self.policy == "lazy":
            level = min(value.level for value in ciphertexts)
            result_level = level-(kind == "mul")
            if result_level < self.boot_in_level:
                raise BackendError("Arithmetic result level is below boot_in_level")
        else:
            level, result_level = self._levels(ciphertexts, dead, multiply=kind == "mul")
        result = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}[kind](*raw)
        if level is not None:
            key = ("mul" if kind == "mul" else "add")+("_double" if len(ciphertexts) == 2 else "_single")
            self._record(key, level, rescale=kind == "mul")
        self._seq += 1
        return RValue(self, result, result_level)

    def _unary(self, kind, x, step=None):
        """Negation and rotation under the record and plan policies."""
        if kind == "rot":
            if type(step) is not int:
                raise BackendError("Rotation step must be an int, excluding bool")
            if step % self.slots == 0:
                return x
        operands = [x]
        dead = self._substituted(operands) if self.policy == "plan" else False
        x = operands[0]
        if self.policy == "record":
            self._touched = []
        value = -x.value if kind == "neg" else x.value.rotate(step)
        if self.policy == "record":
            extra = tuple(node for node in self._touched if node != value.nid)
            self._touched = None
            self.calls.append(((x.value.nid,), value.nid, extra))
            self._seq += 1
            return RValue(self, value, None)
        level, _ = self._levels([x], dead, multiply=False)
        if level is not None:
            self._record("negate_single" if kind == "neg" else "rotate_single", level)
        self._seq += 1
        return RValue(self, value, level)

    def stats(self) -> dict:
        result = dict(boots=self._boots, min_level=self._min_level, latency_us=self._latency_us,
                      level_hist=dict(self._level_hist))
        if self.policy != "lazy":
            result.update(calls=self._seq, policy=self.policy)
        return result
