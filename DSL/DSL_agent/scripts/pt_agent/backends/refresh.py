"""Lazy refresh placement around the existing arithmetic backends."""

from collections import Counter
import operator

from .base import BackendError, _Plain


class RValue(_Plain):
    # Plain ancestry gives reflected operations priority over an inner Plain on the left.
    def __init__(self, backend, value, level):
        self.backend, self.value, self.level = backend, value, level
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
        value = -self.value
        self.backend._record("negate_single", self.level)
        return RValue(self.backend, value, self.level)

    def rotate(self, k):
        value = self.value.rotate(k)
        if value is self.value:
            return self
        self.backend._record("rotate_single", self.level)
        return RValue(self.backend, value, self.level)

    def bootstrap(self):
        if self.level < self.backend.boot_in_level:
            raise BackendError("bootstrap input level is below boot_in_level")
        value = self.value.bootstrap()
        self.backend._boots += 1
        self.backend._record("bootstrap_single", self.backend.boot_in_level)
        return RValue(self.backend, value, self.backend.boot_out_level)


class RefreshBackend:
    def __init__(self, inner, *, input_level: int, boot_in_level: int, boot_out_level: int,
                 latency_table: dict | None = None):
        self.inner, self.slots = inner, inner.slots
        self.input_level, self.boot_in_level, self.boot_out_level = input_level, boot_in_level, boot_out_level
        self.latency_table = latency_table
        self._boots = 0
        self._min_level = input_level
        self._latency_us = None if latency_table is None else 0
        self._level_hist = Counter()
        self._cache = {}

    def input(self, i):
        return RValue(self, self.inner.input(i), self.input_level)

    def const(self, vec):
        return self.inner.const(vec)

    def output(self, values):
        self.inner.output([value.value for value in values])

    def apply_exact(self, x, fn):
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

    def _binary(self, kind, left, right):
        operands = [left, right]
        for index, value in enumerate(operands):
            if isinstance(value, RValue):
                if value.backend is not self:
                    raise BackendError("Values belong to different refresh backends")
                if kind == "mul" and value.level <= self.boot_in_level:
                    operands[index] = self._refreshed(value)
        ciphertexts = [value for value in operands if isinstance(value, RValue)]
        level = min(value.level for value in ciphertexts)
        result_level = level-(kind == "mul")
        if result_level < self.boot_in_level:
            raise BackendError("Arithmetic result level is below boot_in_level")
        raw = [value.value if isinstance(value, RValue) else value for value in operands]
        result = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}[kind](*raw)
        key = ("mul" if kind == "mul" else "add")+("_double" if len(ciphertexts) == 2 else "_single")
        self._record(key, level, rescale=kind == "mul")
        return RValue(self, result, result_level)

    def stats(self) -> dict:
        return dict(boots=self._boots, min_level=self._min_level, latency_us=self._latency_us,
                    level_hist=dict(self._level_hist))
