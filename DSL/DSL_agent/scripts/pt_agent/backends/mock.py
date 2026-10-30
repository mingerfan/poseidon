"""Immutable float64 slot vectors for development arithmetic checks."""

import hashlib
import gc

import numpy as np

from .base import Backend, BackendError, _Cipher, _Plain, _vector


class MockExpr(_Cipher):
    def __init__(self, backend, value, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.value = backend._value(value)


class MockBackend(Backend):
    def __init__(self, slots, batch=None, *, watch=None, noise=None, stream_constants=True):
        from ..cost import release_estimate_caches
        release_estimate_caches()
        super().__init__(slots)
        if batch is not None and (type(batch) is not int or batch < 1):
            raise BackendError("batch must be a positive integer or None")
        self.batch = batch
        self._bound = None
        self._nid = 0
        self.watch = set(watch or ())
        self.noise = noise
        self.watch_max = {}
        self.stream_constants = stream_constants

    def const(self, vec):
        if not self.stream_constants:
            return super().const(vec)
        data = _vector(vec, self.slots, broadcast=True)
        key = hashlib.blake2b(data.tobytes(), digest_size=16).digest()
        index = self._constants.setdefault(key, len(self._constants))
        return _Plain(self, data, f"k{index}")

    def _record_node(self, value):
        nid = self._nid
        self._nid += 1
        if nid in self.watch:
            self.watch_max[nid] = float(np.max(np.abs(value.value)))
            if self.noise is not None:
                sigma, seed = self.noise
                noise = np.random.default_rng(seed+nid).normal(0, sigma, value.value.shape)
                return MockExpr(self, value.value+noise, is_input=value.is_input)
        return value

    def _value(self, value):
        if self.batch is None:
            return _vector(value, self.slots, broadcast=False)
        try:
            data = np.array(value, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BackendError("Expected a finite batch array") from exc
        if data.shape != (self.batch, self.slots) or not np.isfinite(data).all():
            raise BackendError(f"Expected a finite array of shape ({self.batch}, {self.slots})")
        data.flags.writeable = False
        return data

    def bind(self, inputs):
        if not isinstance(inputs, list):
            raise BackendError("bind requires a list of slot vectors")
        self._bound = [self._value(value) for value in inputs]

    def release_batch(self):
        """Drop batch ownership and collect unreachable arithmetic closures."""
        self._bound = None
        self._inputs.clear()
        self._outputs = None
        self._constants.clear()
        self._constant_objects.clear()
        gc.collect()

    def _make_input(self, index):
        if self._bound is None or index >= len(self._bound):
            raise BackendError("Bind an input vector before requesting its index")
        return self._record_node(MockExpr(self, self._bound[index], is_input=True))

    def _apply(self, kind, left, right=None, *, step=None):
        def data(value):
            return value.data if isinstance(value, _Plain) else value.value

        try:
            with np.errstate(over="raise", invalid="raise"):
                x = data(left)
                if kind == "add":
                    out = x + data(right)
                elif kind == "sub":
                    out = x - data(right)
                elif kind in ("mul_cc", "mul_cp"):
                    out = x * data(right)
                elif kind == "neg":
                    out = -x
                elif kind == "rot":
                    out = np.roll(x, -step, axis=-1)
                elif kind == "boot":
                    out = x.copy()
                else:
                    raise BackendError(f"Unknown operation {kind}")
                return self._record_node(MockExpr(self, out))
        except FloatingPointError as exc:
            raise BackendError("Mock arithmetic produced a nonfinite value") from exc

    def apply_exact(self, value, fn: str):
        from ..kernels.poly import EXACT

        if not isinstance(value, MockExpr) or value.backend is not self:
            raise BackendError("Exact activation requires a local Mock ciphertext")
        return MockExpr(self, EXACT[fn](value.value))

    @property
    def outputs(self):
        return [] if self._outputs is None else [value.value for value in self._outputs]
