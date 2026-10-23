"""Immutable float64 slot operations for package self-checks."""

import numpy as np

from .core.base import Backend, BackendError, _Cipher, _Plain, _vector


class NumpyExpr(_Cipher):
    def __init__(self, backend, value, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.value = backend._value(value)


class NumpyBackend(Backend):
    def __init__(self, slots, batch=None):
        super().__init__(slots)
        if batch is not None and (type(batch) is not int or batch < 1):
            raise BackendError("batch must be a positive integer or None")
        self.batch, self._bound = batch, None

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

    def _make_input(self, index):
        if self._bound is None or index >= len(self._bound):
            raise BackendError("Bind an input vector before requesting its index")
        return NumpyExpr(self, self._bound[index], is_input=True)

    def _apply(self, kind, left, right=None, *, step=None):
        def data(value):
            return value.data if isinstance(value, _Plain) else value.value

        try:
            with np.errstate(over="raise", invalid="raise"):
                x = data(left)
                if kind == "add":
                    out = x+data(right)
                elif kind == "sub":
                    out = x-data(right)
                elif kind in ("mul_cc", "mul_cp"):
                    out = x*data(right)
                elif kind == "neg":
                    out = -x
                elif kind == "rot":
                    out = np.roll(x, -step, axis=-1)
                elif kind == "boot":
                    out = x.copy()
                else:
                    raise BackendError(f"Unknown operation {kind}")
                return NumpyExpr(self, out)
        except FloatingPointError as exc:
            raise BackendError("Mock arithmetic produced a nonfinite value") from exc
