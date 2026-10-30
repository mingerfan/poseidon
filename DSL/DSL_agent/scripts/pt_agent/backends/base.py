"""Arithmetic dispatch, ownership checks, constants and shared operation counts."""

import hashlib

import numpy as np


class BackendError(Exception):
    pass


class LevelError(BackendError):
    def __init__(self, op_index: int, msg: str):
        self.op_index, self.msg = op_index, msg
        super().__init__(f"op {op_index}: {msg}")


class _Value:
    def __init__(self, backend):
        self.backend = backend

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


class _Cipher(_Value):
    def __init__(self, backend, *, is_input=False):
        super().__init__(backend)
        self.is_input = is_input

    def __neg__(self):
        return self.backend._apply("neg", self)

    def rotate(self, k):
        if type(k) is not int:
            raise BackendError("Rotation step must be an int, excluding bool")
        k %= self.backend.slots
        return self if k == 0 else self.backend._apply("rot", self, step=k)

    def bootstrap(self):
        return self.backend._apply("boot", self)


class _Plain(_Value):
    def __init__(self, backend, data, name):
        super().__init__(backend)
        self.data, self.name = data, name

    def __neg__(self):
        raise BackendError("Plain values participate only in binary ciphertext operations")

    def rotate(self, k):
        raise BackendError("Plain values do not support rotation")

    def bootstrap(self):
        raise BackendError("Plain values do not support bootstrap")


def _vector(vec, slots, *, broadcast):
    try:
        data = np.array(vec, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendError("Expected a finite numeric vector") from exc
    lengths = (1, slots) if broadcast else (slots,)
    if data.ndim != 1 or data.size not in lengths or not np.isfinite(data).all():
        raise BackendError(f"Expected a finite vector of length {lengths}")
    if data.size == 1 and broadcast:
        data = np.full(slots, data[0], dtype=np.float64)
    data.flags.writeable = False
    return data


class _Counters:
    def __init__(self):
        self.counts = dict(add=0, sub=0, neg=0, mul_cc=0, mul_cp=0, rot=0, boot=0)
        self.rot_steps = set()

    @property
    def total(self):
        return sum(self.counts.values())

    def record(self, kind, step=None):
        self.counts[kind] += 1
        if kind == "rot":
            self.rot_steps.add(step)

    def stats(self, distinct_consts):
        return dict(counts=dict(self.counts), ops_total=self.total, distinct_consts=distinct_consts)


class Backend:
    _keep_const_data = True

    def __init__(self, slots):
        if type(slots) is not int or slots < 2 or slots & (slots-1):
            raise BackendError("slots must be a power of two >= 2")
        self.slots = slots
        self.op_context = None
        self._inputs = []
        self._constants = {}
        # Keep the original object alive for every identity-cache entry.
        self._constant_objects = {}
        self._outputs = None

    def input(self, index: int):
        if type(index) is not int or index != len(self._inputs):
            raise BackendError("Input indices must be consecutive integers starting at 0")
        value = self._make_input(index)
        self._inputs.append(value)
        return value

    def const(self, vec):
        return self._const(vec, keep_data=self._keep_const_data)

    def _const(self, vec, *, keep_data):
        readonly = isinstance(vec, np.ndarray) and not vec.flags.writeable
        if readonly:
            previous = self._constant_objects.get(id(vec))
            if previous is not None and previous[0] is vec:
                return previous[1]
        data = _vector(vec, self.slots, broadcast=True)
        key = hashlib.blake2b(data.tobytes(), digest_size=16).digest()
        if key not in self._constants:
            self._constants[key] = _Plain(self, data if self._keep_const_data else None,
                                          f"k{len(self._constants)}")
        named = self._constants[key]
        result = _Plain(self, data, named.name) if keep_data and named.data is None else named
        if readonly:
            self._constant_objects[id(vec)] = (vec, result)
        return result

    def _operand(self, value):
        if isinstance(value, _Value):
            if value.backend is not self:
                raise BackendError("Values belong to different backend instances")
            return value
        if type(value) in (int, float):
            return self.const([value])
        raise BackendError("Operand must be a local ciphertext, Plain or Python number excluding bool")

    def _binary(self, kind, left, right):
        left, right = self._operand(left), self._operand(right)
        if not isinstance(left, _Cipher) and not isinstance(right, _Cipher):
            raise BackendError("At least one operand must be a ciphertext")
        if kind in ("add", "mul") and isinstance(left, _Plain):
            left, right = right, left
        if kind == "mul":
            kind = "mul_cc" if isinstance(left, _Cipher) and isinstance(right, _Cipher) else "mul_cp"
        return self._apply(kind, left, right)

    def output(self, values: list) -> None:
        if self._outputs is not None:
            raise BackendError("Outputs may be declared only once")
        if not isinstance(values, list) or not values:
            raise BackendError("Declare at least one output in a list")
        for value in values:
            if not isinstance(value, _Cipher) or value.backend is not self or value.is_input:
                raise BackendError("Each output must be a computed ciphertext from this backend")
        self._outputs = list(values)

    def _make_input(self, index):
        raise NotImplementedError

    def _apply(self, kind, left, right=None, *, step=None):
        raise NotImplementedError


def run_program(backend: Backend, fn, n_inputs: int):
    if type(n_inputs) is not int or n_inputs < 0:
        raise BackendError("n_inputs must be a nonnegative integer")
    xs = [backend.input(i) for i in range(n_inputs)]
    outputs = fn(backend, *xs)
    outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    backend.output(outputs)
    return outputs
