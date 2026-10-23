"""Arithmetic mapped directly to a caller-supplied Hecate frontend."""

from .core.base import Backend, BackendError, _Cipher, _Plain


class HecateExpr(_Cipher):
    def __init__(self, backend, value, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.value = value


class HecateBackend(Backend):
    def __init__(self, slots, hc, inputs):
        super().__init__(slots)
        self.hc, self.inputs, self.plains = hc, inputs, {}

    def _make_input(self, index):
        return HecateExpr(self, self.inputs[index], is_input=True)

    def _apply(self, kind, left, right=None, *, step=None):
        def data(value):
            if isinstance(value, _Plain):
                if value.name not in self.plains:
                    self.plains[value.name] = self.hc.Plain(value.data)
                return self.plains[value.name]
            return value.value

        x = data(left)
        if kind == "add":
            result = x+data(right)
        elif kind == "sub":
            result = x-data(right)
        elif kind in ("mul_cc", "mul_cp"):
            result = x*data(right)
        elif kind == "neg":
            result = -x
        elif kind == "rot":
            result = x.rotate(step)
        elif kind == "boot":
            result = self.hc.bootstrap(x)
        else:
            raise BackendError(f"Unknown operation {kind}")
        return HecateExpr(self, result)
