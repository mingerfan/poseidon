"""Operation counts and per-value depth/level bookkeeping."""

from .base import Backend, BackendError, LevelError, _Cipher, _Counters


class CountExpr(_Cipher):
    def __init__(self, backend, depth=0, total_depth=0, level=None, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.depth, self.total_depth, self.level = depth, total_depth, level


class CountBackend(Backend):
    _keep_const_data = False

    def __init__(self, slots, input_level=None, boot_out_level=None):
        super().__init__(slots)
        for value in (input_level, boot_out_level):
            if value is not None and (type(value) is not int or value < 0):
                raise BackendError("Levels must be nonnegative integers or None")
        self.input_level, self.boot_out_level = input_level, boot_out_level
        self._counter = _Counters()
        self._max_depth = self._max_total_depth = 0
        self._min_level = None

    def _track(self, value):
        self._max_depth = max(self._max_depth, value.depth)
        self._max_total_depth = max(self._max_total_depth, value.total_depth)
        if value.level is not None:
            self._min_level = value.level if self._min_level is None else min(self._min_level, value.level)
        return value

    def _make_input(self, index):
        return self._track(CountExpr(self, level=self.input_level, is_input=True))

    def _apply(self, kind, left, right=None, *, step=None):
        ciphertexts = [value for value in (left, right) if isinstance(value, _Cipher)]
        depth = max(value.depth for value in ciphertexts)
        total_depth = max(value.total_depth for value in ciphertexts)
        level = None if self.input_level is None else min(value.level for value in ciphertexts)
        if kind in ("mul_cc", "mul_cp"):
            depth += 1
            total_depth += 1
            if level is not None:
                level -= 1
        elif kind == "boot":
            if self.boot_out_level is None:
                raise BackendError("bootstrap requires boot_out_level")
            depth = 0
            level = self.boot_out_level if self.input_level is not None else None
        if level is not None and level < 0:
            raise LevelError(self._counter.total, "Ciphertext level would be negative")
        self._counter.record(kind, step)
        return self._track(CountExpr(self, depth, total_depth, level))

    def stats(self):
        result = self._counter.stats(len(self._constants))
        result.update(rot_steps=sorted(self._counter.rot_steps), max_depth=self._max_depth,
                      max_total_depth=self._max_total_depth, min_level=self._min_level,
                      outputs=0 if self._outputs is None else len(self._outputs))
        return result
