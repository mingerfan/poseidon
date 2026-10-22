"""Arithmetic DAG with PT-DSL source ownership and CountBackend statistics."""

from .base import Backend, _Cipher, _Counters


class DagExpr(_Cipher):
    def __init__(self, backend, nid, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.nid = nid


class DagBackend(Backend):
    _keep_const_data = False

    def __init__(self, slots):
        super().__init__(slots)
        self.kind: list[str] = []
        self.args: list[tuple[int, ...]] = []
        self.prog_op: list[int | None] = []
        self.counts = _Counters()
        self._depth = []
        self._total_depth = []

    def _append(self, kind, args, *, is_input=False):
        nid = len(self.kind)
        depth = max((self._depth[arg] for arg in args), default=0)
        total_depth = max((self._total_depth[arg] for arg in args), default=0)
        if kind in ('mul_cc', 'mul_cp'):
            depth += 1
            total_depth += 1
        elif kind == 'boot':
            depth = 0
        self.kind.append(kind)
        self.args.append(args)
        self.prog_op.append(self.op_context)
        self._depth.append(depth)
        self._total_depth.append(total_depth)
        return DagExpr(self, nid, is_input=is_input)

    def _make_input(self, index):
        return self._append('input', (), is_input=True)

    def _apply(self, kind, left, right=None, *, step=None):
        args = tuple(value.nid for value in (left, right) if isinstance(value, _Cipher))
        self.counts.record(kind, step)
        return self._append(kind, args)

    def stats(self):
        result = self.counts.stats(len(self._constants))
        result.update(rot_steps=sorted(self.counts.rot_steps), max_depth=max(self._depth, default=0),
                      max_total_depth=max(self._total_depth, default=0), min_level=None,
                      outputs=0 if self._outputs is None else len(self._outputs), nodes=len(self.kind))
        return result
