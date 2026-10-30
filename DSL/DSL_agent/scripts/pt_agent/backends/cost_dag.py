"""U4 §7.8 frontend normalization, used exclusively by the cost model."""

from collections import Counter

from .base import BackendError, _Cipher, _Counters, _Plain
from .dag import DagBackend, DagExpr


class CostDag(DagBackend):
    """Frontend subtraction, ordered value numbering, then output reachability."""
    def __init__(self, slots):
        super().__init__(slots)
        self.numbering = {}
        self.operands = []
        self.steps = []
        self.finalized = False

    def const(self, vec):
        # Keep the payload with the live Plain operand, while the existing
        # Backend.const deduplication table retains only its k name. This
        # permits sub(x,k) -> add(x,const(-k)) without retaining every array.
        return self._const(vec, keep_data=True)

    def _make_input(self, index):
        value = super()._make_input(index)
        self.operands.append(())
        self.steps.append(None)
        return value

    def _apply(self, kind, left, right=None, *, step=None):
        if self.finalized:
            raise BackendError('Cost DAG has already been finalized')
        if kind == 'sub':
            negative = self.const(-right.data) if isinstance(right,_Plain) else self._apply('neg',right)
            return self._apply('add',left,negative)
        operands = tuple(('cipher',v.nid) if isinstance(v,_Cipher) else ('plain',v.name)
                         for v in (left,right) if v is not None)
        key = kind,operands,step
        if key in self.numbering:
            return self.numbering[key]
        value = super()._apply(kind,left,right,step=step)
        self.operands.append(operands)
        self.steps.append(step)
        self.numbering[key] = value
        return value

    def finalize(self):
        """Retain output ancestors in their original order and renumber them."""
        if self.finalized:
            return
        if self._outputs is None:
            raise BackendError('Finalize requires declared cost DAG outputs')
        live, pending = set(),[v.nid for v in self._outputs]
        while pending:
            nid = pending.pop()
            if nid not in live:
                live.add(nid)
                pending.extend(self.args[nid])
        keep = [nid for nid in range(len(self.kind)) if nid in live]
        remap = {old:new for new,old in enumerate(keep)}
        before = Counter(self.prog_op)
        reachable = Counter(self.prog_op[nid] for nid in keep)
        self.normalization = dict(value_numbered_nodes=len(self.kind),live_nodes=len(keep),
            unreachable_nodes=len(self.kind)-len(keep),
            by_op=[dict(id=owner,before_dce=total,reachable=reachable[owner],
                        unreachable=total-reachable[owner]) for owner,total in before.items()])
        self.kind = [self.kind[nid] for nid in keep]
        self.args = [tuple(remap[a] for a in self.args[nid]) for nid in keep]
        self.prog_op = [self.prog_op[nid] for nid in keep]
        self.operands = [tuple((tag,remap[value] if tag=='cipher' else value) for tag,value in self.operands[nid])
                         for nid in keep]
        self.steps = [self.steps[nid] for nid in keep]
        self._inputs = [DagExpr(self,remap[v.nid],is_input=True) for v in self._inputs if v.nid in remap]
        self._outputs = [DagExpr(self,remap[v.nid]) for v in self._outputs]
        self._depth,self._total_depth = [],[]
        self.counts = _Counters()
        self.live_constants = set()
        for nid,(kind,args) in enumerate(zip(self.kind,self.args)):
            depth = max((self._depth[a] for a in args),default=0)
            total = max((self._total_depth[a] for a in args),default=0)
            if kind in ('mul_cc','mul_cp'):
                depth += 1
                total += 1
            elif kind == 'boot':
                depth = 0
            self._depth.append(depth)
            self._total_depth.append(total)
            if kind != 'input':
                self.counts.record(kind,self.steps[nid])
            self.live_constants.update(value for tag,value in self.operands[nid] if tag=='plain')
        self.numbering.clear()
        self.finalized = True

    def stats(self):
        if not self.finalized:
            raise BackendError('Cost DAG statistics require finalize')
        result = super().stats()
        result['distinct_consts'] = len(self.live_constants)
        return result
