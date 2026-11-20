"""Shared paper policy for saved qwen48 serialized-operation table sums."""
from collections import Counter
import math

POLICY_ID='qwen48-operator-sum-omit-relinearize-20261117-1425'
EXCLUDED_OP='relinearize'
SOURCE_SCOPE=('Saved serialized execution operator-table sum, with relinearize omitted by the '
              '2026-11-17 14:25 task policy. Original observer scope for setup/release is retained. '
              'This operator sum has no full-inference calibration, encoding, keygen or host-overhead estimate.')


def profile_sum(prediction):
    """Keep other unpriced terms explicit; omit only the named policy operation."""
    result=dict(pred_latency_s=None,known_cost_partial_s=None,excluded_counts={},missing_remaining=[],
        source_scope=SOURCE_SCOPE,policy_id=POLICY_ID,available=False,error=None,
        excluded_operation=EXCLUDED_OP)
    if prediction is None or prediction=={}:return result
    known=prediction.get('table_known_sum_us')if isinstance(prediction,dict)else None
    missing=prediction.get('missing')if isinstance(prediction,dict)else None
    if (type(known)not in(int,float)or not math.isfinite(known)or known<0
            or not isinstance(missing,list)or any(not isinstance(r,dict)or not isinstance(r.get('op'),str)for r in missing)):
        result['error']='Saved prediction lacks a finite nonnegative known sum or explicit missing-operation inventory'
        return result
    # The current observer has no relinearize mapping. Reject an incompatible
    # already-priced representation instead of silently retaining that price.
    by_op=prediction.get('by_op_us')or{}
    if not isinstance(by_op,dict)or by_op.get(EXCLUDED_OP,0)!=0:
        result['error']='Saved known sum already includes relinearize; a separate documented projection is required'
        return result
    result.update(available=True,known_cost_partial_s=known/1e6,
        excluded_counts=dict(Counter(r['op']for r in missing if r['op']==EXCLUDED_OP)),
        missing_remaining=[dict(r)for r in missing if r['op']!=EXCLUDED_OP])
    if not result['missing_remaining']:result['pred_latency_s']=known/1e6
    return result
