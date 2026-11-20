"""Original fresh Rule63 chain with the current paper-only pricing projection.

The frozen inner phases and their scientific metadata remain unchanged. Only
the independent paper-row projection and source snapshot use this module.
"""
from collections import Counter
from pathlib import Path
import shutil
from types import FunctionType

from . import cold_rule63_whole_model as original
from . import a_group_profile_sum_policy as pricing


def paper_record(args,result,completion,timing):
    row=original.paper_record(args,result,completion,timing)
    prediction=(result['phases'].get('compile')or{}).get('prediction')
    policy=pricing.profile_sum(prediction)
    row.update(pred_latency_s=policy['pred_latency_s'],known_cost_partial_s=policy['known_cost_partial_s'],
        missing_price_ops=dict(Counter(r['op']for r in policy['missing_remaining'])),
        excluded_price_ops=policy['excluded_counts'],prediction_policy=policy)
    original.write_json(args.out/'paper-prediction-policy.json',policy)
    return row


def snapshot(args):
    original.snapshot(args)
    target=args.out/'source'
    for source in(Path(__file__),Path(pricing.__file__)):
        shutil.copyfile(source,target/source.name)
    original.write_json(target/'paper-pricing-policy.json',dict(policy_id=pricing.POLICY_ID,
        excluded_operation=pricing.EXCLUDED_OP,source_scope=pricing.SOURCE_SCOPE,
        frozen_inner_phases_unchanged=True))


# This private dictionary leaves imported original entry defaults untouched.
_namespace=original.__dict__.copy()
_namespace.update(paper_record=paper_record,snapshot=snapshot)
execute=FunctionType(original.execute.__code__,_namespace,original.execute.__name__,
                     original.execute.__defaults__,original.execute.__closure__)
execute.__kwdefaults__=original.execute.__kwdefaults__


def main(argv=None):
    args=original.arguments(argv)
    if args.worker_phase:
        from .cold_rule63_worker import run
        return run(args)
    return int(execute(args)['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
