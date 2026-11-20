"""Original VGG16 fresh cold compilation with a16GiB sampled RSS allowance.

This resource-only recovery keeps the original trace/native workers, algorithm,
source admission, compiler argv and continuous clock. The existing adapter and
its14GiB policy remain unchanged. Actual execution belongs to the serial queue.
"""
from pathlib import Path
import shutil

from . import dacapo_fresh_cold as base


MODULE = 'pt_agent.experiments.dacapo_vgg_fresh_cold_rss16'
VARIANT = 'dacapo-original-vgg16-fresh-cold-rss16-resource-only-recovery-v1'
POLICIES = {'vgg16': dict(base.POLICIES['vgg16'], RSS_gib=16)}
RECOVERY_REASON = 'Original fresh VGG16 attempt stopped at its14GiB sampled RSS allowance'


def resource_policy(args):
    if getattr(args, 'model', None) != 'vgg16':
        raise ValueError('The RSS16 fresh recovery accepts only original VGG16')
    private = dict(vars(base), POLICIES=POLICIES)
    policy = base._ordinary(base.resource_policy, private)(args)
    policy.update(resource_only_recovery=True, recovery_reason=RECOVERY_REASON,
        original_adapter_module=base.MODULE,
        original_adapter_source=str(Path(base.__file__).resolve()),
        original_adapter_policy=dict(base.POLICIES['vgg16']),
        original_adapter_snapshot=str(Path(args.out)/'source'/Path(base.__file__).name),
        recovery_adapter_source=str(Path(__file__).resolve()),
        recovery_adapter_snapshot=str(Path(args.out)/'source'/Path(__file__).name))
    return policy


def phase_worker(args, phase, work, *, command_file=None, policy):
    """Add the reused adapter source beside the new wrapper's own snapshot."""
    target = Path(policy['original_adapter_snapshot'])
    if not target.exists():
        shutil.copyfile(base.__file__, target)
    return base.phase_worker(args, phase, work, command_file=command_file, policy=policy)


def _namespace():
    # Copy plain-function globals only; the original context managers, pure
    # workers and physical supervision implementation keep their own globals.
    return dict(vars(base), MODULE=MODULE, VARIANT=VARIANT, POLICIES=POLICIES,
        __file__=__file__, resource_policy=resource_policy, phase_worker=phase_worker,
        run=run)


def run(args):
    return base._ordinary(base.run, _namespace())(args)


def main(argv=None):
    return base._ordinary(base.main, _namespace())(argv)


if __name__ == '__main__':
    raise SystemExit(main())
