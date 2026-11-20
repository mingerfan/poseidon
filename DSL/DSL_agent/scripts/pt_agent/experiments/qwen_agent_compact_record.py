"""Detached fallback entry using exact packed refresh-record storage.

The original lifetime planner and retained-reply transaction remain the entry.
Only the private recording factory replaces call/stage containers, preserving
their constructor contents and order. This module grants no scheduler or frozen
source admission and does not change the currently running lifetime entry.
"""
from pathlib import Path
import shutil

from . import qwen_agent_cost_lifetime as base
from . import qwen_record_storage as storage
from .qwen_record_storage import PackedCalls, InternedStages

MODULE = 'pt_agent.experiments.qwen_agent_compact_record'
ROUND = 'autonomous-qwen-guarded-replay-compact-record'
MODE = 'original-full-decoder-reply13-private-compact-record'
VARIANT = 'qwen-original-schedule-packed-calls-interned-stages-v1'


def private_planner(lifetime, plan_module=None, dag_class=None, count_class=None, numeric=None):
    """Replace only the original private factory's returned recording tables."""
    namespace = base.private_planner(lifetime, plan_module=plan_module,
        dag_class=dag_class, count_class=count_class, numeric=numeric)
    original_refresh = namespace['RefreshBackend']

    def refresh(inner, **kwargs):
        active = original_refresh(inner, **kwargs)
        if kwargs.get('policy') == 'record':
            calls, stages = PackedCalls(), InternedStages()
            for row in active.calls:
                calls.append(row)
            for pair in active.node_stage:
                stages.append(pair)
            active.calls, active.node_stage = calls, stages
        return active

    namespace['RefreshBackend'] = refresh
    return namespace


def private_cost_namespace(lifetime, cost_module=None, planner=None):
    if planner is None:
        planner = private_planner(lifetime)
    return base.private_cost_namespace(lifetime, cost_module=cost_module, planner=planner)


def source_gate(args):
    source = base.source_gate(args)
    source.update(compact_record_variant=VARIANT,
        compact_record_source=str(Path(__file__).resolve()),
        compact_record_storage_source=str(Path(storage.__file__).resolve()),
        compact_record_scope='Original call values/order and exact statement-stage pairs; storage only',
        compact_record_extra_passes=0)
    out = getattr(args, 'out', None)
    if out is not None and out.is_dir():
        target = out/'variant-source'
        target.mkdir(exist_ok=True)
        for path in (Path(__file__), Path(storage.__file__)):
            shutil.copyfile(path, target/path.name)
    return source


def parser():
    return base.parser()


def session_outcome(directory):
    return base.session_outcome(directory)


def _namespace():
    from .qwen_bridge_agent import _clone_namespace
    private = _clone_namespace(base, dict(MODULE=MODULE, ROUND=ROUND, MODE=MODE,
        __file__=__file__, source_gate=source_gate,
        private_planner=private_planner, private_cost_namespace=private_cost_namespace))
    return private['_namespace']()


def main(argv=None):
    return _namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
