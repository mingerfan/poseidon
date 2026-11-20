"""Detached source-only fallback grouping exact private CSE outer keys."""
from pathlib import Path
import shutil

from . import qwen_agent_compact_record as compact
from . import qwen_grouped_ids as storage
from . import qwen_append_observation as observation
from .qwen_grouped_ids import GroupedIds

MODULE = 'pt_agent.experiments.qwen_agent_grouped_record'
ROUND = 'autonomous-qwen-guarded-replay-grouped-record'
MODE = 'original-full-decoder-reply13-private-grouped-record'
VARIANT = 'qwen-original-schedule-grouped-cse-outer-keys-v1'


def private_planner(lifetime, plan_module=None, dag_class=None, count_class=None, numeric=None):
    namespace = compact.private_planner(lifetime, plan_module=plan_module,
        dag_class=dag_class, count_class=count_class, numeric=numeric)
    original_cost = namespace['CostDag']
    graph_sequence = 0
    def cost_dag(*args, **kwargs):
        nonlocal graph_sequence
        active = original_cost(*args, **kwargs)
        numbering = active.numbering
        if (not isinstance(numbering, compact.base.ScalarNumbering)
                or type(numbering.ids) is not dict or numbering.ids):
            raise ValueError('Grouped CSE requires the original empty ScalarNumbering ids')
        numbering.ids = GroupedIds()
        graph_sequence += 1
        observation.install_append_observation(active, lifetime.diagnostics.recorder, graph_sequence)
        return active
    namespace['CostDag'] = cost_dag
    return namespace


def private_cost_namespace(lifetime, cost_module=None, planner=None):
    if planner is None:
        planner = private_planner(lifetime)
    return compact.private_cost_namespace(lifetime, cost_module=cost_module, planner=planner)


def source_gate(args):
    source = compact.source_gate(args)
    source.update(grouped_record_variant=VARIANT,
        grouped_record_source=str(Path(__file__).resolve()),
        grouped_record_storage_source=str(Path(storage.__file__).resolve()),
        grouped_record_observation_source=str(Path(observation.__file__).resolve()),
        grouped_record_observation_scheme=observation.SCHEME,
        grouped_record_observation_interval_nodes=observation.INTERVAL_NODES,
        grouped_record_scope='Exact original CSE keys/nids and global insertion order; borrowed operand tuples',
        grouped_record_extra_passes=0)
    out = getattr(args, 'out', None)
    if out is not None and out.is_dir():
        target = out/'variant-source'
        target.mkdir(exist_ok=True)
        for path in (Path(__file__), Path(storage.__file__), Path(observation.__file__)):
            shutil.copyfile(path, target/path.name)
    return source


def parser():
    return compact.parser()


def session_outcome(directory):
    return compact.session_outcome(directory)


def _namespace():
    from .qwen_bridge_agent import _clone_namespace
    private = _clone_namespace(compact, dict(MODULE=MODULE, ROUND=ROUND, MODE=MODE,
        __file__=__file__, source_gate=source_gate,
        private_planner=private_planner, private_cost_namespace=private_cost_namespace))
    return private['_namespace']()


def main(argv=None):
    return _namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
