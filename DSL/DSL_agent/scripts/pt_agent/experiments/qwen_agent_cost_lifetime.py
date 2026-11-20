"""Private original-schedule Qwen ownership and function-boundary diagnostics.

The existing guarded replay remains the orchestration and transaction.  This
variant changes temporary ownership, preserves CSE node identities, and records
existing function boundaries.  It never selects new coefficients or changes a
model, arithmetic order, placement algorithm, partition or acceptance policy.
"""
import ast
from collections.abc import MutableMapping
from contextlib import contextmanager
import functools
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import textwrap
from types import FunctionType, ModuleType
import weakref

from . import qwen_agent_resources as resources

MODULE = 'pt_agent.experiments.qwen_agent_cost_lifetime'
ROUND = 'autonomous-qwen-guarded-replay-lifetime'
MODE = 'original-full-decoder-reply13-private-cost-lifetime'
VARIANT = 'qwen-original-schedule-scalar-cse-lifetime-v1'
_MISSING = object()


class ScalarNumbering(MutableMapping):
    """Original CSE keys and nids, with canonical handles only while live.

    A caller retaining a handle always gets that identical handle on a CSE hit.
    Reconstructing a dead handle creates no DAG node and changes no node number.
    The owner reference is weak, including when cyclic GC is disabled.
    """
    def __init__(self, backend, expression):
        self.owner = weakref.ref(backend)
        self.expression = expression
        self.ids = {}
        self.live = weakref.WeakValueDictionary()

    def __getitem__(self, key):
        nid = self.ids[key]
        value = self.live.get(nid)
        if value is None:
            owner = self.owner()
            if owner is None:
                raise RuntimeError('Numbered DAG owner has been released')
            value = self.expression(owner, nid)
            self.live[nid] = value
        return value

    def __setitem__(self, key, value):
        if value.backend is not self.owner() or value.is_input:
            raise ValueError('Numbering requires a local noninput DAG expression')
        self.ids[key] = value.nid
        self.live[value.nid] = value

    def __contains__(self, key):
        return key in self.ids

    def __delitem__(self, key):
        del self.ids[key]

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def clear(self):
        self.ids.clear()
        self.live.clear()


def _sample(backend=None, recorder=None):
    """O(1) counters and small public process metadata; no payload or graph scan."""
    result = dict(self_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        wanted = {'VmRSS', 'VmHWM'}
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in wanted:
                result[key+'_kib'] = int(value.split()[0])
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                result['host_available_bytes'] = int(line.split()[1])*1024
                break
    except (OSError, ValueError, IndexError):
        result['process_sample_unavailable'] = True
    for name in ('kind', 'numbering', '_constant_objects', '_constants'):
        if backend is not None and hasattr(backend, name):
            result[name+'_length'] = len(getattr(backend, name))
    for name in ('calls', 'node_stage'):
        if recorder is not None and hasattr(recorder, name):
            result[name+'_length'] = len(getattr(recorder, name))
    return result


class Diagnostics:
    def __init__(self, path=None):
        from .r10_paper_metrics import StageRecorder
        self.recorder = StageRecorder(path) if path is not None else None

    @contextmanager
    def span(self, name, backend=None, recorder=None):
        if self.recorder is None:
            yield
            return
        log = self.recorder
        log.sequence += 1
        event = dict(event_id=f'{os.getpid()}:{log.sequence}', stage_id=name,
            parent_id=log.stack[-1]['event_id'] if log.stack else None,
            start_ns=log.clock(), pid=os.getpid(), ppid=os.getppid(),
            metadata=dict(ownership=_sample(backend, recorder)), cache_hit=False, artifact=None)
        log.emit(dict(event, type='begin')); log.stack.append(event)
        status, error = 'ok', None
        try:
            yield
        except BaseException as exc:
            status, error = 'error', type(exc).__name__
            raise
        finally:
            if not log.stack or log.stack[-1] is not event:
                raise RuntimeError('Diagnostic spans must close in stack order')
            log.stack.pop()
            event['metadata']['ownership_end'] = _sample(backend, recorder)
            log.emit(dict(event, type='end', end_ns=log.clock(), status=status, error=error))

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)
        return measured


def release_record(inner, recorder, previous=_MISSING):
    """Release only a finished private recorder, retaining the original DAG."""
    if previous is _MISSING:
        inner.__dict__.pop('_apply', None)
    else:
        inner._apply = previous
    for name in ('calls', 'node_stage', '_statements', '_stages', '_cache', '_planned_cache'):
        value = getattr(recorder, name, None)
        if value is not None:
            value.clear()
    recorder.output_nodes = None
    recorder._touched = None


def release_backend(backend):
    """Detach own handle tables after all statistics/details/simulation finish."""
    if hasattr(backend, 'numbering'):
        backend.numbering.clear()
    # Replace owned containers instead of mutating any retained result alias.
    backend._inputs, backend._outputs = [], None
    backend._constants, backend._constant_objects = {}, {}


class Lifetime:
    def __init__(self, diagnostics=None):
        self.diagnostics = diagnostics or Diagnostics()
        self.backends = weakref.WeakSet()
        self.records = []
        self.active = False

    def track_backend(self, backend):
        if self.active:
            self.backends.add(backend)

    def finish_records(self, start=0, *, keep=False):
        records, self.records = self.records[start:], self.records[:start]
        if not keep:
            for inner, recorder, previous in records:
                release_record(inner, recorder, previous)

    @contextmanager
    def estimating(self):
        if self.active:
            raise RuntimeError('Private cost lifetime scopes must not overlap')
        self.active = True
        try:
            yield
        finally:
            self.finish_records()
            for backend in list(self.backends):
                release_backend(backend)
            self.backends.clear()
            self.active = False


def scalar_cost_dag(base, numeric, lifetime):
    from ..backends.dag import DagExpr
    class ScalarCostDag(base):
        def __init__(self, slots):
            super().__init__(slots)
            self.numbering = ScalarNumbering(self, DagExpr)
            lifetime.track_backend(self)

        def const(self, vector):
            if isinstance(vector, numeric.ndarray) and not vector.flags.writeable:
                vector = numeric.array(vector, copy=True)
            return self._const(vector, keep_data=True)

        def finalize(self):
            with lifetime.diagnostics.span('normalize_finalize', self):
                if not self.finalized and self._outputs is not None:
                    # Original finalize never consults numbering.  Inputs and
                    # outputs retain their handles; all topology is in arrays.
                    self.numbering.clear()
                return super().finalize()
    return ScalarCostDag


def private_planner(lifetime, plan_module=None, dag_class=None, count_class=None, numeric=None):
    from .qwen_bridge_agent import _clone_namespace
    if plan_module is None:
        from .. import plan as plan_module
    if dag_class is None:
        from ..backends.cost_dag import CostDag as dag_class
    if count_class is None:
        from ..backends.streaming_count import StreamingCountBackend as count_class
    if numeric is None:
        import numpy as numeric
    original_refresh = plan_module.RefreshBackend
    def tracked_refresh(inner, **kwargs):
        previous = inner.__dict__.get('_apply', _MISSING)
        active = original_refresh(inner, **kwargs)
        lifetime.track_backend(inner)
        if kwargs.get('policy') == 'record':
            lifetime.records.append((inner, active, previous))
        return active
    namespace = _clone_namespace(plan_module, dict(
        CostDag=scalar_cost_dag(dag_class, numeric, lifetime),
        CountBackend=count_class, RefreshBackend=tracked_refresh))
    for name, stage in (('record_dag', 'record_lower'), ('_placement', 'placement'),
                        ('express', 'plan_express'), ('verify', 'conditional_plan_replay')):
        namespace[name] = lifetime.diagnostics.wrap(stage, namespace[name])
    original_program = namespace['plan_program']
    @functools.wraps(original_program)
    def plan_program(*args, **kwargs):
        start = len(lifetime.records)
        succeeded = False
        try:
            result = original_program(*args, **kwargs)
            succeeded = True
            return result
        finally:
            lifetime.finish_records(start, keep=succeeded and kwargs.get('keep_recorder', False))
    namespace['plan_program'] = plan_program
    return namespace


def private_cost_namespace(lifetime, cost_module=None, planner=None):
    planner = private_planner(lifetime) if planner is None else planner
    namespace = resources.private_cost_namespace(cost_module, planner)
    original_estimate = namespace['estimate']
    @functools.wraps(original_estimate)
    def estimate(*args, **kwargs):
        with lifetime.estimating(), lifetime.diagnostics.span('cost_estimate'):
            return original_estimate(*args, **kwargs)
    namespace['estimate'] = estimate
    return namespace


def _evaluation_delegate(function, namespace):
    """Bind just the existing local numerical-preflight dependency privately."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    found = []
    class Dependencies(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if (node.level == 1 and node.module == 'approximation_failures'
                    and tuple(alias.name for alias in node.names) == ('program_failures',)):
                found.append(True)
                return None
            return node
    tree = Dependencies().visit(tree)
    if len(found) != 1:
        raise ValueError('Original evaluate numerical-preflight import changed')
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno-1)
    exec(compile(tree, inspect.getsourcefile(function) or __file__, 'exec'), namespace)
    delegated = namespace[function.__name__]
    delegated.__defaults__, delegated.__kwdefaults__ = function.__defaults__, function.__kwdefaults__
    return delegated


def adapt_agent_functions(functions, *, diagnostics=None, costs=None, scope=None, program_failures=None):
    from .qwen_bridge_agent import _clone_namespace
    diagnostics = diagnostics or Diagnostics()
    lifetime = Lifetime(diagnostics)
    costs = private_cost_namespace(lifetime) if costs is None else costs
    callback = functions['evaluate']
    evaluator = inspect.getclosurevars(callback).nonlocals.get('original_evaluate')
    if (not isinstance(evaluator, FunctionType)
            or evaluator.__globals__.get('__name__') != 'pt_agent.planners.agent_tools'):
        raise ValueError('Require original private bridge Agent evaluator')
    if program_failures is None:
        from ..planners.approximation_failures import program_failures
    module = ModuleType(evaluator.__module__)
    module.__dict__.update(evaluator.__globals__)
    replacements = {name: diagnostics.wrap(stage, module.__dict__[name]) for name, stage in
                    (('_exact_check', 'independent_exact_math'), ('cost_feedback', 'cost'),
                     ('validate', 'heldout_validation'))}
    replacements['program_failures'] = diagnostics.wrap('approximation_preflight', program_failures)
    tools = _clone_namespace(module, replacements)
    measured = _evaluation_delegate(evaluator, tools)
    agent = ModuleType(functions['__name__']); agent.__dict__.update(functions)
    prepared = _clone_namespace(agent, dict(evaluate=resources._replace_closure(
        callback, 'original_evaluate', measured)))
    # Reuse the already audited complete same-globals dispatch adaptation.
    return resources.adapt_agent_functions(prepared, costs=costs, scope=scope)


def source_gate(args):
    from . import qwen_guarded_replay as original
    source = original.source_gate(args)
    source.update(cost_lifetime_variant=VARIANT, cost_lifetime_source=str(Path(__file__).resolve()),
        diagnostic_scope='Existing fit/exact/cost/record/finalize/placement/express/heldout function boundaries only',
        prior_RSS16_case_clock_reused=False, extra_numeric_or_Count_passes=0)
    out = getattr(args, 'out', None)
    if out is not None and out.is_dir():
        shutil.copy2(__file__, out/'variant-source/qwen_agent_cost_lifetime.py')
    return source


def parser():
    from .qwen_guarded_replay import parser as original
    return original()


def session_outcome(directory):
    from .qwen_guarded_replay import session_outcome as original
    return original(directory)


def _namespace():
    from . import qwen_guarded_replay as original
    from .qwen_bridge_agent import _clone_namespace
    private_bounded = ModuleType(original.bounded.__name__)
    bounded = _clone_namespace(original.bounded, {})
    private_bounded.__dict__.update(bounded)
    guarded = _clone_namespace(original, dict(bounded=private_bounded, MODULE=MODULE,
        ROUND=ROUND, MODE=MODE, __file__=__file__, source_gate=source_gate, parser=parser,
        CLOCK_SCOPE=original.CLOCK_SCOPE+'; prior928.387656277s RSS16 attempt excluded'))
    original_phase = guarded['_agent_phase']
    def phase(args, recorder, result):
        diagnostics = Diagnostics(args.work_dir/'evaluation-stage-events.jsonl')
        def functions(directory):
            selected = adapt_agent_functions(original.bridge._agent_functions(), diagnostics=diagnostics)
            selected['feedback_message'] = original.bounded.FeedbackPresenter(
                Path(directory)/'full-feedback', selected['feedback_message'])
            return selected
        # Ordinary cloned functions share bounded, rather than a detached copy.
        bounded['_agent_functions'] = functions
        private_bounded._agent_functions = functions
        result['cost_lifetime_variant'] = VARIANT
        return original_phase(args, recorder, result)
    guarded['_agent_phase'] = phase
    return guarded['_namespace']()


def main(argv=None):
    return _namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
