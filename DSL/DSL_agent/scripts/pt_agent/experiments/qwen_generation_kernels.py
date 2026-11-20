"""Source-bound trusted generation kernels; no model/CLI/native runner.

First20->22:19 logits plus the original token0 handle produce19 selections,
one mixed embedding and two updated public masks. Terminal19->19 omits unused
embedding work. Callers own the original weights, physical backend and later
three real mixed/mask boundary bootstraps. Shared helpers remain unchanged.
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

VARIANT = 'qwen-cubic52-first20to22-terminal19to19-v1'
LEVELS = dict(input_level=5, boot_in_level=1, boot_out_level=5)
SLOTS, VOCABULARY, HIDDEN, CAPACITY = 16384, 151936, 896, 32
WEIGHT = 'original.model.embed_tokens.weight'


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _state(state, *, vocabulary=VOCABULARY, capacity=CAPACITY):
    state = _json(state)
    length = state.get('prefix_length');ids = state.get('prefix_token_ids')
    if (type(length) is not int or not 1 <= length <= capacity-2
            or type(state.get('step')) is not int or state['step'] != 0
            or not isinstance(ids, list) or len(ids) != capacity
            or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
            or not isinstance(state.get('sample_id'), str)
            or state.get('attention_mask') != [[1.]*length+[0.]*(capacity-length)]
            or state.get('sel') != [[0.]*(length-1)+[1.]+[0.]*(capacity-length)]):
        raise ValueError('Require the saved step0 right-padded prefix and its exact public masks')
    return state


@dataclass(frozen=True)
class SourceBinding:
    _record: dict

    def snapshot(self):
        return copy.deepcopy(self._record)


def bind_source(source_case, initial_state):
    """Read closed source/generation metadata only; do not load weights/samples."""
    from .qwen_prefill_compile import frozen_source
    case = Path(source_case).resolve();frozen = frozen_source(case)
    generation = json.loads((case/'generation-input.json').read_text())
    state = _state(initial_state)
    if (frozen.get('candidate_frozen') is not True or frozen.get('source_program_ast_verified') is not True
            or frozen.get('full_model_layers') != 24 or frozen.get('sequence_capacity') != CAPACITY
            or generation.get('partition') != 'calibration' or generation.get('sequence_length') != CAPACITY
            or type(generation.get('index')) is not int or generation['index'] < 0
            or generation.get('sample_id') != state['sample_id']
            or generation.get('prompt_length') != state['prefix_length']):
        raise ValueError('Tie the exact frozen source and saved first eligible calibration prefix')
    return SourceBinding(_json(dict(format='qwen-generation-kernel-source-v1', variant=VARIANT,
        frozen=frozen, generation=generation, initial_state=state,
        initial_state_scope='Caller reconstructs the original same saved sample; no selection/tokenization in this adapter',
        geometry=dict(slots=SLOTS, vocabulary=VOCABULARY, hidden=HIDDEN, capacity=CAPACITY),
        input_payloads_verified=False, independent_C2_gate='original frozen_source',
        standard_lm_quality=None, native_compiled=False)))


def _geometry(slots, vocabulary, hidden, capacity):
    if (any(type(x) is not int for x in (slots, vocabulary, hidden, capacity))
            or slots < capacity or slots < 4 or slots & (slots-1) or vocabulary < 2
            or not 1 <= hidden <= slots//2 or capacity < 3):
        raise ValueError('Use finite original packing geometry')
    sizes = [min(slots//2, vocabulary-start) for start in range(0, vocabulary, slots//2)]
    return dict(slots=slots, vocabulary=vocabulary, hidden=hidden, capacity=capacity,
                vocabulary_blocks=len(sizes), valid_vocabulary_sizes=sizes)


class Kernel:
    """Own only the caller's readonly embedding reference, never cache operands."""
    def __init__(self, kind, binding, matrix, geometry):
        if kind not in ('first', 'terminal') or not isinstance(binding, SourceBinding):
            raise ValueError('Require a bound first or terminal trusted kernel')
        if geometry != _geometry(*(geometry[key] for key in ('slots', 'vocabulary', 'hidden', 'capacity'))):
            raise ValueError('Retain the exact packing geometry and valid vocabulary tail')
        self.kind, self._binding, self.geometry = kind, binding.snapshot(), dict(geometry)
        self._initial = _state(self._binding['initial_state'], vocabulary=geometry['vocabulary'], capacity=geometry['capacity'])
        self._matrix = matrix if kind == 'first' else None
        self.closed = False
        if kind == 'first':
            import numpy as np
            if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64
                    or matrix.shape != (geometry['vocabulary'], geometry['hidden']) or matrix.flags.writeable):
                raise ValueError('Retain the exact readonly original embedding matrix')
        elif matrix is not None:
            raise ValueError('The terminal kernel owns no unused embedding')
        from .qwen_monotone_selector import config
        self._source = _json(dict(binding=self._binding, kind=kind, geometry=geometry,
            selector=config(), embedding_weight=WEIGHT if kind == 'first' else None,
            original_token0_input=geometry['vocabulary_blocks'] if kind == 'first' else None,
            output_roles=self.output_roles(), future_boundary_bootstraps=3 if kind == 'first' else 0,
            boundary_scope='Caller emits real bootstraps only for mixed embedding and both masks; no output-level relabel'))

    def output_roles(self):
        blocks = self.geometry['vocabulary_blocks']
        roles = ['one_hot:'+str(i) for i in range(blocks)]
        return roles+['mixed_embedding', 'attention_mask', 'sel'] if self.kind == 'first' else roles

    @property
    def input_count(self):
        return self.geometry['vocabulary_blocks']+(self.kind == 'first')

    def source(self):
        if self.closed:raise ValueError('Kernel reference lifetime has ended')
        return copy.deepcopy(self._source)

    def __call__(self, active):
        if self.closed or active.slots != self.geometry['slots']:
            raise ValueError('Use this live kernel and its original physical slot geometry')
        from .qwen_monotone_selector import analytic_coefficients
        from ..kernels.encrypted_generation import select_maximum
        from ..kernels import compact_linear
        import numpy as np
        inputs = [active.input(i) for i in range(self.input_count)]
        blocks = self.geometry['vocabulary_blocks']
        with _kernel_scope():
            selected = select_maximum(active, inputs[:blocks], self.geometry['vocabulary'],
                bound=32., normalized_gap=2.**-27, coefficients=analytic_coefficients(np))
            output = list(selected['one_hot'])
            if self.kind == 'first':
                mixed, layout = compact_linear.embedding_from_selection(active, selected, self._matrix)
                if len(mixed) != 1 or layout.size != self.geometry['hidden']:
                    raise ValueError('The actual selected original embedding occupies one token block')
                # Preserve staged.masks' source operand: this is the original
                # token0 handle, supplied explicitly as logical input19.
                zero = inputs[-1]*active.const([0.])
                masks = []
                for name in ('attention_mask', 'sel'):
                    vector = np.zeros(active.slots, dtype=np.float64)
                    length = self._initial['prefix_length']+1
                    if name == 'attention_mask':vector[:length] = 1.
                    else:vector[length-1] = 1.
                    masks.append(zero+active.const(vector))
                output.extend([mixed[0], *masks])
            active.output(output)

    def close(self):
        """Release our reference; the caller still owns any original mmap owner."""
        self._matrix = None;self.closed = True


@contextmanager
def _kernel_scope():
    from .qwen_staged_generation import compact_scope
    with compact_scope():yield


def make_kernels(binding, original_weights):
    """Pass the exact info.graph.weights returned by original _load_program."""
    if (not isinstance(binding, SourceBinding) or len(original_weights) != 559
            or binding._record.get('geometry') != dict(slots=SLOTS, vocabulary=VOCABULARY, hidden=HIDDEN, capacity=CAPACITY)):
        raise ValueError('Retain the original559-weight frozen source and full vocabulary geometry')
    geometry = _geometry(SLOTS, VOCABULARY, HIDDEN, CAPACITY)
    return (Kernel('first', binding, original_weights[WEIGHT], geometry),
            Kernel('terminal', binding, None, geometry))


@dataclass(frozen=True)
class PlannedKernel:
    source: dict
    refresh_plan: dict
    replay: dict


def _check_plan(kernel, planned):
    if not isinstance(planned, PlannedKernel) or planned.source != kernel.source():
        raise ValueError('The exact source-bound kernel and plan must agree')
    p = planned.refresh_plan
    if (p.get('format') != 'pt-agent-refresh-plan-v1' or p.get('levels') != LEVELS
            or any(type(p.get(k)) is not int or p[k] < 0 for k in ('calls', 'boots'))
            or any(not isinstance(p.get(k), list) for k in ('substitutions', 'outputs', 'dead'))
            or planned.replay.get('calls') != p['calls'] or planned.replay.get('boots') != p['boots']
            or planned.replay.get('inner_boots') != p['boots']):
        raise ValueError('Require actual original planner expression and independent replay counts')
    return p


@contextmanager
def _metadata(template):
    if not isinstance(template, list) or any(not isinstance(x, str) for x in template):
        raise ValueError('The saved compiler argv must be a JSON list of strings')
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(template):yield


def _api():
    # Called only after the metadata context, before any cost/plan import.
    from .qwen_prefill_compile import _planner_namespace
    from ..backends import RefreshBackend
    from ..backends.streaming_count import StreamingCountBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    return dict(planner=_planner_namespace(), Refresh=RefreshBackend,
                Count=StreamingCountBackend, Emit=StreamingEmitBackend)


def _release_active(active):
    if active is None:return
    active.__dict__.pop('output', None)
    for key in ('_planned_cache', '_cache'):
        value = getattr(active, key, None)
        if value is not None:value.clear()
    active._plan = None


def _release_record(dag, recorder):
    """Only fresh owned recording objects are cleared, never a caller plan."""
    _release_active(recorder)
    if recorder is not None:
        if hasattr(recorder, 'calls'):recorder.calls.clear()
        recorder.output_nodes = None
    if dag is not None:
        dag.__dict__.pop('_apply', None)
        for key in ('numbering', '_constant_objects', '_constants', '_inputs'):
            value = getattr(dag, key, None)
            if value is not None:value.clear()
        dag._outputs = None


def plan_kernel(kernel, *, metadata_template):
    """Original callback record -> depth-DP -> express -> independent verify."""
    dag = recorder = replay_active = replay_inner = None;started = time.perf_counter_ns()
    with _metadata(metadata_template):
        api = _api();planner = api['planner']
        def recorded(active):
            nonlocal dag, recorder
            # Retain owned objects before the callback, including if lowering
            # fails before original record_dag can return its tuple.
            recorder, dag = active, active.inner
            kernel(active)
        try:
            # Kernel owns its single dispatch scope on every callback. Avoid
            # recursively cloning the already private embedding type guard.
            returned_dag, returned_recorder, _ = planner['record_dag'](recorded, kernel.geometry['slots'],
                frontend_normalize=True, profile='qwen48')
            if returned_dag is not dag or returned_recorder is not recorder:
                raise ValueError('Original recording callback ownership changed')
            placement = planner['_placement'](dag, 'qwen48')
            p = planner['express'](dag, recorder, placement, profile='qwen48')
            planner['_merge'](placement, p, 'qwen48')
            source = kernel.source();saved = copy.deepcopy(p)
        finally:
            _release_record(dag, recorder)
        # The large recorder and DAG are no longer needed for independent replay.
        dag = recorder = returned_dag = returned_recorder = placement = None
        def replayed(active):
            nonlocal replay_active, replay_inner
            replay_active, replay_inner = active, active.inner
            kernel(active)
        try:
            replay = planner['verify'](replayed, kernel.geometry['slots'], saved, profile='qwen48')
        finally:
            _release_active(replay_active)
            _release_physical(replay_inner)
    planned = PlannedKernel(source, saved, copy.deepcopy(replay));_check_plan(kernel, planned)
    return planned, dict(planner='original record_dag/depth_dp/express/verify',
        frontend_normalize=True, profile='qwen48', elapsed_ns=time.perf_counter_ns()-started,
        fresh_model_loads=0, coefficient_fits=0, full_model_Count_runs=0)


def run_kernel(physical, kernel, planned, *, refresh_class=None):
    """Retain actual post-substitution raw outputs; caller owns their last use."""
    plan = _check_plan(kernel, planned)
    if physical.slots != kernel.geometry['slots'] or physical._inputs or physical._outputs is not None:
        raise ValueError('Standalone adapter requires a fresh physical backend')
    if refresh_class is None:
        from ..backends import RefreshBackend
        refresh_class = RefreshBackend
    active = refresh_class(physical, policy='plan', plan=plan, **LEVELS)
    declared = [];called = False;original_output = active.output
    def capture(values):
        nonlocal called
        if called:raise ValueError('Declare kernel outputs once')
        called = True;declared.extend(values);original_output(declared)
    active.output = capture
    try:
        from .qwen_component_cost import _dead_count_scope
        with _dead_count_scope(active):
            kernel(active)
        outputs = list(physical._outputs or [])
        if (not called or len(outputs) != len(kernel.output_roles()) or len(declared) != len(outputs)
                or len(physical._inputs) != kernel.input_count):
            raise ValueError('Complete source-bound input/output ABI required')
        levels = []
        for index, (before, raw) in enumerate(zip(declared, outputs)):
            node = active._plan.outputs.get(index)
            selected = before if node is None else active._planned_cache[node]
            if selected.backend is not active or selected.value is not raw or type(selected.level) is not int or not 1 <= selected.level <= 5:
                raise ValueError('Retain actual planned output substitution and its valid effective level')
            levels.append(selected.level)
        stats = active.stats()
        if stats['calls'] != plan['calls'] or stats['boots'] != plan['boots']:
            raise ValueError('Original Refresh call/boot counts differ from planned kernel')
        return outputs, dict(variant=VARIANT, source=kernel.source(), kind=kernel.kind,
            inputs=kernel.input_count, outputs=len(outputs), output_roles=kernel.output_roles(),
            symbolic_output_levels=levels, input_levels=[5]*kernel.input_count, refresh=stats,
            explicit_boundary_bootstraps_in_this_adapter=0,
            caller_required_boundary_positions=list(range(kernel.geometry['vocabulary_blocks'], kernel.geometry['vocabulary_blocks']+3)) if kernel.kind == 'first' else [],
            original_token0_input=kernel.geometry['vocabulary_blocks'] if kernel.kind == 'first' else None,
            native_scales=None, native_context=None, native_ABI_verified=False,
            full_model_verified=False, C2_source_gate='external original frozen_source',
            final_global_output='first19 plus terminal19; one future final decode',
            output_lifetime='Caller retains raw handles through their true last consumer; no wrapper/RValue escapes')
    finally:
        _release_active(active)


def _release_physical(physical):
    if physical is None:return
    for key in ('_inputs', '_constants', '_constant_objects'):
        value = getattr(physical, key, None)
        if value is not None:value.clear()
    physical._outputs = None


def count_kernel(kernel, planned, *, metadata_template):
    with _metadata(metadata_template):
        api = _api();physical = api['Count'](kernel.geometry['slots'], input_level=5, boot_out_level=5)
        try:
            _, record = run_kernel(physical, kernel, planned, refresh_class=api['Refresh'])
            stats = physical.stats()
            if stats['counts']['boot'] != planned.refresh_plan['boots']:
                raise ValueError('Independent physical Count and original plan boot inventory differ')
            return dict(interface=record, stats=stats)
        finally:_release_physical(physical)


def emit_kernel(kernel, planned, *, metadata_template, directory, frontend, max_constant_bytes, counted):
    if type(max_constant_bytes) is not int or max_constant_bytes <= 0:
        raise ValueError('Explicit positive finite constant-file allowance required')
    with _metadata(metadata_template):
        api = _api();physical = api['Emit'](kernel.geometry['slots'], directory=Path(directory),
            frontend=frontend, max_constant_bytes=max_constant_bytes)
        try:
            _, record = run_kernel(physical, kernel, planned, refresh_class=api['Refresh'])
            stats = physical.stats()
            if (counted['interface']['source'] != kernel.source()
                    or any(stats[k] != counted['stats'][k] for k in ('counts', 'ops_total', 'distinct_consts'))):
                raise ValueError('Same exact kernel Count/Emit schedule and constant inventory required')
            source = physical.write()
            return dict(interface=record, stats=stats, trace_source=str(source), native_compiled=False)
        finally:
            try:
                if not physical.finished:physical.close()
            finally:_release_physical(physical)
