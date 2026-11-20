"""Cached packed prefill -> packed selector -> mixed embedding -> second forward.

All new packed operations use one backend.  The second forward consumes the
actual embedding expression, while its independent Tensor reference receives
the same mixed values.  Original HF/discrete outputs remain quality references.
"""

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from . import qwen_precision_quality as fixed
from . import qwen_selected_embedding as embedding
from . import qwen_monotone_selector as monotone
from . import qwen_own_output_packed as cached
from .r10_paper_metrics import StageRecorder, read_events, summarize_events


VARIANT = 'cached-packed-prefill-cubic52-compact-linked-step1-v1'
VOCABULARY = 151936
PHASE_LIMITS = dict(selector_embedding=dict(rss_gib=2, as_gib=6, seconds=900),
                    tensor=dict(rss_gib=6, as_gib=12, seconds=300),
                    packed_forward=dict(rss_gib=5, as_gib=12, seconds=5400))
TOTAL_SECONDS = sum(row['seconds'] for row in PHASE_LIMITS.values())
PATH_FIELDS = ('model_info', 'weights_archive', 'count_result', 'reference',
               'own_output_result', 'prefill_result', 'prefill_logits',
               'selector_result', 'coefficients')
COUNTERS = ('numpy_selection_attempts', 'numpy_selection_completed',
            'packed_selection_attempts', 'packed_selection_completed',
            'clear_embedding_attempts', 'clear_embedding_completed',
            'packed_embedding_attempts', 'packed_embedding_completed',
            'state_update_attempts', 'state_update_completed',
            'tensor_forward_attempts', 'tensor_forward_completed',
            'mock_forward_attempts', 'mock_forward_completed')
ZERO_FIELDS = ('fresh_hf_forwards', 'api_calls', 'selector_coefficient_fits', 'approximation_selection_runs', 'count_runs',
               'emit_runs', 'compiler_runs', 'keygen_runs', 'ckks_executions',
               'intermediate_decryptions', 'heldout_samples', 'confirm_samples', 'report_samples',
               'new_installations', 'new_downloads')


def initial_result():
    return dict(status='running', phase='linked-packed-generation', variant=VARIANT,
        model='Qwen2-0.5B', source_checked=False, candidate_frozen=False,
        same_backend_embedding_handle_consumed=False, linked_packed_math_verified=False,
        selector_quality_passed=None, training_quality_passed=None,
        one_hot_error_acceptance=None, full_ciphertext_model_binding_verified=False,
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None,
        resource_guards={}, coefficient_reconstruction_calls=None,
        deterministic_fit_cache_misses=None,
        coefficient_policy='Reuse the fixed selected program; deterministic coefficient reconstruction call count unmeasured',
        scope='One raw training prompt. Cached actual packed prefill followed by one new packed selector, compact embedding, independent mixed-input Tensor forward and second packed forward. Cold two-step and cryptographic clocks unmeasured.',
        **{key: 0 for key in COUNTERS + ZERO_FIELDS})


def source_gate(args, np):
    rows, discrete_reference, previous = cached.source_gate(args)
    selected = embedding.check_sources(args, np)
    if Path(selected['own_result']).resolve() != args.own_output_result.resolve():
        raise ValueError('The analytic selector and cached prefill must share the original own-output trajectory')
    coefficients, artifact, _ = monotone.load_artifact(args.coefficients, args.own_output_result, np)
    first_source = selected['selections'][0]
    if (first_source['sample_id'], first_source['step']) != ('train:0', 0):
        raise ValueError('Preserve the first original training prompt and its actual prefill')
    count, program_path, hf = fixed.source(args)
    if hf['trajectories'][0]['sample_id'] != 'train:0':
        raise ValueError('The independent original HF trajectory differs')
    hf_row = hf['trajectories'][0]['steps'][1]
    hf_reference = args.reference.parent / hf_row['logits_file']
    if not hf_reference.is_file():
        raise ValueError('The original complete second HF reference is absent')
    return rows, coefficients, dict(cached_sources=previous, selected_sources=selected,
        coefficients=str(args.coefficients.resolve()), coefficient_config=artifact['config'],
        fixed_program=str(program_path.resolve()), discrete_reference=str(discrete_reference.resolve()),
        original_hf_step1=str(hf_reference.resolve()), count_source_status=count['status'],
        actual_state_source='Original step0 token embeddings plus actual packed mixed embedding at its public next position; argmax is diagnostic only')


@contextmanager
def phase_guard(name, supervisor_pid, observations, *, apply_as=True):
    limits = PHASE_LIMITS[name]
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise ValueError('A dedicated phase resource timer is required')
    original_signal = signal.getsignal(signal.SIGALRM)
    previous_soft, previous_hard = resource.getrlimit(resource.RLIMIT_AS)
    maximum = limits['as_gib'] * 1024**3
    if apply_as:
        if previous_hard != resource.RLIM_INFINITY and previous_hard < maximum:
            raise ValueError('The outer address-space allowance is smaller than this explicit phase')
        resource.setrlimit(resource.RLIMIT_AS, (maximum, previous_hard))
    row = dict(limits, peak_observed_tree_rss_kib=0, samples=0, stopped=False,
               started_ns=time.perf_counter_ns())
    observations[name] = row
    def check(*_):
        rss = cached._resident_snapshot(supervisor_pid)
        row['samples'] += 1
        row['peak_observed_tree_rss_kib'] = max(row['peak_observed_tree_rss_kib'], rss)
        if rss > limits['rss_gib'] * 1024**2:
            row.update(stopped=True, reason='resident_memory')
            raise MemoryError('Linked diagnostic phase resident allowance exceeded: ' + name)
        if (time.perf_counter_ns()-row['started_ns'])/1e9 > limits['seconds']:
            row.update(stopped=True, reason='phase_time')
            raise TimeoutError('Linked diagnostic phase wall time exceeded: ' + name)
    signal.signal(signal.SIGALRM, check)
    try:
        check()
        signal.setitimer(signal.ITIMER_REAL, .1, .1)
        yield row
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, original_signal)
        if apply_as:
            resource.setrlimit(resource.RLIMIT_AS, (previous_soft, previous_hard))
        row['ended_ns'] = time.perf_counter_ns()
        row['seconds_observed'] = (row['ended_ns']-row['started_ns'])/1e9


@contextmanager
def observe_packed_stages(backend, vocabulary, stage_count, rows, np):
    """Observe actual approximate parents while delegating every arithmetic call."""
    from ..kernels import poly
    original = poly.cheb_eval
    capacity, width, block, stage, round_index = backend.slots//2, vocabulary, 0, 0, 0
    def observed(owner, value, coefficients):
        nonlocal width, block, stage, round_index
        if owner is not backend or width <= 1:
            raise ValueError('Unexpected selector observation ownership or stage')
        parents = (width+1)//2
        active = max(0, min(capacity, width//2-block*capacity))
        raw = value.value[..., :active]
        finite = bool(np.isfinite(value.value).all())
        row = dict(round=round_index, width=width, block=block, stage=stage, actual_pairs=active,
            finite=finite, input_min=float(raw.min()) if active else None,
            input_max=float(raw.max()) if active else None,
            inputs_outside_unit=int((np.abs(raw)>1.).sum()),
            comparisons_below_gap=int((np.abs(raw)<embedding.base.GAP).sum()) if stage == 0 else None)
        rows.append(row)
        output = original(owner, value, coefficients)
        measured = output.value[..., :active]
        bound = .5 if stage == stage_count-1 else 1.
        row.update(output_finite=bool(np.isfinite(output.value).all()),
            output_min=float(measured.min()) if active else None,
            output_max=float(measured.max()) if active else None,
            outputs_outside_interval=int((np.abs(measured)>bound).sum()))
        stage += 1
        if stage == stage_count:
            stage = 0
            block += 1
            if block == (parents+capacity-1)//capacity:
                width, block, round_index = parents, 0, round_index+1
        return output
    poly.cheb_eval = observed
    try:
        yield
        if width != 1 or stage or block:
            raise ValueError('The complete actual packed tournament was not observed')
    finally:
        poly.cheb_eval = original


def decode_selection(selection, np):
    from ..kernels.layout import layout_blocks
    result = np.empty(selection['metadata']['vocabulary'], dtype=np.float64)
    for value, (start, stop, _) in zip(selection['one_hot'], layout_blocks(selection['layout'])):
        if value.value.shape != (1, value.backend.slots):
            raise ValueError('One diagnostic batch must preserve its original packed layout')
        result[start:stop] = value.value[0, :stop-start]
    if not np.isfinite(result).all():
        raise ValueError('The actual packed selector output is nonfinite')
    return result


def mixed_inputs(initial, matrix, mixed, np):
    """Public-length append with actual mixed values; no token lookup for feedback."""
    ids = np.asarray(initial['prefix_token_ids'], dtype=np.int64)
    length = initial['prefix_length']
    capacity = len(ids)
    if (type(length) is not int or not 1 <= length <= capacity-2 or initial['step'] != 0
            or initial['attention_mask'] != [[1.]*length+[0.]*(capacity-length)]
            or initial['sel'] != [[0.]*(length-1)+[1.]+[0.]*(capacity-length)]
            or mixed.dtype != np.float64 or mixed.shape != (matrix.shape[1],)
            or not np.isfinite(mixed).all() or ids.min() < 0 or ids.max() >= matrix.shape[0]):
        raise ValueError('Preserve a valid original prefix and actual finite mixed embedding')
    inputs = np.asarray(matrix[ids])[None, :, :].copy()
    inputs[0, length] = mixed
    attention = np.zeros((1, capacity), dtype=np.float64)
    attention[0, :length+1] = 1.
    selection = np.zeros_like(attention)
    selection[0, length] = 1.
    public_state = copy.deepcopy(initial)
    public_state.update(step=1, prefix_length=length+1, attention_mask=attention.tolist(), sel=selection.tolist())
    public_state['prefix_token_ids'][length] = None
    public_state['mixed_embedding_position'] = length
    return dict(inputs_embeds=inputs, attention_mask=attention, sel=selection), public_state


def embedding_input_index(program, position):
    from ..lower import _pack_layout
    from ..kernels.layout import TokenLayout, layout_blocks
    offset, found = 0, None
    for param in program.params:
        packs = [op for op in program.ops if op.kind == 'pack' and op.inputs == (param.name,)]
        if len(packs) != 1:
            raise ValueError('Every original input requires exactly one pack operation')
        layout = _pack_layout(packs[0], param.shape, program.header.slots)
        count = len(layout_blocks(layout))
        if param.name == 'inputs_embeds':
            if not isinstance(layout, TokenLayout) or len(layout_blocks(layout.token)) != 1 or not 0 <= position < layout.L:
                raise ValueError('The actual generated embedding must occupy one original token block')
            found = offset+position
        offset += count
    if found is None:
        raise ValueError('The original inputs_embeds parameter is absent')
    return found


@contextmanager
def bind_embedding_handle(backend, packed_inputs, index, handle, record, np):
    """Bind one trusted existing expression directly, preserving object identity."""
    if (handle.backend is not backend or backend.batch != 1 or not 0 <= index < len(packed_inputs)
            or not np.array_equal(packed_inputs[index], handle.value)):
        raise ValueError('Actual packed embedding and independent mixed-input packing differ')
    sentinel = object()
    previous = backend.__dict__.get('input', sentinel)
    original = backend.input
    backend.bind(packed_inputs)
    backend._inputs.clear()
    record.update(index=index, calls=0, same_object=False, restored=False)
    def input_value(requested):
        if requested == index:
            if type(requested) is not int or requested != len(backend._inputs):
                raise ValueError('Preserve the original consecutive parameter ordering')
            backend._inputs.append(handle)
            record['calls'] += 1
            record['same_object'] = True
            return handle
        return original(requested)
    backend.input = input_value
    try:
        yield
        if record['calls'] != 1 or not record['same_object']:
            raise ValueError('The complete next forward must consume the actual embedding handle exactly once')
    finally:
        if previous is sentinel:
            del backend.input
        else:
            backend.input = previous
        record['restored'] = True


def stats_delta(before, after):
    keys = ('add', 'sub', 'neg', 'mul_cc', 'mul_cp', 'rot', 'boot')
    counts = {key: after['counts'][key]-before['counts'][key] for key in keys}
    return dict(counts=counts, ops_total=sum(counts.values()),
        distinct_constants_added=after['distinct_consts']-before['distinct_consts'],
        rotation_steps_total_after=after['rot_steps'])


def fit_cache_snapshot():
    """Observe the original deterministic coefficient caches, without refitting."""
    from ..kernels import approx
    return {name: getattr(approx, name).cache_info()._asdict()
        for name in ('_remez', '_relative_remez', '_composite')}


def measure_tensor(args, info, program, arrays, public_state, recorder, result, np):
    from ..transformer_tensor import run, polynomial
    from ..metrics import compare, passes
    points, operations, softmax, exact = [], [], [], {}
    expected = sum(1 if op.kind in ('act', 'rmsnorm', 'layernorm') else
        2+op.attrs.get('steps', 0) if op.kind == 'softmax' else 0 for op in program.ops)
    def point(op, role, value, attrs, mask):
        row = fixed.domain_record(value, attrs, np, mask)
        row.update(op_id=op.id, role=role, line=op.lineno)
        points.append(row)
        observed = polynomial(value, attrs, False)
        row['polynomial_output_finite'] = bool(np.isfinite(observed).all())
        if not row['polynomial_output_finite']:
            raise ValueError('The fixed polynomial produced nonfinite mixed-input values')
        if op.kind == 'softmax' and role == 'exp':
            data = np.asarray(value)*(2.**op.attrs.get('steps', 0))
            selected = np.ones_like(data, dtype=bool) if mask is None else mask
            if not selected.any(axis=-1).all():
                raise ValueError('Every original mixed-input attention row requires an effective key')
            data = np.where(selected, data, -np.inf)
            exponentials = np.exp(data-data.max(axis=-1, keepdims=True))
            exact[op.id] = exponentials/exponentials.sum(axis=-1, keepdims=True)
        return observed
    def step(op, value):
        data = np.asarray(value)
        finite = bool(np.isfinite(data).all())
        row = dict(op_id=op.id, kind=op.kind, shape=list(data.shape), finite=finite)
        operations.append(row)
        if not finite:
            raise ValueError('The original mixed-input program produced nonfinite values')
        if op.kind == 'softmax':
            reference = exact.pop(op.id)
            softmax.append(dict(op_id=op.id, max_absolute_difference=float(np.abs(data-reference).max()),
                maximum_sum_error=float(np.abs(data.sum(-1)-1.).max()), minimum_probability=float(data.min())))
    result.update(point_rows=points, op_rows=operations, softmax_rows=softmax)
    result['tensor_forward_attempts'] += 1
    with recorder.span('independent_actual_mixed_input_tensor'):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            value = run(program, info.graph.weights, arrays, point_evaluator=point, step_observer=step)
    if (value.dtype != np.float64 or value.shape != (1, 1, VOCABULARY) or not np.isfinite(value).all()
            or len(points) != expected or len(softmax) != 24 or exact):
        raise ValueError('The complete original mixed-input Tensor measurement is incomplete')
    result['tensor_forward_completed'] += 1
    np.save(args.work_dir/'mixed-step1-tensor-logits.npy', value, allow_pickle=False)
    result['all_domains_contained'] = all(row['contained'] for row in points)
    result['historical_quality'] = {}
    for name, source in (('hf', result['source']['original_hf_step1']),
                         ('discrete_tensor', result['source']['discrete_reference'])):
        reference = np.load(source, mmap_mode='r', allow_pickle=False)
        measured = compare(dict(values=reference, task=info.task,
            lengths=[public_state['prefix_length']], selected_last=True), value, None)
        result['historical_quality'][name] = dict(metrics=measured,
            policy=dict(info.threshold), policy_passed=passes(measured, info.threshold),
            max_absolute_difference=float(np.abs(reference-value).max()))
    result['training_quality_passed'] = (result['all_domains_contained']
        and result['historical_quality']['hf']['policy_passed'])
    return value


def run_linked(args, recorder, result, np):
    from .qwen_mock_progress import ObservedStreamingMockBackend, progress_records
    from ..kernels.encrypted_generation import select_maximum, numpy_selection
    from ..kernels.compact_linear import compact_transformer_projections
    from ..lower import lower, pack_inputs, decode_outputs
    backend, matrix = None, None
    checkpoint = lambda: embedding.base.save(args.work_dir/'worker-result.json', result)
    try:
        with phase_guard('selector_embedding', args.supervisor_pid, result['resource_guards']):
            with recorder.span('packed_selector_and_actual_mixed_embedding'):
                rows, coefficients, result['source'] = source_gate(args, np)
                result['source_checked'] = True
                logits = np.load(args.prefill_logits, mmap_mode='r', allow_pickle=False)
                if logits.dtype != np.float64 or logits.shape != (1, 1, VOCABULARY) or not np.isfinite(logits).all():
                    raise ValueError('The cached actual full-vocabulary packed prefill output differs')
                if int(logits.argmax()) != rows[0]['approximate_greedy_token']:
                    raise ValueError('The cached packed output greedy identity differs from its saved trajectory')
                result['actual_prefill_greedy_token'] = int(logits.argmax())
                result['numpy_selection_attempts'] += 1
                result['numpy_rounds'] = []
                with recorder.span('independent_numpy_selection'), monotone.observe_rounds(np, logits.size, result['numpy_rounds'], checkpoint):
                    reference_w, contract = numpy_selection(logits.reshape(-1), bound=embedding.base.BOUND,
                        normalized_gap=embedding.base.GAP, coefficients=coefficients)
                result['numpy_selection_completed'] += 1
                result['numpy_contract'] = contract
                backend = ObservedStreamingMockBackend(embedding.SLOTS, batch=1)
                inputs = embedding.packed_selection(backend, logits.reshape(-1), np)['one_hot']
                result['packed_stages'] = []
                result['packed_selection_attempts'] += 1
                with recorder.span('actual_packed_selector'), observe_packed_stages(backend, logits.size, len(coefficients), result['packed_stages'], np):
                    selection = select_maximum(backend, inputs, logits.size, bound=embedding.base.BOUND,
                        normalized_gap=embedding.base.GAP, coefficients=coefficients)
                result['packed_selection_completed'] += 1
                actual_w = decode_selection(selection, np)
                result['selector_math'] = fixed.strict_math(reference_w, actual_w, np)
                result['selector_quality'] = embedding.base.one_hot_quality(logits.reshape(-1), actual_w, np)
                result['selector_quality_passed'] = result['selector_quality']['token_agreement']
                result['selector_parent_domains_passed'] = (contract['comparison_gap_contract_passed']
                    and all(row['finite'] and row['output_finite'] and not row['inputs_outside_unit']
                        and not row['outputs_outside_interval'] and (row['comparisons_below_gap'] in (None, 0))
                        for row in result['packed_stages']))
                result['selector_stats'] = backend.stats()
                np.save(args.work_dir/'actual-packed-selection.npy', actual_w, allow_pickle=False)
                np.save(args.work_dir/'independent-numpy-selection.npy', reference_w, allow_pickle=False)
                checkpoint()
                if not result['selector_math']['passed']:
                    raise ValueError('Actual packed selector differs from the independent original NumPy arithmetic')
                matrix, mapping = embedding.map_stored_member(args.weights_archive, embedding.MEMBER)
                result['embedding_mapping'] = mapping
                if matrix.shape != embedding.SHAPE or matrix.flags.writeable:
                    raise ValueError('Keep the unchanged original embedding readonly')
                result['clear_embedding_attempts'] += 1
                with recorder.span('independent_actual_weight_mixture'):
                    mixture = embedding.clear_mixture(actual_w, matrix, np)
                result['clear_embedding_completed'] += 1
                result['packed_embedding_attempts'] += 1
                before = backend.stats()
                with recorder.span('actual_packed_compact_embedding'):
                    values, layout = embedding.compact_embedding(backend, selection, matrix)
                result['packed_embedding_completed'] += 1
                if len(values) != 1 or layout.size != embedding.SHAPE[1]:
                    raise ValueError('Preserve the original896-dimensional generated token layout')
                handle = values[0]
                mixed = handle.value[0, :layout.size].copy()
                result['embedding_math'] = fixed.strict_math(mixture, mixed, np)
                result['embedding_stats'] = stats_delta(before, backend.stats())
                result['hard_greedy_embedding_max_difference'] = float(np.abs(mixed-matrix[int(actual_w.argmax())]).max())
                np.save(args.work_dir/'actual-packed-mixed-embedding.npy', mixed, allow_pickle=False)
                np.save(args.work_dir/'independent-clear-mixed-embedding.npy', mixture, allow_pickle=False)
                if not result['embedding_math']['passed']:
                    raise ValueError('The actual packed mixed embedding failed original strict arithmetic')
                result['state_update_attempts'] += 1
                arrays, public_state = mixed_inputs(rows[0]['input_state'], matrix, mixed, np)
                result.update(actual_input_state=public_state, actual_input_shapes={key: list(value.shape) for key, value in arrays.items()})
                np.save(args.work_dir/'actual-step1-inputs-embeds.npy', arrays['inputs_embeds'], allow_pickle=False)
                result['state_update_completed'] += 1
                checkpoint()
                # Mixed arrays and expression values own their storage.  Close
                # this sole-member view before the full readonly mapping is
                # opened, avoiding a second simultaneous embedding mapping.
                matrix._mmap.close()
                matrix = None
                result['prestage_embedding_mapping_closed_before_full_model'] = True
        with phase_guard('tensor', args.supervisor_pid, result['resource_guards']):
            with recorder.span('load_original_fixed_model_readonly'):
                loaded = {}
                info, program, _ = fixed._load(args, loaded)
                if program.header.slots != backend.slots:
                    raise ValueError('The original full program and selector slot counts differ')
                result['model_source'] = loaded.pop('source')
                result.update(loaded)
            result['coefficient_cache_before_tensor'] = fit_cache_snapshot()
            reference = measure_tensor(args, info, program, arrays, public_state, recorder, result, np)
            result['coefficient_cache_after_tensor'] = fit_cache_snapshot()
            checkpoint()
        with phase_guard('packed_forward', args.supervisor_pid, result['resource_guards']):
            packed_inputs = pack_inputs(program, [arrays[param.name] for param in program.params])
            index = embedding_input_index(program, public_state['mixed_embedding_position'])
            result['handle_handoff'] = {}
            before = backend.stats()
            result['mock_forward_attempts'] += 1
            checkpoint()
            with recorder.span('actual_linked_second_packed_forward', sample_id='train:0', step=1):
                with bind_embedding_handle(backend, packed_inputs, index, handle, result['handle_handoff'], np), \
                        progress_records(backend, program, args.work_dir, result), compact_transformer_projections():
                    values, layouts = lower(program, info.graph.weights, backend)
                observed = decode_outputs([value.value for value in values], layouts)[0]
                result['full_forward_math'] = fixed.strict_math(reference, observed, np)
                result['full_forward_stats'] = stats_delta(before, backend.stats())
                result['combined_stats'] = backend.stats()
                result['coefficient_cache_after_mock'] = fit_cache_snapshot()
                result['deterministic_fit_cache_misses'] = {name: row['misses']-result['coefficient_cache_before_tensor'][name]['misses']
                    for name, row in result['coefficient_cache_after_mock'].items()}
                result['output_layout_blocks'] = [dict(start=start, end=end, valid_size=block.size)
                    for output in layouts for start, end, block in output.blocks]
                result['mock_forward_completed'] += 1
                result['same_backend_embedding_handle_consumed'] = result['handle_handoff']['same_object']
                np.save(args.work_dir/'linked-step1-mock-logits.npy', observed, allow_pickle=False)
            counted = json.loads(args.count_result.read_text())['worker']['count_stats']
            result['full_forward_operation_schedule_matched'] = all(result['full_forward_stats'][key] == counted[key] for key in ('counts', 'ops_total'))
            result['retained_constant_arrays'] = sum(value.data is not None for value in backend._constants.values())
            result['constant_identity_entries'] = len(backend._constant_objects)
            if (not result['full_forward_math']['passed'] or not result['full_forward_operation_schedule_matched']
                    or result['retained_constant_arrays'] or result['constant_identity_entries']):
                raise ValueError('The actual linked second forward failed original packed arithmetic or schedule')
        result.update(status='ok', linked_packed_math_verified=True,
            quality_status='training_diagnostic_passed' if result['training_quality_passed'] and result['selector_quality_passed']
                and result['selector_parent_domains_passed'] else 'quality_rejected')
    finally:
        if backend is not None:
            result['partial_combined_stats'] = backend.stats()
            backend.release_batch()
        if matrix is not None:
            matrix._mmap.close()


def worker(args):
    if (args.work_dir/'worker-result.json').exists():
        raise FileExistsError('Preserve the previous linked continuation result')
    np = embedding.base._numeric()
    result = initial_result()
    result['started_ns'] = time.perf_counter_ns()
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    try:
        with recorder.span('cached_prefill_linked_generation'):
            run_linked(args, recorder, result, np)
    except BaseException as exc:
        result.update(status='error', error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        result.update(ended_ns=time.perf_counter_ns(), worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            timing=summarize_events(read_events(args.work_dir/'events.jsonl')))
        result['worker_seconds'] = (result['ended_ns']-result['started_ns'])/1e9
        embedding.base.save(args.work_dir/'worker-result.json', result)
    return result


def command(args, supervisor_pid):
    from .. import paths
    result = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(TOTAL_SECONDS),
        '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-m', __spec__.name, '--phase', 'worker',
        '--out', str(args.out), '--supervisor-pid', str(supervisor_pid)]
    for field in PATH_FIELDS:
        result.extend(['--'+field.replace('_', '-'), str(getattr(args, field))])
    return result


def append_registry(out, observed, observation, registry, *, revision=None):
    """One outer-only eight-field record for this independent continuation."""
    out, registry = Path(out), Path(registry)
    if revision is None:
        from ..registry import git_rev
        revision = git_rev()
    metrics = {key: observed.get(key) for key in COUNTERS + ZERO_FIELDS}
    status = 'ok' if (observed.get('status') == 'ok' and observation.get('exit_code') == 0
        and observation.get('timed_out') is False and observation.get('remaining_pids') == []) else 'error'
    metrics.update(status=status, linked_packed_math_verified=observed.get('linked_packed_math_verified'),
        training_quality_passed=observed.get('training_quality_passed'),
        quality_status=observed.get('quality_status'),
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None,
        clock_scope='Cached first-step continuation; cold two-step clock unmeasured')
    row = dict(run_id=out.name, round='autonomous-qwen-linked-generation', model='Qwen2-0.5B',
        mode=VARIANT, git_rev=revision, metrics=metrics, artifacts_dir=str(out.resolve()),
        ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing = [json.loads(line) for line in stream if line.strip()]
        matches = [item for item in existing if item.get('run_id') == row['run_id'] and item.get('mode') == VARIANT]
        if matches:
            if len(matches) != 1 or matches[0]['metrics'] != metrics or matches[0]['artifacts_dir'] != row['artifacts_dir']:
                raise ValueError('A different continuation record already exists')
            return matches[0]
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()
    return row


def terminal_result(out, result):
    """Count the complete case, including this result's final JSON bytes."""
    target = out/'result.json'
    for _ in range(20):
        payload = json.dumps(embedding.base._safe(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        total = sum(path.stat().st_size for path in out.rglob('*') if path.is_file() and path != target)+len(payload.encode())
        if total > 4*1024**3 and result.get('status') != 'artifact_budget_exceeded':
            result.update(status='artifact_budget_exceeded', error=dict(type='ArtifactBudgetExceeded', message='Complete case exceeds4GiB'))
            continue
        if result.get('case_artifact_bytes') == total:
            target.write_text(payload)
            return total
        result['case_artifact_bytes'] = total
    raise RuntimeError('The complete linked case metadata size did not stabilize')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'worker'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    for field in PATH_FIELDS:
        parser.add_argument('--'+field.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.work_dir = args.out/'worker'
    if args.phase == 'worker':
        if args.supervisor_pid is None:
            parser.error('The explicit supervisor ancestor PID is required')
        return 0 if worker(args)['status'] == 'ok' else 1
    protected = [getattr(args, name).resolve() for name in PATH_FIELDS]
    if any(args.out == path or path in args.out.parents or args.out in path.parents for path in protected):
        raise ValueError('Preserve all cached original model and experiment directories')
    args.out.mkdir(parents=True, exist_ok=False)
    args.work_dir.mkdir()
    if args.phase == 'prepare':
        _, _, source = source_gate(args, embedding.base._numeric())
        embedding.base.save(args.out/'preparation.json', dict(status='prepared', variant=VARIANT, source=source,
            new_packed_selectors_started=0, new_embedding_runs=0, tensor_forwards=0, mock_forwards=0,
            phase_limits=PHASE_LIMITS, total_inner_seconds=TOTAL_SECONDS, outer_seconds=TOTAL_SECONDS+60,
            clock_scope='Cached first packed step; new linked continuation only', command=command(args, os.getpid())))
        return 0
    from .r10_paper_metrics import run_process
    from .. import paths
    if shutil.disk_usage(args.out).free < 10*1024**3:
        raise RuntimeError('Preserve10GiB free disk before this diagnostic')
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(('LD_', 'NIX_', '__NIX_')) and key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'IN_NIX_SHELL')}
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    source_root = Path(__file__).resolve().parents[1]
    for name in ('experiments/qwen_linked_generation.py', 'experiments/qwen_selected_embedding.py',
                 'experiments/qwen_monotone_selector.py', 'experiments/qwen_own_output_packed.py',
                 'experiments/qwen_precision_quality.py', 'experiments/qwen_mock_progress.py',
                 'kernels/encrypted_generation.py', 'kernels/compact_linear.py', 'backends/streaming_mock.py'):
        destination = args.out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root/name, destination)
    argv = command(args, os.getpid())
    embedding.base.save(args.out/'command.json', dict(argv=argv, phase_limits=PHASE_LIMITS))
    observation = run_process(argv, cwd=Path(__file__).resolve().parents[2], log=args.out/'worker.log',
        resources=args.out/'resources.jsonl', timeout=TOTAL_SECONDS+60, mem_gib=12, fsize_gib=1, env=environment)
    observed = json.loads((args.work_dir/'worker-result.json').read_text()) if (args.work_dir/'worker-result.json').is_file() else {}
    embedding.base.save(args.out/'supervision.json', dict(observation=observation, total_seconds=TOTAL_SECONDS+60))
    terminal_result(args.out, observed)
    append_registry(args.out, observed, observation, paths.RESULTS_DIR/'registry.jsonl')
    return 0 if (observed.get('status') == 'ok' and observation['exit_code'] == 0
        and not observation['timed_out'] and not observation['remaining_pids']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
