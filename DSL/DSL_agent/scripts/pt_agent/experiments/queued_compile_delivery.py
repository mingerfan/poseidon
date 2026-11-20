"""Finite read-only metadata export for explicitly queued baseline/extension cases."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import time

from .experiment_delivery_refresh import atomic_write, DEADLINE
from .live_execution_observer import _Selector, Unavailable

MIB = 1024**2
BASE = 'BASELINE_EXTENSION_COMPILE_STATUS'
MANIFEST_SCHEMA = 'queued-compile-delivery-cases-v1'
READINESS_SCHEMA = 'queued-compile-delivery-readiness-v1'
LABEL = re.compile(r'[A-Za-z0-9_.:/+-]{1,160}\Z')
ERROR_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_.]{0,79}\Z')
ERROR_CLASS = object()
KINDS = {
    'orbit_frontend': 'Original Orbit frontend trace and TDAG parse; no model native compiler',
    'orbit_optimizer': 'Original Orbit optimizer and placement validation; no model native compiler',
    'orbit_interpreter': 'Official generic runtime interpreter build; model-specific compile and boot count unmeasured',
    'dacapo_compile': 'Saved original trace compilation continuation; earlier trace and failed attempts excluded',
    'qwen_recovery': 'Cached Agent session recovery; inherited API attempts and historical clocks remain separate',
    'qwen_compile': 'Frozen full24-layer single-prefill compilation; no own-output generation or cold Agent clock',
    'bert_continuation': 'Retained trained-classifier session continuation; prior session clocks excluded',
    'bert_compile': 'Frozen trained-classifier compilation; Agent, confirmation and report clocks excluded',
}
ACTIVE = {None, 'prepared', 'prepared_not_executed', 'running', 'unobserved', 'not_started', 'detached_started'}
PROCESS = {k: True for k in ('exit_code', 'timed_out', 'signal', 'started_ns', 'ended_ns', 'outer_peak_rss_kib')}
PROCESS.update(remaining_pids={'*': True}, signals_sent={'*': {'pid': True, 'signal': True}},
               peak_rss_kib_by_pid={'*': True}, limits={'memory_gib': True, 'file_gib': True, 'wall_seconds': True})
STATE = {k: True for k in ('phase', 'iteration', 'calls', 'max_iters', 'tuner')}
PARAMS = {k: True for k in (
    'poly_degree', 'poly_deg', 'max_slot', 'slots', 'logN', 'logSlots',
    'declared_polynomial_degree', 'original_poly_deg', 'original_max_slot',
    'Sf', 'Sw', 'Csw', 'CSw', 'max_level', 'bpsdepth', 'threads',
    'l_low', 'l_high', 'boot_l_low', 'boot_l_high', 'rescalingFactor',
    'polynomialDegree', 'levelLowerBound', 'levelUpperBound',
    'bootstrapLevelLowerBound', 'bootstrapLevelUpperBound')}
PARAMS.update(profile=True, profile_use=True, runtime=True)
CONTRACT = dict(context=PARAMS, levels={'lower_bound': True, 'upper_bound': True},
                boot_profiles={'*': {k: True for k in ('input_level_min', 'input_level_max', 'output_level', 'output_scale_log2')}})
LIMITS = {k: True for k in ('memory_gib', 'address_space_gib', 'trace_memory_gib', 'parse_memory_gib',
    'optimize_memory_gib', 'validate_memory_gib', 'step_AS_gib', 'subtree_RSS_gib', 'subtree_rss_gib',
    'subtree_plus_supervisor_RSS_gib', 'physical_subtree_RSS_gib', 'trace_virtual_AS_max_gib',
    'minimum_host_available_gib', 'minimum_available_gib', 'minimum_available_RAM_gib',
    'minimum_free_disk_gib', 'free_reserve_gib', 'minimum_free_bytes', 'file_gib', 'case_gib', 'case_bytes',
    'trace_seconds', 'parse_seconds', 'optimize_seconds', 'validate_seconds', 'compile_seconds',
    'build_seconds', 'complete_seconds', 'complete_batch_seconds', 'source_metadata_seconds',
    'overall_seconds', 'total_seconds', 'wrapper_seconds', 'concurrency', 'single_concurrency')}
GUARD = {k: True for k in ('stop', 'available_ram_bytes', 'available_bytes', 'disk_free_bytes', 'free_bytes',
    'artifact_bytes', 'case_bytes', 'subtree_rss_bytes', 'subtree_plus_supervisor_rss_bytes', 'elapsed_seconds')}
GUARD['latest_process_sample'] = {k: True for k in ('sample_ts_ns', 'subtree_rss_bytes', 'supervisor_rss_bytes', 'subtree_plus_supervisor_rss_bytes')}
SPAN = {k: True for k in ('event_id', 'stage_id', 'parent_id', 'inclusive_ns', 'exclusive_ns', 'status')}
COMPLETION = {k: True for k in ('round', 'status', 'started_ns', 'ended_ns', 'complete_case_elapsed_ns',
    'complete_case_elapsed_seconds', 'elapsed_scope', 'required_artifacts_persisted', 'registry_persisted', 'timing_status')}
COMPLETION['phase_spans'] = {'*': SPAN}
SESSION_OUTCOME = dict(STATE, status=True, done=True, frozen=True, inherited_attempts=True, new_attempts=True)
WORKER = {k: True for k in ('status', 'phase', 'elapsed_ns', 'worker_maxrss_kib', 'compile_attempts', 'compile_completed',
    'compiled_bootstraps', 'compiled_operation_count', 'compiled_outputs', 'runtime_plan', 'placement_bootstrap_nodes',
    'placement_bootstrap_weight_sum', 'candidate_frozen', 'api_calls', 'error_type')}
WORKER.update(error=ERROR_CLASS, process=PROCESS, params=PARAMS, profile_metadata=PARAMS, session=SESSION_OUTCOME)
PHASE = dict(WORKER, phase=True, process=PROCESS, observation=PROCESS, worker=WORKER)
PROCESS_ROW = dict(phase=True, stage=True, observation=PROCESS, source_state_before=STATE, source_state_after=STATE)
RESULT = {k: True for k in ('round', 'model', 'mode', 'status', 'failed_stage', 'failed_phase', 'error_type',
    'compiled_bootstraps', 'bootstrap_count', 'compiled_operation_count', 'runtime_plan', 'compile_attempts', 'compile_completed',
    'native_compile_attempts', 'native_compile_completed', 'runtime_interpreter_compiled', 'model_backend_compiled',
    'go_build_attempts', 'go_build_completed', 'native_optimizer_completed', 'placement_artifact_checked',
    'orbit_model_trace_passed', 'tdag_parse_passed', 'placement_bootstrap_nodes', 'placement_bootstrap_weight_sum',
    'candidate_frozen', 'frozen', 'api_calls', 'llm_calls', 'actual_api_calls', 'api_delta', 'driver_provider_calls',
    'incoming_signal', 'guard_stop_signal', 'timeout_signal', 'timeout_stop_signal', 'wall_timeout_observed',
    'cleanup_unknown', 'source_checked', 'timing_scope', 'scope', 'cold_end_to_end_run')}
RESULT.update(error=ERROR_CLASS, monitor_error=ERROR_CLASS, resource_monitor_error=ERROR_CLASS,
    accounting_error=ERROR_CLASS, process=PROCESS, compile_process=PROCESS, compile_worker=WORKER,
    processes={'*': PROCESS_ROW}, phase_results={'*': WORKER}, phases={'*': PHASE},
    attempted={'*': True}, completed={'*': True}, limits=LIMITS,
    session=SESSION_OUTCOME, final_session=STATE, selection_outcome={'frozen': True, 'outcome': True},
    source={'initial_session': STATE, 'optimizer_params': PARAMS, 'contract': CONTRACT, 'profile': True, 'compiler_profile': True},
    plan={'path': True}, guard_trigger=GUARD, resource_guard=GUARD, memory_guard_observation=GUARD,
    termination={'cause': True, 'signal': True, 'error': ERROR_CLASS},
    resource_adaptation={'phases': {'*': {'stop_observed': True}}},
    timing={'spans': {'*': SPAN}})
ENTRY = {k: True for k in ('id', 'case', 'kind', 'model', 'parameter_group')}
MANIFEST = dict(schema=True, results_root=True, cases={'*': ENTRY})


class MetadataSelector(_Selector):
    """Capture only an exception class prefix, then discard its message text."""
    def value(self, selected, depth=0):
        if selected is not ERROR_CLASS:
            return super().value(selected, depth)
        self.space()
        if self.peek() != '"':
            super().value(None, depth)
            return None
        self.take(); prefix = []; collecting = True; escaped = False
        while True:
            char = self.take()
            if escaped:
                escaped = False; collecting = False
            elif char == '\\':
                escaped = True; collecting = False
            elif char == '"':
                break
            elif ord(char) < 32:
                raise Unavailable('invalid_json')
            elif collecting:
                if char == ':': collecting = False
                elif len(prefix) < 80: prefix.append(char)
                else: collecting = False; prefix = []
        name = ''.join(prefix)
        return name if ERROR_NAME.fullmatch(name) and (name.endswith('Error') or name in ('SystemExit', 'KeyboardInterrupt')) else None


def read_projection(path, fields, maximum=32*MIB):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise Unavailable('missing_or_nonregular')
    if path.stat().st_size > maximum: raise Unavailable('size_limit')
    with path.open(encoding='utf-8') as stream:
        return MetadataSelector(stream, maximum).project(fields)


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalar_map(value):
    return {k: v for k, v in (value or {}).items() if isinstance(k, str) and LABEL.fullmatch(k)
            and (v is None or type(v) in (bool, int) or type(v) is float and math.isfinite(v)
                 or isinstance(v, str) and len(v) <= 512)} if isinstance(value, dict) else {}


def manifest_value(value, root):
    if value.get('schema') == READINESS_SCHEMA: value = value.get('manifest') or {}
    root = Path(root).resolve()
    if (value.get('schema') != MANIFEST_SCHEMA or set(value) != {'schema', 'results_root', 'cases'}
            or Path(value.get('results_root', '')).resolve() != root):
        raise ValueError('Explicit matching results root and queued case manifest required')
    rows = value['cases']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8: raise ValueError('Use one through eight explicit cases')
    seen_ids = set(); seen_paths = set(); out = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(ENTRY): raise ValueError('Only explicit case identity fields are accepted')
        if any(not isinstance(row[k], str) or not LABEL.fullmatch(row[k]) for k in ('id', 'kind', 'model', 'parameter_group')):
            raise ValueError('Short case labels required')
        if row['kind'] not in KINDS or not isinstance(row['case'], str): raise ValueError('Unknown queued case kind')
        path = Path(row['case']).resolve()
        if path == root or not path.is_relative_to(root) or path in seen_paths or row['id'] in seen_ids:
            raise ValueError('Distinct explicit cases below the matching result root required')
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in seen_paths):
            raise ValueError('Queued case identities must be disjoint')
        seen_ids.add(row['id']); seen_paths.add(path); out.append(dict(row, case=str(path)))
    return out


def clean(obs):
    return (isinstance(obs, dict) and type(obs.get('exit_code')) is int and obs['exit_code'] == 0
        and obs.get('timed_out') is False and obs.get('signal') is None
        and obs.get('remaining_pids') == [] and obs.get('signals_sent') == [])


def process_clock(obs):
    a, b = obs.get('started_ns'), obs.get('ended_ns')
    return b-a if integer(a) and integer(b) and b >= a else None


def stage_rows(marker, record):
    spans = (marker or {}).get('phase_spans')
    if not isinstance(spans, list): spans = (record.get('timing') or {}).get('spans') or []
    out = []
    for span in spans:
        if not isinstance(span, dict): continue
        row = {k: span.get(k) for k in ('event_id', 'stage_id', 'parent_id', 'status')}
        row.update({k: span[k] if integer(span.get(k)) else None for k in ('inclusive_ns', 'exclusive_ns')})
        out.append(row)
    return out


def observations(record):
    """Return saved wrappers and native child observations with distinct scopes."""
    out = []
    def add(label, obs, native=False):
        if isinstance(obs, dict): out.append(dict(stage=label, observation=obs, native=native))
    for key in ('process', 'compile_process'): add(key, record.get(key), key == 'process')
    for item in record.get('processes') or []:
        add(item.get('phase') or item.get('stage') or 'wrapper', item.get('observation'))
    for name, child in (record.get('phase_results') or {}).items():
        if isinstance(child, dict): add(name, child.get('process'), True)
    phases = record.get('phases') or {}
    if isinstance(phases, list):
        for row in phases:
            add(row.get('phase', 'wrapper'), row.get('process'))
            add(row.get('phase', 'native'), (row.get('worker') or {}).get('process'), True)
    elif isinstance(phases, dict):
        for name, child in phases.items():
            if isinstance(child, dict): add(name, child.get('process'), True)
    add('compile', (record.get('compile_worker') or {}).get('process'), True)
    return out


def child_record(record, phase):
    for key in ('phase_results', 'phases'):
        phases = record.get(key) or {}
        if isinstance(phases, dict) and isinstance(phases.get(phase), dict): return phases[phase]
    return {}


def phase_worker_observations(record):
    """Project saved Qwen phase workers without importing the original driver."""
    phases = record.get('phases') or []
    if not isinstance(phases, list): return [], {}, None
    workers = [p['worker'] for p in phases if isinstance(p, dict) and isinstance(p.get('worker'), dict)]
    state = next((w['session'] for w in reversed(workers) if isinstance(w.get('session'), dict)), {})
    frozen = next((w['candidate_frozen'] for w in reversed(workers) if type(w.get('candidate_frozen')) is bool), None)
    if frozen is None and type(state.get('frozen')) is bool: frozen = state['frozen']
    return [w['api_calls'] for w in workers if integer(w.get('api_calls'))], state, frozen


def failure(record, obs, status):
    cause = (record.get('termination') or {}).get('cause')
    guard = (record.get('guard_trigger') is not None or record.get('resource_guard') is not None
        or record.get('memory_guard_observation') is not None
        or any(p.get('stop_observed') is True for p in (record.get('resource_adaptation') or {}).get('phases', [])))
    if guard or cause in ('resource_guard', 'physical_or_storage_guard', 'driver_guard', 'own_budget_guard'):
        category = 'recorded_resource_or_budget_guard'
    elif (record.get('wall_timeout_observed') is True or record.get('timeout_signal') is not None
            or cause == 'wall_timeout' or any(o['observation'].get('timed_out') is True for o in obs)):
        category = 'recorded_timeout'
    elif record.get('incoming_signal') is not None or cause == 'incoming_signal': category = 'recorded_incoming_signal'
    elif record.get('monitor_error') or record.get('resource_monitor_error'): category = 'supervision_error'
    elif any(o['observation'].get('signal') is not None for o in obs): category = 'process_signal_origin_unresolved'
    elif status not in ACTIVE and status != 'ok': category = 'saved_execution_or_quality_outcome'
    else: category = None
    return dict(category=category, saved_termination_cause=cause, error_class=record.get('error_type') or record.get('error'),
        incoming_signal=record.get('incoming_signal') if record.get('incoming_signal') is not None else (record.get('termination') or {}).get('signal') if cause == 'incoming_signal' else None,
        guard_stop_signal=record.get('guard_stop_signal'), timeout_signal=record.get('timeout_signal'),
        timeout_stop_signal=record.get('timeout_stop_signal'), monitor_error_class=record.get('monitor_error'),
        resource_monitor_error_class=record.get('resource_monitor_error'),
        process_signals=[o['observation']['signal'] for o in obs if o['observation'].get('signal') is not None])


def parameters(record):
    out = []
    for label, value in [('source.optimizer_params', (record.get('source') or {}).get('optimizer_params'))]:
        if isinstance(value, dict): out.append(dict(source_field=label, values=scalar_map(value)))
    source = record.get('source') or {}
    for key in ('profile', 'compiler_profile'):
        if isinstance(source.get(key), str): out.append(dict(source_field='source.'+key, profile_path=source[key]))
    if isinstance(source.get('contract'), dict): out.append(dict(source_field='source.contract', values=source['contract']))
    for phase, worker in (record.get('phase_results') or {}).items():
        if isinstance(worker, dict) and isinstance(worker.get('params'), dict):
            out.append(dict(source_field='phase_results.'+phase+'.params', values=scalar_map(worker['params'])))
    return out


def observe(entry):
    case = Path(entry['case']); missing = []; record = {}; marker = None
    for filename, fields in (('result.json', RESULT), ('completion.json', COMPLETION)):
        try:
            value = read_projection(case/filename, fields)
            if filename == 'result.json': record = value
            else: marker = value
        except (OSError, ValueError, UnicodeError) as exc: missing.append(dict(file=filename, observation=type(exc).__name__))
    status = record.get('status', 'unobserved'); marker_status = (marker or {}).get('status')
    elapsed_ns = (marker or {}).get('complete_case_elapsed_ns'); start = (marker or {}).get('started_ns'); end = (marker or {}).get('ended_ns')
    closed = bool(marker and marker_status not in ACTIVE)
    closure = (closed and bool(record) and marker_status == status
        and marker.get('required_artifacts_persisted') is True and marker.get('registry_persisted') is True
        and integer(elapsed_ns) and integer(start) and integer(end) and end-start == elapsed_ns
        and number(marker.get('complete_case_elapsed_seconds')) and marker['complete_case_elapsed_seconds'] == elapsed_ns/1e9)
    obs = observations(record)
    # BERT saves failed native process observations before raising out of its worker.
    if entry['kind'] == 'bert_compile':
        for phase in ('trace', 'compile'):
            try:
                native = read_projection(case/'work'/(phase+'-process.json'), PROCESS, maximum=MIB)
                if not any(x['native'] and x['stage'] == phase for x in obs): obs.append(dict(stage=phase, observation=native, native=True))
            except (OSError, ValueError, UnicodeError): pass
    clean_children = bool(obs) and all(clean(x['observation']) for x in obs) and record.get('cleanup_unknown') is not True
    admitted = closure and status == 'ok' and clean_children and marker.get('timing_status') == 'ok'
    kind = entry['kind']; model_compile = None; interpreter = None
    compiler = child_record(record, 'compile')
    native_compile = [x for x in obs if x['native'] and x['stage'] == 'compile']
    count = record.get('compiled_bootstraps', record.get('bootstrap_count'))
    if kind in ('dacapo_compile', 'qwen_compile', 'bert_compile'):
        success = False
        if kind == 'dacapo_compile':
            worker = record.get('compile_worker') or {}
            success = (record.get('compile_attempts') == 1 and record.get('compile_completed') == 1
                and record.get('native_compile_attempts') == 1 and record.get('native_compile_completed') == 1
                and worker.get('status') == 'ok' and worker.get('compile_completed') == 1)
        else:
            required = ('placement', 'emit', 'trace', 'compile') if kind == 'qwen_compile' else ('emit', 'trace', 'compile')
            success = (all((record.get('attempted') or {}).get(p) == 1 and (record.get('completed') or {}).get(p) == 1 for p in required)
                and all(child_record(record, p).get('status') == 'ok' for p in required))
        model_compile = bool(admitted and success and native_compile and integer(count)) if closed else None
    elif kind == 'orbit_interpreter':
        interpreter = bool(admitted and record.get('runtime_interpreter_compiled') is True
            and record.get('go_build_attempts') == 1 and record.get('go_build_completed') == 1) if closed else None
    stage_success = admitted
    if kind in ('dacapo_compile', 'qwen_compile', 'bert_compile'): stage_success = model_compile is True
    elif kind == 'orbit_interpreter': stage_success = interpreter is True
    elif kind in ('orbit_frontend', 'orbit_optimizer'):
        phases = ('trace', 'parse') if kind == 'orbit_frontend' else ('optimize', 'validate')
        flags = ('orbit_model_trace_passed', 'tdag_parse_passed') if kind == 'orbit_frontend' else ('native_optimizer_completed', 'placement_artifact_checked')
        stage_success = bool(admitted and all(record.get(k) is True for k in flags)
            and all((record.get('attempted') or {}).get(p) == 1 and (record.get('completed') or {}).get(p) == 1
                    and child_record(record, p).get('status') == 'ok' for p in phases))
    native_rows = [dict(stage=x['stage'], elapsed_ns=process_clock(x['observation']), clean=clean(x['observation']),
        exit_code=x['observation'].get('exit_code'), timed_out=x['observation'].get('timed_out'),
        signal=x['observation'].get('signal'), remaining_PID_count=len(x['observation']['remaining_pids']) if isinstance(x['observation'].get('remaining_pids'), list) else None)
        for x in obs if x['native']]
    compiled_ns = next((x['elapsed_ns'] for x in native_rows if x['stage'] == 'compile'), None)
    worker_calls, worker_state, worker_frozen = phase_worker_observations(record)
    first = (record.get('source') or {}).get('initial_session') or {}
    final = record.get('final_session') or record.get('session') or worker_state
    if kind == 'bert_continuation' and not final:
        for p in record.get('processes') or []: final = p.get('source_state_after') or p.get('source_state_before') or final
    calls = final.get('calls'); before = first.get('calls', final.get('inherited_attempts'))
    api = next((record[k] for k in ('api_calls', 'api_delta', 'actual_api_calls', 'llm_calls') if integer(record.get(k))), None)
    if status in ACTIVE and not obs and not any(integer(v) and v > 0 for v in (record.get('attempted') or {}).values()): api = None
    delta = calls-before if integer(calls) and integer(before) and calls >= before else None
    outer_frozen = record.get('candidate_frozen', record.get('frozen'))
    frozen = outer_frozen
    if frozen is None: frozen = (record.get('selection_outcome') or {}).get('frozen', worker_frozen)
    placement = record.get('placement_bootstrap_nodes')
    if placement is None: placement = (child_record(record, 'validate')).get('placement_bootstrap_nodes')
    hwm = [v for x in obs for v in (x['observation'].get('peak_rss_kib_by_pid') or {}).values() if number(v)]
    return dict(entry, result_path=str(case/'result.json'), completion_path=str(case/'completion.json'),
        result_status=status, completion_status=marker_status, closed=closed, closure_verified=bool(closure),
        clean_children=clean_children if obs else None, metadata_missing_or_invalid=missing,
        admitted_closed_stage_success=bool(stage_success), model_native_compile_success=model_compile,
        runtime_interpreter_build_success=interpreter, compiled_bootstraps=count if model_compile is True else None,
        reported_compile_count=count if kind in ('dacapo_compile', 'qwen_compile', 'bert_compile') and closed and status not in ACTIVE and integer(count) else None,
        compiled_operation_count=record.get('compiled_operation_count') if model_compile is True else None,
        optimizer_placement_bootstrap_nodes=placement if kind == 'orbit_optimizer' and integer(placement) else None,
        optimizer_placement_bootstrap_weight_sum=record.get('placement_bootstrap_weight_sum') if kind == 'orbit_optimizer' else None,
        optimizer_placement_scope='Saved TDAG placement inventory; native compiled count remains unmeasured' if kind == 'orbit_optimizer' else None,
        candidate_frozen=frozen if type(frozen) is bool else None,
        outer_candidate_frozen=outer_frozen if type(outer_frozen) is bool else None,
        last_saved_worker_candidate_frozen=worker_frozen,
        frozen_observation_disagreement=(outer_frozen != worker_frozen) if type(outer_frozen) is bool and type(worker_frozen) is bool else None,
        API_new_attempts=api, API_new_attempts_scope='Saved outer API/ledger counter; no HTTP success inferred' if api is not None else None,
        API_worker_reported_attempts=sum(worker_calls) if worker_calls else None,
        API_worker_counter_observations=len(worker_calls),
        API_worker_counter_scope='Sum of saved per-phase worker counters; may be partial, distinct from the outer ledger interval and HTTP outcomes.',
        session_attempts_before=before if integer(before) else None, session_attempts_after=calls if integer(calls) else None,
        session_inherited_attempts=final.get('inherited_attempts') if integer(final.get('inherited_attempts')) else None,
        session_reported_new_attempts=final.get('new_attempts') if integer(final.get('new_attempts')) else None,
        session_attempt_delta=delta, session_attempt_scope='Saved session attempt counters; distinct from transport success and inherited attempts',
        elapsed_ns=elapsed_ns if closure else None, complete_case_seconds=marker.get('complete_case_elapsed_seconds') if closure else None,
        elapsed_scope=KINDS[kind], saved_elapsed_scope=record.get('timing_scope') or record.get('scope') or (marker or {}).get('elapsed_scope'),
        native_compile_elapsed_ns=compiled_ns if kind in ('dacapo_compile', 'qwen_compile', 'bert_compile') else None,
        native_compile_attempt_ns=compiled_ns if kind in ('dacapo_compile', 'qwen_compile', 'bert_compile') else None,
        native_compile_attempt_seconds=compiled_ns/1e9 if compiled_ns is not None and kind in ('dacapo_compile', 'qwen_compile', 'bert_compile') else None,
        native_compile_success_ns=compiled_ns if model_compile is True else None,
        native_compile_success_seconds=compiled_ns/1e9 if compiled_ns is not None and model_compile is True else None,
        generic_interpreter_build_elapsed_ns=process_clock(record.get('process') or {}) if kind == 'orbit_interpreter' else None,
        native_process_clocks=native_rows, stage_clocks=stage_rows(marker, record), stage_clock_rule='Nested inclusive spans overlap; no parent or child clocks are summed.',
        attempted=scalar_map(record.get('attempted')), completed=scalar_map(record.get('completed')),
        failure=failure(record, obs, status), failed_stage=record.get('failed_stage') or record.get('failed_phase'),
        resources=dict(limits=scalar_map(record.get('limits')), single_PID_HWM_KiB=max(hwm) if hwm else None,
            guard_trigger=record.get('guard_trigger'), resource_guard=record.get('resource_guard'), memory_guard_observation=record.get('memory_guard_observation')),
        native_parameters=parameters(record), native_parameter_scope='Saved compiler/profile scalar metadata; no parameter objects, primes or cryptographic construction inspected.',
        count_scope=KINDS[kind], real_CKKS_scheduled_by_exporter=False,
        prediction_transferred_from_other_plan=False)


def snapshot(entries, now):
    rows = []
    for entry in entries:
        try: rows.append(observe(entry))
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            rows.append(dict(entry, result_status='unobserved', closed=False, closure_verified=False,
                model_native_compile_success=None, runtime_interpreter_build_success=None,
                compiled_bootstraps=None, metadata_error_class=type(exc).__name__))
    return dict(schema='queued-baseline-extension-compile-status-v1', observed_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        scope='Explicit saved metadata only. Generic interpreter, optimizer placement and model native compilation remain separate. No model/API/crypto/report access.',
        no_stage_clock_summation=True, cases=rows)


def publish(directory, value):
    directory = Path(directory)
    fields = ('observed_utc', 'id', 'case', 'kind', 'model', 'parameter_group', 'result_status', 'completion_status',
        'closed', 'closure_verified', 'clean_children', 'admitted_closed_stage_success', 'model_native_compile_success',
        'runtime_interpreter_build_success', 'compiled_bootstraps', 'reported_compile_count', 'compiled_operation_count',
        'optimizer_placement_bootstrap_nodes', 'optimizer_placement_bootstrap_weight_sum', 'candidate_frozen',
        'outer_candidate_frozen', 'last_saved_worker_candidate_frozen', 'frozen_observation_disagreement',
        'API_new_attempts', 'API_new_attempts_scope', 'API_worker_reported_attempts', 'API_worker_counter_observations',
        'session_attempts_before', 'session_attempts_after', 'session_attempt_delta', 'session_inherited_attempts', 'session_reported_new_attempts',
        'elapsed_ns', 'complete_case_seconds', 'elapsed_scope', 'saved_elapsed_scope', 'native_compile_elapsed_ns',
        'native_compile_attempt_ns', 'native_compile_attempt_seconds', 'native_compile_success_ns', 'native_compile_success_seconds',
        'generic_interpreter_build_elapsed_ns', 'failed_stage', 'failure_class', 'error_class', 'incoming_signal', 'guard_stop_signal',
        'timeout_signal', 'timeout_stop_signal', 'native_parameters_json', 'resources_json', 'native_process_clocks_json',
        'stage_clocks_json', 'result_path', 'completion_path')
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for case in value['cases']:
        row = {k: case.get(k) for k in fields}; row['observed_utc'] = value['observed_utc']
        failure_record = case.get('failure') or {}; row['failure_class'] = failure_record.get('category')
        row.update({k: failure_record.get(k) for k in ('error_class', 'incoming_signal', 'guard_stop_signal', 'timeout_signal', 'timeout_stop_signal')})
        for field in ('native_parameters', 'resources', 'native_process_clocks', 'stage_clocks'):
            row[field+'_json'] = json.dumps(case.get(field), separators=(',', ':'), allow_nan=False)
        writer.writerow(row)
    atomic_write(directory/(BASE+'.csv'), stream.getvalue())
    atomic_write(directory/(BASE+'.json'), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def execute(args, *, now=time.time, sleep=time.sleep):
    if type(args.poll_seconds) is not int or args.poll_seconds != 60: raise ValueError('Use exactly60second polling')
    cutoff = datetime.fromisoformat(args.until.replace('Z', '+00:00'))
    maximum = datetime.fromisoformat(DEADLINE.replace('Z', '+00:00')); start = now()
    if cutoff.tzinfo is None or cutoff > maximum or cutoff.timestamp() > start+7*86400:
        raise ValueError('Finite timezone-aware cutoff no later than2026-11-17T10:00Z required')
    root, paper, out = (Path(getattr(args, k)).resolve() for k in ('results_root', 'paper_dir', 'publish_dir'))
    if out != paper and not out.is_relative_to(paper): raise ValueError('Publication must stay in the explicit PAPER directory')
    if root == paper or root.is_relative_to(paper) or paper.is_relative_to(root): raise ValueError('PAPER and raw results must be disjoint')
    fields = dict(MANIFEST, manifest=MANIFEST)
    entries = manifest_value(read_projection(args.manifest, fields, maximum=MIB), root)
    if now() >= cutoff.timestamp(): return 0
    out.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out/'.queued-compile-delivery.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB); polls = 0
        while now() < cutoff.timestamp():
            publish(out, snapshot(entries, now())); polls += 1
            if args.once: break
            sleep(min(60, max(0, cutoff.timestamp()-now())))
        return polls
    finally: os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'results-root', 'paper-dir', 'publish-dir'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60); parser.add_argument('--until', default=DEADLINE)
    parser.add_argument('--once', action='store_true'); args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS, (256*MIB, 256*MIB)); execute(args); return 0


if __name__ == '__main__': raise SystemExit(main())
