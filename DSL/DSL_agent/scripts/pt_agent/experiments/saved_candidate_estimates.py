"""Project saved candidate metric scalars after a cold BERT producer closes.

SESSION JSON contains embedded prompt/program text. Only whitelisted metric
scalars are returned; request/response files, arrays and programs are not opened.
This module performs no Agent or numerical execution.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from . import bert_three_hour_terminal_metadata as terminal

METRICS = ('iteration', 'feasible', 'latency_pred_us', 'boots_pred', 'agree', 'max_u',
           'top1', 'exact_top1', 'acc_drop_pp', 'coarse_evals', 'full_evals',
           'seconds', 'sweeps', 'converged')
IDENTITIES = {
    'BERT-FRESH-COLD-2H': 'fresh-original-model-v11-wall-budget-c2-compile-no-report',
    'BERT-FRESH-COLD-3H': 'fresh-original-model-v11-three-hour-wall-budget-c2-compile-no-report'}


def metric_row(value, *, pointer, transaction, origin):
    metrics, unavailable = {}, []
    for key in METRICS:
        item = value.get(key)
        if item is None or type(item) in (int, bool):
            metrics[key] = item
        elif type(item) is float and math.isfinite(item):
            metrics[key] = item
        else:
            metrics[key] = None
            unavailable.append(key)
    stage = value.get('stage')
    return dict(source_json_pointer=pointer, transaction=transaction, origin=origin,
        evaluator_stage=stage if type(stage) is str else None, metrics=metrics,
        nonfinite_or_nonscalar_original_fields=unavailable,
        estimate_kind='Saved pre-compilation candidate estimate and internal validation observation',
        compiled_bootstraps=None, final_program_selection_inferred=False,
        pending_transaction_is_committed=False if transaction == 'pending_uncommitted' else None,
        evaluator_stage_is_cold_pipeline_status=False)


def project_session(state):
    rows = []
    history = state.get('history') or []
    if not isinstance(history, list):
        raise ValueError('Original session history must be a list')
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError('Original completed candidate metric row required')
        pointer = '/history/'+str(index)
        rows.append(metric_row(row, pointer=pointer, transaction='committed_history', origin='llm'))
        if isinstance(row.get('tuned'), dict):
            rows.append(metric_row(row['tuned'], pointer=pointer+'/tuned',
                                   transaction='committed_history', origin='tuned'))
    pending = state.get('pending') or {}
    if not isinstance(pending, dict):
        raise ValueError('Original pending transaction must be an object')
    if isinstance(pending.get('row'), dict):
        rows.append(metric_row(pending['row'], pointer='/pending/row',
                               transaction='pending_uncommitted', origin='llm'))
        if isinstance(pending['row'].get('tuned'), dict):
            rows.append(metric_row(pending['row']['tuned'], pointer='/pending/row/tuned',
                                   transaction='pending_uncommitted', origin='tuned'))
    return rows


def export_case(case, *, process_reader=terminal.closure._process):
    case = Path(case).resolve()
    if not (case/'completion.json').is_file():
        raise terminal.Pending('Original cold case is still running; no terminal candidate publication')
    registry = terminal.closure.registry_rows(case.parent/'registry.jsonl', {case.name: case})
    result, completion = terminal.closed(case, registry)
    if (result.get('round') not in IDENTITIES or result.get('model') != 'bert-tiny-sst2'
            or result.get('mode') != IDENTITIES[result['round']]):
        raise ValueError('An original cold BERT two/three-hour source is required')
    closure = terminal.births_closed(case, process_reader=process_reader)
    path = case/'session/SESSION.json'
    state = terminal.closure.read(path)
    rows = project_session(state)
    return dict(schema='closed-cold-BERT-saved-candidate-estimates-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), source_case=str(case),
        source_session=str(path), source_result=str(case/'result.json'),
        source_completion=str(case/'completion.json'), source_status=result['status'],
        source_complete_case_s=completion['complete_case_elapsed_seconds'],
        original_compiled_bootstraps=result.get('compiled_bootstraps'),
        original_compile_accepted=result.get('compile_accepted'), rows=rows,
        committed_history_rows=len(state.get('history') or []),
        pending_metric_rows=sum(r['transaction'] == 'pending_uncommitted' for r in rows),
        source_registry_fields=8, original_producer_closure=closure,
        candidate_validation_and_native_compilation_are_separate=True,
        candidate_counts_are_actual_CKKS_execution=False,
        standalone_Count_or_Mock_wall_times=None, standalone_Count_or_Mock_invocations=None,
        missing_timing_scope='Original session scalars supply no independent Count/Mock clocks or invocation counters',
        input_scope='One bounded original session JSON contains embedded prompt/program text; output projects numeric metrics and evaluator stage only. API request/response files and array/weight/program payload files are unopened.',
        original_scientific_records_modified=False, API=0, model_runs=0, Count_runs=0,
        compiler_runs=0, CKKS_runs=0, feedback_to_Agent=False)
