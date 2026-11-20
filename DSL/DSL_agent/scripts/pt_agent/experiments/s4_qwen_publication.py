"""Publish S4 provenance and close unsupported/source-unavailable cells once."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

from .. import paths
from .a_group_runs import append_record, empty_record
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry
from .s4_qwen_block import read, DEADLINE

Q0 = 's4-q0-qwen-layer0-rule63-depth-20261118'
Q1 = 's4-q1-qwen-original-layer0-cold3h-20261118'
Q2 = 's4-q2-qwen-layer0-rule63-optimized-20261118'


def terminal(args, identity, system, status, detail, task, budget):
    case = paths.RESULTS_DIR / identity
    if (case / 'paper-record.json').is_file():
        return read(case / 'paper-record.json')
    started = time.perf_counter_ns()
    case.mkdir(exist_ok=True)
    result = dict(round='S4-QWEN-ENTRY-ADMISSION', run_id=identity, model='Qwen2-0.5B-layer0',
        system=system, status=status, status_detail=detail, compiler_attempts=0,
        model_forwards=0, sample_reads=0, API_calls=0, CKKS_runs=0,
        scope='Existing frontend/source availability; no compiler outcome is inferred')
    if (case / 'result.json').is_file():
        if read(case / 'result.json') != result:
            raise ValueError('Preserve previously recorded entry admission evidence')
    else:
        write_json(case / 'result.json', result)
    record_registry(case, result['round'], result['model'], 'existing-entry-admission-only', status,
                    dict(compiler_attempts=0))
    completion = persist_case_completion(case, round_label=result['round'], started_ns=started,
        status=status, timing=dict(status='ok', attribution_ready=False, spans=[], diagnostics=[]),
        required_artifact_bytes=artifact_bytes(case))
    row = empty_record(run_id=identity, task=task, model='qwen2_0_5b_block', system=system,
        param_group='ours-N16-boot5', status=status, status_detail=detail, budget_s=budget,
        source=str(case / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
        api_calls=0, compiler_attempts=0, complete_admission_case_s=completion['complete_case_elapsed_seconds'],
        paper_eligible=time.time() <= DEADLINE, full_model=False, layer=0, sequence_capacity=32)
    write_json(case / 'paper-record.json', row)
    append_record(args.paper / 'runs.jsonl', row)
    return row


def scientific_records(args):
    rows = [json.loads(line) for line in (args.paper / 'runs.jsonl').read_text().splitlines() if line.strip()]
    current = {row['system']: row for row in rows if row.get('task', '').startswith('S4-')}
    if args.q1_case.name != Q1:
        eligible = [row for row in rows if row.get('system') == 'arranger' and
                    (row.get('scientific_run_id') or row['run_id']) == args.q1_case.name]
        if eligible:
            current['arranger'] = eligible[-1]
        else:
            current.pop('arranger', None)
    return rows, current


def append_metadata(args, row, original):
    """Keep initial observations and append a unique documented correction."""
    file = args.paper / 'runs.jsonl'
    with file.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        matched = [json.loads(line) for line in stream if line.strip() and
                   json.loads(line).get('run_id') == row['run_id']]
        if matched and matched != [row]:
            raise ValueError('Preserve prior S4 metadata-correction identity')
        if not matched:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    entry = dict(old_run_id=original['run_id'], new_run_id=row['run_id'], scientific_run_id=original['run_id'],
        reason=row['metadata_correction_reason'], metadata_only=True, new_scientific_runs=0,
        published_utc=row['finished_utc'])
    target = args.paper / 'supersessions.jsonl'
    with target.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        found = [json.loads(line) for line in stream if line.strip() and
                 json.loads(line).get('new_run_id') == row['run_id']]
        if found and found != [entry]:
            raise ValueError('Preserve the original supersession identity')
        if not found:
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def finalize(args):
    """Apply the latest task's explicit Q2 n/a and expose the original causes."""
    for identity in (Q0, Q1, Q2):
        original = read(paths.RESULTS_DIR / identity / 'paper-record.json')
        suffix = '-latest-task-metadata-20261118'
        row = dict(original, run_id=identity+suffix, scientific_run_id=identity,
                   metadata_only=True, new_scientific_runs=0,
                   original_paper_record=str(paths.RESULTS_DIR / identity / 'paper-record.json'))
        if identity == Q0:
            row.update(compiler_attempts=0, failure_phase='rule-program-generation',
                original_R0_recipe_feasible=False,
                metadata_correction_reason='Expose original R0 approximation-selection rejection before any compiler attempt')
        elif identity == Q1:
            case = paths.RESULTS_DIR / identity
            worker = read(case / 'phases/001-prepare/worker_result.json')
            row.update(status_detail=worker['error'], compiler_attempts=0, failure_phase='raw-calibration',
                agent_attempts=0, heldout_samples_selected=256, heldout_samples_evaluated=0,
                confirm_samples_evaluated=0, report_samples=0,
                actual_original_model_intake_completed=True,
                original_C2_executed=False, frozen=False, complete_10800_seconds_consumed=False,
                three_hour_availability_established=False,
                metadata_correction_reason='Expose the saved inner nonfinite-calibration cause; no Agent, compiler, confirmation or three-hour exhaustive attempt occurred')
        else:
            row.update(status='n/a', status_detail='no rule program to place (Q0)',
                failure_phase=None, upstream_source=Q0,
                metadata_correction_reason='Latest CODEX_TASKS S4 update explicitly makes Q2 not applicable when Q0 produced no rule program; original source-admission record preserved')
        target = args.out / (identity + '-current-paper-record.json')
        if target.is_file():
            row['finished_utc'] = read(target)['finished_utc']
        else:
            row['finished_utc'] = datetime.now(timezone.utc).isoformat()
        append_metadata(args, row, original)
        write_json(target, row)
    write_json(args.out / 'conditional-unrun.json', dict(
        independent_report=dict(status='not_run', reason='No frozen candidate or usable Q1 program'),
        Qwen_Fig2_component_count=dict(status='not_run', reason='Latest conditional count requires a Q1 frozen program; none exists'),
        new_API_calls=0, new_CKKS=0, compiler_attempts=0,
        Q0_and_Q1_retry_count=0, full_decoder_quality=None))
    refresh(args, dict(status='complete', q1_status='driver_error', q1_phase='raw-calibration',
                      q2_status='n/a', q2_compiler_attempts=0, heldout_samples=256,
                      conditional_unrun=str(args.out / 'conditional-unrun.json')), complete=True)
    return 0


def refresh(args, state, complete=False):
    now = datetime.now(timezone.utc).isoformat()
    ledger, records = scientific_records(args)
    state = dict(state, updated_utc=now, complete=complete, task_source=str(args.task),
        task_source_mtime_ns=args.task.stat().st_mtime_ns, rows=list(records.values()),
        source_q0=str(paths.RESULTS_DIR / Q0), source_q1=str(args.q1_case),
        publication_cutoff_utc='2026-11-18T12:00:00+00:00', report_feedback_permitted=False,
        parameter_group='ours-N16-boot5', full_Qwen_or_whole_CKKS_scheduled=False)
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 'S4_QWEN_PROGRESS.json', state)
        write_json(root / 'S4_QWEN_RESULTS.json', dict(status=state['status'], complete=complete,
            rows=list(records.values()), task_source=str(args.task), cutoff_utc=state['publication_cutoff_utc']))
        target = root / 'EXPERIMENT_DELIVERY.json'
        delivery = read(target)
        delivery.update(status='current_required_experiment_scope_complete' if complete else 'S4_Qwen_block_work_active',
            current_required_experiment_scope_complete=complete, task_source_mtime_ns=state['task_source_mtime_ns'],
            last_metadata_revision_utc=now, S4_Qwen_block=state,
            current_scientific_background_processes=[] if complete else
                [dict(case=str(args.q1_case), status=state.get('q1_status'), phase=state.get('q1_phase'))])
        delivery['links']['S4_Qwen_progress'] = str(args.paper / 'S4_QWEN_PROGRESS.json')
        delivery['links']['S4_Qwen_results'] = str(args.paper / 'S4_QWEN_RESULTS.json')
        selector = read(root / 'paper_cell_selector.json')
        if not any(row['model'] == 'qwen2_0_5b_block' and row['system'] == 'orion' for row in selector['cells']):
            selector['cells'].append(dict(model='qwen2_0_5b_block', system='orion', param_group='ours-N16-boot5'))
        for cell in selector['cells']:
            if cell['model'] != 'qwen2_0_5b_block' or cell['system'] == 'orbit':
                continue
            if 'pre_S4_policy' not in cell:
                cell['pre_S4_policy'] = {key: cell.get(key) for key in ('cell_state', 'reason', 'new_work_policy')}
            cell.update(new_work_policy='S4_original_first_block_only', current_quantitative_inclusion=True,
                        figures=cell.get('historical_figures') or ['Fig3', 'Fig5', 'TableIII'])
            row = records.get(cell['system'])
            if row:
                scientific_id = row.get('scientific_run_id') or row['run_id']
                cell.update(cell_state='terminal', compilation_status=row['status'],
                    selected_run_id=row['run_id'], scientific_run_id=scientific_id, source_result=row['source'],
                    source_completion=str(Path(row['source']).parent / 'completion.json'),
                    reason=row.get('status_detail'), compiler_attempts=row.get('compiler_attempts'),
                    clock_kind='cold' if row.get('cold') else 'source_admission_or_rule_generation',
                    output_kind='compiled RuntimePlan' if row['status'] == 'ok' else None,
                    current_selected_metadata_only=row.get('metadata_only', False),
                    time_outcome_scope='not_applicable' if row['status'] == 'n/a' else 'frontend_failure'
                        if row.get('compiler_attempts') == 0 and row['status'] == 'driver_error' else 'existing_entry_admission')
                for field in ('compile_s', 'generation_e2e_s', 'boots_compiled', 'exec_ops', 'pred_latency_s',
                              'peak_rss_gib', 'budget_s'):
                    cell[field] = row.get(field)
            elif cell['system'] == 'arranger':
                cell.update(cell_state='running', compilation_status=None, reason='One original-model cold chain is active',
                            budget_s=10800, expected_scientific_run_id=args.q1_case.name,
                            selected_run_id=None, scientific_run_id=None,
                            source_result=str(args.q1_case / 'result.json'),
                            source_completion=str(args.q1_case / 'completion.json'))
                for field in ('compile_s', 'generation_e2e_s', 'boots_compiled', 'exec_ops', 'pred_latency_s', 'peak_rss_gib'):
                    cell[field] = None
            elif cell['system'] == 'dacapo_search_ours':
                cell.update(cell_state='queued', reason='After Q1: check the same Q0 rule source',
                            expected_scientific_run_id=Q2, budget_s=7200)
        selector.update(generated_utc=now, S4_Qwen_progress=str(args.paper / 'S4_QWEN_PROGRESS.json'))
        counts = dict(Counter(row['cell_state'] for row in selector['cells']))
        outcomes = dict(Counter(row.get('compilation_status') for row in selector['cells']
                                if row['cell_state'] == 'terminal'))
        delivery.update(matrix_states=counts, measured_outcomes=outcomes)
        write_json(root / 'paper_cell_selector.json', selector)
        write_json(target, delivery)
        current = read(root / 'current_runs.json')
        original = {row['run_id']: row for row in current['runs']}
        old_ids = {row.get('scientific_run_id') for row in records.values() if row.get('metadata_only')}
        original = {key: row for key, row in original.items() if key not in old_ids}
        original.update({row['run_id']: row for row in records.values()})
        current.update(runs=list(original.values()), refreshed_utc=now,
                       total_ledger_records=len(ledger), current_records=len(original), S4_Qwen_progress=state)
        write_json(root / 'current_runs.json', current)
        evidence = read(root / 'current_experiment_evidence.json')
        evidence.update(status=delivery['status'], task_source_mtime_ns=state['task_source_mtime_ns'],
            S4_Qwen_block=state, cell_states=counts, compilation_outcomes=outcomes, last_metadata_revision_utc=now)
        write_json(root / 'current_experiment_evidence.json', evidence)
        progress = read(root / 'A_GROUP_PROGRESS.json')
        progress.update(status=delivery['status'], S4_Qwen_block=state, current_required_experiment_scope_complete=complete,
            current_task_source_mtime_ns=state['task_source_mtime_ns'], updated_utc=now,
            Qwen_block_compiler_grid='S4 original firstblock only; Q0 then fresh Q1 then same-source Q2')
        write_json(root / 'A_GROUP_PROGRESS.json', progress)
        policy = read(root / 'paper_comparison_policy.json')
        policy.update(updated_utc=now, Qwen_block='S4 original layer0/32token only; full decoder remains cancelled',
            S4_Qwen_block=state, current_required_experiment_scope_complete=complete)
        write_json(root / 'paper_comparison_policy.json', policy)
    mainline = args.mirror.parent / 'MAINLINE_PROGRESS.json'
    if mainline.is_file():
        value = read(mainline)
        value.update(status='current_required_experiment_scope_complete' if complete else 'S4_Qwen_block_work_active',
                     S4_Qwen_block=state, current_required_experiment_scope_complete=complete, updated_utc=now)
        write_json(mainline, value)
    return state


def main(args):
    if args.finalize:
        return finalize(args)
    args.out.mkdir(exist_ok=True)
    (args.out / 'previous-metadata').mkdir(exist_ok=True)
    for prefix, root in (('paper', args.paper), ('mirror', args.mirror)):
        for name in ('EXPERIMENT_DELIVERY.json', 'paper_cell_selector.json', 'current_runs.json',
                     'current_experiment_evidence.json', 'A_GROUP_PROGRESS.json', 'paper_comparison_policy.json'):
            target = args.out / 'previous-metadata' / (prefix + '-' + name)
            if not target.exists():
                shutil.copyfile(root / name, target)
    terminal(args, 's4-orion-qwen-existing-entry-20261118', 'orion', 'unsupported',
        'Existing orion.nn has no attention, softmax or RMSNorm path for the original Qwen block; no adapter is added',
        'S4-Orion-Qwen-entry', 0)
    terminal(args, 's4-heir-qwen-existing-entry-20261118', 'heir', 'unsupported',
        'Existing project HEIR exporter is CNN-only and has no complete original Qwen block path (RMSNorm/attention/softmax/rotary); official HEIR capabilities are not classified by this project-entry check',
        'S4-HEIR-Qwen-entry', 0)
    case = args.q1_case
    previous = None
    while time.time() <= DEADLINE:
        result = read(case / 'result.json')
        state = dict(status='running', q1_status=result['status'], q1_phase=result.get('active_phase'),
                     heldout_samples=256, cold_result=str(case / 'result.json'),
                     api_before=result.get('api_before'), api_after=result.get('api_after'),
                     authorized_Q1_additional_attempt=case.name != Q1,
                     retained_initial_Q1_failure=str(paths.RESULTS_DIR / Q1 / 'result.json'))
        key = (result['status'], result.get('active_phase'), len(result.get('processes', [])))
        if key != previous:
            refresh(args, state)
            previous = key
        if (case / 'completion.json').is_file() and (case / 'paper-record.json').is_file():
            for row in result.get('processes', []):
                observation = row.get('observation') or {}
                if observation.get('remaining_pids'):
                    raise RuntimeError('Preserve active descendants before continuing the S4 serial queue')
            # Q0 stopped before producing any program. This is a source check,
            # without an invented optimized-compiler attempt or timeout.
            q0 = read(paths.RESULTS_DIR / Q0 / 'result.json')
            if q0['status'] == 'ok':
                raise RuntimeError('Unexpected Q0 success: require same-source compiler entry, do not substitute a program')
            terminal(args, Q2, 'dacapo_search_ours', 'driver_error',
                'Q0 original R0 recipe rejected _softmax::inv before producing program/trace; the required identical Q0 input is unavailable, optimized compiler attempts=0',
                'S4-Q2', 7200)
            refresh(args, dict(state, status='complete', q1_status=result['status'],
                               q2_status='n/a', q2_compiler_attempts=0), complete=True)
            write_json(args.out / 'result.json', dict(status='ok', S4_required_cells_closed=True,
                authorized_Q1_additional_attempts=int(case.name != Q1), Q0_retries=0,
                initial_Q1_failure_preserved=True))
            return 0
        time.sleep(5)
    refresh(args, dict(status='deadline_reached', q1_status='deadline_observation'), complete=False)
    return 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--q1-case', type=Path, default=paths.RESULTS_DIR / Q1)
    for key in ('out', 'paper', 'mirror', 'task'):
        parser.add_argument('--' + key, type=Path, required=True)
    return parser


if __name__ == '__main__':
    raise SystemExit(main(cli().parse_args()))
