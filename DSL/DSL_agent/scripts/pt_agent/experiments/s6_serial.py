"""Finite serial S6 no-provider controls, ending at the submission cutoff."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

from .. import paths
from . import s5_serial as previous
from . import s5_tuner_off as off
from . import s6_no_llm as driver
from . import s4_qwen_block as s4
from .a_group_runs import empty_record, append_record
from .a_group_profile_sum_policy import profile_sum
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry

read = s4.read
MODELS = previous.MODEL_NAMES
MODULE = 'pt_agent.experiments.s6_serial'


def case_for(model):
    return paths.RESULTS_DIR / ('s6-'+previous.PAPER_MODELS[model]+'-fresh-rule63-no-llm-20261118')


def command(args, model, seconds):
    selected = previous.settings(args, model)
    case = case_for(model)
    argv = [sys.executable, '-B', '-m', driver.MODULE]
    if model == 'bert_tiny':
        src = selected['source']
        return argv + ['--bert', '--model', src['model'], '--inputs', src['inputs'], '--out', str(case),
            '--checkpoint-out', str(case.with_name(case.name+'-checkpoint')), '--metadata-template', str(args.template),
            '--paper-runs', str(case / 'driver-paper-record.jsonl'),
            '--calibration-snapshot', str(args.calibration_snapshot)]
    argv += ['--cnn', '--model', model, '--out', str(case), '--seconds', str(seconds),
             '--agent-seconds', str(seconds), '--calls', '1', '--tuner-seconds', str(driver.TUNER_SECONDS),
             '--inputs', selected['source']['records'], '--selections', selected['source']['selections'],
             '--metadata-template', selected['source']['compiler_metadata_recipe']]
    for name in ('memory_gib', 'case_gib', 'trace_seconds', 'compile_seconds'):
        argv += ['--'+name.replace('_', '-'), selected[name]]
    if selected['archive_emit']:
        argv.append('--archive-emit-constants-after-trace')
    if selected['diagnostic_elide']:
        argv.append('--diagnostic-elide-final-ir')
    return argv


def report_command(args, model):
    case = case_for(model)
    target = case.with_name(case.name+'-isolated-report')
    argv = [sys.executable, '-B', '-m', driver.MODULE]
    if model == 'bert_tiny':
        return argv + ['--bert-report', '--source-case', str(case), '--out', str(target),
            '--metadata-template', str(args.template), '--calibration-snapshot', str(args.calibration_snapshot)], target
    return argv + ['--cnn-report', '--model', model, '--source-case', str(case), '--out', str(target),
        '--seconds', str(s4.remaining_before_deadline(1800)), '--memory-gib', '6'], target


def recipes(text):
    import ast
    rows = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ('act', 'softmax', 'layernorm', 'rmsnorm'):
            continue
        attrs = {}
        for field in call.keywords:
            try:
                attrs[field.arg] = ast.literal_eval(field.value)
            except (ValueError, TypeError):
                attrs[field.arg] = dict(reference_expression=ast.unparse(field.value))
        rows.append(dict(output=ast.unparse(node.targets[0]), kind=call.func.id,
                         line=node.lineno, attrs=attrs))
    return rows


def details(case, model):
    session = case / ('session' if model == 'bert_tiny' else 'work/agent')
    seed, frozen = session / 'initial-rule63.ptdsl', session / 'program.ptdsl'
    result = dict(initial_program=str(seed) if seed.is_file() else None,
                  frozen_program=str(frozen) if frozen.is_file() else None,
                  before=recipes(seed.read_text()) if seed.is_file() else [],
                  after=recipes(frozen.read_text()) if frozen.is_file() else [],
                  initial_prediction=None, final_internal_prediction=None, heldout=None,
                  session_calls_are_local_replies=True)
    trace = session / 'agent_trace.jsonl'
    if trace.is_file():
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        if rows:
            result['initial_prediction'] = rows[0].get('fb', {}).get('cost')
    state = session / 'SESSION.json'
    if state.is_file():
        value = read(state)
        result['session_policy'] = {k: value.get(k) for k in ('max_iters', 'tuner', 'tuner_budget_s',
            'tune_policy', 'tune_rule', 'confirm_rule', 'focus_retune', 'numeric_lock', 'placement', 'split_version')}
        result['history'] = value.get('history')
        result['final_internal_prediction'] = value.get('best')
        candidates = value.get('records') or {}
        # All of these records are internal held-out observations; no report is returned to the Agent.
        result['internal_candidates'] = candidates
        for row in reversed(value.get('history') or []):
            if row.get('tuned'):
                result['heldout'] = row['tuned']
                break
        if result['heldout'] is None and value.get('history'):
            result['heldout'] = value['history'][-1]
    write_json(case / 's6-approximation-details.json', result)
    return result


def grouped_stages(stages):
    result = dict(tuning=0., simulation_and_counting=0., placement_and_compile=0., setup_and_checks=0.)
    for key, value in stages.items():
        group = ('tuning' if key == 'candidate_tuning' else 'simulation_and_counting'
            if key == 'candidate_parse_math_and_cost_tools' else 'placement_and_compile'
            if key in ('placement', 'emit_and_constant_archive', 'trace', 'native_compile_stage')
            else 'setup_and_checks')
        result[group] += value
    return result


def row_for(args, model, observed):
    case = case_for(model)
    result = read(case / 'result.json') if (case / 'result.json').is_file() else {}
    complete = read(case / 'completion.json') if (case / 'completion.json').is_file() else {}
    is_bert = model == 'bert_tiny'
    compiled = (result.get('phases') or {}).get('compile') or {}
    process = compiled.get('process') or {}
    worker_errors = {k: v.get('error') for k, v in result.get('phases', {}).items() if v and v.get('error')}
    termination = result.get('termination') or {}
    timed = (observed.get('timed_out') or time.time() >= s4.DEADLINE or
        termination.get('cause') == 'wall_budget' or any((p.get('observation') or {}).get('timed_out') or
        (p.get('observation') or {}).get('exit_code') == 124 for p in result.get('processes', [])))
    guard = args.out / (model+'-resources.physical-guard.json')
    status = ('ok' if result.get('status') == complete.get('status') == 'ok' and compiled else
        'resource_limit' if guard.is_file() or termination.get('cause') in ('physical_memory', 'resource_guard')
        or 'MemoryError' in str(worker_errors) else 'timeout' if timed else 'compiler_crash'
        if process.get('exit_code') not in (None, 0) else 'driver_error')
    quality = result.get('confirmation_quality') or {}
    audit = case / 'post-run-quality-audit.json'
    if audit.is_file():
        quality = read(audit).get('confirmation') or quality
    policy = profile_sum(result.get('prediction') if is_bert else compiled.get('prediction'))
    stages, stage_error = {}, None
    if complete:
        try:
            stages = previous.standard_stages(case, is_bert)
        except Exception as exc:
            stage_error = type(exc).__name__+': '+str(exc)
    peak = 0
    resource = args.out / (model+'-resources.jsonl')
    if resource.is_file():
        for line in resource.read_text().splitlines():
            item = json.loads(line)
            peak = max(peak, sum(p.get('vmrss_kib', 0)*1024 for p in item.get('processes', [])))
    detail = details(case, model) if case.is_dir() else {}
    baseline = detail.get('initial_prediction') or {}
    pre = detail.get('final_internal_prediction') or {}
    elapsed = complete.get('complete_case_elapsed_seconds')
    row = empty_record(run_id=case.name, task='S6-no-LLM', model=previous.PAPER_MODELS[model],
        system='arranger_no_llm', param_group='ours-N16-boot5', status=status,
        status_detail=str(worker_errors) if worker_errors else result.get('error'),
        budget_s=10800 if is_bert else 7200, compile_s=(process['ended_ns']-process['started_ns'])/1e9
        if process.get('ended_ns') else None, generation_e2e_s=elapsed if status == 'ok' else None,
        observed_complete_case_s=elapsed, peak_rss_gib=peak/s4.GIB if peak else None,
        boots_compiled=result.get('compiled_bootstraps') if is_bert else compiled.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'), pred_latency_s=policy['pred_latency_s'],
        pred_table=str(paths.PROFILES_DIR / 'qwen48/compiler-config.json'), prediction_policy=policy,
        api_calls=result.get('api_calls', result.get('api_delta', 0)), source=str(case / 'result.json'),
        finished_utc=datetime.now(timezone.utc).isoformat(), stages_s=stages,
        s6_stages_s=grouped_stages(stages), stage_attribution_error=stage_error,
        tuner=True, tuner_budget_s=driver.TUNER_SECONDS, max_rule_replies=1, cold=True,
        rule_initial_boots_pred=baseline.get('boots_pred'),
        rule_initial_latency_pred_us=baseline.get('latency_pred_us'),
        latency_pred_us=pre.get('latency_pred_us'),
        quality=dict(metric='classification_top1', split='confirmation1000', value=quality.get('top1'),
            original=quality.get('exact_top1'), passed=quality.get('passed'), total=quality.get('total')),
        approximation_details=str(case / 's6-approximation-details.json'),
        heldout=detail.get('heldout'), report=None, paper_eligible=time.time() <= s4.DEADLINE,
        source_control=previous.CONTROL_CASES[model],
        budget_comparison=dict(total_wall_same=True, tuner_budget_same=is_bert,
            original_tuner_s=2400 if is_bert else 600, S6_tuner_s=2400,
            note='S6 explicitly requests2400s; original paper CNN uses600s. This difference is retained.'))
    row['boots'] = row['boots_compiled']
    if result.get('status') == 'quality_rejected':
        row.update(outcome='quality_rejected', status_detail='Original confirmation accepted no candidate; compiler not attempted',
            status_classification_note='Five-class schema has no separate quality-rejection class; this is not a compiler result')
    write_json(case / 'paper-record.json', row)
    append_record(args.paper / 'runs.jsonl', row)
    return row


def publish(args, state):
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(), task_source=str(args.task),
        task_source_mtime_ns=args.task.stat().st_mtime_ns, cutoff_utc='2026-11-18T12:00:00+00:00',
        system='arranger_no_llm', param_group='ours-N16-boot5', heavy_cases_parallel=False,
        real_API_calls=0, confirm_report_compile_feedback_to_Agent=False,
        scope_note='One offline Rule63 reply in the original Agent loop, original tuner and program-plan placement',
        tuner_budget_comparison='S6 explicit2400s; paper CNN600s/BERT2400s; total2h/3h allocations preserved')
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 's6_no_llm.json', state)
        p = root / 'EXPERIMENT_DELIVERY.json'
        if p.is_file():
            value = read(p)
            value.update(status='S6_no_LLM_'+state['status'], current_required_experiment_scope_complete=False,
                updated_utc=state['updated_utc'], task_source_mtime_ns=state['task_source_mtime_ns'],
                S6_no_LLM=dict(status=state['status'], source=str(args.paper / 's6_no_llm.json')),
                Qwen_update4=dict(status='stopped_by_user', paper_eligible=False, source=str(args.stop_receipt)))
            value['links']['S6_no_LLM'] = str(args.paper / 's6_no_llm.json')
            write_json(p, value)


def main(args):
    from .a5_a6_rule63_search import guarded_process
    from ..autonomous_budget import call_records
    stop = read(args.stop_receipt)
    if stop.get('remaining_scientific_pids') or not stop.get('heavy_lock_available'):
        raise ValueError('Close every Qwen worker and release its lease before S6')
    args.out.mkdir(exist_ok=False)
    state = dict(status='starting', rows=[], reports={}, models=list(MODELS), active_model=None,
                 active_phase=None, completed_models=[], tests=str(args.tests))
    tests = read(args.tests)
    if tests.get('status') != 'ok':
        raise ValueError('S6 entry regression tests must pass before scientific runs')
    ledger = paths.RESULTS_DIR / 'llm_calls.jsonl'
    before = len(call_records(ledger))
    state['api_ledger_before'] = before
    publish(args, state)
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for model in MODELS:
            if time.time() >= s4.DEADLINE-180:
                state.update(status='cutoff_reached', unrun_models=[m for m in MODELS if m not in state['completed_models']])
                break
            budget = s4.remaining_before_deadline(10800 if model == 'bert_tiny' else 7200)
            state.update(status='running', active_model=model, active_phase='cold')
            publish(args, state)
            argv = command(args, model, budget)
            write_json(args.out / (model+'-command.json'), argv)
            observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (model+'-cold.log'),
                resources=args.out / (model+'-resources.jsonl'), timeout=budget, mem_gib=24, fsize_gib=16)
            write_json(args.out / (model+'-process.json'), observed)
            if observed.get('remaining_pids'):
                raise RuntimeError('Original scientific descendants must close before advancing S6')
            if len(call_records(ledger)) != before:
                raise ValueError('S6 must add zero provider ledger rows')
            row = row_for(args, model, observed)
            state['rows'].append(row)
            state['completed_models'].append(model)
            publish(args, state)
            if row['status'] == 'ok' and time.time() < s4.DEADLINE-120:
                state.update(active_phase='isolated_report')
                publish(args, state)
                argv, report_case = report_command(args, model)
                write_json(args.out / (model+'-report-command.json'), argv)
                observation = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                    log=args.out / (model+'-report.log'), resources=args.out / (model+'-report-resources.jsonl'),
                    timeout=s4.remaining_before_deadline(1900), mem_gib=24, fsize_gib=8)
                write_json(args.out / (model+'-report-process.json'), observation)
                report = read(report_case / 'result.json') if (report_case / 'result.json').is_file() else dict(status='error',error='Report did not close')
                state['reports'][model] = dict(status=report['status'], source=str(report_case / 'result.json'),
                    report_samples=report.get('report_samples') or report.get('report_samples_completed'),
                    task_policy_passed=report.get('report_policy_passed', report.get('task_policy_passed',
                        (report.get('worker') or {}).get('task_policy_passed'))))
                row['report'] = state['reports'][model]
                publish(args, state)
                if observation.get('remaining_pids'):
                    raise RuntimeError('Report descendants must close before the next S6 model')
            else:
                state['reports'][model] = dict(status='not_run', reason='No accepted compiled program or cutoff reached')
            if time.time() < s4.DEADLINE:
                previous.archive_closed_intermediates(args, case_for(model), model)
        else:
            state.update(status='complete', active_model=None, active_phase=None)
    state['api_ledger_after'] = len(call_records(ledger))
    if state['api_ledger_after'] != before:
        raise ValueError('No-LLM controls added real API records')
    write_json(args.out / 'result.json', state)
    publish(args, state)
    record_registry(args.out, 'S6-NO-LLM-SERIAL', 'four-model-control', 'rule63-local-reply-original-agent-loop',
                    state['status'], dict(models=len(state['rows']), API_calls=0))
    return 0 if state['status'] == 'complete' else 1


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'paper', 'mirror', 'task', 'tests', 'stop-receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--template', type=Path, default=paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    p.add_argument('--calibration-snapshot', type=Path, default=paths.RESULTS_DIR / 'fresh-vgg16-hybrid-main-cold-r30-20261116/source/pt_agent/analyzer.py')
    return p


if __name__ == '__main__':
    args = cli().parse_args()
    try:
        result = main(args)
    except Exception as exc:
        if args.out.is_dir() and (args.out / 'progress.json').is_file():
            value = read(args.out / 'progress.json')
            value.update(status='driver_error', error=type(exc).__name__+': '+str(exc))
            write_json(args.out / 'result.json', value)
            publish(args, value)
        raise
    raise SystemExit(result)
