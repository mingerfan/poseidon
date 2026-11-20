"""Independent report access after the requested S6/S6b cold sequence."""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import sys
import time

from .. import paths
from . import s6_serial as base
from . import s6_no_llm as driver
from . import s4_qwen_block as s4
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry


def report_metrics(value):
    worker = value.get('worker') or {}
    measured = worker.get('validation_metrics') or {}
    return dict(status=value.get('status'),
        samples=value.get('report_samples_completed', value.get('report_samples', worker.get('report_samples'))),
        original_top1_percent=value.get('reference_top1_percent', measured.get('exact_top1')),
        frozen_top1_percent=value.get('approximate_top1_percent', measured.get('top1')),
        accuracy_drop_pp=value.get('accuracy_drop_pp', measured.get('acc_drop_pp')),
        passed=value.get('report_policy_passed', worker.get('task_policy_passed')))


def same_program(a, b):
    try:
        return ast.dump(ast.parse(a), include_attributes=False) == ast.dump(ast.parse(b), include_attributes=False)
    except (SyntaxError, TypeError):
        return False


def heldout_metrics(case, model):
    session = case / ('session' if model == 'bert_tiny' else 'work/agent')
    frozen = session / 'program.ptdsl'
    if not frozen.is_file():
        return dict(available=False, reason='No frozen program')
    source = frozen.read_text()
    observed = session / 'observed-quality.jsonl'
    if observed.is_file():
        matched = []
        for line in observed.read_text().splitlines():
            row = json.loads(line)
            metrics = row['metrics']
            if (row['split'] in ('heldout', 'test[9000:10000]') and metrics.get('total') == 1000
                    and metrics.get('top1') is not None and same_program(source, row['program'])):
                matched.append(row)
        if matched:
            return dict(available=True, metric='classification_top1', split=matched[-1]['split'],
                **matched[-1]['metrics'], source=str(observed), measured_in_original_cold_loop=True,
                extra_model_evaluations=0)
    trace = session / 'agent_trace.jsonl'
    if not trace.is_file():
        return dict(available=False, reason='No retained internal quality observation')
    state_path = session / 'SESSION.json'
    selected = ((s4.read(state_path).get('confirmation') or {}).get('selected') or {}) if state_path.is_file() else {}
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        initial = (row.get('fb') or {}).get('validation') or {}
        if (selected.get('origin') == 'llm' and row.get('iteration') == selected.get('iteration')
                and same_program(source, selected.get('source')) and initial.get('total') == 1000
                and initial.get('top1') is not None):
            return dict(available=True, metric='classification_top1', split='heldout',
                **{k: initial.get(k) for k in ('top1', 'exact_top1', 'agree', 'total', 'acc_drop_pp')},
                source=str(trace), selected_origin='offline_rule_reply_in_legacy_llm_slot',
                source_binding='Original completed session selects this exact program and iteration; fb.validation is its original heldout evaluation',
                measured_in_original_cold_loop=True, extra_model_evaluations=0)
        tuned = row.get('tuned') or {}
        if not same_program(source, tuned.get('source')):
            continue
        full = [item for item in tuned.get('history', []) if item.get('phase') == 'result']
        if full and full[-1].get('agree') == full[-1].get('total') == 1000:
            exact = initial.get('exact_top1')
            if exact is not None and math.isfinite(exact):
                return dict(available=True, metric='classification_top1', split='heldout', total=1000,
                    agree=1000, top1=exact, exact_top1=exact, acc_drop_pp=0.,
                    source=str(trace), measured_in_original_cold_loop=True,
                    derivation='Retained final-program validation agrees with the exact reference on all1000 labels; accuracy equals the recorded exact accuracy',
                    extra_model_evaluations=0)
    return dict(available=False, reason='Original selected-program heldout accuracy was not retained; no extra validation is inferred')


def can_report(row):
    if row.get('status') != 'ok':
        return False
    case = Path(row['source']).parent
    result, completion = case / 'result.json', case / 'completion.json'
    if not result.is_file() or not completion.is_file():
        return False
    actual, closed = s4.read(result), s4.read(completion)
    return actual.get('status') == closed.get('status') == 'ok'


def publish(args, state):
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(), source_cold_controller=str(args.prior_case),
                 real_API_calls=0, feedback_to_Agent=False)
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 'S6_FINAL_REPORTS_PROGRESS.json', state)


def enrich(row):
    case = Path(row['source']).parent
    result_path = case / 'result.json'
    actual = s4.read(result_path) if result_path.is_file() else {}
    not_started = actual.get('metadata_only') and actual.get('scientific_attempts') == 0
    if not_started:
        row.update(status='not_run', status_detail=actual.get('reason'), metadata_only=True,
            scientific_attempts=0, cold=False, paper_eligible=False,
            full_three_hour_availability_test_run=False, actual_allocated_scientific_s=0,
            optional_admission_source=str(result_path))
    model = 'bert_tiny' if row['model'] == 'bert_tiny' else row['model']
    row['heldout_quality'] = heldout_metrics(case, model)
    row['runtime_plan_latency_pred_us'] = row['pred_latency_s']*1e6 if row.get('pred_latency_s') is not None else None
    row['runtime_plan_latency_pred_s'] = row.get('pred_latency_s')
    row['offline_replies'] = 0 if not_started else 1
    row['real_API_calls'] = 0
    row['report_feedback_to_Agent'] = False
    return row


def main(args):
    from .a5_a6_rule63_search import guarded_process
    from ..autonomous_budget import call_records
    args.out.mkdir(exist_ok=False)
    state = dict(status='waiting_for_all_cold_controls', reports=[], original_sources_preserved=True)
    publish(args, state)
    while not (args.prior_case / 'result.json').is_file():
        if time.time() >= s4.DEADLINE:
            state.update(status='cutoff_before_reports', reason='Cold controls retained priority until the hard submission cutoff')
            write_json(args.out / 'result.json', state)
            publish(args, state)
            return 1
        time.sleep(2)
    # The scheduler releases the heavy lease before publishing its final cold rows.
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        prior = s4.read(args.prior_case / 'result.json')
        if 'groups' not in prior:
            raise ValueError('Retain the original cold scheduler failure; reports require completed source groups')
        before = len(call_records(paths.RESULTS_DIR / 'llm_calls.jsonl'))
        for group, filename in [('S6', 's6_no_llm.json'), ('S6b', 's6b_no_llm_nofloor.json')]:
            data = s4.read(args.paper / filename)
            for row in data['rows']:
                model = next((k for k,v in base.previous.PAPER_MODELS.items() if v == row['model']), None)
                if model is None:
                    raise ValueError('Original model mapping required for independent report')
                row = enrich(row)
                old = data.get('reports', {}).get(model) or {}
                old_path = Path(old['source']) if old.get('source') else None
                if old_path is not None and old_path.is_file() and s4.read(old_path).get('status') == 'ok':
                    report = s4.read(old_path)
                    row['report'] = dict(report_metrics(report), source=str(old_path))
                    continue
                if old_path is not None and old_path.is_file():
                    failure = s4.read(old_path)
                    if failure.get('status') == 'error':
                        worker = failure.get('worker') or {}
                        if (worker.get('error') != 'ValueError: Fresh model policy, completed session or actual call accounting differs'
                                or worker.get('report_samples_read') != 0):
                            row['report'] = dict(report_metrics(failure), source=str(old_path),
                                retry_permitted=False, reason='Retain the actual once-only report result')
                            continue
                if not can_report(row):
                    continue
                if time.time() >= s4.DEADLINE-30:
                    state.update(status='cutoff_reached')
                    row['report'] = dict(status='not_run', reason='Hard publication cutoff before independent report dispatch')
                    continue
                case = Path(row['source']).parent
                target = case.with_name(case.name+'-final-isolated-report')
                if target.exists():
                    raise ValueError('Every final report is a fresh once-only attempt')
                rule = row.get('tune_rule', 'v8-6.23' if group == 'S6b' else 'v8-6.25')
                argv = [sys.executable, '-B', '-m', driver.MODULE,
                    '--bert-report' if model == 'bert_tiny' else '--cnn-report', '--final-report',
                    '--source-case', str(case), '--out', str(target), '--tune-rule', rule]
                if model == 'bert_tiny':
                    argv += ['--metadata-template', str(args.template), '--calibration-snapshot', str(args.calibration_snapshot)]
                else:
                    argv += ['--model', model, '--seconds', str(s4.remaining_before_deadline(1800)), '--memory-gib', '6']
                key = group.lower()+'-'+model
                state.update(status='running', active_group=group, active_model=model)
                publish(args, state)
                write_json(args.out / (key+'-command.json'), argv)
                observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (key+'.log'),
                    resources=args.out / (key+'-resources.jsonl'), timeout=s4.remaining_before_deadline(1900), mem_gib=24, fsize_gib=8)
                write_json(args.out / (key+'-process.json'), observed)
                if observed.get('remaining_pids'):
                    raise RuntimeError('Close every report child before another frozen source')
                report = s4.read(target / 'result.json')
                summary = dict(report_metrics(report), source=str(target / 'result.json'))
                row['report'] = summary
                data.setdefault('reports', {})[model] = summary
                state['reports'].append(dict(group=group, model=model, **summary))
                publish(args, state)
                if len(call_records(paths.RESULTS_DIR / 'llm_calls.jsonl')) != before:
                    raise ValueError('Independent reports must add zero API ledger rows')
            data.update(final_independent_reports_source=str(args.out / 'progress.json'))
            for root in (args.paper, args.mirror):
                write_json(root / filename, data)
        if state['status'] != 'cutoff_reached':
            state.update(status='complete', active_group=None, active_model=None)
    write_json(args.out / 'result.json', state)
    publish(args, state)
    record_registry(args.out, 'S6-S6B-INDEPENDENT-REPORTS', 'four-model-controls', 'once-after-all-cold-controls',
                    state['status'], dict(API_calls=0, reports=len(state['reports'])))
    return int(state['status'] != 'complete')


if __name__ == '__main__':
    p = base.cli()
    p.add_argument('--prior-case', type=Path, required=True)
    args = p.parse_args()
    try:
        result = main(args)
    except Exception as exc:
        if args.out.is_dir():
            write_json(args.out / 'result.json', dict(status='driver_error', error=type(exc).__name__+': '+str(exc)))
        raise
    raise SystemExit(result)
