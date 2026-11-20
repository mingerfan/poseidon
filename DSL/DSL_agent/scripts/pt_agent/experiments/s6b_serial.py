"""Insert requested no-floor CNN controls before the untouched BERT entry."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from .. import paths
from . import s6_serial as base
from . import s6_no_llm as driver
from . import s5_tuner_off as off
from . import s4_qwen_block as s4
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry
from .a_group_runs import empty_record, append_record

MODULE = 'pt_agent.experiments.s6b_serial'
RULE = 'v8-6.23'
ORDER = [('S6b', m) for m in base.MODELS[:3]] + [('S6', 'bert_tiny'), ('S6b', 'bert_tiny')]


def case_for(group, model):
    original = base.case_for(model)
    if group == 'S6' and model == 'bert_tiny':
        return original.with_name(original.name+'-ordered-after-s6b')
    return original.with_name(original.name.replace('s6-', 's6b-', 1)) if group == 'S6b' else original


def rejection_counts(case, model):
    session = case / ('session' if model == 'bert_tiny' else 'work/agent')
    path = session / 'agent_trace.jsonl'
    counts = Counter()
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for event in (row.get('tuned') or {}).get('history', []):
                if event.get('rejected'):
                    counts[str(event['rejected'])] += 1
                if event.get('phase') == 'skip_cost' or event.get('skip_cost') or event.get('rejected') == 'cost':
                    counts['skip_cost'] += 1
                if event.get('phase') == 'coarse' and event.get('accepted') is False and not event.get('rejected'):
                    counts['coarse_failed'] += 1
                if 'fallback' in event:
                    counts['prefix_fallback_attempts'] += len(event['fallback'])
                    counts['prefix_fallback_accepted'] += len(event.get('accepted_lines', []))
    for key in ('skip_cost', 'coarse_failed', 'prefix_fallback_attempts', 'prefix_fallback_accepted'):
        counts.setdefault(key, 0)
    return dict(counts)


def namespace(group):
    def tagged_write(path, value):
        if Path(path).name == 'paper-record.json':
            value.update(task=group+'-no-LLM', system='arranger_no_llm_nofloor' if group == 'S6b' else 'arranger_no_llm',
                tune_rule=RULE if group == 'S6b' else 'v8-6.25',
                baseline_actual_tune_rule='v8-6.25', requested_baseline_label_in_task='v8-6.24',
                rule_scope='Existing rule selection only; all algorithm implementations remain unchanged')
            value['rejection_counts'] = rejection_counts(Path(path).parent,
                'bert_tiny' if value['model'] == 'bert_tiny' else next(k for k,v in base.previous.PAPER_MODELS.items() if v == value['model']))
            value['rejection_count_scope'] = 'Retained explicit events. The existing degree-sweep silent cost-continue has no event; total skip_cost is unmeasured.'
            value['total_skip_cost_count'] = None
        write_json(path, value)
    return off.clone_module(base, dict(case_for=lambda model: case_for(group, model), write_json=tagged_write))


def command(args, group, model, seconds):
    argv = namespace(group)['command'](args, model, seconds)
    return argv + ['--tune-rule', RULE if group == 'S6b' else 'v8-6.25', '--s6-continue']


def process_identity(module, out):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
            stat = (path / 'stat').read_text().split()
            if module in argv and str(out) in argv and stat[2] != 'Z':
                found.append(dict(pid=int(path.name), birth=stat[21]))
        except (OSError, ValueError, IndexError):
            continue
    return found


def terminate_identity(identity):
    path = Path('/proc') / str(identity['pid'])
    try:
        if (path / 'stat').read_text().split()[21] != identity['birth']:
            raise ValueError('The original scheduler process identity changed')
        os.kill(identity['pid'], signal.SIGTERM)
    except FileNotFoundError:
        return


def scheduler_handoff(args):
    gate = paths.RESULTS_DIR / 's6-sequence-handoff-20261118'
    receipt = dict(status='waiting_after_S6_CNN', original_batch=str(args.prior_case),
                   scientific_runs_repeated=0, BERT_model_reads=0, signals=[])
    write_json(args.out / 'handoff.json', receipt)
    # The former report waiter has not dispatched any numerical work.
    waiting = args.report_waiter
    if waiting.is_dir() and not (waiting / 'result.json').exists():
        progress = s4.read(waiting / 'progress.json')
        if progress.get('status') != 'waiting_for_S6_batch' or (waiting / 'resnet20-independent-report').exists():
            raise ValueError('Only the unused report wait entry may be reordered')
        for identity in process_identity('pt_agent.experiments.s6_report_repair', waiting):
            terminate_identity(identity)
            receipt['signals'].append(dict(identity, role='unstarted_report_waiter', signal='SIGTERM'))
    while not (gate / 'BERT-entry-waiting.json').is_file():
        if time.time() >= s4.DEADLINE:
            raise TimeoutError('Publication cutoff before the requested scheduler boundary')
        time.sleep(1)
    before = s4.read(args.prior_case / 'progress.json')
    if ([r['model'] for r in before.get('rows', [])] != ['resnet20', 'vgg16', 'alexnet']
            or base.case_for('bert_tiny').exists() or case_for('S6', 'bert_tiny').exists()
            or any(not (case_for('S6', m) / 'completion.json').is_file() for m in base.MODELS[:3])):
        raise ValueError('Handoff requires three closed CNNs and no BERT source entry')
    write_json(args.out / 'prior-three-CNN-state.json', before)
    identities = process_identity('pt_agent.experiments.s6_serial', args.prior_case)
    if len(identities) != 1:
        raise ValueError('Exactly the original S6 scheduler must own the safe waiting boundary')
    terminate_identity(identities[0])
    receipt['signals'].append(dict(identities[0], role='scheduler_at_unstarted_BERT_gate', signal='SIGTERM'))
    for _ in range(45):
        if not process_identity('pt_agent.experiments.s6_serial', args.prior_case):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Wait for the original owner to reap the unused BERT launcher')
    if case_for('S6', 'bert_tiny').exists():
        raise ValueError('The held BERT entry must still have zero scientific attempts')
    stub = base.case_for('bert_tiny')
    if stub.exists():
        if {p.name for p in stub.iterdir()} != {'paper-record.json'}:
            raise ValueError('Preserve and inspect any unexpected BERT source-entry artifacts')
        receipt['unused_scheduler_entry_stub'] = str(stub)
        receipt['stub_scope'] = 'Original scheduler closes its unused launcher; no model/session/result/forward exists'
    receipt.update(status='reordered_before_BERT_source_entry', utc=datetime.now(timezone.utc).isoformat(),
        prior_rows=3, original_scheduler_raw_result_preserved=True)
    write_json(args.out / 'handoff.json', receipt)
    return before


def publication(args, state):
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(), task_source=str(args.task),
        task_source_mtime_ns=args.task.stat().st_mtime_ns, cutoff_utc='2026-11-18T12:00:00+00:00',
        real_API_calls=0, heavy_cases_parallel=False, ordered_controls=ORDER,
        original_S6_tune_rule='v8-6.25', S6b_tune_rule=RULE)
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        for group, name in [('S6', 's6_no_llm.json'), ('S6b', 's6b_no_llm_nofloor.json')]:
            payload = dict(state['groups'][group], status=state['status'], active_group=state.get('active_group'),
                active_model=state.get('active_model'), active_phase=state.get('active_phase'),
                source_progress=str(args.out / 'progress.json'), updated_utc=state['updated_utc'],
                task_source=str(args.task), task_source_mtime_ns=state['task_source_mtime_ns'],
                real_API_calls=0, confirm_report_compile_feedback_to_Agent=False,
                tune_rule=RULE if group == 'S6b' else 'v8-6.25',
                tuner_budget_s=2400, baseline_actual_tune_rule='v8-6.25',
                requested_baseline_label_in_task='v8-6.24', paper_cutoff_utc=state['cutoff_utc'])
            write_json(root / name, payload)
        write_json(root / 'S6_S6B_PROGRESS.json', state)
        p = root / 'EXPERIMENT_DELIVERY.json'
        value = s4.read(p)
        value.update(status='S6_S6b_'+state['status'], current_required_experiment_scope_complete=False,
            S6_S6b=dict(status=state['status'], progress=str(args.paper / 'S6_S6B_PROGRESS.json')),
            updated_utc=state['updated_utc'])
        value['links']['S6b_no_LLM_nofloor'] = str(args.paper / 's6b_no_llm_nofloor.json')
        write_json(p, value)


def independent_report(args, state, group, model):
    from .a5_a6_rule63_search import guarded_process
    n = namespace(group)
    argv, target = n['report_command'](args, model)
    if target.is_dir():
        old = s4.read(target / 'result.json')
        if old.get('status') == 'ok':
            return old, target
        target = args.out / (group.lower()+'-'+model+'-report-entry-continuation')
        argv[argv.index('--out')+1] = str(target)
    argv += ['--tune-rule', RULE if group == 'S6b' else 'v8-6.25', '--s6-continue']
    key = group.lower()+'-'+model
    write_json(args.out / (key+'-report-command.json'), argv)
    state.update(active_group=group, active_model=model, active_phase='isolated_report')
    publication(args, state)
    observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (key+'-report.log'),
        resources=args.out / (key+'-report-resources.jsonl'), timeout=s4.remaining_before_deadline(1900), mem_gib=24, fsize_gib=8)
    write_json(args.out / (key+'-report-process.json'), observed)
    if observed.get('remaining_pids'):
        raise RuntimeError('Report descendants must close before another control')
    return s4.read(target / 'result.json'), target


def report_summary(value, target):
    return dict(status=value['status'], source=str(target / 'result.json'),
        report_samples=value.get('report_samples') or value.get('report_samples_completed'),
        task_policy_passed=value.get('report_policy_passed', value.get('task_policy_passed',
            (value.get('worker') or {}).get('task_policy_passed'))))


def main(args):
    from .a5_a6_rule63_search import guarded_process
    from ..autonomous_budget import call_records
    tests = s4.read(args.tests)
    if tests.get('status') != 'ok' or tests.get('s6b_unique_tests_passed') != 4:
        raise ValueError('Four S6b parameter and accounting tests must pass before orchestration')
    args.out.mkdir(exist_ok=False)
    before = scheduler_handoff(args)
    state = dict(status='running', active_group=None, active_model=None, active_phase=None,
        groups=dict(S6=dict(rows=before['rows'], reports=before.get('reports', {})), S6b=dict(rows=[], reports={})))
    ledger = paths.RESULTS_DIR / 'llm_calls.jsonl'
    api_before = len(call_records(ledger))
    state['API_ledger_before'] = api_before
    for group in ('S6', 'S6b'):
        (args.out / group.lower()).mkdir()
    publication(args, state)
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        # Finish the untouched ResNet report after the single wrapper correction.
        if time.time() < s4.DEADLINE-120:
            report, target = independent_report(args, state, 'S6', base.MODELS[0])
            summary = report_summary(report, target)
            state['groups']['S6']['reports'][base.MODELS[0]] = summary
            state['groups']['S6']['rows'][0]['report'] = summary
            publication(args, state)
        for group, model in ORDER:
            if time.time() >= s4.DEADLINE-180:
                state.update(status='cutoff_reached')
                break
            allocated = s4.remaining_before_deadline(10800 if model == 'bert_tiny' else 7200)
            state.update(active_group=group, active_model=model, active_phase='cold')
            publication(args, state)
            key = group.lower()+'-'+model
            argv = command(args, group, model, allocated)
            write_json(args.out / (key+'-command.json'), argv)
            observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (key+'-cold.log'),
                resources=args.out / group.lower() / (model+'-resources.jsonl'), timeout=allocated, mem_gib=24, fsize_gib=16)
            write_json(args.out / (key+'-process.json'), observed)
            if observed.get('remaining_pids') or len(call_records(ledger)) != api_before:
                raise RuntimeError('Close all owned children and preserve zero API ledger additions')
            selected = SimpleNamespace(**dict(vars(args), out=args.out / group.lower()))
            row = namespace(group)['row_for'](selected, model, observed)
            row.update(actual_allocated_seconds_before_cutoff=allocated)
            state['groups'][group]['rows'].append(row)
            publication(args, state)
            if row['status'] == 'ok' and time.time() < s4.DEADLINE-120:
                report, target = independent_report(args, state, group, model)
                summary = report_summary(report, target)
                state['groups'][group]['reports'][model] = summary
                row['report'] = summary
                publication(args, state)
            else:
                state['groups'][group]['reports'][model] = dict(status='not_run', reason='No accepted compiled program or cutoff reached')
            if time.time() < s4.DEADLINE:
                base.previous.archive_closed_intermediates(args, case_for(group, model), key)
        else:
            state.update(status='complete', active_group=None, active_model=None, active_phase=None)
    state['API_ledger_after'] = len(call_records(ledger))
    write_json(args.out / 'result.json', state)
    publication(args, state)
    record_registry(args.out, 'S6-S6B-SERIAL', 'four-model-controls', 'existing-v8-6.23-rule-selection-and-original-S6',
        state['status'], dict(API_calls=0, rows=sum(len(v['rows'])for v in state['groups'].values())))
    return int(state['status'] != 'complete')


def cli():
    p = base.cli()
    p.add_argument('--prior-case', type=Path, required=True)
    p.add_argument('--report-waiter', type=Path, required=True)
    return p


if __name__ == '__main__':
    args = cli().parse_args()
    try:
        code = main(args)
    except Exception as exc:
        if args.out.is_dir():
            write_json(args.out / 'result.json', dict(status='driver_error', error=type(exc).__name__+': '+str(exc)))
        raise
    raise SystemExit(code)
