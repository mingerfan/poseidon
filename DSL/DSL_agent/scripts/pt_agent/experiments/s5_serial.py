"""Serial, deadline-bounded S5 tuner-off controls after the current Qwen run."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType

from .. import paths
from . import s4_qwen_block as s4
from . import s5_tuner_off as off
from .a_group_runs import append_record, empty_record
from .a_group_profile_sum_policy import profile_sum
from .r10_paper_metrics import write_json, read_events, summarize_events, run_process
from .s2_heir_serial import record_registry
from .completion_metrics import persist_case_completion, artifact_bytes

read = s4.read
MODULE = 'pt_agent.experiments.s5_serial'
MODEL_NAMES = ('resnet20_silu', 'vgg16_silu_avgpool', 'alexnet_silu_avgpool', 'bert_tiny')
PAPER_MODELS = dict(resnet20_silu='resnet20', vgg16_silu_avgpool='vgg16', alexnet_silu_avgpool='alexnet', bert_tiny='bert_tiny')
CONTROL_CASES = dict(resnet20_silu='fresh-resnet20-agent-e2e-r30-authorized-20261116',
    vgg16_silu_avgpool='fresh-vgg16-hybrid-main-cold-r30-20261116',
    alexnet_silu_avgpool='fresh-alexnet-agent-e2e-r30-20261116',
    bert_tiny='bert-sst2-fresh-cold-agent-3h-logquota-recovery-20261117')


def case_for(model):
    return paths.RESULTS_DIR / ('s5-' + PAPER_MODELS[model] + '-fresh-cold-tuner-off-20261118')


def settings(args, model):
    source = read(paths.RESULTS_DIR / CONTROL_CASES[model] / 'result.json')
    if model == 'bert_tiny':
        return dict(source_case=CONTROL_CASES[model], source=source['source'])
    argv = source['processes'][0]['argv']
    def option(name):
        if argv.count(name) != 1:
            raise ValueError('One original paper argument required: '+name)
        return argv[argv.index(name)+1]
    return dict(source_case=CONTROL_CASES[model], source=source['source'],
        calls=option('--calls'), tuner_seconds=option('--tuner-seconds'),
        memory_gib=option('--memory-gib'), case_gib=option('--case-gib'),
        trace_seconds=option('--trace-seconds'), compile_seconds=option('--compile-seconds'),
        archive_emit=source.get('archive_emit_constants_after_trace') is True,
        diagnostic_elide=source.get('diagnostic_elide_final_ir') is True)


def command(args, model):
    case = case_for(model)
    selected = settings(args, model)
    argv = [sys.executable, '-B', '-m', off.MODULE]
    if model == 'bert_tiny':
        src = selected['source']
        return argv + ['--bert', '--model', src['model'], '--inputs', src['inputs'], '--out', str(case),
            '--checkpoint-out', str(case.with_name(case.name+'-checkpoint')), '--metadata-template', str(args.template),
            '--paper-runs', str(args.paper / 'runs.jsonl'), '--calibration-snapshot', str(args.calibration_snapshot)]
    argv += ['--cnn', '--model', model, '--out', str(case), '--seconds', '7200', '--agent-seconds', '7200',
        '--no-tuner', '--inputs', selected['source']['records'], '--selections', selected['source']['selections'],
        '--metadata-template', selected['source']['compiler_metadata_recipe']]
    for name in ('calls', 'tuner_seconds', 'memory_gib', 'case_gib', 'trace_seconds', 'compile_seconds'):
        argv += ['--'+name.replace('_', '-'), selected[name]]
    if selected['archive_emit']:
        argv.append('--archive-emit-constants-after-trace')
    if selected['diagnostic_elide']:
        argv.append('--diagnostic-elide-final-ir')
    return argv


def publication(args, state):
    now = datetime.now(timezone.utc).isoformat()
    state.update(updated_utc=now, task_source=str(args.task), task_source_mtime_ns=args.task.stat().st_mtime_ns,
        cutoff_utc='2026-11-18T12:00:00+00:00', system='arranger_tuner_off', param_group='ours-N16-boot5',
        heavy_cases_parallel=False, source_Qwen=str(args.wait_case))
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 'S5_TUNER_OFF_PROGRESS.json', state)
        write_json(root / 'S5_TUNER_OFF_RESULTS.json', dict(status=state['status'], rows=state.get('rows', []),
            reports=state.get('reports', {}), source_progress=str(args.out / 'progress.json')))
        path = root / 'EXPERIMENT_DELIVERY.json'
        value = read(path)
        value.update(status='S5_tuner_off_work_active' if state['status'] != 'complete' else 'S5_finite_controls_closed',
            current_required_experiment_scope_complete=False, task_source_mtime_ns=state['task_source_mtime_ns'],
            S5_tuner_off=dict(status=state['status'], progress=str(args.paper / 'S5_TUNER_OFF_PROGRESS.json')),
            last_metadata_revision_utc=now)
        value['links']['S5_tuner_off_progress'] = str(args.paper / 'S5_TUNER_OFF_PROGRESS.json')
        value['links']['S5_tuner_off_results'] = str(args.paper / 'S5_TUNER_OFF_RESULTS.json')
        write_json(path, value)
        progress = read(root / 'A_GROUP_PROGRESS.json')
        progress.update(S5_tuner_off=state, current_required_experiment_scope_complete=False, updated_utc=now)
        write_json(root / 'A_GROUP_PROGRESS.json', progress)


def archive_worker(args):
    from .r34_archive_text import archive_one
    manifest = read(args.archive_manifest)
    for entry in manifest['files']:
        path = Path(entry['path'])
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError('Archive requires an inactive original regular constant file')
        case = Path(entry['case'])
        result_status = read(case / 'result.json').get('status')
        if result_status not in ('ok', 'error', 'quality_rejected') or read(case / 'completion.json').get('status') != result_status:
            raise ValueError('Archive source experiment must already be complete')
        # A live reader keeps the file in place; no data inspection or digest.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                handles = list((proc / 'fd').iterdir())
            except OSError:
                continue
            for handle in handles:
                try:
                    if os.readlink(handle) == str(path):
                        raise RuntimeError('An original constant file still has an active reader')
                except FileNotFoundError:
                    pass
                except PermissionError:
                    pass
        target = path.with_name(path.name + '.gz')
        temporary = target.with_name(target.name + '.partial')
        observed = archive_one(path, target, temporary)
        write_json(Path(entry['receipt']), observed)
    return 0


def archive_inactive(args):
    files = []
    entries = [(model, 'a2-'+model+'-frozen-native-profile-four-placement-pars-20261117',
                'emit/constants.npz') for model in ('vgg16', 'alexnet')]
    entries.append(('bert-trace', CONTROL_CASES['bert_tiny'], 'compile-work/emit/candidate_trace.mlir'))
    for model, case_name, relative in entries:
        case = paths.RESULTS_DIR / case_name
        path = case / relative
        if not path.is_file() or path.with_name(path.name+'.gz').exists():
            continue
        files.append(dict(path=str(path), case=str(case), receipt=str(args.out / (model+'-pre-cold-archive.json')),
            restore_argv=['gzip', '-dk', '--', str(path)+'.gz']))
    if not files:
        return
    manifest = args.out / 'pre-cold-capacity-manifest.json'
    write_json(manifest, dict(files=files, hashes=0,
        purpose='Recover capacity before new cold clocks; old scientific results and compiled bundles retained',
        prospective_restore=True))
    argv = [sys.executable, '-B', '-m', MODULE, '--archive-worker', '--archive-manifest', str(manifest)]
    value = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / 'pre-cold-archive.log',
        resources=args.out / 'pre-cold-archive-resources.jsonl', timeout=600, mem_gib=1, fsize_gib=16)
    write_json(args.out / 'pre-cold-archive-process.json', value)
    if value['exit_code'] != 0 or value['timed_out'] or value['remaining_pids']:
        raise RuntimeError('Inactive constant archival failed; retain all original or partial files')


def archive_closed_intermediates(args, case, model):
    """Reversible storage work follows the completed cold/report clocks."""
    files = []
    for path in case.rglob('*'):
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 256*1024**2:
            continue
        if path.name != 'constants.npz' and path.suffix != '.mlir':
            continue
        if path.with_name(path.name+'.gz').exists():
            continue
        receipt = args.out / (model+'-'+str(len(files))+'-closed-archive.json')
        files.append(dict(path=str(path), case=str(case), receipt=str(receipt),
                          restore_argv=['gzip', '-dk', '--', str(path)+'.gz']))
    if not files:
        return
    manifest = args.out / (model+'-closed-archive-manifest.json')
    write_json(manifest, dict(files=files, hashes=0, cold_clock_already_closed=True,
        scope='Only complete inactive generated constants/MLIR; frozen DSL, weights, datasets, RuntimePlan and bundle stay in place'))
    observed = run_process([sys.executable, '-B', '-m', MODULE, '--archive-worker', '--archive-manifest', str(manifest)],
        cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (model+'-archive.log'),
        resources=args.out / (model+'-archive-resources.jsonl'), timeout=s4.remaining_before_deadline(600),
        mem_gib=1, fsize_gib=8)
    write_json(args.out / (model+'-archive-process.json'), observed)
    if observed['exit_code'] or observed['timed_out'] or observed['remaining_pids']:
        raise RuntimeError('Closed-case intermediate archival failed; preserve recovery files')


def standard_stages(case, is_bert):
    from . import agent_cold_stage_partition as standard
    completion = read(case / 'completion.json')
    if not is_bert:
        timing = read(case / 'timing.json')
    else:
        events = read_events(case / 'events.jsonl')
        outer = summarize_events(events)['spans']
        for source in sorted(case.rglob('*events.jsonl')):
            if source == case / 'events.jsonl' or source.is_symlink():
                continue
            rows = read_events(source)
            beginning = {row['event_id']: row for row in rows if row.get('type') == 'begin'}
            ending = {row['event_id']: row for row in rows if row.get('type') == 'end'}
            for row in rows:
                if row.get('type') not in ('begin', 'end') or row.get('parent_id') is not None:
                    continue
                start = beginning[row['event_id']]['start_ns']
                end = ending.get(row['event_id'], {}).get('end_ns', start)
                parents = [span for span in outer if span['start_ns'] <= start and end <= span['end_ns']]
                if parents:
                    row['parent_id'] = min(parents, key=lambda value: value['inclusive_ns'])['event_id']
            events += rows
        timing = summarize_events(events)
    spans = [dict(row) for row in timing['spans']]
    for row in spans:
        if row['stage_id'] == 'fresh_cold_complete' and row['parent_id'] is None:
            row['stage_id'] = 'fresh_end_to_end'
    labels = dict(standard.LABELS, prepare='intake_and_preflight', checkpoint='intake_and_preflight', calibrate='calibration',
        step='agent_context_and_session', confirm='independent_freeze_confirmation',
        agent_confirm='independent_freeze_confirmation', math='strict_packed_arithmetic')
    function = FunctionType(standard.partition.__code__, dict(vars(standard), LABELS=labels),
                            standard.partition.__name__, standard.partition.__defaults__)
    verified = function(spans, completion['complete_case_elapsed_seconds'])
    write_json(case / 's5-standard-stage-partition.json', verified)
    return {key: value/1e9 for key, value in verified['category_ns'].items()}


def cnn_row(args, model, observed):
    case = case_for(model)
    result, completion = read(case / 'result.json'), read(case / 'completion.json')
    compiled = (result.get('phases') or {}).get('compile') or {}
    native = compiled.get('process') or {}
    error = result.get('error')
    worker_errors = {phase: value.get('error') for phase, value in result.get('phases', {}).items() if value and value.get('error')}
    guard = next(iter(args.out.glob(model+'-resources.physical-guard.json')), None)
    status = ('ok' if result['status'] == 'ok' else 'resource_limit' if guard or 'reserve' in str(error) or 'MemoryError' in str(worker_errors)
        else 'timeout' if observed.get('timed_out') or native.get('timed_out') or any(
            (item.get('observation') or {}).get('timed_out') or (item.get('observation') or {}).get('exit_code') == 124
            for item in result.get('processes', [])) else 'compiler_crash'
        if native.get('exit_code') not in (None, 0) else 'driver_error')
    policy = profile_sum(compiled.get('prediction'))
    quality = (read(case / 'post-run-quality-audit.json').get('confirmation')
               if (case / 'post-run-quality-audit.json').is_file() else {}) or {}
    peak = 0
    resources = args.out / (model+'-resources.jsonl')
    for line in resources.read_text().splitlines():
        sample = json.loads(line)
        peak = max(peak, sum(value.get('vmrss_kib', 0)*1024 for value in sample.get('processes', [])))
    stages, stage_error = {}, None
    if status == 'ok':
        try:
            stages = standard_stages(case, False)
        except Exception as exc:
            stage_error = type(exc).__name__+': '+str(exc)
    row = empty_record(run_id=case.name, task='S5-tuner-off', model=PAPER_MODELS[model], system='arranger_tuner_off',
        param_group='ours-N16-boot5', status=status, status_detail=str(worker_errors) if worker_errors else error,
        budget_s=7200, compile_s=(native['ended_ns']-native['started_ns'])/1e9 if native.get('ended_ns') else None,
        generation_e2e_s=completion['complete_case_elapsed_seconds'] if status == 'ok' else None,
        observed_complete_case_s=completion['complete_case_elapsed_seconds'], peak_rss_gib=peak/s4.GIB,
        boots_compiled=compiled.get('compiled_bootstraps'), exec_ops=compiled.get('execution_steps'),
        pred_latency_s=policy['pred_latency_s'], pred_table=str(paths.PROFILES_DIR / 'qwen48/compiler-config.json'),
        prediction_policy=policy, api_calls=result.get('api_delta'),
        source=str(case / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
        stages_s=stages, stage_attribution_error=stage_error, tuner=False, cold=True,
        quality=dict(metric='classification_top1', split='confirmation1000', value=quality.get('top1'),
                     original=quality.get('exact_top1'), passed=quality.get('passed'), total=quality.get('total')),
        paper_eligible=time.time() <= s4.DEADLINE, source_control=CONTROL_CASES[model],
        actual_tuner_only_control=True)
    write_json(case / 'paper-record.json', row)
    append_record(args.paper / 'runs.jsonl', row)
    return row


def report_command(args, model):
    case = case_for(model)
    output = case.with_name(case.name+'-isolated-report')
    if model == 'bert_tiny':
        return [sys.executable, '-B', '-m', off.MODULE, '--bert-report', '--source-case', str(case),
                '--out', str(output), '--metadata-template', str(args.template),
                '--calibration-snapshot', str(args.calibration_snapshot)], output
    return [sys.executable, '-B', '-m', off.MODULE, '--cnn-report', '--model', model,
            '--source-case', str(case), '--out', str(output), '--seconds', str(s4.remaining_before_deadline(1800)),
            '--memory-gib', '6'], output


def main(args):
    if args.archive_worker:
        return archive_worker(args)
    from .a5_a6_rule63_search import guarded_process
    if args.resume_wait:
        old = read(args.out / 'progress.json')
        if old.get('status') != 'waiting_for_Qwen_Q1' or old.get('rows') or old.get('attempted_models'):
            raise ValueError('Only an untouched waiting controller can resume its metadata entry')
        if any(case_for(model).exists() for model in MODEL_NAMES):
            raise ValueError('No S5 scientific case may already exist when resuming the waiting entry')
    else:
        args.out.mkdir(exist_ok=False)
    state = dict(status='waiting_for_Qwen_Q1', rows=[], reports={}, attempted_models=[],
                 ordered_models=list(MODEL_NAMES), tuner=False)
    publication(args, state)
    while not (args.wait_case / 'completion.json').is_file():
        if time.time() >= s4.DEADLINE:
            state.update(status='deadline_reached_before_start')
            write_json(args.out / 'result.json', state)
            publication(args, state)
            return 1
        time.sleep(2)
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        state.update(status='preparing_capacity')
        publication(args, state)
        archive_inactive(args)
        for model in MODEL_NAMES:
            if time.time() >= s4.DEADLINE:
                state.update(status='deadline_reached')
                break
            budget = s4.remaining_before_deadline(10800 if model == 'bert_tiny' else 7200)
            if budget <= 180:
                state.update(status='deadline_closure_reserve', unrun_model=model)
                break
            state.update(status='running', active_model=model, active_phase='cold')
            state['attempted_models'].append(model)
            publication(args, state)
            argv = command(args, model)
            if model != 'bert_tiny' and budget < 7200:
                argv[argv.index('--seconds')+1] = str(budget)
            write_json(args.out / (model+'-command.json'), argv)
            observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (model+'-cold.log'),
                resources=args.out / (model+'-resources.jsonl'), timeout=budget, mem_gib=24, fsize_gib=16)
            write_json(args.out / (model+'-process.json'), observed)
            case = case_for(model)
            if not (case / 'completion.json').is_file() or not (case / 'paper-record.json').is_file() and model == 'bert_tiny':
                raise RuntimeError('Original S5 driver failed to close required terminal artifacts; preserve and stop serial batch')
            row = read(case / 'paper-record.json') if model == 'bert_tiny' else cnn_row(args, model, observed)
            state['rows'].append(row)
            publication(args, state)
            if observed['remaining_pids']:
                raise RuntimeError('Finish the original owned descendants before advancing S5')
            if row['status'] != 'ok' or time.time() >= s4.DEADLINE:
                state['reports'][model] = dict(status='not_run', reason='No accepted compiled candidate or publication cutoff reached')
                if time.time() < s4.DEADLINE:
                    archive_closed_intermediates(args, case, model)
                continue
            state.update(active_phase='isolated_report')
            publication(args, state)
            argv, report_case = report_command(args, model)
            write_json(args.out / (model+'-report-command.json'), argv)
            seconds = s4.remaining_before_deadline(1900)
            observation = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (model+'-report.log'),
                resources=args.out / (model+'-report-resources.jsonl'), timeout=seconds, mem_gib=24, fsize_gib=8)
            write_json(args.out / (model+'-report-process.json'), observation)
            value = read(report_case / 'result.json') if (report_case / 'result.json').is_file() else dict(status='error',reason='Missing report result')
            state['reports'][model] = dict(status=value['status'], source=str(report_case / 'result.json'),
                report_samples=value.get('report_samples') or value.get('report_samples_completed'),
                task_policy_passed=value.get('task_policy_passed') or (value.get('worker') or {}).get('task_policy_passed'))
            publication(args, state)
            if observation['remaining_pids']:
                raise RuntimeError('Report descendants must close before the next S5 case')
            archive_closed_intermediates(args, case, model)
        else:
            state.update(status='complete', active_model=None, active_phase=None)
    write_json(args.out / 'result.json', state)
    publication(args, state)
    record_registry(args.out, 'S5-TUNER-OFF-SERIAL', 'four-model-control', 'finite-cold-controls-and-isolated-reports',
                    state['status'], dict(models=len(state['rows'])))
    return 0 if state['status'] == 'complete' else 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-worker', action='store_true')
    parser.add_argument('--resume-wait', action='store_true')
    parser.add_argument('--archive-manifest', type=Path)
    for name in ('out', 'wait-case', 'paper', 'mirror', 'task'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--template', type=Path, default=paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    parser.add_argument('--calibration-snapshot', type=Path, default=paths.RESULTS_DIR / 'fresh-vgg16-hybrid-main-cold-r30-20261116/source/pt_agent/analyzer.py')
    return parser


if __name__ == '__main__':
    args = cli().parse_args()
    try:
        result = main(args)
    except Exception as exc:
        if args.out and args.out.is_dir() and (args.out / 'progress.json').is_file():
            state = read(args.out / 'progress.json')
            state.update(status='error', error=type(exc).__name__+': '+str(exc))
            write_json(args.out / 'result.json', state)
            publication(args, state)
        raise
    raise SystemExit(result)
