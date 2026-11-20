"""S5 delegates the paper's original cold drivers with numerical tuning off."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import FunctionType

from .. import paths
from . import r30_common_model_e2e as cnn
from . import bert_fresh_cold_3h as bert
from . import s4_qwen_block as s4
from .r10_paper_metrics import write_json

MODULE = 'pt_agent.experiments.s5_tuner_off'
read = s4.read


def clone_module(module, replacements=None):
    source = vars(module)
    result = dict(source)
    for name, value in source.items():
        if isinstance(value, FunctionType) and value.__globals__ is source:
            cloned = FunctionType(value.__code__, result, value.__name__, value.__defaults__, value.__closure__)
            cloned.__kwdefaults__ = value.__kwdefaults__
            result[name] = cloned
    result.update(replacements or {})
    return result


def cached_entry(argv):
    """Retain the pure shell and worker arguments; omit extra digest checks."""
    result = list(argv)
    index = result.index('pt_agent.tools.nixrun')
    result[index:index+1] = [s4.MODULE, '--enter']
    return result


def cnn_command(args, phase, work, *, remaining_seconds):
    argv, seconds = cnn.phase_command(args, phase, work, remaining_seconds=remaining_seconds)
    argv = cached_entry(argv)
    index = argv.index('pt_agent.experiments.r30_common_model_e2e')
    argv[index:index+1] = [MODULE, '--cnn-worker']
    return argv, seconds


def inherited_calibration(snapshot, namespace):
    """Use the trusted paper snapshot for a comparison changing only tuner.

    S4 changed the inverse calibration definition after the paper's BERT run.
    The S5 ablation keeps the paper's original finite BERT definition. This
    loads just its trusted calibrate function, not a generated program.
    """
    tree = ast.parse(Path(snapshot).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'calibrate']
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != ['graph', 'inputs', 'batch']:
        raise ValueError('Original calibration function/signature required')
    body = ast.Module(body=functions, type_ignores=[])
    values = dict(namespace)
    exec(compile(body, str(snapshot), 'exec'), values)
    return values['calibrate']


def disable_tuner(original):
    def state(*args, **kwargs):
        if 'tuner' in kwargs and kwargs['tuner'] is not False:
            raise ValueError('S5 changes exactly the tuner selection to off')
        return original(*args, **dict(kwargs, tuner=False))
    return state


def bert_policy():
    from . import bert_serial_continuation as serial
    serial.STATE_POLICY = dict(serial.STATE_POLICY, tuner=False)


def bert_worker(argv, template, calibration_snapshot):
    from .r29_local_compile_matrix import native_metadata_commands
    bert_policy()
    with native_metadata_commands(read(template)):
        from .. import analyzer
        from ..planners import agent
        from . import bert_sst2_recorded as recorded
        original_state, original_calibrate = agent.new_agent_state, analyzer.calibrate
        agent.new_agent_state = disable_tuner(original_state)
        analyzer.calibrate = inherited_calibration(calibration_snapshot, vars(analyzer))
        try:
            return recorded.worker_delegate(argv, template, step_seconds=10800)
        finally:
            agent.new_agent_state, analyzer.calibrate = original_state, original_calibrate


def bert_command(args, phase, work, remaining):
    base = bert.command(args, phase, work, remaining)
    if phase == 'checkpoint':
        return base
    argv = cached_entry(base)
    # The recorded worker alone creates sessions. Math and native phases stay
    # on the original BERT worker and keep the tuner-off state policy.
    target = 'pt_agent.experiments.bert_sst2_recorded'
    if target in argv:
        index = argv.index(target)
        argv[index:index+1] = [MODULE, '--bert-worker']
    elif bert.MODULE in argv:
        index = argv.index(bert.MODULE)
        argv[index:index+1] = [MODULE, '--bert-native']
    argv += ['--calibration-snapshot', str(args.calibration_snapshot)]
    return argv


def bert_paper(args, result, observation):
    from .a_group_runs import append_record
    delegated = copy.copy(args)
    delegated.paper_runs = args.out / 'original-paper-record.jsonl'
    bert.paper_record(delegated, result, observation)
    rows = [json.loads(line) for line in delegated.paper_runs.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError('One actual original BERT observation required')
    row = dict(rows[0], system='arranger_tuner_off', task='S5-tuner-off', tuner=False,
        paper_eligible=time.time() <= s4.DEADLINE,
        actual_tuner_only_control=True, calibration_policy='Original paper snapshot; S4 normalization change excluded from this control',
        calibration_source_snapshot=str(args.calibration_snapshot))
    row['original_outer_stages_s'] = row['stages_s']
    row['stages_s'], row['stage_attribution_error'] = {}, None
    if row['status'] == 'ok':
        from .s5_serial import standard_stages
        try:
            row['stages_s'] = standard_stages(args.out, True)
            row.update(stage_attribution_ready=True,
                stage_scope='Fig6 categories; linked inner and outer exclusive spans, exact completion partition',
                stage_partition_source=str(args.out / 's5-standard-stage-partition.json'))
        except Exception as exc:
            row.update(stage_attribution_ready=False,
                stage_attribution_error=type(exc).__name__ + ': ' + str(exc))
    cached_timeouts = [dict(phase=value['phase'], exit_code=124,
                           source='Original bounded pure-environment launcher')
        for value in result.get('processes', [])
        if s4.MODULE in value.get('argv', [])
        and (value.get('observation') or {}).get('exit_code') == 124]
    if cached_timeouts and row['status'] == 'driver_error':
        row.update(status='timeout', inner_phase_timeouts=cached_timeouts,
                   timeout_scope='inner_phase_allocation')
    write_json(args.out / 'paper-record.json', row)
    if args.paper_runs:
        append_record(args.paper_runs, row)


def bert_outer(argv, snapshot):
    bert_policy()
    namespace = clone_module(bert, dict(command=bert_command, paper_record=bert_paper))
    original_parser = namespace['parser']
    def parser():
        result = original_parser()
        result.add_argument('--calibration-snapshot', type=Path, default=snapshot)
        return result
    namespace['parser'] = parser
    return namespace['main'](argv)


def cnn_report(argv):
    from . import r30_frozen_cnn_report as report
    from .a5_a6_rule63_search import guarded_process
    namespace = clone_module(report)
    def run(argv, **kwargs):
        return guarded_process(cached_entry(argv), **inherited_file_limit(kwargs))
    namespace['run_process'] = run
    namespace['claim_access'] = lambda out, gate: claim_report_access(report, out, gate)
    return namespace['main'](argv)


def inherited_file_limit(kwargs):
    """A nested launcher respects its inherited hard native file allowance."""
    import resource
    result = dict(kwargs)
    hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    if hard != resource.RLIM_INFINITY:
        result['fsize_gib'] = min(result.get('fsize_gib', 8), hard / 1024**3)
    return result


def claim_report_access(report, out, gate):
    directory = paths.RESULTS_DIR / 'frozen-cnn-report-access'
    previous = directory / (Path(gate['source_case']).name + '.json')
    if not previous.exists():
        return report.claim_access(out, gate)
    marker = report.read(previous)
    old_out = Path(marker['out'])
    failed = report.read(old_out / 'result.json')
    if (marker.get('source_case') != gate['source_case'] or marker.get('model') != gate['model']
            or marker.get('report_started') is not False or marker.get('report_complete') is not False
            or marker.get('report_samples') != 0 or failed.get('status') != 'error'
            or failed.get('report_samples_read') != 0 or failed.get('report_samples_completed') != 0
            or failed.get('error') != 'SubprocessError: Exception occurred in preexec_fn.'
            or (old_out / 'worker-result.json').exists() or (old_out / 'report-rows.jsonl').exists()):
        raise ValueError('Preserve once-only report access after any worker or sample has started')
    continued = directory / (Path(gate['source_case']).name + '-preexec-continuation.json')
    report.durable_json(continued, dict(marker, out=str(Path(out).resolve()),
        previous_unstarted_access=str(previous), entry_failure=str(old_out / 'result.json'),
        status='prepared'), exclusive=True)
    return continued


def bert_report(args):
    """Fork report access exactly as the original isolated BERT protocol."""
    from . import bert_frozen_fork_report as fork
    from . import bert_sst2 as original
    from .r10_paper_metrics import run_process, read_events, summarize_events
    from .completion_metrics import persist_case_completion, artifact_bytes
    from .s2_heir_serial import record_registry
    from .a5_a6_rule63_search import guarded_process
    import time
    bert_policy()
    started = time.perf_counter_ns()
    args.out.mkdir()
    try:
        result = read(args.source_case / 'result.json')
        completion = read(args.source_case / 'completion.json')
        if result.get('compile_accepted') is not True or completion.get('status') != 'ok':
            raise ValueError('Only a completed accepted S5 BERT candidate can enter report')
        fork.prepare_shell(args, {})
        (args.out / 'worker').mkdir()
        metadata = read(args.source_case / 'model/model_info.json')['source']
        seconds = s4.remaining_before_deadline(1800)
        argv = [sys.executable, '-B', '-m', s4.MODULE, '--enter', '--seconds', str(seconds),
            '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-m', MODULE, '--bert-worker',
            '--delegate-worker', '--metadata-template', str(args.metadata_template),
            '--calibration-snapshot', str(args.calibration_snapshot), '--phase', 'report',
            '--work-dir', str(args.out / 'worker'), '--model-dir', str(args.out / 'model'),
            '--session-dir', str(args.out / 'session'), '--model', metadata['local_path'],
            '--inputs', metadata['inputs_path'], '--ledger', str(paths.RESULTS_DIR / 'llm_calls.jsonl'),
            '--allowance', str(paths.RESULTS_DIR / 'autonomous-api-allowance.json'),
            '--data-python', str(paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent / 'paper-data-venv/bin/python')]
        observation = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / 'worker.log',
            resources=args.out / 'resources.jsonl', timeout=seconds, mem_gib=24, fsize_gib=2)
        write_json(args.out / 'process.json', observation)
        measured = read(args.out / 'worker/worker_result.json')
        original.check_worker('report', measured)
        fork.check_report(args, measured)
        if not bert._clean(observation):
            raise RuntimeError('Original isolated BERT report worker failed')
        value = dict(status='ok', worker=measured, source_case=str(args.source_case),
                     report_samples=872, API_calls=0, compiler_attempts=0, feedback_to_Agent=False)
    except BaseException as exc:
        value = dict(status='error', error=type(exc).__name__ + ': ' + str(exc), source_case=str(args.source_case))
    write_json(args.out / 'result.json', value)
    timing = summarize_events(read_events(args.out / 'worker/events.jsonl'))
    record_registry(args.out, 'S5-ISOLATED-BERT-REPORT', 'bert-tiny', 'native872-isolated-once',
                    value['status'], dict(API_calls=0, compiler_attempts=0))
    persist_case_completion(args.out, round_label='S5-ISOLATED-BERT-REPORT', started_ns=started,
        status=value['status'], timing=timing, required_artifact_bytes=artifact_bytes(args.out))
    return int(value['status'] != 'ok')


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ('cnn', 'cnn-worker', 'cnn-report', 'bert', 'bert-worker', 'bert-native', 'bert-report'):
        parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--calibration-snapshot', type=Path)
    args, remaining = parser.parse_known_args(argv)
    if args.cnn_worker:
        return cnn.main(remaining)
    if args.cnn_report:
        return cnn_report(remaining)
    if args.cnn:
        from .a5_a6_rule63_search import guarded_process
        return clone_module(cnn, dict(phase_command=cnn_command, run_process=guarded_process))['main'](remaining)
    if args.bert_worker:
        entry = argparse.ArgumentParser(add_help=False)
        entry.add_argument('--delegate-worker', action='store_true')
        entry.add_argument('--step-seconds', type=int, default=10800)
        entry.add_argument('--metadata-template', type=Path, required=True)
        selected, numeric = entry.parse_known_args(remaining)
        return bert_worker(numeric, selected.metadata_template, args.calibration_snapshot)
    if args.bert_native:
        bert_policy()
        return bert.main(remaining)
    if args.bert:
        return bert_outer(remaining, args.calibration_snapshot)
    if args.bert_report:
        entry = argparse.ArgumentParser()
        for name in ('out', 'source-case', 'metadata-template'):
            entry.add_argument('--'+name, type=Path, required=True)
        report_args = entry.parse_args(remaining)
        report_args.calibration_snapshot = args.calibration_snapshot
        return bert_report(report_args)
    parser.error('Select an S5 driver operation')


if __name__ == '__main__':
    raise SystemExit(main())
