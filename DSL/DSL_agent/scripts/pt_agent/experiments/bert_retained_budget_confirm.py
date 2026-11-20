"""One original C2 after a declared stop at a closed three-round BERT boundary.

This warm recovery keeps the original maximum of ten iterations. It dispatches
the entire saved candidate pool to the original confirmation step, with no new
generation/tuning/report. The original supervisor must already have closed.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import bert_serial_continuation as serial
from . import bert_sst2 as original
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, _scan

ROUND = 'BERT-RETAINED-BUDGET-CONFIRM'
MODEL = original.MODEL
MODE = 'warm-three-completed-round-budget-stop-original-c2-once'
MODULE = 'pt_agent.experiments.bert_retained_budget_confirm'
GIB = 1024**3
CASE_BYTES, FREE_BYTES = 4*GIB, 10*GIB
COMPLETION_RESERVE_BYTES = 1024**2
CONFIRM_SECONDS, TOTAL_SECONDS = 21600, 22200
ZERO = ('new_api_calls', 'new_tune_commands', 'report_samples', 'compiler_runs',
        'keygen_runs', 'ckks_runs', 'downloads', 'installations', 'integrity_checks')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    serial.durable_json(path, value)


def state_summary(state):
    return {key: state.get(key) for key in (*serial.STATE_POLICY, 'phase', 'iteration', 'calls', 'failed')}


def _policy(state):
    if any(state.get(key) != value for key, value in serial.STATE_POLICY.items()):
        raise ValueError('Retain every original saved BERT v11/C2 policy, including max_iters10')


def _closed_process(process, *, successful=False):
    if (not isinstance(process, dict) or type(process.get('exit_code')) is not int
            or type(process.get('timed_out')) is not bool
            or not isinstance(process.get('signals_sent'), list)
            or process.get('remaining_pids') != [] or process.get('cleanup_unknown') is True
            or process.get('signal') is not None and type(process['signal']) is not int):
        raise ValueError('A closed process with explicit terminal and cleanup evidence is required')
    if successful and (process['exit_code'] != 0 or process['timed_out']
                       or process.get('signal') is not None or process['signals_sent']):
        raise ValueError('The retained boundary must follow a clean successful original tune')


def source_gate(args):
    case, continuation = args.source_case, args.continuation_case
    expected = continuation/'phases/002-step/before/session-SESSION.json'
    if args.boundary_snapshot != expected.resolve():
        raise ValueError('Use the original closed tune3 boundary captured before evaluate4')
    state = read(args.boundary_snapshot)
    _policy(state)
    if (state.get('phase') != 'evaluate' or type(state.get('iteration')) is not int or state['iteration'] != 4
            or type(state.get('calls')) is not int or state['calls'] != 3
            or state.get('pending') is not None or state.get('failed') is not None
            or state.get('best_iteration') != 3 or state.get('first_feasible') != 3
            or len(state.get('history', [])) != 3 or len(state.get('pool', [])) != 5
            or not isinstance(state.get('records'), dict) or not state['records']
            or not state.get('best_program') or 'confirmation' in state
            or any(item.get('iteration') != 3 for item in state['pool'])):
        raise ValueError('The retained complete three-round candidate boundary differs')
    run, completion = read(continuation/'result.json'), read(continuation/'completion.json')
    if (run.get('round') != serial.ROUND or run.get('model') != MODEL
            or run.get('status') not in ('ok', 'error') or completion.get('status') != run['status']
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or not serial.same((run.get('source') or {}).get('source_case'), case)):
        raise ValueError('The original continuation must already have a durable terminal result')
    if any(type((run.get('attempted') or {}).get(phase)) is not int
           or run['attempted'][phase] != 0 for phase in ('report', 'compile')):
        raise ValueError('The source continuation must have zero original report/compiler attempts')
    rows = run.get('processes') or []
    if len(rows) < 2:
        raise ValueError('Original closed tune and following evaluate provenance are required')
    for row in rows:
        _closed_process(row.get('observation'))
        for key in ('source_state_before', 'source_state_after'):
            if (row.get(key) or {}).get('phase') in ('confirm', 'done'):
                raise ValueError('Preserve the one confirmation boundary; original branch advanced to C2')
    first = rows[0]
    _closed_process(first['observation'], successful=True)
    before, after = first.get('source_state_before') or {}, first.get('source_state_after') or {}
    if (first.get('phase') != 'step' or before.get('phase') != 'tune'
            or before.get('iteration') != 3 or before.get('calls') != 3
            or after != state_summary(state) or rows[1].get('source_state_before') != after):
        raise ValueError('Snapshot is not the completed original tune3/evaluate4 boundary')
    prior = read(args.boundary_snapshot.with_name('result.json'))
    worker = prior.get('worker') or {}
    _closed_process(prior.get('process'), successful=True)
    if (prior.get('status') != 'ok' or prior.get('model') != MODEL or worker.get('status') != 'ok'
            or worker.get('phase') != 'step' or worker.get('next_step') != 'evaluate'
            or (worker.get('step') or {}).get('session_calls') != 3 or worker.get('llm_calls') != 0
            or worker.get('frozen') is not False or worker.get('report_samples') != 0):
        raise ValueError('Original successful tune worker boundary is not accepted')
    current = read(case/'session/SESSION.json')
    _policy(current)
    if (current.get('phase') not in ('evaluate', 'tune') or current.get('failed') is not None
            or type(current.get('calls')) is not int or current['calls'] != 4
            or type(current.get('iteration')) is not int or current['iteration'] not in (4, 5)
            or 'confirmation' in current):
        raise ValueError('Original source must remain unconfirmed with its fourth call recorded separately')
    forbidden = [case/'session'/name for name in ('AGENT.json', 'FROZEN.json', 'REPORT_ACCESS.json', 'program.ptdsl')]
    forbidden += [case/'model/samples/confirm.npz', case/'model/samples/report.npz']
    if any(path.exists() for path in forbidden):
        raise ValueError('Original confirmation/cache/report access must still be absent')
    if any(path.name.endswith('-confirm') or path.name.endswith('-report') for path in (case/'phases').iterdir()):
        raise ValueError('Original confirm/report phase attempt already exists')
    info = read(case/'model/model_info.json')
    if info.get('graph') != 'graph':
        raise ValueError('Original ModelInfo must retain its directly bound graph directory')
    src = info.get('source') or {}
    from .bert_checkpoint_audit import source_gate as checkpoint_source
    checkpoint = checkpoint_source(src['local_path'], case/'phases/001-prepare/worker_result.json', case/'model/model_info.json')
    partition, sample = info.get('split') or {}, read(case/'model/samples/source.json')
    if (partition.get('kind') != 'sst2_native' or partition.get('version') != 'sst2-native-v1'
            or partition.get('seed') != 20261024 or sample.get('partition') != partition
            or sample.get('native_test_read') is not False or not serial.same(sample.get('source'), src['inputs_path'])
            or any(partition.get(key) != list(range(start, start+count)) for key, start, count in
                   [('calibration', 0, 1000), ('heldout', 1000, 1000), ('confirm', 2000, 1000), ('report', 3000, 872)])):
        raise ValueError('Retain original SST2 task and all calibration/heldout/confirm/report partitions')
    template = read(args.metadata_template)
    from .r29_local_compile_matrix import compiler_argv
    compiler_argv(template, 'pars', binary='/compiler', source='/source', profile='/profile', output='/output')
    for name in ('intake_python', 'data_python'):
        if not (getattr(args, name).is_file() or getattr(args, name).is_symlink()):
            raise FileNotFoundError('The existing configured interpreter is required: '+name)
    for name in ('model_info.json', 'calibration.json', 'samples/source.json', 'samples/calibration.npz', 'samples/heldout.npz', 'graph/graph.json', 'graph/weights.npz'):
        if not (case/'model'/name).is_file():
            raise FileNotFoundError('Required original import/cache metadata or reference is absent: '+name)
    return dict(source_case=str(case), continuation_case=str(continuation), boundary_snapshot=str(args.boundary_snapshot),
        boundary=state_summary(state), original_source_state=state_summary(current),
        model=src['local_path'], inputs=src['inputs_path'], model_info=str(case/'model/model_info.json'),
        original_checkpoint=checkpoint, completed_search_rounds=3, inherited_calls=3,
        original_calls=4, excluded_original_calls=[4], pool_size=5,
        declared_stop='Completed three search rounds; original maximum10 is retained; independent phase dispatch to C2',
        timing_scope='Warm retained-boundary recovery only; no cold clock or sum of historical clocks',
        original_continuation_status=run['status'], original_continuation_clock=completion.get('complete_case_elapsed_seconds'))


def original_inflight():
    """Public process metadata only; never inspect prompts or signal a source."""
    table = _scan()
    own = {os.getpid()}
    while True:
        extended = own | {pid for pid, row in table.items() if row.get('ppid') in own}
        if extended == own:
            break
        own = extended
    found = []
    for pid, row in table.items():
        if pid in own or row.get('state') == 'Z':
            continue
        try:
            argv = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, PermissionError) as exc:
            raise RuntimeError('Cannot establish source-process absence') from exc
        modules = (b'pt_agent.experiments.bert_sst2_worker', b'pt_agent.experiments.bert_sst2_recorded',
                   b'pt_agent.experiments.bert_serial_continuation')
        native = argv[0].rsplit(b'/', 1)[-1] if argv else b''
        compilers = (b'heir-opt', b'heir-translate', b'hecate-opt', b'go', b'compile', b'link', b'cc1plus', b'ld.lld')
        if any(module in argv for module in modules) or native in compilers:
            found.append(dict(pid=pid, starttime_ticks=row.get('starttime')))
    return found


def check_resources(args, *, own_pid=None):
    if artifact_bytes(args.out)+COMPLETION_RESERVE_BYTES > CASE_BYTES or shutil.disk_usage(args.out).free < FREE_BYTES:
        raise RuntimeError('Confirm-only artifact allowance or10GiB disk reserve exhausted')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])*1024
    if available is None or available < 1.25*GIB:
        raise RuntimeError('Host available memory must retain1.25GiB')
    table = _scan(); owned = {os.getpid() if own_pid is None else own_pid}
    while True:
        extended = owned | {pid for pid, row in table.items() if row.get('ppid') in owned}
        if extended == owned:
            break
        owned = extended
    rss = sum(table.get(pid, {}).get('vmrss_kib', 0)*1024 for pid in owned)
    if rss > 14*GIB:
        raise RuntimeError('Own confirm-only process tree RSS exceeds14GiB')
    if original_inflight() or serial.foreign_heavy_processes():
        raise RuntimeError('Source BERT or another heavy process is still active')
    return dict(aggregate_rss_bytes=rss, host_available_bytes=available)


def prepare_branch(args, source):
    """Independent writable metadata/cache shell; original array files stay referenced."""
    model, samples, session = args.out/'model', args.out/'model/samples', args.out/'session'
    samples.mkdir(parents=True); session.mkdir()
    original_model = args.source_case/'model'
    for name in ('model_info.json', 'calibration.json', 'samples/source.json'):
        shutil.copyfile(original_model/name, model/name)
    for name in ('graph', 'samples/calibration.npz', 'samples/heldout.npz'):
        (model/name).symlink_to(original_model/name, target_is_directory=name == 'graph')
    state = read(args.boundary_snapshot)
    branch = copy.deepcopy(state); branch['phase'] = 'confirm'
    save(session/'SESSION.json', branch)
    shutil.copyfile(args.boundary_snapshot, args.out/'retained-boundary-SESSION.json')
    save(args.out/'recovery.json', dict(source=source, only_session_change='phase:evaluate->confirm',
        max_iters_retained=10, full_candidate_pool_retained=True, original_source_changed=False,
        independent_model_samples_shell=True, new_confirm_cache_directory=str(samples),
        inherited_model_and_existing_array_references='Read only; original Graph loader marks its weights readonly',
        selected_by_report=False, new_generation_or_tune_commands=0))
    return branch


def command(args, source, work):
    argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(CONFIRM_SECONDS+60),
            '--python', str(args.intake_python), '--', '-B', '-m', serial.RECORDED,
            '--delegate-worker', '--metadata-template', str(args.metadata_template), '--step-seconds', str(CONFIRM_SECONDS),
            '--phase', 'step', '--work-dir', str(work), '--model-dir', str(args.out/'model'),
            '--session-dir', str(args.out/'session'), '--model', source['model'], '--inputs', source['inputs']]
    for name in ('ledger', 'allowance', 'data_python'):
        argv += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    return argv


def branch_bindings(args):
    """Metadata and reference destinations only; never open an array payload."""
    model, original_model = args.out/'model', args.source_case/'model'
    for name in ('model_info.json', 'calibration.json', 'samples/source.json'):
        if read(model/name) != read(original_model/name):
            raise ValueError('Fork metadata no longer matches the original import: '+name)
    for name in ('graph', 'samples/calibration.npz', 'samples/heldout.npz'):
        if not (model/name).is_symlink() or (model/name).resolve() != (original_model/name).resolve():
            raise ValueError('Fork reference destination differs from the original: '+name)
    cache = model/'samples/confirm.npz'
    if cache.is_symlink() or cache.exists() and not cache.is_file():
        raise ValueError('Confirmation cache must belong to the new independent samples directory')
    if (args.out/'session/REPORT_ACCESS.json').exists() or (model/'samples/report.npz').exists():
        raise ValueError('Confirm-only recovery must retain frozen report access for a later stage')
    return dict(original_model_info=str(original_model/'model_info.json'), fork_model_info=str(model/'model_info.json'),
                original_import_metadata_equal=True, graph_and_existing_array_reference_destinations_equal=True,
                numerical_payloads_opened=0, report_accessed=False)


def accepted_branch(args, before):
    branch_bindings(args)
    state = read(args.out/'session/SESSION.json')
    for key in (*serial.STATE_POLICY, 'calls', 'history', 'pool', 'records', 'messages', 'tune_cache',
                'prompt_tokens', 'completion_tokens'):
        if state.get(key) != before.get(key):
            raise ValueError('Original confirmation changed retained search state: '+key)
    if state.get('phase') != 'done' or state.get('failed') is not None:
        raise ValueError('Original C2 did not finish')
    return serial.frozen_state(args.out)


def _claim(args):
    directory = paths.RESULTS_DIR/'bert-budget-stop-c2-claims'
    directory.mkdir(exist_ok=True)
    path = directory/(args.source_case.name+'.json')
    stream = path.open('x+')
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    json.dump(dict(source_case=str(args.source_case), branch=str(args.out), boundary=str(args.boundary_snapshot),
                   purpose=MODE, retries=0), stream, allow_nan=False)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return path, stream


def terminal_case(args, result):
    """Include result persistence while retaining room for one completion marker."""
    for _ in range(20):
        actual = artifact_bytes(args.out)
        result['terminal_case_budget'] = dict(case_limit_bytes=CASE_BYTES,
            artifact_bytes_before_completion=actual, completion_reserve_bytes=COMPLETION_RESERVE_BYTES,
            within_reserved_case_allowance=actual+COMPLETION_RESERVE_BYTES <= CASE_BYTES)
        if not result['terminal_case_budget']['within_reserved_case_allowance']:
            result.update(status='error', error='Terminal result and reserved completion exceed confirm-only case allowance')
        result['frozen_accepted'] = result['status'] == 'ok' and result['frozen'] is True and result['completed_confirm'] == 1
        result['metrics'] = {key: result[key] for key in ('status','outcome','admission_passed','attempted_confirm',
                           'completed_confirm','frozen','frozen_accepted',*ZERO)}
        save(args.out/'result.json', result)
        if artifact_bytes(args.out) == actual:
            return
    raise RuntimeError('Confirm-only terminal result inventory did not converge')


def execute(args):
    if args.out.exists():
        raise FileExistsError('Preserve every prior independent confirmation attempt')
    args.out.mkdir(parents=True)
    started = time.perf_counter_ns(); deadline = time.monotonic()+TOTAL_SECONDS
    recorder = StageRecorder(args.out/'events.jsonl'); stop = threading.Event(); failures = []
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        source_case=str(args.source_case), continuation_case=str(args.continuation_case),
        admission_passed=False, attempted_confirm=0, completed_confirm=0, frozen=False, frozen_accepted=False,
        process=None, worker=None, outcome=None, timing_scope='Independent warm budget-stop recovery; historical clocks excluded',
        resource_limits=dict(requested_parent_AS_gib=24, child_AS_gib=4, tree_RSS_gib=14, host_available_gib=1.25,
                             file_gib=8, case_gib=4, minimum_free_gib=10,
                             confirm_seconds=CONFIRM_SECONDS, overall_seconds=TOTAL_SECONDS), **dict.fromkeys(ZERO, 0))
    handlers = {}; claim_stream = None; watcher = None
    def interrupt(signum, frame):
        raise InterruptedError('Own confirm-only supervisor received signal'+str(signum))
    def monitor():
        while not stop.wait(.25):
            try:
                check_resources(args)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Finite confirm-only total wall allowance exhausted')
            except Exception as exc:
                failures.append(type(exc).__name__+': '+str(exc))
                os.kill(os.getpid(), signal.SIGTERM)
                return
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
        with recorder.span('retained_closed_source_admission'):
            source = source_gate(args); check_resources(args)
            result['source'] = source; result['admission_passed'] = True
            shutil.copyfile(Path(__file__), args.out/'driver-source.py')
            controls = args.out/'original-control-source'; controls.mkdir()
            for name in ('agent.py', 'transformer_session.py', 'confirm.py'):
                shutil.copyfile(Path(serial.__file__).parent.parent/'planners'/name, controls/name)
            shutil.copyfile(Path(original.__file__), controls/'bert_sst2.py')
            shutil.copyfile(Path(serial.__file__).with_name('bert_sst2_worker.py'), controls/'bert_sst2_worker.py')
            save(args.out/'source.json', source)
        claim_path, claim_stream = _claim(args); result['exclusive_c2_claim'] = str(claim_path)
        with recorder.span('independent_model_and_session_shell'):
            before = prepare_branch(args, source)
            result['branch_bindings'] = branch_bindings(args)
        # Recheck after persistence and before admitting the only numeric step.
        if source_gate(args) != source:
            raise ValueError('Original retained source changed before confirmation')
        check_resources(args)
        work = args.out/'phases/001-confirm'; work.mkdir(parents=True); (work/'tmp').mkdir()
        argv = command(args, source, work); save(work/'command.json', argv)
        result['attempted_confirm'] = 1; save(args.out/'result.json', result)
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        with recorder.span('original_c2_confirmation_once'):
            result['process'] = run_process(argv, cwd=paths.REPO/'DSL/DSL_agent/scripts', log=work/'worker.log',
                resources=work/'resources.jsonl', timeout=min(CONFIRM_SECONDS+120, deadline-time.monotonic()),
                mem_gib=4, fsize_gib=8, env=original.child_environment(args, work))
        if (work/'worker_result.json').exists(): result['worker'] = read(work/'worker_result.json')
        _closed_process(result['process'], successful=True)
        if failures: raise RuntimeError(failures[-1])
        if source_gate(args) != source:
            raise ValueError('Original retained source changed during independent confirmation')
        worker = result['worker'] or {}; original.check_worker('step', worker)
        step = worker.get('step') or {}
        if (step.get('step') != 'confirm' or step.get('next_step') != 'done'
                or step.get('session_calls') != 3 or worker.get('llm_calls') != 0
                or worker.get('report_samples') != 0):
            raise ValueError('Exactly one original C2 step with zero new calls/report was required')
        result['completed_confirm'] = 1
        outcome = accepted_branch(args, before)
        result.update(selection_outcome=outcome, frozen=outcome['frozen'],
                      outcome=outcome['outcome'], status='ok' if outcome['frozen'] else 'quality_rejected')
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if watcher is not None: watcher.join(timeout=3)
        for signum in handlers: signal.signal(signum, signal.SIG_IGN)
        try:
            if result['admission_passed']:
                try: check_resources(args)
                except Exception as exc: failures.append(type(exc).__name__+': '+str(exc))
            if failures: result.update(status='error', error=failures[-1], resource_failures=failures)
            result['timing'] = summarize_events(read_events(args.out/'events.jsonl'))
            result['recorded_before_persistence_seconds'] = (time.perf_counter_ns()-started)/1e9
            terminal_case(args, result)
            from ..registry import git_rev
            row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(),
                       metrics=result['metrics'], artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
                if any(json.loads(line).get('run_id') == args.out.name for line in stream if line.strip()):
                    raise RuntimeError('Duplicate independent confirmation registry identity')
                stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            persist_case_completion(args.out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
            if ((args.out/'completion.json').stat().st_size > COMPLETION_RESERVE_BYTES
                    or artifact_bytes(args.out) > CASE_BYTES and result['status'] != 'error'):
                raise RuntimeError('Actual completion exceeded its reserved successful-case allowance')
        finally:
            if claim_stream is not None: claim_stream.close()
            for signum, handler in handlers.items(): signal.signal(signum, handler)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source-case','continuation-case','boundary-snapshot','out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    cli.add_argument('--data-python', type=Path, default=paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent/'paper-data-venv/bin/python')
    cli.add_argument('--ledger', type=Path, default=paths.RESULTS_DIR/'llm_calls.jsonl')
    cli.add_argument('--allowance', type=Path, default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    cli.add_argument('--metadata-template', type=Path, default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    return cli


def main(argv=None):
    cli = parser(); args = cli.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.expanduser().absolute() if name in ('intake_python','data_python') else value.expanduser().resolve())
    if (any(path.parent != paths.RESULTS_DIR.resolve() for path in (args.source_case,args.continuation_case,args.out))
            or len({args.source_case,args.continuation_case,args.out}) != 3):
        cli.error('Use separate direct results cases for original source, closed continuation and new branch')
    result = execute(args)
    print(json.dumps({key: result.get(key) for key in ('status','outcome','frozen','frozen_accepted','error')}), flush=True)
    return 0 if result['status'] in ('ok','quality_rejected') else 1


if __name__ == '__main__': raise SystemExit(main())
