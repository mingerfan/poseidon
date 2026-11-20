"""One serial, guarded warm BERT execution-equivalence pair after prior closure.

System outer only: closed source/registry/birth admission and original heavy
lease. The separate pure numeric worker receives explicit paths, never a global
registry or an Agent/provider. This experiment makes no cold two-hour claim.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import bert_frozen_fork_report as reference
from . import bert_fresh_cold_2h as fresh
from .archive_producer_closure import closed_case, registry_rows, _process
from .autonomous_serial_batch import acquire_lease
from .closed_success_raw_trace_archive import producer_births
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, _scan, read_events, run_process, summarize_events


ROUND = 'BERT-ORDERED-MOCK-PAIR40'
MODEL = 'bert-tiny-sst2'
MODE = 'warm-original-frozen-heldout40-ordered-ten-row-execution-equivalence'
MODULE = 'pt_agent.experiments.bert_ordered_mock_pair'
WORKER = MODULE+'_worker'
GIB = 1024**3
SECONDS, WORKER_SECONDS = 2400, 1800
CASE_BYTES, RESERVE = 4*GIB, 1024**2
ZERO = ('API', 'Count', 'compiler_runs', 'keygen_runs', 'ckks_runs', 'report_samples',
        'native_test_reads', 'downloads', 'installations', 'integrity_checks')
read, save = reference.read, reference.save


def require(condition, message):
    if not condition:
        raise ValueError(message)


def option(argv, name):
    require(argv.count(name) == 1 and argv.index(name)+1 < len(argv), 'One original option required: '+name)
    return argv[argv.index(name)+1]


def owner_gate(args):
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'), 'Use the actual host PID namespace')
    live = _process(args.prior_pid)
    require(live is None or live['starttime'] != args.prior_starttime,
            'The actual prior controller birth must be gone, including zombies')
    owner, _ = closed_case(args.prior_case, ('ok', 'error', 'completed_with_observations'))
    require(owner.get('round') == 'AUTONOMOUS-SERIAL'
        and owner.get('mode') == 'finite-trusted-serial-experiments', 'Original finite serial owner required')
    ledger = registry_rows(paths.RESULTS_DIR/'registry.jsonl', {args.prior_case.name: args.prior_case})[args.prior_case.name]
    # Serial owner results deliberately omit model/metrics top-level fields.
    # Their registry uses the original queue model and the status/jobs pair.
    require(ledger.get('round') == owner['round'] and ledger.get('mode') == owner['mode']
        and ledger.get('model') == 'original-model-experiment-queue'
        and ledger.get('metrics') == dict(status=owner['status'], jobs=owner.get('jobs')),
        'Actual model-less queue result and eight-field queue registry must agree')
    supervisor = read(args.prior_case/'supervisor.json')
    require((supervisor.get('process') or {}).get('pid') == supervisor.get('pid') == args.prior_pid
        and supervisor['process'].get('starttime') == args.prior_starttime, 'Exact saved prior controller birth required')
    jobs = owner.get('jobs') or []
    declared = read(args.prior_case/'manifest.json').get('jobs') or []
    require(len(jobs) == len(declared) == 2, 'Both original fresh and independent native jobs must close')
    for job, wanted in zip(jobs, declared):
        proc = job.get('process') or {}
        require(job.get('id') == wanted.get('id') and job.get('out') == wanted.get('out')
            and job.get('argv') == wanted.get('argv')
            and Path(option(job['argv'], '--out')).resolve() == Path(job['out']).resolve()
            and read(args.prior_case/(job['id']+'-command.json')) == job['argv'],
            'Original manifest and actual saved job must match')
        require(not job.get('cleanup_unknown') and job.get('signal') is None
            and type(proc.get('exit_code')) is int and type(proc.get('timed_out')) is bool
            and proc.get('signal') is None and proc.get('remaining_pids') == []
            and isinstance(proc.get('signals_sent'), list), 'Every prior job needs resolved actual process cleanup')
    require(Path(jobs[0]['out']).resolve() == args.first_case
        and option(jobs[0]['argv'], '-m') == fresh.MODULE
        and option(jobs[1]['argv'], '-m') == 'pt_agent.experiments.bert_rule63_shared_optimized_compile',
        'Keep the original fresh-then-independent-current-Rule63 owner family')
    native = jobs[1]['argv']
    require(('--native-seconds' not in native or int(option(native, '--native-seconds')) == 7200)
        and ('--seconds' not in native or int(option(native, '--seconds')) == 9600)
        and declared[1].get('seconds') == 9720,
        'Keep the full original7200native plus input-preparation/cleanup allowance')
    return dict(case=str(args.prior_case), pid=args.prior_pid, starttime=args.prior_starttime,
                status=owner['status'], actual_jobs_closed=2, registry_unique_eight_fields=True,
                native_case=str(Path(jobs[1]['out']).resolve()))


def first_failure(args):
    result, completion = closed_case(args.first_case, ('error',))
    reference.registry_row(args.first_case, result)
    record = read(args.first_case/'two-hour-record.json')
    require(result.get('round') == fresh.ROUND and result.get('mode') == fresh.MODE
        and result.get('model') == record.get('model') == MODEL
        and record.get('round') == fresh.ROUND and record.get('mode') == fresh.MODE
        and record.get('compile_accepted') is False
        and result.get('compile_accepted') is False
        and record.get('persisted_pipeline_within_two_hours') is False
        and record.get('started_ns') == completion.get('started_ns')
        and record.get('required_artifacts_and_registry_completed_ns') == completion.get('ended_ns'),
        'Retain the actual same-source failed fresh cold clock')
    last = result['processes'][-1]
    observation = last.get('observation') or {}
    seconds = int(option(last['argv'], '--seconds'))
    elapsed = (observation['ended_ns']-observation['started_ns'])/1e9
    require(last.get('phase') == 'step' and option(last['argv'], '-m') == 'pt_agent.tools.nixrun'
        and observation.get('exit_code') == 124 and observation.get('remaining_pids') == []
        and observation.get('timed_out') is False and observation.get('signal') is None
        and observation.get('signals_sent') == []
        and seconds <= elapsed < seconds+15
        and result.get('wall_policy', {}).get('search_absolute_ceiling_seconds') == 5400
        and result.get('wall_policy', {}).get('total_seconds') == 7200
        and all(result['attempted'][key] == 0 for key in ('confirm', 'math', 'emit', 'trace', 'compile')),
        'The retained actual need must be the original search wrapper timeout before later phases')
    return dict(case=str(args.first_case), authoritative_record=str(args.first_case/'two-hour-record.json'),
                usable_within_two_hours=False, original_status=result['status'],
                actual_case_s=completion['complete_case_elapsed_seconds'], inner_search_seconds=seconds,
                inner_exit_code=124, scientific_rows_modified=False)


def case_paths(args, *, created=False):
    root = paths.RESULTS_DIR.resolve()
    cases = (args.source_case, args.compile_case, args.first_case, args.prior_case, args.out)
    require(len(set(cases)) == 5 and all(p.parent == root and p.resolve() == p for p in cases),
            'Use five distinct canonical direct result cases')
    require(created or not args.out.exists(), 'Preserve any prior execution-equivalence case')
    return root


def source_gate(args, *, created=False):
    root = case_paths(args, created=created)
    owner = owner_gate(args)
    failed = first_failure(args)
    require((args.compile_case/'outer-admission.json').is_file(),
            'The old compiler outer proof must already exist and remain read-only')
    # Reuse the existing full same-weight C2/checkpoint/cal3/native metadata
    # admission. This call reads metadata only; its report worker is not called.
    original = reference.admission(args)
    lineage = original['source']['fork_provenance']
    current_native = Path(owner['native_case'])
    native_result, _ = closed_case(current_native, ('ok', 'error'))
    reference.registry_row(current_native, native_result)
    source_cases = {args.source_case, args.compile_case, args.prior_case, args.first_case, current_native,
        Path(original['native_case']).resolve(), Path(lineage['original_source_case']).resolve(),
        Path(lineage['continuation_case']).resolve()}
    require(all(p.parent == root for p in source_cases), 'All saved producer families stay in the result root')
    births = {str(case): producer_births(case) for case in sorted(source_cases)}
    require(all(type(count) is int and count > 0 for count in births.values()),
            'Every saved source/owner family must have actual closed positive producer births')
    return dict(schema='bert-ordered-original-mock-pair-admission-v1', status='ok',
        source_case=str(args.source_case), model_dir=str(args.source_case/'model'),
        program=original['program'], frozen=str(args.source_case/'session/FROZEN.json'),
        model=original['model'], inputs=original['inputs'], metadata_template=str(args.metadata_template),
        source_closed=True, owner_queue_closed=True, all_source_births_closed=True,
        source_registry_unique_eight_fields=True, logical_heldout_rows=40, report_read_capability=False,
        owner=owner, original_failed_fresh=failed, source_producer_counts=births,
        reference_compile_case=str(args.compile_case), reference_native_case=original['native_case'],
        reference_checkpoint=original['source']['checkpoint_audit'],
        intake_python=str(args.intake_python), data_python=str(args.data_python),
        model_source_unchanged=True, report_or_compiler_feedback_to_Agent=False)


@contextmanager
def heavy_lease(args, result):
    with (paths.RESULTS_DIR/'autonomous-heavy-experiments.lock').open('a') as lease:
        acquire_lease(lease, 60)
        identity = _process(os.getpid())
        require(identity and identity.get('starttime'), 'The actual equivalence owner birth is required')
        result['heavy_lease'] = dict(acquired=True, released=False, owner=identity)
        save(args.out/'supervisor.json', dict(pid=os.getpid(), process=identity,
                                            lease=result['heavy_lease']))
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            result['heavy_lease']['released'] = True


def resources(args):
    from .r27_orbit_lazy_backend import _available_ram_bytes
    from .r31_lattigo_runtime_plan import physical_memory_observation
    table = _scan()
    owned = {os.getpid()}
    while True:
        more = owned | {pid for pid, row in table.items() if row.get('ppid') in owned}
        if more == owned:
            break
        owned = more
    sample = dict(subtree_rss_bytes=sum(table.get(pid, {}).get('vmrss_kib', 0)*1024 for pid in owned))
    value = physical_memory_observation(sample, _available_ram_bytes(), rss_gib=14, minimum_available_gib=1.25)
    value.update(case_bytes=artifact_bytes(args.out), disk_free_bytes=shutil.disk_usage(args.out).free)
    if value['case_bytes']+RESERVE > CASE_BYTES:
        value['reasons'].append('4GiB metadata case with completion reserve exhausted')
    if value['disk_free_bytes'] < 10*GIB:
        value['reasons'].append('10GiB disk reserve exhausted')
    value['stop'] = bool(value['reasons'])
    return value


def command(args, proof, seconds):
    argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds),
            '--python', str(args.intake_python), '--', '-B', '-m', WORKER, '--run-worker',
            '--source-admission', str(args.out/'source-admission.json'), '--work-dir', str(args.out/'worker')]
    for key in ('model_dir', 'program', 'frozen', 'model', 'inputs', 'metadata_template'):
        argv += ['--'+key.replace('_', '-'), proof[key]]
    return argv


def check_worker(args, value, proof):
    require(value.get('schema') == 'bert-ordered-original-mock-pair-result-v1'
        and value.get('status') == 'ok' and value.get('pair_accepted') is True
        and value.get('ordered_execution_observed') is True and value.get('source') == proof
        and value.get('source_admission') == str(args.out/'source-admission.json')
        and value.get('attempted') == value.get('completed') == dict(tensor=1, original_mock=1, ordered_mock=1)
        and value.get('logical_heldout_rows') == 40 and value.get('original_mock_batch_size') == 10
        and value.get('requested_workers') == 3
        and value.get('cold_chain_measurement') is False and value.get('qualifies_two_hour_claim') is False
        and value.get('speedup') is None and all(value.get(key) == 0 for key in ZERO),
        'One complete actual numerical pair with all original scope and zero external actions is required')
    checked = value.get('pair_checks') or {}
    mathematics = checked.get('output_mathematics') or {}
    require(mathematics.get('passed') is True
        and mathematics.get('criterion') == reference.compiled.base.original.CRITERION
        and all(type(mathematics.get(key)) in (int, float) and math.isfinite(mathematics[key])
                and mathematics[key] >= 0 for key in ('max_absolute_error', 'max_tolerance_ratio'))
        and mathematics['max_tolerance_ratio'] <= 1
        and all(checked.get(key) is True for key in ('full_metrics_equal',
            'original_domain_diagnostics_equal', 'original_event_indices_equal')),
        'Original full outputs, metrics and diagnostics must all pass')


def sampled_resources(args):
    peak, samples, births = 0, 0, set()
    with (args.out/'worker-resources.jsonl').open() as stream:
        for line in stream:
            if not line.strip():
                continue
            require(line.endswith('\n'), 'Complete original resource samples are required')
            value = json.loads(line)
            outer = value.get('outer_process') or {}
            require(outer.get('pid') == os.getpid() and isinstance(value.get('processes'), list),
                    'Original numerical-pair resource owner differs')
            rows = value['processes']
            peak = max(peak, sum(row.get('vmrss_kib', 0) for row in [outer, *rows]))
            for row in rows:
                require(type(row.get('pid')) is int and row['pid'] > 0
                    and type(row.get('starttime')) is int and row['starttime'] > 0,
                        'Actual numerical producer PID/birth required')
                births.add((row['pid'], row['starttime']))
            samples += 1
    require(samples > 0 and births, 'The complete paired process tree must be sampled')
    for pid, birth in births:
        current = _process(pid)
        require(current is None or current.get('starttime') != birth,
                'A numerical-pair producer remains present after supervision')
    return dict(max_sampled_subtree_rss_gib=peak/1024**2, snapshots=samples,
                saved_numeric_births_closed=len(births), complete_samples=True,
                sampled_rss_policy_passed=peak/1024**2 <= 14,
                scope='Maximum simultaneous saved sum of owner and child RSS; shared fork pages count per process')


def register(args, result, started):
    from ..registry import git_rev
    result['metrics'] = {key: result[key] for key in ('status', 'pair_accepted', 'attempted_pair',
        'completed_pair', 'cold_chain_measurement', 'qualifies_two_hour_claim', *ZERO)}
    result['timing'] = summarize_events(read_events(args.out/'events.jsonl'))
    save(args.out/'result.json', result)
    require(artifact_bytes(args.out)+RESERVE <= CASE_BYTES or result['status'] == 'error',
            'Reserve completion space for the actual pair case')
    row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(),
               metrics=result['metrics'], artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        require(not any(json.loads(line).get('run_id') == args.out.name for line in stream if line.strip()),
                'Preserve the unique execution-equivalence registry identity')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    persist_case_completion(args.out, round_label=ROUND, started_ns=started, status=result['status'],
        timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))


def execute(args):
    case_paths(args)
    require(time.time()+SECONDS+30 < args.deadline_timestamp, 'The full finite pair allowance must fit the publication deadline')
    args.out.mkdir()
    started = time.perf_counter_ns()
    deadline = time.monotonic()+SECONDS
    recorder = StageRecorder(args.out/'events.jsonl')
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        pair_accepted=False, attempted_pair=0, completed_pair=0, source=None, worker=None,
        process=None, guard_trigger=None, guard_persistence_error=None, incoming_signal=None, monitor_stopped=None,
        cold_chain_measurement=False, qualifies_two_hour_claim=False, speedup=None,
        limits=dict(outer_seconds=SECONDS, worker_seconds=WORKER_SECONDS, address_space_gib=24,
                    subtree_RSS_gib=14, host_reserve_gib=1.25, file_gib=4, case_gib=4, free_disk_gib=10),
        **dict.fromkeys(ZERO, 0))
    stop = threading.Event()
    watcher = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        if result['guard_trigger'] is None:
            result['incoming_signal'] = signum
        raise InterruptedError('Execution-equivalence owner interrupted '+str(signum))

    def monitor():
        while not stop.wait(.25):
            try:
                value = resources(args)
                if time.monotonic() >= deadline:
                    value['reasons'].append('Execution-equivalence outer wall allowance exhausted')
                    value['stop'] = True
                if not value['stop']:
                    continue
                result['guard_trigger'] = value
            except BaseException as exc:
                result['guard_trigger'] = dict(error=type(exc).__name__+': '+str(exc))
            try:
                save(args.out/'guard-stop.json', result['guard_trigger'])
            except BaseException as exc:
                result['guard_persistence_error'] = type(exc).__name__+': '+str(exc)
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
            return

    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        with heavy_lease(args, result), recorder.span('complete_warm_execution_pair'):
            require(not reference.serial.foreign_heavy_processes(), 'Another numeric/compiler job is active')
            with recorder.span('closed_sources_and_actual_producer_admission'):
                proof = source_gate(args, created=True)
            result['source'] = proof
            save(args.out/'source-admission.json', proof)
            result['resource_admission'] = resources(args)
            require(not result['resource_admission']['stop'], 'Execution pair resource admission failed')
            for name in ('bert_ordered_mock_pair.py', 'bert_ordered_mock_pair_worker.py', 'ordered_mock_validation.py'):
                shutil.copyfile(Path(__file__).with_name(name), args.out/name)
            allowance = min(WORKER_SECONDS, int(deadline-time.monotonic()-30))
            require(allowance == WORKER_SECONDS, 'Keep the complete original finite worker allowance')
            argv = command(args, proof, allowance)
            save(args.out/'command.json', argv)
            result['attempted_pair'] = 1
            save(args.out/'result.json', result)
            with recorder.span('original_and_ordered_numeric_pair_once'):
                result['process'] = run_process(argv, cwd=paths.REPO/'DSL/DSL_agent/scripts',
                    log=args.out/'worker.log', resources=args.out/'worker-resources.jsonl',
                    timeout=allowance+15, mem_gib=24, fsize_gib=4)
            require(fresh._clean(result['process']), 'The original numerical pair must close cleanly')
            value = read(args.out/'worker/pair-result.json')
            result['worker'] = value
            check_worker(args, value, proof)
            result['resource_sampling'] = sampled_resources(args)
            require(result['resource_sampling'].get('sampled_rss_policy_passed') is True,
                    'The actual sampled process-tree RSS exceeded the original14GiB policy')
            require(source_gate(args, created=True) == proof, 'Preserve all original source/owner provenance during the pair')
            result['resource_completion'] = resources(args)
            require(not result['resource_completion']['stop'], 'Execution pair completion resources failed')
            require(time.monotonic() < deadline, 'Execution pair outer wall allowance exhausted')
            result.update(status='ok', pair_accepted=True, completed_pair=1)
    except BaseException as exc:
        result.update(status='error', pair_accepted=False, error=type(exc).__name__+': '+str(exc))
    finally:
        # R10 has already reaped the numeric descendants. Keep cancellation
        # masked until this owner has stopped its monitor and persisted closure.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        stop.set()
        if watcher is not None:
            watcher.join(timeout=3)
        result['monitor_stopped'] = watcher is None or not watcher.is_alive()
        if result['guard_trigger'] or not result['monitor_stopped']:
            result.update(status='error', pair_accepted=False)
        if result['worker'] is None and (args.out/'worker/pair-result.json').is_file():
            result['worker'] = read(args.out/'worker/pair-result.json')
        try:
            register(args, result, started)
        finally:
            if result['monitor_stopped']:
                for sig, old in handlers.items():
                    signal.signal(sig, old)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source-case', 'compile-case', 'first-case', 'prior-case', 'out', 'metadata-template'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--prior-pid', type=int, required=True)
    cli.add_argument('--prior-starttime', type=int, required=True)
    cli.add_argument('--host-pid-namespace', required=True)
    cli.add_argument('--deadline-utc', required=True)
    cli.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    cli.add_argument('--data-python', type=Path, default=paths.PAPER_DATA_PYTHON)
    cli.add_argument('--ledger', type=Path, default=paths.RESULTS_DIR/'llm_calls.jsonl')
    cli.add_argument('--allowance', type=Path, default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    return cli


def normalize_args(args):
    if args.data_python is None:
        saved = read(args.source_case/'model/samples/source.json')
        require(isinstance(saved.get('reader_python'), str) and bool(saved['reader_python']),
                'Infer the existing SST-2 reader from its original saved source record')
        args.data_python = Path(saved['reader_python'])
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving a venv interpreter symlink selects its base Python and
            # loses the existing environment's installed numeric packages.
            setattr(args, name, value.expanduser().absolute() if name.endswith('_python')
                    else value.expanduser().resolve())
    return args


def main(argv=None):
    args = normalize_args(parser().parse_args(argv))
    from .autonomous_serial_batch import deadline_timestamp
    args.deadline_timestamp = deadline_timestamp(args.deadline_utc)
    result = execute(args)
    print(json.dumps(dict(status=result['status'], pair_accepted=result['pair_accepted'], error=result['error'])))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
