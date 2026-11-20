"""Archive only the future closed BERT compile case's recovered raw input.

This independent resource job keeps the scientific status, clocks, plans,
bundles, DSL and original source gzip unchanged. The serial scheduler owns the
heavy lease. No future queued Qwen job consumes this recovered BERT input.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import threading
import time

from .. import paths
from . import a_group_future_closed_archive as prior
from . import a_group_cold_closed_archive as closed_helpers
from . import bert_rule63_shared_optimized_compile as producer
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events
from .r34_archive_text import archive_one, archive_state

GIB = 1024 ** 3
ROUND = 'BERT-RULE63-CLOSED-RESTORED-INPUT-ARCHIVE'
MODULE = 'pt_agent.experiments.bert_rule63_restored_input_archive'
SOURCE_ID = 'bert-rule63-current-shared-optimized-qwen48-20261117'


def require(value, message):
    if not value:
        raise ValueError(message)


def admit(args):
    require(args.no_future_restored_input_consumers, 'Explicit absence of future recovered-input consumers required')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'), 'Use the selected host process namespace')
    case = paths.RESULTS_DIR.resolve() / SOURCE_ID
    closure, result = prior.closed(case, case.name)
    require(result.get('round') == producer.ROUND and result.get('mode') == producer.MODE
            and result.get('model') == producer.MODEL and result.get('monitor_stopped') is True,
            'Only the exact completed BERT optimized producer is eligible')
    closed_helpers.clean_cleanup_metadata(result)
    with (paths.RESULTS_DIR / 'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        records = [json.loads(line) for line in stream if line.strip()]
    ledger = next(row for row in records if row.get('run_id') == case.name)
    require(all(ledger.get(k) == result[k] for k in ('round', 'model', 'mode')),
            'Original BERT producer registry identity differs')
    births = prior.producer_births(case)
    raw = case / 'input/candidate_trace.mlir'
    selected = []
    if not os.path.lexists(raw):
        skip = 'No recovered input was produced; scientific failure remains unchanged'
    elif raw.is_symlink():
        skip = 'Original raw was referenced by a symlink; no recovered local copy is selected'
    else:
        require(births > 0, 'Existing recovered input requires closed saved producer birth evidence')
        observation = prior.snapshot(raw, archive_source=True)
        require(observation['bytes'] <= 16 * GIB, 'Recovered input must fit its original16GiB file allowance')
        target = raw.with_name(raw.name + '.gz'); temporary = target.with_name(target.name + '.partial')
        require(not os.path.lexists(target) and not os.path.lexists(temporary),
                'Preserve existing recovered-input archives and partials')
        selected.append(dict(source=str(raw), target=str(target), temporary=str(temporary), stat=observation))
        skip = None
    excluded = {row['source'] for row in selected}
    preserved = [prior.snapshot(path) for path in case.rglob('*') if str(path) not in excluded]
    original = paths.RESULTS_DIR.resolve() / producer.SOURCE_ID / 'work/emit/candidate_trace.mlir.gz'
    require(original.is_file() and not original.is_symlink(), 'The original source gzip must remain present')
    preserved.append(prior.snapshot(original))
    return dict(source_case=str(case), source_closure=closure, scientific_status=result['status'],
        scientific_error=result.get('error'), producer_births_closed=births,
        host_pid_namespace=args.host_pid_namespace, no_future_restored_input_consumers=True,
        selected=selected, preserved=preserved, skip_reason=skip,
        scope='Only the recovered local raw; complete or partial input after clean producer closure; all scientific outcomes retained')


def append_registry(out, result):
    row = dict(run_id=out.name, round=ROUND, model='bert_tiny', mode='resource-only-reversible-restored-input-gzip',
        git_rev=None, metrics={k:result.get(k) for k in ('status','resource_status','scientific_status','released_bytes')},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()),
                'Preserve prior resource archive identity')
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def run(args):
    require(args.run and 1 <= args.seconds <= 1800, 'Explicit finite resource-only archive invocation required')
    out = args.out.absolute()
    require(out.parent == paths.RESULTS_DIR.resolve() and not os.path.lexists(out), 'Fresh direct resource case required')
    out.mkdir(); started = time.perf_counter_ns(); recorder = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, status='running', error=None, resource_status='pending', scientific_status=None,
        source_case=str(paths.RESULTS_DIR.resolve() / SOURCE_ID), scientific_status_upgraded=False,
        files=[], in_progress=None, completed_files=0, released_bytes=0, guard_error=None, monitor_stopped=False,
        heavy_lease_acquired=False, model_runs=0, Count_runs=0, native_runs=0, API_calls=0, CKKS_runs=0,
        CRC_checks=0, readback_checks=0, hash_checks=0,
        clock_scope='Independent resource reclamation after producer closure; original science clocks unchanged')
    stop = threading.Event(); active = threading.Event()
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGXFSZ)
    old = {sig:signal.getsignal(sig) for sig in signals}
    def interrupted(sig, frame):
        result['incoming_signal'] = sig
        raise InterruptedError('Recovered-input archive interrupted by signal ' + str(sig))
    def monitor():
        while not stop.wait(.2):
            try:
                require(shutil.disk_usage(out).free >= 4 * GIB, 'resource_limit: free4GiB reserve exhausted')
            except BaseException as exc:
                result['guard_error'] = type(exc).__name__ + ': ' + str(exc)
                if active.is_set(): os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    try:
        for sig in signals: signal.signal(sig, interrupted)
        for key, cap in ((resource.RLIMIT_AS, GIB), (resource.RLIMIT_FSIZE, 16 * GIB)):
            hard = resource.getrlimit(key)[1]
            bound = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
            resource.setrlimit(key, (bound, bound))
        signal.setitimer(signal.ITIMER_REAL, args.seconds); watcher.start()
        with recorder.span('closed_producer_metadata_admission'): proof = admit(args)
        result['source_admission'] = proof; result['scientific_status'] = proof['scientific_status']
        prior.durable(out / 'admission.json', proof)
        if proof['skip_reason']:
            result.update(status='ok', resource_status='metadata_skip', skip_reason=proof['skip_reason'])
        else:
            row = proof['selected'][0]
            source,target,temporary = (Path(row[key]) for key in ('source','target','temporary'))
            require(prior.snapshot(source, archive_source=True) == row['stat'], 'Recovered raw stat changed after admission')
            require(not os.path.lexists(target) and not os.path.lexists(temporary), 'Fresh archive destinations required')
            prior.preserve(proof['preserved'])
            require(shutil.disk_usage(source.parent).free >= 4 * GIB, 'resource_limit: free4GiB required')
            result['in_progress'] = archive_state(source,target,temporary); prior.durable(out / 'progress.json', result)
            with recorder.span('archive_one_recovered_raw'):
                active.set()
                try: observed = archive_one(source,target,temporary)
                finally: active.clear()
            require(observed['status'] in ('compressed','kept_uncompressed'), 'Original R34 archive primitive failed')
            require(not result['guard_error'], 'Resource guard rejected archival completion')
            prior.preserve(proof['preserved'])
            result.update(status='ok', resource_status='archived_or_kept', files=[observed],
                completed_files=1, released_bytes=observed['released_bytes'], in_progress=None)
    except BaseException as exc:
        result.update(status='error', resource_status='failed', error=type(exc).__name__ + ': ' + str(exc))
        if result['in_progress']: result['interrupted_state'] = archive_state(source,target,temporary)
    finally:
        for sig in signals: signal.signal(sig, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0); stop.set()
        if watcher.is_alive(): watcher.join(timeout=3)
        result['monitor_stopped'] = not watcher.is_alive()
        if result['guard_error'] or not result['monitor_stopped']:
            result.update(status='error', resource_status='failed', error=result['guard_error'] or 'Archive monitor did not stop')
        result['timing'] = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
        prior.durable(out / 'result.json', result); append_registry(out,result)
        marker = persist_case_completion(out, round_label=ROUND, started_ns=started,
            status=result['status'], timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
        prior.durable(out / 'completion.json', marker)
        if result['monitor_stopped']:
            for sig, handler in old.items(): signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--host-pid-namespace', required=True)
    parser.add_argument('--no-future-restored-input-consumers', action='store_true')
    parser.add_argument('--seconds', type=int, default=1740)
    parser.add_argument('--run', action='store_true')
    result = run(parser.parse_args(argv))
    print(json.dumps({k:result[k] for k in ('status','resource_status','scientific_status','released_bytes')}))
    return int(result['status'] != 'ok')


if __name__ == '__main__': raise SystemExit(main())
