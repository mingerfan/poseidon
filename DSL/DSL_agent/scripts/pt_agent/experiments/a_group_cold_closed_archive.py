"""Resource-only raw/NPZ archives for five exact fresh Rule63 cold cases.

The serial scheduler owns the heavy-job lease. Only the completed case's
work/emit/candidate_trace.mlir and adjacent redundant constants.npz are eligible.
All scientific records, ModelInfo, RuntimePlans, bundles and clocks stay intact.
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
import time

from .. import paths
from . import a_group_future_closed_archive as prior
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events
from .r34_archive_text import archive_one, archive_state

GIB = 1024**3
ROUND = 'A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE'
SOURCE_ROUND = 'COLD-RULE63'
CASES = {
    'a2b-resnet20-rule63-fresh-cold-depth-20261117': 'resnet20_silu',
    'a2b-vgg16-rule63-fresh-cold-depth-20261117': 'vgg16_silu_avgpool',
    'a2b-alexnet-rule63-fresh-cold-depth-20261117': 'alexnet_silu_avgpool',
    'a5-vgg16-rule63-fresh-cold-optimized-20261117': 'vgg16_silu_avgpool',
    'a6-alexnet-rule63-fresh-cold-optimized-20261117': 'alexnet_silu_avgpool',
}
RESERVE_BYTES = 4 * GIB
NPZ_ONLY_CASES = (
    'a2b-vgg16-rule63-fresh-cold-depth-20261117',
    'a2b-alexnet-rule63-fresh-cold-depth-20261117',
)


def require(value, message):
    prior.require(value, message)


def clean_cleanup_metadata(value):
    """Accept compiler errors after cleanup; reject unresolved descendants."""
    if isinstance(value, dict):
        require(not value.get('remaining_pids'), 'Saved descendant cleanup remains unresolved')
        require(not value.get('cleanup_unknown'), 'Saved descendant cleanup remains unknown')
        for child in value.values():
            clean_cleanup_metadata(child)
    elif isinstance(value, list):
        for child in value:
            clean_cleanup_metadata(child)


def admit_npz_only(args):
    """Keep the exact VGG/Alex raw input for its later compiler consumer."""
    require(args.case in NPZ_ONLY_CASES, 'NPZ-only selection requires an exact A2b VGG/Alex source')
    require(not args.no_future_consumers, 'Blanket raw/NPZ no-consumer authority is inapplicable')
    require(args.no_future_selected_npz_consumers and args.future_raw_consumers,
            'Require selected-NPZ-only authority and explicit future raw consumers')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
            'Execute in the approved host process namespace')
    case = paths.RESULTS_DIR.resolve() / args.case
    closure, result = prior.closed(case, case.name)
    require(result.get('round') == SOURCE_ROUND and result.get('model') == CASES[args.case]
            and result.get('placement') == 'depth', 'Original A2b depth source identity differs')
    require(result.get('monitor_stopped') is True, 'Source monitor must be closed')
    clean_cleanup_metadata(result)
    births = prior.producer_births(case)
    emit = case / 'work/emit'
    raw, npz = emit / 'candidate_trace.mlir', emit / 'constants.npz'
    raw_exists, npz_exists = os.path.lexists(raw), os.path.lexists(npz)
    selected = []
    if raw_exists:
        prior.snapshot(raw, archive_source=True)
        require(births > 0, 'Existing raw input requires saved producer birth metadata')
        if npz_exists:
            observation = prior.snapshot(npz, archive_source=True)
            target = npz.with_name(npz.name + '.gz')
            temporary = target.with_name(target.name + '.partial')
            require(not os.path.lexists(target) and not os.path.lexists(temporary),
                    'Preserve previous NPZ archives, partials and symlinks')
            selected.append(dict(source=str(npz), target=str(target),
                                 temporary=str(temporary), stat=observation))
    excluded = {row['source'] for row in selected}
    preserved = [prior.snapshot(path) for path in case.rglob('*') if str(path) not in excluded]
    skip = (None if selected else 'Raw input absent; NPZ retained' if not raw_exists
            else 'Redundant Emit NPZ absent; raw input retained')
    return dict(source_case=str(case), source_closure=closure, source_round=SOURCE_ROUND,
        scientific_status=result['status'], scientific_error=result.get('error'),
        producer_births_closed=births, host_pid_namespace=args.host_pid_namespace,
        selection_mode='npz-only-preserve-raw', no_future_selected_NPZ_consumers=True,
        future_raw_consumers=True, preserved_raw=str(raw), selected=selected,
        preserved=preserved, skip_reason=skip,
        scope='Only redundant Emit constants.npz; raw remains for later same-program compilation')


def admit(args):
    if getattr(args, 'npz_only_preserve_raw', False):
        return admit_npz_only(args)
    require(args.case in CASES, 'Only five exact approved cold source cases are eligible')
    require(args.no_future_consumers, 'Explicit no-future-raw/NPZ-consumer authority required')
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
            'Execute in the approved host process namespace')
    case = paths.RESULTS_DIR.resolve() / args.case
    closure, result = prior.closed(case, case.name)
    require(result.get('round') == SOURCE_ROUND and result.get('model') == CASES[args.case],
            'Original fresh Rule63 source round/model differs')
    require(result.get('monitor_stopped') is not False, 'Source monitor remains active')
    clean_cleanup_metadata(result)
    births = prior.producer_births(case)
    emit = case / 'work/emit'
    raw, npz = emit / 'candidate_trace.mlir', emit / 'constants.npz'
    raw_exists = os.path.lexists(raw)
    if raw_exists:
        require(births > 0, 'Existing raw trace requires saved producer birth metadata')
    selected = []
    for path in (raw, npz) if raw_exists else ():
        if not os.path.lexists(path):
            continue
        stat = prior.snapshot(path, archive_source=True)
        target = path.with_name(path.name + '.gz')
        temporary = target.with_name(target.name + '.partial')
        require(not os.path.lexists(target) and not os.path.lexists(temporary),
                'Preserve previous archives, partials and symlinks')
        selected.append(dict(source=str(path), target=str(target),
                             temporary=str(temporary), stat=stat))
    excluded = {row['source'] for row in selected}
    preserved = [prior.snapshot(path) for path in case.rglob('*') if str(path) not in excluded]
    return dict(source_case=str(case), source_closure=closure, source_round=SOURCE_ROUND,
        scientific_status=result['status'], scientific_error=result.get('error'),
        producer_births_closed=births, host_pid_namespace=args.host_pid_namespace,
        no_future_consumers=True, selected=selected, preserved=preserved,
        skip_reason=None if raw_exists else 'Raw trace absent; raw/NPZ both skipped; scientific status retained',
        scope='Only exact work/emit raw/NPZ pair; all other scientific files and directory identities retained')


def append_registry(out, result):
    row = dict(run_id=out.name, round=ROUND, model=result['source_kind'],
        mode='resource-only-reversible-gzip', git_rev=None,
        metrics={key: result.get(key) for key in ('status', 'resource_status',
            'scientific_status', 'completed_files', 'released_bytes')},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        require(not any(json.loads(line).get('run_id') == out.name
            for line in stream if line.strip()), 'Preserve previous resource archive identity')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    require(args.run, 'Explicit resource-only --run required')
    require(1 <= args.seconds <= 1200, 'Finite archive allowance1..1200s required')
    out = args.out.absolute()
    require(out.parent == paths.RESULTS_DIR.resolve() and not os.path.lexists(out),
            'Fresh direct resource-only results child required')
    out.mkdir()
    started = time.perf_counter_ns()
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, source_kind=args.case,
        source_case=str(paths.RESULTS_DIR.resolve() / args.case), status='running', error=None,
        resource_status='pending', scientific_status=None, scientific_status_upgraded=False,
        files=[], completed_files=0, released_bytes=0, in_progress=None,
        api_calls=0, model_Count_compile_CKKS=0, integrity_checks=0,
        output_readback_checks=0, heavy_lease_acquired=False,
        limits=dict(address_space_gib=1, seconds=args.seconds, minimum_free_gib=4, concurrency=1),
        clock_scope='Independent resource reclamation only; original scientific clocks unchanged')
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    def interrupted(sig, frame):
        result['incoming_signal'] = sig
        raise InterruptedError('Cold source archive interrupted by signal ' + str(sig))
    for sig in old:
        signal.signal(sig, interrupted)
    try:
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        limit = GIB if hard == resource.RLIM_INFINITY else min(GIB, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        signal.setitimer(signal.ITIMER_REAL, args.seconds)
        with recorder.span('closed_metadata_admission'):
            admission = admit(args)
        result['source_admission'] = admission
        result['scientific_status'] = admission['scientific_status']
        if admission.get('selection_mode') == 'npz-only-preserve-raw':
            result.update(selection_mode='npz-only-preserve-raw',
                no_future_selected_NPZ_consumers=True, future_raw_consumers=True)
        prior.durable(out / 'admission.json', admission)
        if admission['skip_reason']:
            result.update(status='ok', resource_status='metadata_skip', skip_reason=admission['skip_reason'])
        else:
            for row in admission['selected']:
                source, target, temporary = (Path(row[name]) for name in ('source', 'target', 'temporary'))
                require(prior.snapshot(source, archive_source=True) == row['stat'],
                        'Original raw/NPZ stat changed after admission')
                require(not os.path.lexists(target) and not os.path.lexists(temporary),
                        'Archive destinations must remain fresh')
                prior.preserve(admission['preserved'])
                required = RESERVE_BYTES + row['stat']['bytes'] * 101 // 100 + 1024**2
                require(shutil.disk_usage(source.parent).free >= required,
                        'resource_limit: free4GiB plus101% raw/NPZ temporary allowance')
                result['in_progress'] = archive_state(source, target, temporary)
                prior.durable(out / 'progress.json', result)
                with recorder.span('gzip_one_redundant_file', source=str(source)):
                    observed = archive_one(source, target, temporary)
                require(observed['status'] in ('compressed', 'kept_uncompressed'),
                        'Original archive primitive did not complete')
                prior.preserve(admission['preserved'])
                result['files'].append(observed)
                result['completed_files'] = len(result['files'])
                result['released_bytes'] = sum(item['released_bytes'] for item in result['files'])
                result['in_progress'] = None
                prior.durable(out / 'progress.json', result)
            result.update(status='ok', resource_status='archived_or_kept')
    except BaseException as exc:
        result.update(status='error', resource_status='failed', error=type(exc).__name__ + ': ' + str(exc))
        if result['in_progress']:
            result['interrupted_state'] = archive_state(source, target, temporary)
    finally:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['timing'] = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
        result['resource_only_elapsed_s'] = (time.perf_counter_ns() - started) / 1e9
        prior.durable(out / 'result.json', result)
        append_registry(out, result)
        marker = persist_case_completion(out, round_label=ROUND, started_ns=started,
            status=result['status'], timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
        prior.durable(out / 'completion.json', marker)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--host-pid-namespace', required=True)
    parser.add_argument('--seconds', type=int, default=1140)
    parser.add_argument('--no-future-consumers', action='store_true')
    parser.add_argument('--npz-only-preserve-raw', action='store_true')
    parser.add_argument('--no-future-selected-npz-consumers', action='store_true')
    parser.add_argument('--future-raw-consumers', action='store_true')
    parser.add_argument('--run', action='store_true')
    result = run(parser.parse_args(argv))
    print(json.dumps({key: result[key] for key in ('status', 'resource_status',
        'scientific_status', 'completed_files', 'released_bytes')}))
    return int(result['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
