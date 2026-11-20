"""Restore only the recorded official VGG bundle, then run its existing math case.

The archive inventory supplies names, kinds and byte lengths. It supplies no
content validation. The original model, official bias omission and physical32
polynomial are evaluated only by the unchanged dacapo_vgg16_plan_math entry.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import signal
import stat
import sys
import tarfile
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events

ROUND = 'DACAPO-VGG16-RESTORE-AND-OFFICIAL-MATH'
MODEL = 'dacapo-original-vgg16-silu-avgpool'
MODE = 'recorded-runtime-bundle-restore-then-original-three-train-math'
GIB = 1024**3
RESTORE_SECONDS = 300
MATH_WRAPPER_SECONDS = 2040
TOTAL_SECONDS = 2400
EXPECTED_FILES = 9717
EXPECTED_BYTES = 5095021924
STAGING_GIB = 6
OWN_GIB = .5


def read(path):
    return json.loads(Path(path).read_text())


def durable(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.resolve() != path:
        raise ValueError('Use an absolute path with no symbolic-link components')
    return path


def _member_name(name, *, directory=False):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('Invalid bundle inventory or archive member name')
    if directory:
        name = name.rstrip('/')
    parts = name.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(name).is_absolute():
        raise ValueError('Bundle members must have safe relative names')
    return PurePosixPath(name).as_posix()


def validate_inventory(rows, bundle_name):
    """Return the exact recorded tree, including its single bundle root."""
    if (not isinstance(bundle_name, str) or '/' in bundle_name or '\\' in bundle_name
            or not bundle_name.endswith('.bundle') or bundle_name in ('.', '..')):
        raise ValueError('One named runtime bundle is required')
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty recorded bundle inventory is required')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('kind') not in ('file', 'directory'):
            raise ValueError('Only regular files and directories are restorable')
        name = _member_name(row.get('path'))
        if name.split('/')[0] != bundle_name or name in result:
            raise ValueError('Duplicate or non-bundle inventory member')
        size = row.get('bytes')
        if type(size) is not int or size < 0 or (row['kind'] == 'directory' and size != 0):
            raise ValueError('Inventory byte lengths must be nonnegative integers')
        if row['kind'] == 'file' and size > GIB:
            raise ValueError('A restored file exceeds the1GiB allowance')
        result[name] = dict(row, path=name)
    if result.get(bundle_name, {}).get('kind') != 'directory':
        raise ValueError('The inventory must include its bundle root directory')
    for name in result:
        if name != bundle_name and result.get(str(PurePosixPath(name).parent), {}).get('kind') != 'directory':
            raise ValueError('Every inventory member needs its recorded directory parent')
    if sum(row['bytes'] for row in result.values()) > STAGING_GIB * GIB:
        raise ValueError('The recorded tree exceeds the6GiB staging allowance')
    return result


def _regular_file(path):
    path = _absolute(path)
    item = path.lstat()
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise ValueError('Use a regular non-linked recorded file')
    return item


def tree_metadata(bundle):
    """Only names, kinds and sizes; no payload is opened."""
    bundle = _absolute(bundle)
    if not bundle.is_dir():
        raise ValueError('The bundle target must be a regular directory')
    result = {}
    for parent, directories, files in os.walk(bundle, followlinks=False):
        for name in [''] + directories + files:
            path = Path(parent) / name if name else Path(parent)
            item = path.lstat()
            if stat.S_ISLNK(item.st_mode):
                raise ValueError('Restored trees must contain no symbolic links')
            if stat.S_ISDIR(item.st_mode):
                kind, size = 'directory', 0
            elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                kind, size = 'file', item.st_size
            else:
                raise ValueError('Restored trees must contain only regular unlinked files')
            relative = path.relative_to(bundle.parent).as_posix()
            result[relative] = dict(kind=kind, bytes=size)
    return result


def check_tree(bundle, inventory):
    observed = tree_metadata(bundle)
    expected = {name:dict(kind=row['kind'], bytes=row['bytes']) for name, row in inventory.items()}
    if observed != expected:
        raise ValueError('Restored file names, kinds or sizes differ from the recorded inventory')
    return dict(regular_files=sum(row['kind'] == 'file' for row in observed.values()),
                regular_bytes=sum(row['bytes'] for row in observed.values()),
                metadata_only=True, extra_integrity_checks=0)


def _rename_noreplace(source, target):
    """Linux atomic directory promotion that also rejects a concurrently made target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('Atomic no-replace directory rename is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def restore_bundle(source, staging_root, *, seconds=RESTORE_SECONDS, clock=time.monotonic,
                   progress=None, check=None):
    """Stream a bound archive into a new stage; retain all partial files on failure."""
    bundle, archive = _absolute(source['bundle']), _absolute(source['archive'])
    inventory = validate_inventory(source['inventory'], bundle.name)
    if (source['regular_count'] != sum(row['kind'] == 'file' for row in inventory.values())
            or source['regular_bytes'] != sum(row['bytes'] for row in inventory.values())):
        raise ValueError('Source inventory totals differ')
    if bundle.exists() or bundle.is_symlink():
        checked = check_tree(bundle, inventory)
        return dict(status='already_present', archive_opened=False, restored_files=0,
                    restored_bytes=0, promotion_succeeded=False, target=str(bundle), inventory=checked)
    if archive != bundle.with_name(bundle.name + '.tar.gz'):
        raise ValueError('Only the archive beside this exact runtime bundle is restorable')
    initial = _regular_file(archive)
    if not bundle.parent.is_dir():
        raise ValueError('The original emit directory must already exist')
    staging_root = _absolute(staging_root)
    if staging_root == bundle or staging_root.is_relative_to(bundle):
        raise ValueError('Stage the archive outside the original bundle')
    staging_root.mkdir(parents=True, exist_ok=False)
    staged = staging_root / bundle.name
    began = clock()
    state = dict(status='running', archive_opened=False, target=str(bundle), staging=str(staged),
                 restored_files=0, restored_bytes=0, promotion_succeeded=False,
                 current_member=None, extra_integrity_checks=0)

    def verify():
        if clock() - began >= seconds:
            raise TimeoutError('Bundle restore wall allowance exhausted')
        if check is not None:
            check()

    def report():
        if progress is not None:
            progress(dict(state))

    seen = set()
    try:
        verify()
        descriptor = os.open(str(archive), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as compressed:
            opened = os.fstat(compressed.fileno())
            if (opened.st_ino, opened.st_dev, opened.st_size, opened.st_mtime_ns) != (
                    initial.st_ino, initial.st_dev, initial.st_size, initial.st_mtime_ns):
                raise ValueError('Archive metadata changed before opening')
            state['archive_opened'] = True
            report()
            with tarfile.open(fileobj=compressed, mode='r|gz', bufsize=1024 * 1024) as stream:
                for member in stream:
                    verify()
                    if member.issym() or member.islnk() or member.linkname or member.sparse is not None:
                        raise ValueError('Links and sparse archive members are outside the recorded bundle format')
                    if not (member.isdir() or member.isfile()):
                        raise ValueError('Only ordinary file or directory tar members are supported')
                    name = _member_name(member.name, directory=member.isdir())
                    if name in seen or name not in inventory:
                        raise ValueError('Duplicate or unrecorded archive member')
                    row = inventory[name]
                    kind = 'directory' if member.isdir() else 'file'
                    if kind != row['kind'] or member.size != row['bytes']:
                        raise ValueError('Archive member kind or size differs from the recorded inventory')
                    destination = staging_root.joinpath(*PurePosixPath(name).parts)
                    state['current_member'] = name
                    if member.isdir():
                        destination.mkdir(exist_ok=False)
                    else:
                        if not destination.parent.is_dir():
                            raise ValueError('Archive parent directory must precede its files')
                        extracted = stream.extractfile(member)
                        if extracted is None:
                            raise ValueError('Archive regular file has no stream')
                        count = 0
                        with extracted, destination.open('xb') as sink:
                            while True:
                                verify()
                                chunk = extracted.read(1024 * 1024)
                                if not chunk:
                                    break
                                count += len(chunk)
                                if count > row['bytes']:
                                    raise ValueError('Archive stream exceeds its recorded byte length')
                                sink.write(chunk)
                                state['restored_bytes'] += len(chunk)
                            if count != row['bytes']:
                                raise ValueError('Archive regular file ended before its recorded byte length')
                            sink.flush()
                            os.fsync(sink.fileno())
                        state['restored_files'] += 1
                    seen.add(name)
                    report()
        if seen != set(inventory):
            raise ValueError('The archive is missing recorded bundle members')
        final = _regular_file(archive)
        if (final.st_size, final.st_mtime_ns, final.st_ino, final.st_dev) != (
                initial.st_size, initial.st_mtime_ns, initial.st_ino, initial.st_dev):
            raise ValueError('Archive metadata changed while restoring')
        verified = check_tree(staged, inventory)
        for parent, directories, _ in os.walk(staged, topdown=False):
            for directory in directories:
                _fsync_dir(Path(parent) / directory)
            _fsync_dir(parent)
        verify()
        _rename_noreplace(staged, bundle)
        state['promotion_succeeded'] = True
        _fsync_dir(bundle.parent)
        _fsync_dir(staging_root)
        state.update(status='restored', current_member=None, inventory=verified)
        report()
        return state
    except BaseException as exc:
        state.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
        report()
        raise


def source_gate(args):
    from . import dacapo_vgg16_plan_math as existing
    source, _ = existing.source_gate(args, require_restored=False)
    bundle = _absolute(source['bundle'])
    archive_case = _absolute(args.archive_case)
    manifest = read(archive_case / 'manifest.json')
    recorded = read(archive_case / 'result.json')
    selected = [row for row in manifest['files'] if row.get('kind') == 'runtime_bundle']
    # The original archive helper has a terminal result/registry, with no
    # completion marker. Require its actual protocol rather than invent one.
    if (recorded.get('round') != 'R34-completed' or recorded.get('status') != 'ok'
            or type(recorded.get('elapsed_ns')) is not int or recorded['elapsed_ns'] <= 0
            or not isinstance(recorded.get('ended_utc'), str) or recorded.get('guard_errors') != []
            or recorded.get('in_progress') is not None or recorded.get('source_result_unchanged') is not True
            or recorded.get('selected_files') != len(recorded.get('files', []))
            or recorded.get('completed_files') != recorded.get('selected_files')
            or recorded.get('error') or recorded.get('final_resource_error') or len(selected) != 1
            or _absolute(selected[0]['path']) != bundle):
        raise ValueError('A closed successful same-plan bundle archive is required')
    registry = Path(getattr(args, 'registry', paths.RESULTS_DIR / 'registry.jsonl'))
    with registry.open() as stream:
        archive_rows = [row for line in stream if line.strip()
                        for row in [json.loads(line)] if row.get('run_id') == archive_case.name]
    fields = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}
    if (len(archive_rows) != 1 or set(archive_rows[0]) != fields
            or archive_rows[0].get('round') != 'R34-completed'
            or archive_rows[0].get('mode') != 'reversible-completed-case-archive'
            or _absolute(archive_rows[0]['artifacts_dir']) != archive_case
            or archive_rows[0].get('metrics', {}).get('status') != 'ok'
            or archive_rows[0].get('metrics', {}).get('completed_files') != recorded['completed_files']):
        raise ValueError('The successful source archive needs its unique original eight-field registry row')
    inventory = validate_inventory(selected[0]['inventory'], bundle.name)
    count = sum(row['kind'] == 'file' for row in inventory.values())
    total = sum(row['bytes'] for row in inventory.values())
    archive = bundle.with_name(bundle.name + '.tar.gz')
    rows = [row for row in recorded.get('files', []) if row.get('kind') == 'runtime_bundle']
    if (count != EXPECTED_FILES or total != EXPECTED_BYTES or len(rows) != 1
            or rows[0].get('status') != 'compressed' or _absolute(rows[0]['source']) != bundle
            or _absolute(rows[0]['archive']) != archive or rows[0].get('raw_bytes') != total):
        raise ValueError('Only the9717-file original VGG bundle inventory may be restored')
    item = _regular_file(archive)
    if item.st_size != rows[0].get('compressed_bytes'):
        raise ValueError('Recorded archive size differs from current stat metadata')
    return dict(bundle=str(bundle), archive=str(archive), inventory=list(inventory.values()),
        regular_count=count, regular_bytes=total, official_source=source,
        archive_case=str(archive_case), archive_manifest=str(archive_case / 'manifest.json'),
        archive_stat=dict(size_bytes=item.st_size, mtime_ns=item.st_mtime_ns),
        payload_reads_in_source_gate=0, extra_integrity_checks=0)


def math_command(args):
    """The existing outer entry retains its Nix/4GiB worker and original argv."""
    return [sys.executable, '-B', '-m', 'pt_agent.experiments.dacapo_vgg16_plan_math',
        '--compile-case', str(args.compile_case), '--archive-case', str(args.archive_case),
        '--out', str(args.math_out), '--seconds', '1800', '--memory-gib', '4',
        '--case-gib', '0.5', '--minimum-free-gib', '10']


def _tree_bytes(directory, *, excluded=None):
    total = 0
    if not Path(directory).exists():
        return total
    for parent, directories, files in os.walk(directory, followlinks=False):
        if excluded is not None:
            directories[:] = [name for name in directories if Path(parent) / name != excluded]
        for name in files:
            try:
                total += (Path(parent) / name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def _available_ram():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def _latest_rss(path):
    if path is None or not Path(path).is_file():
        return None
    with Path(path).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 262144))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get('processes'), list):
            processes = sum(p.get('vmrss_kib', 0) * 1024 for p in row['processes'])
            return dict(ts_ns=row.get('ts_ns'), subtree_rss_bytes=processes,
                supervisor_rss_bytes=(row.get('outer_process') or {}).get('vmrss_kib', 0) * 1024)
    return None


def resource_check(out, staging_root, resource_file, deadline, *, clock=time.monotonic,
                   disk_usage=shutil.disk_usage, available_ram=None):
    sample = _latest_rss(resource_file)
    observed = dict(ts_ns=time.perf_counter_ns(), own_artifact_bytes=_tree_bytes(out, excluded=staging_root),
        staging_bytes=_tree_bytes(staging_root), disk_free_bytes=disk_usage(out).free,
        available_ram_bytes=_available_ram() if available_ram is None else available_ram(), rss=sample)
    reasons = []
    if clock() >= deadline:
        reasons.append('Total2400second wall allowance exhausted')
    if observed['own_artifact_bytes'] > OWN_GIB * GIB:
        reasons.append('Own artifacts exceed0.5GiB excluding restore staging')
    if observed['staging_bytes'] > STAGING_GIB * GIB:
        reasons.append('Restore staging exceeds6GiB')
    if observed['disk_free_bytes'] < 10 * GIB:
        reasons.append('Free disk below10GiB')
    if observed['available_ram_bytes'] < 1.25 * GIB:
        reasons.append('Host MemAvailable below1.25GiB')
    if sample and sample['subtree_rss_bytes'] + sample['supervisor_rss_bytes'] > 14 * GIB:
        reasons.append('Subtree plus supervisor RSS exceeds14GiB')
    if reasons:
        error = RuntimeError('; '.join(reasons))
        error.observation = dict(observed, reasons=reasons)
        raise error
    return observed


def _clean(record):
    return (record.get('exit_code') == 0 and record.get('timed_out') is False
        and record.get('signal') is None and record.get('remaining_pids') == []
        and record.get('signals_sent') == [])


def _parent_limits():
    hard_as = resource.getrlimit(resource.RLIMIT_AS)[1]
    hard_file = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    if ((hard_as != resource.RLIM_INFINITY and hard_as < 24 * GIB)
            or (hard_file != resource.RLIM_INFINITY and hard_file < GIB)):
        raise RuntimeError('Parent must permit24GiB virtual AS and1GiB files; restore child alone is1GiB AS')
    return dict(parent_AS_hard_bytes=None if hard_as == resource.RLIM_INFINITY else hard_as,
                parent_file_hard_bytes=None if hard_file == resource.RLIM_INFINITY else hard_file)


def restore_worker(args):
    result = dict(status='running', archive_restore_attempts=0, archive_restores=0, restore=None,
                  extra_integrity_checks=0, model_forwards=0, api_calls=0, compiler_runs=0)
    started = time.perf_counter_ns()
    target = Path(args.worker_result)
    target.parent.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder(target.parent / 'restore-events.jsonl')
    try:
        source = read(args.source_record)
        result['archive_restore_attempts'] = int(not Path(source['bundle']).exists())
        def progress(state):
            result['restore'] = state
            durable(target, result)
        def check():
            if shutil.disk_usage(args.staging_root.parent).free < 10 * GIB:
                raise RuntimeError('Free disk below10GiB during bundle restore')
        durable(target, result)
        with recorder.span('restore-stream-and-atomic-promotion'):
            result['restore'] = restore_bundle(source, args.staging_root, progress=progress, check=check)
        result['archive_restores'] = int(result['restore']['status'] == 'restored')
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:4000])
    finally:
        if result['restore'] is not None:
            result['archive_restores'] = int(result['restore'].get('promotion_succeeded', False))
        result.update(elapsed_ns=time.perf_counter_ns() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            staging_preserved_on_failure=True)
        durable(target, result)
    return result


def _register(registry, out, result):
    from ..registry import git_rev
    row = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(),
        metrics={key:result.get(key) for key in ('status', 'archive_restore_attempts', 'archive_restores',
            'math_attempts', 'math_completed', 'accuracy_acceptance', 'security_acceptance', 'extra_integrity_checks')},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise RuntimeError('Restore driver run id is already registered')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _accept_math(args, source):
    from . import dacapo_vgg16_plan_math as existing
    result = read(Path(args.math_out) / 'result.json')
    closed = read(Path(args.math_out) / 'completion.json')
    expected_source = dict(source['official_source'], restored_bundle_available=True)
    if (result.get('round') != existing.ROUND or result.get('model') != MODEL
            or result.get('status') != 'ok' or result.get('worker_attempts') != 1
            or result.get('worker_completed') != 1 or closed.get('status') != 'ok'
            or closed.get('required_artifacts_persisted') is not True or closed.get('registry_persisted') is not True
            or result.get('source') != expected_source):
        raise ValueError('Original math case did not persist its complete same-source success')
    existing.accepted_worker(result.get('worker') or {})
    return dict(result=str(Path(args.math_out) / 'result.json'), completion=str(Path(args.math_out) / 'completion.json'),
                status=result['status'], scope='Original official bias-omitted physical32 arithmetic; no task acceptance',
                complete_case_elapsed_seconds=closed['complete_case_elapsed_seconds'])


def _restore_observation(record):
    """An interrupted checkpoint cannot establish a zero promotion total."""
    terminal = record.get('status') in ('ok', 'error') and type(record.get('elapsed_ns')) is int
    promoted = (record.get('restore') or {}).get('promotion_succeeded') is True
    return dict(archive_restore_attempts=record.get('archive_restore_attempts'),
        archive_restores=1 if promoted else (record.get('archive_restores') if terminal else None),
        restore_terminal_counters_available=terminal,
        restore_counter_scope='terminal restore worker' if terminal else 'partial checkpoint; total unknown unless promotion was recorded')


def execute(args):
    started, deadline = time.perf_counter_ns(), time.monotonic() + TOTAL_SECONDS
    out, math_out = _absolute(args.out), _absolute(args.math_out)
    root = paths.RESULTS_DIR.resolve()
    if out.parent != root or math_out.parent != root or out == math_out or out.exists() or math_out.exists():
        raise ValueError('Use two distinct new direct results cases for restore and original mathematics')
    args.compile_case, args.archive_case = _absolute(args.compile_case), _absolute(args.archive_case)
    args.out, args.math_out = out, math_out
    out.mkdir()
    work = out / 'work'
    work.mkdir()
    staging = work / 'restore-staging'
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        archive_restore_attempts=0, archive_restores=0, restore_worker_attempts=0, restore_worker_completed=0,
        math_attempts=0, math_completed=0, restore=None, math=None, processes=[], source=None,
        incoming_signal=None, guard_trigger=None, monitor_error=None, stage_ns={},
        extra_integrity_checks=0, api_calls=0, compiler_runs=0, keygen_runs=0, ckks_runs=0,
        accuracy_acceptance=None, security_acceptance=None,
        original_archive_preserved=True, original_math_entry_modified=False, bias_correction_implemented=False,
        limits=dict(restore_child_AS_gib=1, math_outer_AS_gib=24, original_math_worker_AS_gib=4,
            restore_seconds=300, original_math_worker_seconds=1800, original_math_total_seconds=2040,
            total_seconds=2400, own_artifacts_gib=.5, restore_staging_gib=6, file_gib=1,
            RSS_gib=14, minimum_available_RAM_gib=1.25, minimum_free_disk_gib=10,
            retries=0, concurrency=1))
    stop = threading.Event()
    active = [None]
    old = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, frame):
        if result['guard_trigger'] is None and result['monitor_error'] is None:
            result['incoming_signal'] = signum
        raise InterruptedError('Restore/math supervisor received signal ' + str(signum))
    def monitor():
        while not stop.wait(.5):
            try:
                resource_check(out, staging, active[0], deadline)
            except BaseException as exc:
                result['guard_trigger'] = getattr(exc, 'observation', None)
                if result['guard_trigger'] is None:
                    result['monitor_error'] = type(exc).__name__ + ': ' + str(exc)
                try:
                    durable(out / 'resource-stop.json', dict(error=str(exc), observation=result['guard_trigger']))
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    for sig in old:
        signal.signal(sig, interrupt)
    def phase(name, argv, *, timeout, memory, attempt_field):
        if result['guard_trigger'] is not None or result['monitor_error'] or result['incoming_signal'] is not None:
            raise RuntimeError('A prior supervision stop prevents the next phase')
        resource_check(out, staging, None, deadline)
        active[0] = out / (name + '-resources.jsonl')
        row = dict(stage=name, argv=argv, observation=None, supervision_exception=None)
        result['processes'].append(row)
        durable(out / 'result.json', result)
        began = time.perf_counter_ns()
        try:
            with recorder.span(name):
                try:
                    result[attempt_field] += 1
                    durable(out / 'result.json', result)
                    row['observation'] = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                        log=out / (name + '.log'), resources=active[0], timeout=timeout,
                        mem_gib=memory, fsize_gib=1, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                except BaseException as exc:
                    row['supervision_exception'] = type(exc).__name__ + ': ' + str(exc)
                    raise
                if not _clean(row['observation']):
                    raise RuntimeError(name + ' process failed; no retry')
                if result['guard_trigger'] is not None or result['monitor_error'] or result['incoming_signal'] is not None:
                    raise RuntimeError(name + ' ended after a supervision stop')
        finally:
            active[0] = None
            result['stage_ns'][name] = time.perf_counter_ns() - began
    try:
        with recorder.span('source-and-budget-admission'):
            result['parent_limits'] = _parent_limits()
            source = source_gate(args)
            result['source'] = source
            for module_path in (Path(__file__), Path(__file__).with_name('dacapo_vgg16_plan_math.py')):
                shutil.copy2(module_path, out / module_path.name)
            durable(out / 'source.json', source)
            initial = resource_check(out, staging, None, deadline)
            needed = (0 if Path(source['bundle']).exists() else source['regular_bytes']) + GIB
            if initial['disk_free_bytes'] < 10 * GIB + needed:
                raise RuntimeError('Disk admission requires recorded restore bytes plus1GiB for the two small cases and10GiB reserve')
            result['initial_resource'] = initial
        watcher.start()
        worker_result = work / 'restore-worker-result.json'
        restore_argv = [sys.executable, '-B', '-m', __spec__.name, '--restore-worker',
            '--source-record', str(out / 'source.json'), '--staging-root', str(staging),
            '--worker-result', str(worker_result)]
        result['archive_restore_attempts'] = None
        result['archive_restores'] = None
        phase('restore', restore_argv, timeout=RESTORE_SECONDS, memory=1, attempt_field='restore_worker_attempts')
        result['restore'] = read(worker_result)
        result.update(_restore_observation(result['restore']))
        if result['restore'].get('status') != 'ok':
            raise RuntimeError('Bundle restoration worker did not complete')
        result['restore_worker_completed'] = 1
        check_tree(Path(source['bundle']), validate_inventory(source['inventory'], Path(source['bundle']).name))
        remaining = deadline - time.monotonic() - 30
        if remaining <= 0:
            raise TimeoutError('No total allowance remains for the unchanged math entry')
        phase('math', math_command(args), timeout=min(MATH_WRAPPER_SECONDS, remaining), memory=24,
              attempt_field='math_attempts')
        result['math'] = _accept_math(args, source)
        result.update(status='ok', math_completed=1)
    except BaseException as exc:
        if getattr(exc, 'observation', None) is not None:
            result['guard_trigger'] = exc.observation
        status = 'resource_stopped' if result['guard_trigger'] is not None else (
            'supervision_error' if result['monitor_error'] else ('interrupted' if result['incoming_signal'] else 'error'))
        result.update(status=status, error=type(exc).__name__ + ': ' + str(exc)[:4000])
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        try:
            worker_result = work / 'restore-worker-result.json'
            if result['restore'] is None and worker_result.is_file():
                result['restore'] = read(worker_result)
            if result['restore'] is not None:
                result.update(_restore_observation(result['restore']))
            if result['math'] is None and (math_out / 'result.json').is_file():
                measured = read(math_out / 'result.json')
                result['math'] = dict(result=str(math_out / 'result.json'), status=measured.get('status'),
                    error=measured.get('error'), partial_retained=True)
            try:
                final = resource_check(out, staging, None, deadline)
                result['final_resource_before_persistence'] = final
                if (final['own_artifact_bytes'] + 65536 > OWN_GIB * GIB
                        or final['disk_free_bytes'] < 10 * GIB + 65536):
                    raise RuntimeError('Keep64KiB of own allowance and free-disk reserve for terminal metadata')
            except BaseException as exc:
                if getattr(exc, 'observation', None) is not None:
                    result['guard_trigger'] = exc.observation
                result.update(status='resource_stopped', final_resource_error=type(exc).__name__ + ': ' + str(exc))
            result['staging_retained_bytes'] = _tree_bytes(staging)
            result['own_artifact_bytes_before_persistence'] = _tree_bytes(out, excluded=staging)
            result['recorded_elapsed_before_persistence_seconds'] = (time.perf_counter_ns() - started) / 1e9
            timing = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
            result['timing'] = timing
            for span in timing['spans']:
                result['stage_ns'].setdefault(span['stage_id'], span.get('inclusive_ns'))
            durable(out / 'result.json', result)
            _register(args.registry, out, result)
            persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=timing, required_artifact_bytes=artifact_bytes(out))
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-case', type=Path)
    parser.add_argument('--archive-case', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--math-out', type=Path)
    parser.add_argument('--registry', type=Path, default=paths.RESULTS_DIR / 'registry.jsonl')
    parser.add_argument('--restore-worker', action='store_true')
    parser.add_argument('--source-record', type=Path)
    parser.add_argument('--staging-root', type=Path)
    parser.add_argument('--worker-result', type=Path)
    args = parser.parse_args(argv)
    if args.restore_worker:
        if any(getattr(args, name) is None for name in ('source_record', 'staging_root', 'worker_result')):
            parser.error('Restore worker requires explicit source-record/staging-root/worker-result')
        result = restore_worker(args)
    else:
        if any(getattr(args, name) is None for name in ('compile_case', 'archive_case', 'out', 'math_out')):
            parser.error('Use explicit compile-case/archive-case/out/math-out')
        result = execute(args)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
