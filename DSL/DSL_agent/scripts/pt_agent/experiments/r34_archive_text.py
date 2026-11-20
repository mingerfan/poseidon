"""Reversibly archive inactive historical compiler text, with explicit input list."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import time

from .r10_paper_metrics import write_json


def validate_files(root, rows):
    root = Path(root).resolve()
    selected = []
    for row in rows:
        source = Path(row['path'])
        relative = source.relative_to(root)
        if (not relative.parts[0].startswith(('A1-', 'E', 'u3', 'u4', 's1-'))
                or not (source.suffix == '.mlir' or source.name.endswith('.runtime-plan.json'))
                or source.is_symlink() or source.resolve() != source
                or not source.is_file() or source.stat().st_size != row['bytes']):
            raise ValueError('Only unchanged inactive historical compiler text is eligible')
        target = source.with_name(source.name + '.gz')
        temporary = target.with_name(target.name + '.partial')
        if target.exists() or temporary.exists():
            raise ValueError('Archive destination already exists')
        selected.append((source, target, temporary))
    if not selected or len({str(x[0]) for x in selected}) != len(selected):
        raise ValueError('A nonempty unique explicit file list is required')
    return selected


def archive_one(source, target, temporary):
    """Keep the original until compression, close, fsync and final rename succeed."""
    began = time.perf_counter_ns()
    raw_bytes = source.stat().st_size
    with source.open('rb') as reader, temporary.open('xb') as sink:
        with gzip.GzipFile(filename=source.name, mode='wb', compresslevel=1, fileobj=sink) as encoded:
            shutil.copyfileobj(reader, encoded, length=1024 * 1024)
        sink.flush()
        os.fsync(sink.fileno())
    compressed_bytes = temporary.stat().st_size
    if compressed_bytes >= raw_bytes:
        temporary.unlink()
        return dict(source=str(source), status='kept_uncompressed', raw_bytes=raw_bytes,
                    compressed_bytes=None, released_bytes=0, elapsed_ns=time.perf_counter_ns()-began)
    temporary.rename(target)
    directory = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
        source.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(source=str(source), archive=str(target), status='compressed', raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes, released_bytes=raw_bytes-compressed_bytes,
                restore_argv=['gzip', '-dk', '--', str(target)],
                elapsed_ns=time.perf_counter_ns()-began)


def archive_state(source, target, temporary):
    return dict(source=str(source), archive=str(target), temporary=str(temporary),
                restore_argv=['gzip', '-dk', '--', str(target)],
                source_exists=source.exists(), archive_exists=target.exists(),
                temporary_exists=temporary.exists(),
                archive_bytes=target.stat().st_size if target.exists() else None)


def main(argv=None):
    from .. import paths
    from ..registry import git_rev
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    memory = 512 * 1024**2
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory = memory if hard == resource.RLIM_INFINITY else min(memory, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    manifest = json.loads(args.manifest.read_text())
    selected = validate_files(paths.RESULTS_DIR, manifest['files'])
    args.out.mkdir(exist_ok=False)
    write_json(args.out/'manifest.json', manifest)
    began = time.perf_counter_ns()
    result = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  files=[], address_space_limit_bytes=memory,
                  memory_enforcement='RLIMIT_AS bounds all process mappings and resident memory', reserve_disk_gib=10,
                  source_scope=manifest['scope'], checksums_or_byte_comparisons=0,
                  experiment_or_report_evaluations=0, api_calls=0)
    try:
        for source, target, temporary in selected:
            if shutil.disk_usage(paths.RESULTS_DIR).free < 10*1024**3:
                raise RuntimeError('Disk reserve unavailable before next archive')
            result['in_progress'] = archive_state(source, target, temporary)
            write_json(args.out/'progress.json', result)
            row = archive_one(source, target, temporary)
            result['files'].append(row)
            result['in_progress'] = None
            write_json(args.out/'progress.json', result)
        result['status'] = 'ok'
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if result.get('in_progress'):
            result['interrupted_file'] = archive_state(source, target, temporary)
    result.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_ns=time.perf_counter_ns()-began,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  released_bytes=sum(x['released_bytes'] for x in result['files']),
                  completed_files=len(result['files']))
    write_json(args.out/'result.json', result)
    row = dict(run_id=args.out.name, round='R34', model='historical-compiler-text', mode='reversible-gzip',
               git_rev=git_rev(), metrics={k:result[k] for k in ('status','completed_files','released_bytes','elapsed_ns','max_rss_kib')},
               artifacts_dir=str(args.out), ts=result['ended_utc'])
    with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    print(json.dumps(row,ensure_ascii=False))
    return 0 if result['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
