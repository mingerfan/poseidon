"""Publish bounded, explicit-case experiment metadata without model execution."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import tempfile
import time

from .. import paths

DEADLINE = '2026-11-17T10:00:00Z'
MAX_METADATA = 16 * 1024**2
SAFE_KEY = re.compile(r'[A-Za-z0-9_.:-]{1,96}\Z')
IDENTITY = ('round', 'model', 'mode', 'status', 'variant', 'failed_stage', 'active_stage')
SCALARS = ('compiled_bootstraps', 'bootstrap_count', 'static_native_bootstraps',
           'api_calls', 'llm_calls', 'actual_api_calls', 'api_delta',
           'keygen_runs', 'real_ckks_attempts', 'real_ckks_completed',
           'report_samples', 'report_samples_read', 'report_samples_completed',
           'confirm_samples', 'heldout_samples', 'reference_top1_percent',
           'approximate_top1_percent', 'accuracy_drop_pp', 'all_finite',
           'finite_samples', 'report_completed', 'formal_matrix_complete',
           'trace_attempts', 'trace_completed', 'compile_attempts', 'compile_completed',
           'native_compile_attempts', 'native_compile_completed',
           'recorded_elapsed_seconds', 'recorded_elapsed_ns')
COMPLETION = ('status', 'complete_case_elapsed_seconds', 'complete_case_elapsed_ns',
              'timing_status', 'required_artifacts_persisted', 'registry_persisted',
              'exclusive_attribution_ready', 'started_ns', 'ended_ns')


def finite_scalar(value):
    return value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value)


def load_small(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Metadata symlink is refused')
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > MAX_METADATA:
        raise ValueError('Only small regular metadata records are supported')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Metadata object required')
    return value


def validate_manifest(value, results_root):
    if value.get('schema') != 'explicit-experiment-metadata-cases-v1':
        raise ValueError('Explicit experiment case manifest required')
    entries = value.get('cases')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 256:
        raise ValueError('Finite explicit case list required')
    root = Path(results_root).resolve()
    checked = []
    seen = set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {'id', 'case', 'model', 'system', 'variant_role'}:
            raise ValueError('Only case identity and metadata path fields are accepted')
        if any(not isinstance(row[k], str) or not SAFE_KEY.fullmatch(row[k]) for k in ('id', 'model', 'system', 'variant_role')):
            raise ValueError('Short explicit identity labels required')
        case = Path(row['case']).resolve()
        if not case.is_relative_to(root) or case == root or row['id'] in seen:
            raise ValueError('Unique cases below the configured result root required')
        seen.add(row['id'])
        checked.append(dict(row, case=str(case)))
    return checked


def metadata_counts(value):
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items()
            if isinstance(k, str) and SAFE_KEY.fullmatch(k) and finite_scalar(v)}


def summarize_record(record):
    if record is None:
        return None
    out = {k: record[k] for k in IDENTITY
           if isinstance(record.get(k), str) and SAFE_KEY.fullmatch(record[k])}
    out.update({k: record[k] for k in SCALARS if k in record and finite_scalar(record[k])})
    for key in ('attempted', 'attempts', 'completed'):
        if key in record:
            out[key] = metadata_counts(record[key])
    if isinstance(record.get('limits'), dict):
        out['limits'] = metadata_counts(record['limits'])
    # Error messages may contain candidate text. Publish only their class.
    error = record.get('error')
    if isinstance(error, str):
        error_class = error.split(':', 1)[0]
        if SAFE_KEY.fullmatch(error_class):
            out['error_class'] = error_class
    process_rows = record.get('processes', [])
    if not isinstance(process_rows, list):
        process_rows = []
    if isinstance(record.get('process'), dict):
        process_rows = process_rows + [{'stage': 'process', 'observation': record['process']}]
    out['processes'] = []
    for row in process_rows:
        if not isinstance(row, dict):
            continue
        obs = row.get('observation', row.get('process', {}))
        if not isinstance(obs, dict):
            continue
        phase = row.get('stage', row.get('phase'))
        phase = phase if isinstance(phase, str) and SAFE_KEY.fullmatch(phase) else None
        obs_out = {k: obs[k] for k in ('exit_code', 'timed_out', 'signal', 'started_ns', 'ended_ns')
                   if k in obs and finite_scalar(obs[k])}
        hwm = metadata_counts(obs.get('peak_rss_kib_by_pid'))
        nums = [v for v in hwm.values() if type(v) in (int, float)]
        obs_out['single_PID_HWM_KiB'] = max(nums) if nums else None
        remaining = obs.get('remaining_pids')
        obs_out['remaining_PID_count'] = len(remaining) if isinstance(remaining, list) else None
        out['processes'].append(dict(stage=phase, **obs_out))
    return out


def snapshot(entries, now):
    rows = []
    for entry in entries:
        case = Path(entry['case'])
        row = dict(entry, result_path=str(case/'result.json'), completion_path=str(case/'completion.json'))
        try:
            record = load_small(case/'result.json')
            marker = load_small(case/'completion.json')
            row['result'] = summarize_record(record)
            row['completion'] = {k: marker[k] for k in COMPLETION
                                 if k in marker and (finite_scalar(marker[k]) or k in ('status', 'timing_status') and isinstance(marker[k], str) and SAFE_KEY.fullmatch(marker[k]))} if marker else None
            row['terminal_observed'] = marker is not None and marker.get('status') not in (None, 'running', 'prepared')
            spans = marker.get('phase_spans', []) if marker else []
            if not spans and isinstance(record, dict):
                spans = record.get('timing', {}).get('spans', [])
            row['stage_clocks'] = [{k: s[k] for k in ('stage_id', 'parent_id', 'inclusive_ns', 'exclusive_ns', 'status')
                                    if k in s and (finite_scalar(s[k]) or isinstance(s[k], str) and SAFE_KEY.fullmatch(s[k]))}
                                   for s in spans if isinstance(s, dict)]
        except (OSError, ValueError, TypeError) as exc:
            row.update(result=None, completion=None, terminal_observed=False, metadata_read_error_class=type(exc).__name__)
        rows.append(row)
    return dict(schema='compile-first-experiment-metadata-view-v1', observed_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                scope='Explicit saved result/completion metadata only. No SESSION/provider/messages/data/weights/keys or model execution.', cases=rows)


def atomic_write(path, content):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Publication symlink refused')
    fd, temporary = tempfile.mkstemp(prefix='.metadata-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError('Publication symlink refused')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish(directory, value):
    directory = Path(directory)
    atomic_write(directory/'LIVE_CASE_METADATA.json', json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    buffer = io.StringIO()
    fields = ('id','model','system','variant_role','result_status','completion_status','terminal_observed','complete_case_seconds','compiled_bootstraps','bootstrap_count','API_calls','result_path','completion_path')
    writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
    for item in value['cases']:
        record, marker = item.get('result') or {}, item.get('completion') or {}
        writer.writerow({**{k:item[k] for k in ('id','model','system','variant_role','terminal_observed','result_path','completion_path')},
                         'result_status':record.get('status'), 'completion_status':marker.get('status'),
                         'complete_case_seconds':marker.get('complete_case_elapsed_seconds'),
                         'compiled_bootstraps':record.get('compiled_bootstraps'), 'bootstrap_count':record.get('bootstrap_count'),
                         'API_calls':record.get('api_delta',record.get('actual_api_calls',record.get('api_calls',record.get('llm_calls'))))})
    atomic_write(directory/'LIVE_CASE_METADATA.csv', buffer.getvalue())


def execute(args, *, now=time.time, sleep=time.sleep):
    if type(args.poll_seconds) is not int or args.poll_seconds not in (30, 60):
        raise ValueError('Polling interval must be30or60seconds')
    cutoff = datetime.fromisoformat(args.until.replace('Z','+00:00'))
    maximum = datetime.fromisoformat(DEADLINE.replace('Z','+00:00'))
    if cutoff.tzinfo is None or cutoff > maximum:
        raise ValueError('Explicit timezone-aware stop no later than the experiment deadline required')
    start = now()
    if not args.once and not start < cutoff.timestamp() <= start + 7*86400:
        raise ValueError('Future finite deadline within7days required')
    entries = validate_manifest(load_small(args.manifest), args.results_root)
    out = Path(args.publish_dir).resolve()
    root = Path(args.results_root).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Publication must stay separate from all raw experiment evidence')
    if any(out == Path(e['case']) or out.is_relative_to(Path(e['case'])) or Path(e['case']).is_relative_to(out) for e in entries):
        raise ValueError('Publication and source cases must be disjoint')
    out.mkdir(parents=True, exist_ok=True)
    polls = 0
    while True:
        current = now()
        if not args.once and current >= cutoff.timestamp():
            break
        publish(out, snapshot(entries, current)); polls += 1
        if args.once:
            break
        sleep(min(args.poll_seconds, max(0, cutoff.timestamp()-now())))
    return polls


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--publish-dir',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,default=paths.RESULTS_DIR)
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--until',default=DEADLINE)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    execute(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
