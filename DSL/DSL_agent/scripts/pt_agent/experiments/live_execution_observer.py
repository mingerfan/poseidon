"""Read-only, finite observation of an explicitly selected serial experiment queue.

Only this process's two publication files are written. Session conversations,
programs, API requests, samples, model payloads and numerical outputs are never
deserialized or published. Unknown JSON values are skipped by the selector.
"""
import argparse
from datetime import datetime, timezone
import html
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
from zoneinfo import ZoneInfo

MIB = 1024**2
MAX_SECONDS = 7 * 86400
MAX_METADATA = 2 * MIB
MAX_SESSION = 32 * MIB
MAX_LEDGER = 32 * MIB
MAX_LINE = 64 * 1024
PUBLIC_JSON = 'LIVE_EXECUTION_STATUS.json'
PUBLIC_HTML = 'LIVE_EXECUTION_STATUS.html'
PHASES = ('intake', 'calibration', 'agent', 'placement', 'emit', 'archive_emit_constants',
          'trace', 'compile', 'prepare', 'prompt', 'step', 'report', 'parse',
          'optimize', 'validate', 'codegen')
STATUSES = {'running', 'pending', 'not_started', 'ok', 'error', 'failed', 'prepared',
            'quality_rejected', 'dependency_skipped', 'resource_deferred',
            'preserved_existing_case', 'completed_with_observations', 'resource_stopped',
            'interrupted', 'artifact_budget_exceeded'}
TERMINAL = {'ok', 'error', 'failed', 'prepared', 'quality_rejected', 'completed_with_observations',
            'resource_stopped', 'interrupted', 'artifact_budget_exceeded'}
RESOURCE_FLAGS = {'--memory-gib': 'address_space_gib', '--rss-gib': 'rss_gib',
                  '--mem-gib': 'address_space_gib',
                  '--case-gib': 'artifact_gib', '--seconds': 'wall_seconds',
                  '--trace-seconds': 'trace_seconds', '--compile-seconds': 'compile_seconds',
                  '--minimum-free-gib': 'disk_reserve_gib', '--file-gib': 'file_gib'}
SESSION_FIELDS = {name: True for name in ('phase', 'iteration', 'calls', 'max_iters', 'tuner')}
PROCESS_FIELDS = {name: True for name in ('pid', 'exit_code', 'signal', 'timed_out')}
JOB_FIELDS = {name: True for name in ('id', 'out', 'kind', 'status', 'experiment_status', 'seconds')}
JOB_FIELDS.update(argv={'*': True}, requires={'*': True}, process=PROCESS_FIELDS)
CASE_FIELDS = dict(status=True, phase=True, attempted={'*': True}, completed={'*': True},
                   api_before=True, api_after=True, api_delta=True, api_calls=True,
                   completed_samples=True, evaluated_pairs=True,
                   limits={name: True for name in ('memory_gib', 'rss_gib', 'case_gib', 'seconds')})
COMPLETION_FIELDS = {name: True for name in ('status', 'required_artifacts_persisted',
                    'registry_persisted', 'complete_case_elapsed_seconds')}


class Unavailable(ValueError):
    """A small metadata observation could not be completed; no source is changed."""


class _Selector:
    """Bounded streaming JSON projection; ignored strings are never accumulated."""
    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, maximum
        self.buffer, self.index, self.read = '', 0, 0

    def peek(self):
        if self.index == len(self.buffer):
            self.buffer = self.stream.read(min(8192, self.maximum - self.read + 1))
            self.read += len(self.buffer)
            self.index = 0
            if self.read > self.maximum:
                raise Unavailable('size_limit')
        return self.buffer[self.index] if self.index < len(self.buffer) else ''

    def take(self):
        value = self.peek()
        if not value:
            raise Unavailable('incomplete_json')
        self.index += 1
        return value

    def space(self):
        while self.peek() and self.peek() in ' \t\r\n':
            self.take()

    def string(self, capture):
        if self.take() != '"':
            raise Unavailable('invalid_json')
        out = ['"'] if capture else None
        escaped = False
        while True:
            char = self.take()
            if out is not None:
                out.append(char)
                if len(out) > 4096:
                    raise Unavailable('selected_string_limit')
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                return json.loads(''.join(out)) if out is not None else None
            elif ord(char) < 32:
                raise Unavailable('invalid_json')

    def value(self, selected, depth=0):
        if depth > 128:
            raise Unavailable('nesting_limit')
        self.space()
        char = self.peek()
        if char == '"':
            return self.string(selected is True)
        if char in ('{', '['):
            is_object = char == '{'
            self.take(); self.space()
            close = '}' if is_object else ']'
            result = ({} if is_object else []) if isinstance(selected, dict) else None
            count = 0
            while self.peek() != close:
                if count:
                    if self.take() != ',':
                        raise Unavailable('invalid_json')
                    self.space()
                if is_object:
                    key = self.string(isinstance(selected, dict)); self.space()
                    if self.take() != ':':
                        raise Unavailable('invalid_json')
                    child = selected.get(key, selected.get('*')) if isinstance(selected, dict) else None
                else:
                    key = None
                    child = selected.get('*') if isinstance(selected, dict) else None
                value = self.value(child, depth + 1)
                if child is not None and result is not None:
                    if count >= 1000:
                        raise Unavailable('selected_items_limit')
                    if is_object:
                        result[key] = value
                    else:
                        result.append(value)
                count += 1
                self.space()
            self.take()
            return result
        token = []
        while self.peek() and self.peek() not in ' \t\r\n,]}':
            token.append(self.take())
            if len(token) > 64:
                raise Unavailable('invalid_scalar')
        try:
            value = json.loads(''.join(token), parse_constant=lambda _: None)
        except (ValueError, TypeError):
            raise Unavailable('invalid_json') from None
        return value if selected is True else None

    def project(self, fields):
        result = self.value(fields)
        self.space()
        if self.peek() or not isinstance(result, dict):
            raise Unavailable('invalid_json')
        return result


def projection(path, fields, *, maximum=MAX_METADATA):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Unavailable('missing_or_nonregular')
    if path.stat().st_size > maximum:
        raise Unavailable('size_limit')
    with path.open(encoding='utf-8') as stream:
        return _Selector(stream, maximum).project(fields)


def _read(path, fields, missing, label, *, maximum=MAX_METADATA):
    try:
        return projection(path, fields, maximum=maximum)
    except (OSError, ValueError, UnicodeError):
        missing.append(label)
        return {}


def _number(value, *, integral=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value if not integral or type(value) is int else None


def _label(value):
    # IDs and paths are trusted local metadata; newlines/control characters are
    # excluded and the publication remains bounded even for malformed files.
    return value[:160] if isinstance(value, str) and value and all(ord(c) >= 32 for c in value) else None


def _status(value):
    return value if isinstance(value, str) and value in STATUSES else None


def _resources(argv, seconds=None):
    result = {}
    if isinstance(argv, list):
        for i, value in enumerate(argv[:-1]):
            if isinstance(value, str) and value in RESOURCE_FLAGS:
                try:
                    number = float(argv[i+1])
                except (TypeError, ValueError):
                    continue
                if _number(number) is not None:
                    result[RESOURCE_FLAGS[value]] = number
    if _number(seconds) is not None:
        result['outer_wall_seconds'] = seconds
    return result


def ledger_summary(ledger, allowance, missing):
    """Count only complete accounting rows; take no lock held across an API call."""
    count = succeeded = invalid = 0
    partial = False
    path = Path(ledger)
    known = path.is_file() and not path.is_symlink()
    if known:
        try:
            if path.stat().st_size > MAX_LEDGER:
                raise Unavailable('ledger_size_limit')
            with path.open(encoding='utf-8') as stream:
                while True:
                    line = stream.readline(MAX_LINE + 1)
                    if not line:
                        break
                    if len(line) > MAX_LINE:
                        raise Unavailable('ledger_line_limit')
                    if not line.endswith('\n'):
                        partial = True
                        break
                    if not line.strip():
                        continue
                    try:
                        row = _Selector(io.StringIO(line), MAX_LINE).project({'status': True})
                    except ValueError:
                        invalid += 1
                        continue
                    count += 1
                    succeeded += type(row.get('status')) is int and row['status'] == 200
        except (OSError, ValueError, UnicodeError):
            known = False
    if not known:
        missing.append('api.ledger_complete_count')
    policy = _read(allowance, {'version': True, 'ledger': True, 'ceiling': True,
                             'observed_attempts': True, 'increment': True},
                   missing, 'api.allowance')
    binding = (type(policy.get('version')) is int and policy['version'] == 1
               and isinstance(policy.get('ledger'), str)
               and Path(policy['ledger']).is_absolute()
               and Path(policy['ledger']).resolve() == path.resolve())
    if not binding:
        missing.append('api.allowance_ledger_binding')
    ceiling = _number(policy.get('ceiling'), integral=True) if binding else None
    if ceiling is None:
        missing.append('api.allowance.ceiling')
    return dict(complete_accounting_rows=count if known else None,
                http_200_rows=succeeded if known else None, malformed_rows=invalid if known else None,
                partial_tail=partial if known else None, absolute_allowance=ceiling,
                remaining_recorded_allowance=max(0, ceiling-count) if known and ceiling is not None and not invalid else None,
                allowance_observed_attempts=_number(policy.get('observed_attempts'), integral=True) if binding else None,
                allowance_increment=_number(policy.get('increment'), integral=True) if binding else None,
                scope='Completed ledger rows; an in-flight request may not yet have an accounting row')


def case_summary(case, missing, label):
    case = Path(case)
    terminal = _read(case/'result.json', CASE_FIELDS, missing, label+'.result')
    progress = _read(case/'progress.json', CASE_FIELDS, [], label+'.progress')
    value = terminal or progress
    attempted, completed = value.get('attempted'), value.get('completed')
    attempted = attempted if isinstance(attempted, dict) else {}
    completed = completed if isinstance(completed, dict) else {}
    phases = {name: dict(attempted=_number(attempted.get(name), integral=True),
                         completed=_number(completed.get(name), integral=True))
              for name in PHASES if name in attempted or name in completed}
    active = [name for name, row in phases.items() if row['attempted'] and not row['completed']]
    phase = value.get('phase') if value.get('phase') in PHASES else active[-1] if active else None
    session = {}
    for relative in ('work/agent/SESSION.json', 'work/session/SESSION.json', 'session/SESSION.json', 'SESSION.json'):
        path = case/relative
        if path.exists():
            raw = _read(path, SESSION_FIELDS, missing, label+'.session', maximum=MAX_SESSION)
            session = dict(phase=raw.get('phase') if raw.get('phase') in ('evaluate','tune','confirm','done') else None,
                           iteration=_number(raw.get('iteration'), integral=True),
                           calls=_number(raw.get('calls'), integral=True),
                           max_iters=_number(raw.get('max_iters'), integral=True),
                           tuner=raw.get('tuner') if type(raw.get('tuner')) is bool else None)
            break
    completion = _read(case/'completion.json', COMPLETION_FIELDS, [], label+'.completion')
    closed = (bool(completion) and _status(terminal.get('status')) in TERMINAL
              and completion.get('status') == terminal.get('status')
              and completion.get('required_artifacts_persisted') is True
              and completion.get('registry_persisted') is True)
    return dict(case_id=case.name, status=_status(value.get('status')), phase=phase,
                phase_counts=phases, session=session or None, durable_completion=closed,
                complete_case_elapsed_seconds=_number(completion.get('complete_case_elapsed_seconds')) if closed else None,
                api_calls=_number(value.get('api_delta', value.get('api_calls')), integral=True))


def collect(args):
    missing = []
    batch = Path(args.batch_case)
    manifest = _read(batch/'manifest.json', {'jobs': {'*': JOB_FIELDS}}, missing, 'queue.manifest')
    progress = _read(batch/'progress.json', {'status': True, 'jobs': {'*': JOB_FIELDS}}, [], 'queue.progress')
    result = _read(batch/'result.json', {'status': True, 'jobs': {'*': JOB_FIELDS}}, [], 'queue.result')
    active = _read(batch/'active.json', JOB_FIELDS, [], 'queue.active')
    completion = _read(batch/'completion.json', COMPLETION_FIELDS, [], 'queue.completion')
    jobs = manifest.get('jobs')
    jobs = jobs if isinstance(jobs, list) else []
    observed_jobs = (result or progress).get('jobs')
    observed_jobs = observed_jobs if isinstance(observed_jobs, list) else []
    observed = {row.get('id'): row for row in observed_jobs
                if isinstance(row, dict) and isinstance(row.get('id'), str)}
    active_id = _label(active.get('id'))
    if active.get('status') != 'running' or observed.get(active_id, {}).get('status') not in (None, 'running'):
        active_id = None
    rows = []
    active_case = None
    for job in jobs:
        if not isinstance(job, dict):
            missing.append('queue.job_schema'); continue
        ident = _label(job.get('id'))
        if ident is None:
            missing.append('queue.job_id'); continue
        row = observed.get(ident) or (active if active_id == ident else {})
        rows.append(dict(id=ident, kind=job.get('kind') if job.get('kind') in ('archive','experiment') else 'experiment',
                         status=_status(row.get('status')) or 'pending',
                         experiment_status=_status(row.get('experiment_status')),
                         recorded_resource_limits=_resources(job.get('argv'), job.get('seconds'))))
        # Only a case declared in the explicit queue manifest is followed. No
        # filename from a SESSION or API artifact can redirect the observer.
        if active_id == ident and isinstance(job.get('out'), str):
            candidate = Path(job['out']).resolve()
            if candidate.is_relative_to(batch.resolve().parent) and candidate != batch.resolve().parent:
                active_case = candidate
            else:
                missing.append('active.case_path_outside_results')
    status = _status(result.get('status', progress.get('status'))) or 'running'
    done = (bool(result) and status in ('ok','error','completed_with_observations') and completion.get('status') == status
            and completion.get('required_artifacts_persisted') is True
            and completion.get('registry_persisted') is True)
    watch = case_summary(args.watch_case, missing, 'watch')
    current = (case_summary(active_case, missing, 'active') if active_case is not None
               and active_case.resolve() != Path(args.watch_case).resolve() else None)
    if done:
        active_id = None
    observed_at = datetime.now(timezone.utc)
    return dict(schema_version=1, observed_at_utc=observed_at.isoformat(),
                observed_at_local=observed_at.astimezone(ZoneInfo('Asia/Taipei')).isoformat(),
                display_timezone='Asia/Taipei',
                observer_status='observing', queue=dict(case_id=batch.name, status=status,
                  durable_completion=done, active_job_id=_label(active_id), jobs=rows),
                watched_case=watch, active_case=current,
                api=ledger_summary(args.ledger, args.allowance, missing),
                missing=sorted(set(missing)), experiment_actions=0,
                observer_limits=dict(address_space_mib=256, poll_seconds=args.poll_seconds,
                                     total_seconds=args.seconds, child_processes=0),
                scope='Read-only status and accounting metadata; no quality values or experiment payloads')


def _atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Publication files must be regular local files')
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def render(value, poll_seconds):
    escape = lambda value: html.escape('unknown' if value is None else str(value))
    rows = ''.join('<tr><td>'+escape(row['id'])+'</td><td>'+escape(row['status'])+
                   '</td><td>'+escape(row['recorded_resource_limits'])+'</td></tr>'
                   for row in value['queue']['jobs'])
    details = json.dumps({key: value[key] for key in ('watched_case','active_case','api','missing','observer_limits')},
                         ensure_ascii=False, indent=2, allow_nan=False)
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="'+str(int(poll_seconds))+'">'
            '<title>Live execution status</title><style>body{font:16px Arial,sans-serif;max-width:1100px;'
            'margin:32px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%}'
            'td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}pre{white-space:pre-wrap;'
            'overflow-wrap:anywhere;background:#f3f5f7;padding:16px}</style><h1>Live execution status</h1>'
            '<p>Observed '+escape(value['observed_at_local'])+' (Asia/Taipei) · '+escape(value['observer_status'])+'</p>'
            '<p>Evidence UTC: '+escape(value['observed_at_utc'])+'</p>'
            '<p>Queue '+escape(value['queue']['case_id'])+' · '+escape(value['queue']['status'])+
            ' · active job '+escape(value['queue']['active_job_id'])+'</p>'
            '<table><tr><th>Job</th><th>Status</th><th>Recorded resource limits</th></tr>'+rows+'</table>'
            '<pre>'+html.escape(details)+'</pre><p>This page only observes saved status metadata. '
            'Unknown fields remain explicit; stopping observation leaves experiments running.</p></html>\n')


def publish(directory, value, poll_seconds):
    _atomic(Path(directory)/PUBLIC_JSON, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    _atomic(Path(directory)/PUBLIC_HTML, render(value, poll_seconds))


def observe(args, *, clock=time.monotonic, sleep=time.sleep, collector=collect, publisher=publish, stopped=None):
    began = clock(); count = 0
    while True:
        value = collector(args); count += 1
        elapsed = max(0, clock()-began)
        reason = ('queue_complete' if value['queue']['durable_completion'] else
                  'observer_signal' if stopped is not None and stopped() else
                  'observation_limit_reached' if elapsed >= args.seconds else None)
        value.update(observer_status=reason or 'observing', observer_elapsed_seconds=elapsed, observations=count)
        publisher(args.publish_dir, value, args.poll_seconds)
        if reason:
            return value
        sleep(min(args.poll_seconds, max(0, args.seconds-(clock()-began))))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch-case','watch-case','ledger','allowance','publish-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.poll_seconds) or not 10 <= args.poll_seconds <= 3600
            or not math.isfinite(args.seconds) or not 0 < args.seconds <= MAX_SECONDS):
        parser.error('Use a finite observation up to seven days and polling from 10 to 3600 seconds')
    for name in ('batch_case','watch_case','ledger','allowance','publish_dir'):
        setattr(args, name, getattr(args, name).resolve())
    # Publications may not overwrite or enter any experiment source directory.
    for source in (args.batch_case, args.watch_case):
        if args.publish_dir == source or args.publish_dir.is_relative_to(source):
            parser.error('The publication directory must be separate from observed experiment cases')
    if any(path in (args.publish_dir/PUBLIC_JSON, args.publish_dir/PUBLIC_HTML)
           for path in (args.ledger,args.allowance)):
        parser.error('Publication names must differ from accounting input files')
    return args


def main(argv=None):
    args = parse_args(argv)
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    limit = min(256*MIB, hard) if hard != resource.RLIM_INFINITY else 256*MIB
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    interrupted = []
    handlers = {sig: signal.signal(sig, lambda signum, frame: interrupted.append(signum))
                for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        observe(args, stopped=lambda: bool(interrupted))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
