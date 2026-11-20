"""R10 serial stage records and Linux child supervision (standard library only)."""

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class StageRecorder:
    def __init__(self, path, clock=time.perf_counter_ns):
        self.path = Path(path)
        self.clock = clock
        self.stack = []
        self.sequence = 0

    def emit(self, value):
        with self.path.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()

    @contextmanager
    def span(self, stage_id, **metadata):
        self.sequence += 1
        event = dict(event_id=f'{os.getpid()}:{self.sequence}', stage_id=stage_id,
                     parent_id=self.stack[-1]['event_id'] if self.stack else None,
                     start_ns=self.clock(), pid=os.getpid(), ppid=os.getppid(),
                     metadata=metadata, cache_hit=bool(metadata.get('cache_hit', False)),
                     artifact=metadata.get('artifact'))
        self.emit(dict(event, type='begin'))
        self.stack.append(event)
        status, error = 'ok', None
        try:
            yield event
        except BaseException as exc:
            status, error = 'error', f'{type(exc).__name__}: {exc}'[:2000]
            raise
        finally:
            if not self.stack or self.stack[-1] is not event:
                raise RuntimeError('Stage spans must close in stack order')
            self.stack.pop()
            self.emit(dict(event, type='end', end_ns=self.clock(), status=status, error=error))


def read_events(path):
    if not Path(path).exists():
        return []
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed writer can leave one incomplete last line.
            events.append(dict(type='invalid', reason='incomplete event line'))
    return events


def summarize_events(events, termination_observed_ns=None):
    spans, invalid = {}, []
    stacks = {}
    for event in events:
        kind, key = event.get('type'), event.get('event_id')
        if kind == 'begin':
            if key in spans:
                invalid.append('duplicate begin: ' + str(key))
            spans[key] = dict(event, end_ns=None, inclusive_ns=None, exclusive_ns=None,
                              status='interrupted', termination_observed_ns=termination_observed_ns)
            stacks.setdefault(event.get('pid'), []).append(key)
        elif kind == 'end':
            if key not in spans or spans[key]['end_ns'] is not None:
                invalid.append('unmatched end: ' + str(key))
                continue
            pid = spans[key].get('pid')
            if event.get('pid', pid) != pid:
                invalid.append('end PID differs from begin: ' + str(key))
            # Event files from separate workers can be appended after the
            # outer file. Only ordering within the same PID is a serial stream.
            stack = stacks.setdefault(pid, [])
            if not stack or stack[-1] != key:
                invalid.append('non-LIFO end: ' + str(key))
                if key in stack:
                    stack.remove(key)
            else:
                stack.pop()
            spans[key].update(event)
            duration = event['end_ns'] - spans[key]['start_ns']
            spans[key]['inclusive_ns'] = duration
            if duration < 0:
                invalid.append('negative duration: ' + str(key))
        else:
            invalid.append(event.get('reason', 'unknown event'))
    for key, span in spans.items():
        parent = span.get('parent_id')
        if parent is not None and parent not in spans:
            invalid.append('missing parent: ' + str(parent))
        children = sorted((child for child in spans.values() if child.get('parent_id') == key),
                          key=lambda item: item['start_ns'])
        end = span['end_ns']
        complete = end is not None
        last_end = span['start_ns']
        child_total = 0
        for child in children:
            child_end = child['end_ns']
            if child['start_ns'] < span['start_ns'] or child['start_ns'] < last_end:
                invalid.append('overlapping or out-of-parent child: ' + str(child['event_id']))
            if child_end is None:
                complete = False
            else:
                if end is not None and child_end > end:
                    invalid.append('child ends after parent: ' + str(child['event_id']))
                last_end = max(last_end, child_end)
                child_total += child_end - child['start_ns']
        if complete:
            span['exclusive_ns'] = span['inclusive_ns'] - child_total
    valid = not invalid
    if not valid:
        for span in spans.values():
            span['exclusive_ns'] = None
    return dict(status='ok' if valid else 'timing_invalid', diagnostics=invalid,
                spans=list(spans.values()), attribution_ready=valid and all(
                    span['end_ns'] is not None for span in spans.values()))


class UnsupportedSupervision(RuntimeError):
    pass


class _Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        raise UnsupportedSupervision('PR_GET_CHILD_SUBREAPER unavailable')
    if value is not None and libc.prctl(36, int(value), 0, 0, 0) != 0:
        raise UnsupportedSupervision('PR_SET_CHILD_SUBREAPER unavailable')
    return current.value


def _process(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        values = dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                      pgid=int(fields[2]), starttime=int(fields[19]))
        status = (Path('/proc') / str(pid) / 'status').read_text()
        for line in status.splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
                name, amount, *_ = line.split()
                values[name.rstrip(':').lower() + '_kib'] = int(amount)
        return values
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return None


def _scan():
    return {item['pid']: item for path in Path('/proc').iterdir()
            if path.name.isdigit() for item in [_process(path.name)] if item is not None}


def run_process(argv, *, cwd, log, resources, timeout, mem_gib,
                fsize_gib=8, env=None, term_grace=5):
    """Run one phase; reap its descendants even when they create new sessions."""
    prior_subreaper = _subreaper(True)
    original_children = {pid for pid, item in _scan().items() if item['ppid'] == os.getpid()}
    known, peak, signals_sent = {}, {}, []
    outer_peak = 0
    started = time.perf_counter_ns()
    process = None
    timed_out, interrupted = False, None
    old_handlers = {}

    def interrupt(signum, frame):
        raise _Interrupted(signum)

    def limits():
        memory = int(mem_gib * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        size = int(fsize_gib * 1024**3)
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))

    def update():
        table = _scan()
        family = {pid for pid, birth in known.items()
                  if pid in table and table[pid]['starttime'] == birth}
        family.update(pid for pid, item in table.items() if item['ppid'] == os.getpid()
                      and pid not in original_children)
        changed = True
        while changed:
            extra = {pid for pid, item in table.items() if item['ppid'] in family}
            changed = not extra.issubset(family)
            family.update(extra)
        rows = []
        for pid in family:
            item = table.get(pid)
            if item is None:
                continue
            known[pid] = item['starttime']
            peak[pid] = max(peak.get(pid, 0), item.get('vmhwm_kib', 0))
            rows.append(item)
        return rows

    def send(signum):
        groups = set()
        for item in update():
            if item['state'] == 'Z':
                continue
            fresh = _process(item['pid'])
            if fresh is None or fresh['starttime'] != known[item['pid']]:
                continue
            try:
                if fresh['pgid'] != os.getpgrp() and fresh['pgid'] not in groups:
                    os.killpg(fresh['pgid'], signum)
                    groups.add(fresh['pgid'])
                else:
                    os.kill(fresh['pid'], signum)
                signals_sent.append(dict(pid=fresh['pid'], signal=signum))
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupt)
        with Path(log).open('w') as output, Path(resources).open('a') as samples:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=limits)
            item = _process(process.pid)
            if item:
                known[process.pid] = item['starttime']
            try:
                while True:
                    rows = update()
                    outer = _process(os.getpid())
                    outer_peak = max(outer_peak, (outer or {}).get('vmhwm_kib', 0))
                    samples.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), processes=rows,
                                                  outer_process=outer)) + '\n')
                    samples.flush()
                    if process.poll() is not None:
                        break
                    if (time.perf_counter_ns() - started) / 1e9 >= timeout:
                        timed_out = True
                        break
                    time.sleep(0.1)
            except _Interrupted as exc:
                interrupted = exc.signum
            finally:
                # Ignore repeated cancellation while finishing this phase's cleanup.
                for signum in old_handlers:
                    signal.signal(signum, signal.SIG_IGN)
                if any(item['state'] != 'Z' for item in update()):
                    send(signal.SIGTERM)
                    deadline = time.monotonic() + term_grace
                    while time.monotonic() < deadline and any(
                            item['state'] != 'Z' for item in update()):
                        time.sleep(0.05)
                    if any(item['state'] != 'Z' for item in update()):
                        send(signal.SIGKILL)
                process.wait(timeout=5)
                for pid in known:
                    if pid != process.pid:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
        remaining = [item['pid'] for item in update() if item['state'] != 'Z']
        return dict(exit_code=process.returncode, timed_out=timed_out, signal=interrupted,
                    started_ns=started, ended_ns=time.perf_counter_ns(),
                    termination_observed_ns=time.perf_counter_ns(),
                    peak_rss_kib_by_pid={str(k): v for k, v in peak.items()},
                    outer_peak_rss_kib=outer_peak,
                    limits=dict(memory_gib=mem_gib, file_gib=fsize_gib, wall_seconds=timeout),
                    signals_sent=signals_sent, remaining_pids=remaining)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        _subreaper(prior_subreaper)
