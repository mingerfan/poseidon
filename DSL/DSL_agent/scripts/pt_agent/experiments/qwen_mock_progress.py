"""Counts and small progress records around unchanged streaming Mock arithmetic."""

from contextlib import contextmanager
import json
from pathlib import Path
import resource
import threading
import time

from ..backends.streaming_mock import StreamingMockBackend
from .r10_paper_metrics import write_json


class ObservedStreamingMockBackend(StreamingMockBackend):
    """Count completed operations without retaining their arrays or expressions."""

    def __init__(self, slots, batch=None):
        super().__init__(slots, batch=batch)
        self.attempted = dict(add=0, sub=0, neg=0, mul_cc=0, mul_cp=0, rot=0, boot=0)
        self.completed = dict(self.attempted)
        self.rot_steps = set()

    def _apply(self, kind, left, right=None, *, step=None):
        self.attempted[kind] += 1
        value = super()._apply(kind, left, right, step=step)
        self.completed[kind] += 1
        if kind == 'rot':
            self.rot_steps.add(step)
        return value

    def stats(self):
        return dict(counts=dict(self.completed), counts_attempted=dict(self.attempted),
                    ops_total=sum(self.completed.values()),
                    ops_attempted=sum(self.attempted.values()),
                    distinct_consts=len(self._constants), rot_steps=sorted(self.rot_steps))


def memory_record():
    result = dict(worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmSize:')):
            key, value = line.split(':', 1)
            result[{'VmRSS':'worker_rss_kib', 'VmSize':'worker_vmsize_kib'}[key]] = int(value.split()[0])
    return result


@contextmanager
def progress_records(backend, program, work, result, *, interval=10.):
    """Write metadata only, in a file separate from the worker result."""
    stop = threading.Event()
    work = Path(work)
    timeline = work/'mock-progress.jsonl'
    latest = work/'mock-progress.json'
    started = time.monotonic()
    operations = {op.id:dict(id=op.id, kind=op.kind, line=op.lineno) for op in program.ops}
    errors = []

    def capture(status):
        counts = backend.stats()
        row = dict(status=status, elapsed_seconds=time.monotonic()-started,
                   observed_ns=time.perf_counter_ns(), active_op=operations.get(backend.op_context),
                   counts=counts['counts'], counts_attempted=counts['counts_attempted'],
                   ops_total=counts['ops_total'], ops_attempted=counts['ops_attempted'],
                   distinct_constants=counts['distinct_consts'], memory=memory_record())
        write_json(latest, row)
        with timeline.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        result['mock_progress'] = row

    def observe():
        while not stop.wait(interval):
            try:
                capture('running')
            except BaseException as exc:
                errors.append(repr(exc))
                return

    capture('running')
    thread = threading.Thread(target=observe, name='qwen-mock-progress', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        capture('stopped')
        if errors:
            raise RuntimeError('Mock progress metadata could not be preserved: '+errors[0])
