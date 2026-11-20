"""Optional ordered execution of the original held-out Mock validation batches.

This module is dormant unless a new worker explicitly calls it. It preserves
the original validation function, batch boundaries and parent-side observers.
Only independent original execute calls run in owned fork children. Child spans
overlap; only the original parent validation span is an additive wall clock.
It exposes
no model, compiler, provider, checkpoint, report or filesystem entry point.
"""
import copy
import multiprocessing
import os
import threading
import time
from types import FunctionType


MIB = 1024**2
MAX_CAPTURE_BYTES = 256*MIB
MAX_EVENTS = 8192


class CaptureLimit(RuntimeError):
    pass


def _snapshot(value, budget):
    """Own a callback value at its original observation time, with bounded RAM."""
    if value is None or type(value) in (bool, int, float):
        budget[0] += 32
        result = value
    elif type(value) is str:
        budget[0] += 4*len(value)+64
        result = value
    elif type(value) in (list, tuple):
        budget[0] += 64+16*len(value)
        result = type(value)(_snapshot(item, budget) for item in value)
    elif type(value) is dict:
        budget[0] += 128+64*len(value)
        result = {_snapshot(key, budget): _snapshot(item, budget)
                  for key, item in value.items()}
    elif type(value).__module__.split('.')[0] == 'numpy' and hasattr(value, 'nbytes'):
        budget[0] += int(value.nbytes)+128
        if budget[0] > budget[1]:
            raise CaptureLimit('Original observer data exceed the private IPC allowance')
        result = value.copy()
    else:
        raise CaptureLimit('Original callback contains an unsupported snapshot type')
    if budget[0] > budget[1]:
        raise CaptureLimit('Original observer data exceed the private IPC allowance')
    return result


def _child(connection, execute, program, weights, batch, start, stop, flags, limit):
    # Every child belongs to this one validation. Native arithmetic is delegated.
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    events = []
    budget = [0, limit]
    began = time.perf_counter_ns()

    def record(kind, *args):
        if len(events) >= MAX_EVENTS:
            raise CaptureLimit('Original observer event count exceeds the private allowance')
        events.append((kind, _snapshot(args, budget)))

    def observe(name, value, cfg):
        record('point', name, value, cfg)

    if flags[0]:
        def sample_event(op_id, role, value, mask=None):
            record('sample', op_id, role, value, mask)
        observe.sample_event = sample_event
    if flags[1]:
        def inverse_event(op_id, role, value, inverse):
            record('inverse', op_id, role, value, inverse)
        observe.inverse_event = inverse_event
    try:
        output = execute(program, weights, batch, engine='mock', observer=observe)
        output = _snapshot(output, budget)
        message = dict(kind='ok', events=events, output=output)
    except CaptureLimit as exc:
        # Re-execute this batch in the parent with the original observer, then
        # use the original sequential route for all remaining batches.
        message = dict(kind='capture_limit', error_class=type(exc).__name__)
    except BaseException as exc:
        # Do not pickle exception instances: some original numerical exceptions
        # require keyword-only construction. Parent replay raises the original.
        message = dict(kind='execution_error', error_class=type(exc).__name__)
    message.update(start=start, stop=stop, pid=os.getpid(), captured_bytes=budget[0],
                   execute_ns=time.perf_counter_ns()-began)
    try:
        connection.send(message)
    finally:
        connection.close()


class OrderedExecutor:
    """Replay complete original batches in order; preserve failure grouping."""

    def __init__(self, execute, program, ctx, total, batch_size, *, workers=3,
                 capture_bytes=MAX_CAPTURE_BYTES, telemetry=None, before_batch=None):
        if type(workers) is not int or not 2 <= workers <= 3:
            raise ValueError('Use two or three owned original-batch workers')
        if type(capture_bytes) is not int or not 0 < capture_bytes <= MAX_CAPTURE_BYTES:
            raise ValueError('Use a bounded original-observer capture allowance')
        self.execute = execute
        self.program = program
        self.ctx = ctx
        self.total = total
        self.batch_size = batch_size
        self.workers = workers
        self.capture_bytes = capture_bytes
        self.telemetry = telemetry
        self.before_batch = before_batch
        self.context = multiprocessing.get_context('fork')
        self.cursor = 0
        self.submitted = 0
        self.flags = None
        self.jobs = []
        self.closed = False
        self.disabled = threading.active_count() != 1
        if self.disabled:
            self.report('sequential_fallback', reason='parent_has_python_threads')

    def report(self, event, **fields):
        if self.telemetry is not None:
            self.telemetry(dict(event=event, **fields))

    def _prime(self):
        while len(self.jobs) < self.workers and self.submitted < self.total:
            start = self.submitted
            stop = min(start+self.batch_size, self.total)
            batch = {key: value[start:stop] for key, value in self.ctx.val_inputs.items()}
            receiving, sending = self.context.Pipe(duplex=False)
            process = self.context.Process(target=_child,
                args=(sending, self.execute, self.program, self.ctx.weights, batch,
                      start, stop, self.flags, self.capture_bytes))
            try:
                process.start()
            except BaseException:
                receiving.close()
                sending.close()
                raise
            sending.close()
            self.jobs.append((start, stop, process, receiving))
            self.submitted = stop
            self.report('batch_started', start=start, stop=stop, pid=process.pid)

    def close(self):
        if self.closed:
            return
        self.closed = True
        pending, self.jobs = self.jobs, []
        cleanup = []
        errors = []
        for _, _, process, connection in pending:
            try:
                connection.close()
                if process.is_alive():
                    cleanup.append(dict(pid=process.pid, signal=15))
                    process.terminate()
            except Exception as exc:
                errors.append(type(exc).__name__)
        for _, _, process, _ in pending:
            try:
                process.join(timeout=2)
                if process.is_alive():
                    cleanup.append(dict(pid=process.pid, signal=9))
                    process.kill()
                    process.join(timeout=2)
                if process.is_alive():
                    errors.append('OwnedWorkerStillAlive')
                else:
                    process.close()
            except Exception as exc:
                errors.append(type(exc).__name__)
        # Record only after every child is closed, so a logger failure cannot
        # interrupt owned-process cleanup.
        for event in cleanup:
            self.report('unused_prefetch_cleanup', **event)
        if errors:
            self.report('owned_cleanup_error', errors=errors)
            raise RuntimeError('Owned original-batch cleanup was incomplete: '+','.join(errors))

    def __call__(self, program, weights, inputs, *, engine='tensor', exact=False,
                 observer=None):
        began = time.perf_counter_ns()
        if (engine != 'mock' or exact or self.disabled or observer is None):
            return self.execute(program, weights, inputs, engine=engine,
                                exact=exact, observer=observer)
        if program is not self.program or weights is not self.ctx.weights:
            raise ValueError('Ordered execution requires the original program and weight objects')
        flags = (hasattr(observer, 'sample_event'), hasattr(observer, 'inverse_event'))
        if self.flags is None:
            self.flags = flags
        if flags != self.flags:
            raise ValueError('Original observer event interfaces changed within validation')
        expected_stop = min(self.cursor+self.batch_size, self.total)
        if (set(inputs) != set(self.ctx.val_inputs)
                or {len(value) for value in inputs.values()} != {expected_stop-self.cursor}):
            raise ValueError('Keep the complete original validation batch boundaries')
        if self.before_batch is not None:
            self.before_batch()
        self._prime()
        start, stop, process, connection = self.jobs[0]
        if start != self.cursor or stop != expected_stop:
            raise ValueError('Keep the original ordered sample indices')
        try:
            while not connection.poll(0.2):
                if not process.is_alive():
                    raise RuntimeError('An original-batch worker exited without a result')
            message = connection.recv()
            if message.get('pid') != process.pid:
                raise ValueError('Original-batch result is from a different owned child')
            process.join(timeout=2)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError('An original-batch worker did not exit cleanly')
            connection.close()
            process.close()
            self.jobs.pop(0)
            if (message.get('start'), message.get('stop')) != (start, stop):
                raise ValueError('Original-batch result indices differ')
            self.cursor = stop
            self.report('batch_received', start=start, stop=stop, kind=message.get('kind'),
                        execute_ns=message.get('execute_ns'),
                        captured_bytes=message.get('captured_bytes'),
                        child_span_scope='original_execute_and_callback_capture; overlapping_work_not_additive_wall')
            if message['kind'] in ('capture_limit', 'execution_error'):
                self.close()
                self.disabled = True
                # No child event has been replayed for this failing batch.
                # Original execution reconstructs its precise partial observers
                # and exception class, including original stop_on_divergence.
                output = self.execute(program, weights, inputs, engine=engine,
                                      exact=exact, observer=observer)
                if message['kind'] == 'execution_error':
                    raise RuntimeError('Original parent execution did not reproduce the child failure')
                self.report('batch_replayed', start=start, stop=stop,
                            parent_call_ns=time.perf_counter_ns()-began, route='original_sequential_fallback')
                return output
            if message['kind'] != 'ok':
                raise ValueError('Unknown private original-batch result')
            for kind, event_args in message['events']:
                target = (observer if kind == 'point' else observer.sample_event if kind == 'sample'
                          else observer.inverse_event if kind == 'inverse' else None)
                if target is None:
                    raise ValueError('Unknown original observer event')
                target(*event_args)
            output = message['output']
            del message
            self._prime()
            self.report('batch_replayed', start=start, stop=stop,
                        parent_call_ns=time.perf_counter_ns()-began, route='ordered_original_batch')
            return output
        except BaseException:
            self.close()
            self.disabled = True
            raise


def run_original_validate(validate, execute, program, ctx, n=None, *, batch_size,
                          workers=3, capture_bytes=MAX_CAPTURE_BYTES,
                          telemetry=None, before_batch=None, **kwargs):
    """Clone only validation globals; the shared module and code stay intact."""
    if kwargs.get('engine', 'tensor') != 'mock':
        return validate(program, ctx, n, **kwargs)
    total = len(ctx.val_images) if n is None else min(n, len(ctx.val_images))
    if total <= batch_size:
        return validate(program, ctx, n, **kwargs)
    if validate.__globals__.get('execute') is not execute:
        raise ValueError('Original validate must use the provided original execute')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Preserve a positive original batch size')
    executor = OrderedExecutor(execute, program, ctx, total, batch_size,
                              workers=workers, capture_bytes=capture_bytes,
                              telemetry=telemetry, before_batch=before_batch)
    namespace = dict(validate.__globals__, execute=executor)
    clone = FunctionType(validate.__code__, namespace, validate.__name__,
                         validate.__defaults__, validate.__closure__)
    clone.__kwdefaults__ = copy.copy(validate.__kwdefaults__)
    try:
        return clone(program, ctx, n, **kwargs)
    finally:
        executor.close()


def parallel_mock_validate(program, ctx, n=None, *, workers=3, telemetry=None,
                           **kwargs):
    """Optional fresh-worker held-out wrapper; no current run is modified."""
    from .. import transformer_validation as original
    from ..cost import release_estimate_caches
    from ..planners.agent_tools import VAL_BATCH, MOCK_VAL_BATCH
    if kwargs.get('engine', 'tensor') == 'mock' and ctx.val_split != 'heldout':
        raise ValueError('This optional entry is restricted to internal held-out validation')
    if kwargs.get('engine', 'tensor') == 'mock' and not callable(telemetry):
        raise ValueError('A fresh worker must persist ordered-batch telemetry')
    return run_original_validate(original.validate, original.execute, program, ctx, n,
        batch_size=min(VAL_BATCH, MOCK_VAL_BATCH), workers=workers,
        telemetry=telemetry, before_batch=release_estimate_caches, **kwargs)
