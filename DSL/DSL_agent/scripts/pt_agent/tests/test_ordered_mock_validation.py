"""Real fork fixtures for ordered callbacks, failure groups and cleanup."""
import copy
import multiprocessing
import os
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from pt_agent.experiments import ordered_mock_validation as entry


class FixtureArithmeticError(FloatingPointError):
    def __init__(self, message, *, original_batch):
        super().__init__(message)
        self.original_batch = original_batch


def fixture_execute(program, weights, inputs, *, engine='tensor', exact=False,
                    observer=None):
    x = inputs['x'].copy()
    start = int(x[0, 0])
    cfg = dict(domain=(-10., 100.), step=1)
    if observer is not None:
        if hasattr(observer, 'sample_event'):
            observer.sample_event(7, 'inv', x, mask=np.ones_like(x, dtype=bool))
        observer('first', x, cfg)
    cfg['step'] = 2
    x += weights['offset']
    if program.error_start == start:
        if program.error_kind == 'exit':
            os._exit(23)
        raise FixtureArithmeticError('original arithmetic failed', original_batch=start)
    output = x*1.125+.25
    if observer is not None:
        if hasattr(observer, 'inverse_event'):
            observer.inverse_event(7, 'inv', output, np.full_like(output, .5))
        observer('last', output, cfg)
    return output


execute = fixture_execute


def fixture_validate(program, ctx, n=None, *, engine='tensor', stop_on_divergence=True,
                     output_callback=None, diagnostic_callback=None, batch_callback=None):
    total = len(ctx.val_images) if n is None else min(n, len(ctx.val_images))
    output = np.full_like(ctx.val_images[:total], np.nan)
    events = []
    diverged = []
    done = 0
    failures = []
    violations = None
    if program.original_domain_class:
        from pt_agent.domain_criteria import ViolationOrder
        point = SimpleNamespace(op=SimpleNamespace(id=7, kind='softmax', attrs={'steps': 0}),
                                role='inv', attrs={'fn': 'inv'})
        row = dict(domain=(.1, 5.), domain_source='analytic',
                   program_line=1, graph_tensor='original_inv')
        violations = ViolationOrder([point], [row], total, version='r8-v11')
    for start in range(0, total, 10):
        stop = min(start+10, total)

        def record(kind, *args):
            events.append((len(events), start, kind, copy.deepcopy(args)))

        def observer(name, value, cfg):
            record('point', name, value, cfg)
            if program.callback_error and start == 10:
                raise FloatingPointError('original parent observer failed')

        if program.with_events:
            # Keep the original named-mask callback interface.
            def sample_event(op_id, role, value, mask=None):
                record('sample', op_id, role, value, mask)
                if violations is not None:
                    violations.observe(op_id, role, value, mask, start=start)
            observer.sample_event = sample_event
            def inverse_event(op_id, role, value, inverse):
                record('inverse', op_id, role, value, inverse)
                if violations is not None:
                    violations.inverse_result(op_id, role, value, inverse, start=start)
            observer.inverse_event = inverse_event
        try:
            output[start:stop] = execute(program, ctx.weights,
                {key: value[start:stop] for key, value in ctx.val_inputs.items()},
                engine=engine, observer=observer)
        except FloatingPointError as exc:
            diverged.extend(range(start, stop))
            failures.append(dict(exception=type(exc).__name__,
                                 original_batch=getattr(exc, 'original_batch', None)))
            if stop_on_divergence:
                break
        done = stop
        if batch_callback is not None:
            batch_callback(start, stop)
    if output_callback is not None:
        output_callback(output.copy())
    if diagnostic_callback is not None:
        diagnostic_callback(events)
    result = dict(output=output, events=events, diverged=diverged, done=done, failures=failures)
    if violations is not None:
        result['violation_report'] = violations.report(engine=engine, split='heldout')
    return result


def fixture(count=35, **updates):
    program = SimpleNamespace(error_start=None, error_kind='arithmetic',
                              callback_error=False, with_events=True, original_domain_class=False)
    for key, value in updates.items():
        setattr(program, key, value)
    values = np.arange(count, dtype=np.float64).reshape(-1, 1)
    ctx = SimpleNamespace(weights={'offset': .125}, val_inputs={'x': values},
                          val_images=values, val_split='heldout')
    return program, ctx


class OrderedValidationTests(unittest.TestCase):
    def assertSame(self, left, right):
        np.testing.assert_equal(left['output'], right['output'])
        for key in ('diverged', 'done', 'failures'):
            self.assertEqual(left[key], right[key])
        if 'violation_report' in left:
            self.assertEqual(left['violation_report'], right['violation_report'])
        self.assertEqual(len(left['events']), len(right['events']))
        for a, b in zip(left['events'], right['events']):
            self.assertEqual(a[:3], b[:3])
            self.assertEqual(len(a[3]), len(b[3]))
            for x, y in zip(a[3], b[3]):
                if isinstance(x, np.ndarray):
                    np.testing.assert_equal(x, y)
                else:
                    self.assertEqual(x, y)

    def runPair(self, count=35, *, run_kwargs=None, entry_kwargs=None, **updates):
        program, ctx = fixture(count, **updates)
        kwargs = {'engine': 'mock'}
        kwargs.update(run_kwargs or {})
        reference = fixture_validate(program, ctx, **kwargs)
        telemetry = []
        current = entry.run_original_validate(fixture_validate, execute, program, ctx,
            batch_size=10, telemetry=telemetry.append, **(entry_kwargs or {}), **kwargs)
        self.assertSame(reference, current)
        return current, telemetry

    def test_original_batches_partial_tail_callbacks_and_values(self):
        before = {p.pid for p in multiprocessing.active_children()}
        result, telemetry = self.runPair()
        self.assertEqual(result['done'], 35)
        self.assertEqual([(e['start'], e['stop']) for e in telemetry
                          if e['event'] == 'batch_received'], [(0, 10), (10, 20), (20, 30), (30, 35)])
        self.assertEqual([e['event'] for e in telemetry[:3]], ['batch_started']*3)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        self.assertIs(fixture_validate.__globals__['execute'], fixture_execute)

    def test_no_domain_event_interface_additions(self):
        result, _ = self.runPair(with_events=False)
        self.assertTrue(all(event[2] == 'point' for event in result['events']))

    def test_original_keyword_only_numerical_exception_and_stop_group(self):
        result, telemetry = self.runPair(error_start=10)
        self.assertEqual(result['diverged'], list(range(10, 20)))
        self.assertEqual(result['done'], 10)
        self.assertEqual(result['failures'], [dict(exception='FixtureArithmeticError', original_batch=10)])
        self.assertTrue(any(e.get('kind') == 'execution_error' for e in telemetry))

    def test_continue_after_original_failed_batch(self):
        result, _ = self.runPair(error_start=10, run_kwargs={'stop_on_divergence': False})
        self.assertEqual(result['done'], 35)
        self.assertEqual(result['diverged'], list(range(10, 20)))

    def test_parent_observer_failure_keeps_original_event_prefix(self):
        result, _ = self.runPair(callback_error=True)
        self.assertEqual(result['diverged'], list(range(10, 20)))

    def test_parent_observer_failure_can_continue_sequentially(self):
        result, _ = self.runPair(callback_error=True, run_kwargs={'stop_on_divergence': False})
        self.assertEqual(result['done'], 35)

    def test_capture_allowance_uses_original_route_without_partial_replay(self):
        result, telemetry = self.runPair(entry_kwargs={'capture_bytes': 1})
        self.assertEqual(result['done'], 35)
        self.assertEqual([e.get('kind') for e in telemetry if e['event'] == 'batch_received'], ['capture_limit'])

    def test_tensor_and_single_original_batch_do_not_fork(self):
        for count, engine in ((35, 'tensor'), (9, 'mock')):
            with self.subTest(count=count, engine=engine):
                _, telemetry = self.runPair(count, run_kwargs={'engine': engine})
                self.assertEqual(telemetry, [])

    def test_caller_n_and_all_three_original_callbacks_preserved(self):
        program, ctx = fixture()
        outputs = []
        diagnostics = []
        batches = []
        current = entry.run_original_validate(fixture_validate, execute, program, ctx, 23,
            batch_size=10, engine='mock', output_callback=outputs.append,
            diagnostic_callback=diagnostics.append, batch_callback=lambda *x: batches.append(x))
        reference = fixture_validate(program, ctx, 23, engine='mock')
        self.assertSame(reference, current)
        np.testing.assert_equal(outputs[0], reference['output'])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(batches, [(0, 10), (10, 20), (20, 23)])

    def test_cache_release_hook_and_configuration_bounds(self):
        releases = []
        self.runPair(entry_kwargs={'before_batch': lambda: releases.append(True)})
        self.assertEqual(len(releases), 4)
        program, ctx = fixture()
        for kwargs in ({'workers': 0}, {'workers': 4}, {'capture_bytes': 0},
                       {'capture_bytes': entry.MAX_CAPTURE_BYTES+1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                entry.run_original_validate(fixture_validate, execute, program, ctx,
                    batch_size=10, engine='mock', **kwargs)

    def test_existing_python_thread_keeps_original_sequential_route(self):
        event = threading.Event()
        thread = threading.Thread(target=event.wait)
        thread.start()
        try:
            _, telemetry = self.runPair()
            self.assertEqual([e['event'] for e in telemetry], ['sequential_fallback'])
        finally:
            event.set()
            thread.join(timeout=2)

    def test_lost_owned_child_is_driver_error_and_all_children_close(self):
        program, ctx = fixture(error_start=10, error_kind='exit')
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaises((RuntimeError, EOFError)):
            entry.run_original_validate(fixture_validate, execute, program, ctx,
                                        batch_size=10, engine='mock')
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_cleanup_error_still_attempts_all_owned_handles(self):
        class Handle:
            def __init__(self, pid, stuck=False):
                self.pid, self.stuck, self.alive, self.joined, self.closed = pid, stuck, True, 0, False
            def is_alive(self):return self.alive
            def terminate(self):self.alive = self.stuck
            def join(self, timeout):self.joined += 1
            def kill(self):self.alive = self.stuck
            def close(self):self.closed = True
        program, ctx = fixture()
        owner = entry.OrderedExecutor(execute, program, ctx, 35, 10)
        first, second = Handle(1, stuck=True), Handle(2)
        owner.jobs = [(0, 10, first, SimpleNamespace(close=lambda: None)),
                      (10, 20, second, SimpleNamespace(close=lambda: None))]
        with self.assertRaises(RuntimeError):owner.close()
        self.assertEqual(first.joined, 2)
        self.assertEqual(second.joined, 1)
        self.assertTrue(second.closed)

    def test_logger_failure_cannot_leave_owned_children_running(self):
        program, ctx = fixture()
        before = {p.pid for p in multiprocessing.active_children()}
        def logger(event):raise OSError('tiny fixture logger failed')
        with self.assertRaises(OSError):
            entry.run_original_validate(fixture_validate, execute, program, ctx,
                                        batch_size=10, engine='mock', telemetry=logger)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_original_violation_order_class_and_global_event_indices(self):
        result, _ = self.runPair(original_domain_class=True)
        samples = result['violation_report']['samples']
        self.assertEqual(samples[10]['first_violation']['event_index'], 1)
        self.assertEqual(samples[30]['first_violation']['event_index'], 3)

    def test_original_domain_reports_discard_unconsumed_prefetch(self):
        for keep_going in (False, True):
            with self.subTest(keep_going=keep_going):
                result, _ = self.runPair(original_domain_class=True, error_start=10,
                    run_kwargs={'stop_on_divergence': not keep_going})
                self.assertEqual(result['violation_report']['samples'][20]['observed'], keep_going)


if __name__ == '__main__':
    unittest.main()
