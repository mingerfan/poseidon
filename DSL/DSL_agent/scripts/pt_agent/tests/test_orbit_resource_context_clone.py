"""Enter the real private contexts without any model or frontend execution."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_common_resource_retry as subject
from pt_agent.experiments import orbit_common_model_frontend as frontend
from pt_agent.experiments import orbit_common_model_worker as worker


class OrbitResourceContextCloneChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ('subprocess.Popen', 'socket.getaddrinfo',
                       'socket.socket.connect', 'urllib.request.urlopen'):
            blocker = patch(target, side_effect=AssertionError('No model, trace or transport in context tests'))
            blocker.start()
            self.addCleanup(blocker.stop)

    @contextmanager
    def alarm(self, *, active=False):
        handler = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        signal.setitimer(signal.ITIMER_REAL, 0)
        def existing(signum, frame):
            raise AssertionError('Existing timer must not expire in this short test')
        signal.signal(signal.SIGALRM, existing)
        if active:
            signal.setitimer(signal.ITIMER_REAL, 1200, 13)
        try:
            yield existing, signal.getitimer(signal.ITIMER_REAL)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            if timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *timer)

    def test_stdlib_wrapper_globals_and_private_generator_globals_stay_separate(self):
        namespace = subject.worker_namespace()
        for name in ('_wall_limit', '_phase'):
            original = getattr(worker, name)
            cloned = namespace[name]
            self.assertIs(cloned.__code__, original.__code__)
            self.assertIs(cloned.__globals__, original.__globals__)
            self.assertIn('_GeneratorContextManager', cloned.__globals__)
            self.assertIs(cloned.__wrapped__.__code__, original.__wrapped__.__code__)
            self.assertIs(cloned.__wrapped__.__globals__, namespace)
            self.assertIs(original.__wrapped__.__globals__, vars(worker))
        self.assertIs(namespace['trace_model'].__globals__, namespace)

    def test_real_phase_body_uses_private_checkpoint_before_and_after_yield(self):
        namespace = subject.worker_namespace()
        calls = []
        args = object()
        result = {}
        namespace['_checkpoint'] = lambda a, r: calls.append(('checkpoint', a, dict(r)))
        class Recorder:
            @contextmanager
            def span(self, name):
                calls.append(('begin', name))
                try:
                    yield
                finally:
                    calls.append(('end', name))
        with patch.object(worker, '_checkpoint', side_effect=AssertionError('Original module hook must stay unused')):
            with namespace['_phase'](args, Recorder(), result, 'synthetic-context'):
                calls.append(('body', result['failed_phase']))
                result['changed_in_body'] = True
        self.assertEqual([row[0] for row in calls], ['checkpoint', 'begin', 'body', 'end', 'checkpoint'])
        self.assertIs(calls[0][1], args)
        self.assertTrue(calls[-1][2]['changed_in_body'])

    def test_real_phase_exception_preserves_original_exit_behavior(self):
        namespace = subject.worker_namespace()
        events = []
        namespace['_checkpoint'] = lambda *args: events.append('checkpoint')
        class Recorder:
            @contextmanager
            def span(self, name):
                events.append('begin')
                try:
                    yield
                finally:
                    events.append('end')
        with self.assertRaisesRegex(ValueError, 'synthetic failure'):
            with namespace['_phase'](None, Recorder(), {}, 'synthetic-context'):
                raise ValueError('synthetic failure')
        self.assertEqual(events, ['checkpoint', 'begin', 'end'])

    def test_real_wall_context_installs_and_restores_existing_alarm_and_timer(self):
        namespace = subject.worker_namespace()
        with self.alarm(active=True) as (existing, before):
            with namespace['_wall_limit'](600):
                self.assertIsNot(signal.getsignal(signal.SIGALRM), existing)
                remaining, interval = signal.getitimer(signal.ITIMER_REAL)
                self.assertGreater(remaining, 0)
                self.assertLessEqual(remaining, 600)
                self.assertEqual(interval, 0)
            self.assertIs(signal.getsignal(signal.SIGALRM), existing)
            after = signal.getitimer(signal.ITIMER_REAL)
            self.assertGreater(after[0], 0)
            self.assertLessEqual(after[0], before[0])
            self.assertEqual(after[1], before[1])

    def test_real_wall_timeout_handler_raises_original_failure_and_restores_alarm(self):
        namespace = subject.worker_namespace()
        with self.alarm() as (existing, before):
            with self.assertRaises(worker.WorkerFailure) as caught:
                with namespace['_wall_limit'](600):
                    signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
            self.assertEqual(caught.exception.status, 'timeout')
            self.assertIs(signal.getsignal(signal.SIGALRM), existing)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), before)

    def run_worker(self, phase, memory, *, fail=False):
        namespace = subject.worker_namespace()
        work = self.root / (phase + ('-error' if fail else '-ok'))
        originals = {name: getattr(worker, name) for name in
                     ('main', '_wall_limit', '_phase', '_checkpoint', 'trace_model', 'parse_model')}
        originals['outer_main'] = frontend.main
        entered = []
        namespace['_initialize_environment'] = lambda: None
        def forbidden(*args, **kwargs):
            raise AssertionError('No numeric module/model/source loading in this context test')
        for name in ('_numeric_modules', '_load_model', '_load_expr', '_check_sources'):
            namespace[name] = forbidden
        def fake_body(args, recorder, result):
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            # Both contexts are the actual original generator bodies; no stub
            # stands in for either of the two previously broken decorators.
            with namespace['_phase'](args, recorder, result, 'synthetic-only-body'):
                entered.append((args.phase, args.memory_gib))
                result['synthetic_body_entered'] = True
                if fail:
                    raise ValueError('synthetic body failure')
        namespace['trace_model' if phase == 'trace' else 'parse_model'] = fake_body
        argv = ['--phase', phase, '--model', frontend.MODEL_RECORDS['alexnet']['factory'],
                '--memory-gib', str(memory), '--work-dir', str(work)]
        required = (('frontend', 'model-source', 'benchmark', 'checkpoint', 'poly-dir',
                     'deps-dir', 'orbit-root', 'r13-result', 'r16-result')
                    if phase == 'trace' else ('orbit-root', 'profile', 'mlir'))
        for key in required:
            argv.extend(('--' + key, str(self.root / key)))
        stderr = io.StringIO()
        with self.alarm() as (existing, before), \
                patch.object(worker.resource, 'setrlimit') as limits, \
                patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)), \
                patch('sys.stderr', stderr):
            code = namespace['main'](argv)
            self.assertIs(signal.getsignal(signal.SIGALRM), existing)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), before)
        self.assertEqual(code, 1 if fail else 0, stderr.getvalue())
        self.assertEqual(entered, [(phase, float(memory))])
        self.assertIn((worker.resource.RLIMIT_AS, (memory * subject.GIB, memory * subject.GIB)),
                      [call.args for call in limits.call_args_list])
        record = json.loads((work / 'phase_result.json').read_text())
        self.assertEqual(record['status'], 'error' if fail else 'ok')
        self.assertTrue(record['synthetic_body_entered'])
        self.assertEqual(record['address_space_limit_gib'], memory)
        self.assertEqual(record['physical_RSS_guard_gib'], 14)
        self.assertEqual(record['host_available_reserve_gib'], 1.25)
        self.assertEqual(record['resource_adapter'], subject.VARIANT)
        events = [json.loads(line) for line in (work / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['type'] for event in events], ['begin', 'begin', 'end', 'end'])
        self.assertTrue(any(event.get('stage_id') == 'synthetic-only-body' for event in events))
        for name, original in originals.items():
            self.assertIs(frontend.main if name == 'outer_main' else getattr(worker, name), original)

    def test_trace24_main_enters_real_wall_and_phase_contexts_and_saves_resource_record(self):
        self.run_worker('trace', 24)

    def test_trace24_main_error_restores_real_contexts_and_preserves_error_terminal(self):
        self.run_worker('trace', 24, fail=True)

    def test_parse4_main_uses_real_contexts_without_expanding_parse_allowance(self):
        self.run_worker('parse', 4)


if __name__ == '__main__':
    unittest.main()
