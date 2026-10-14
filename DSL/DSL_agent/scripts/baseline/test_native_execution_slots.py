"""Offline process-level resource and CLI default checks; never calls an API."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from native_execution_slots import native_slot


def contend(directory, active, maximum, ready):
    ready.wait(10)
    with native_slot(directory, timeout=10):
        with active.get_lock():
            active.value += 1
            maximum.value = max(maximum.value, active.value)
        time.sleep(0.1)
        with active.get_lock():
            active.value -= 1


@unittest.skipUnless(os.name == 'posix', 'Linux flock execution gate')
class NativeSlotTests(unittest.TestCase):
    def test_ten_processes_never_exceed_two_native_stages(self):
        ctx = multiprocessing.get_context('spawn')
        active, maximum = ctx.Value('i', 0), ctx.Value('i', 0)
        ready = ctx.Event()
        with tempfile.TemporaryDirectory() as directory:
            workers = [ctx.Process(target=contend, args=(directory, active, maximum, ready)) for _ in range(10)]
            try:
                for worker in workers:
                    worker.start()
                ready.set()
                for worker in workers:
                    worker.join(15)
                    self.assertEqual(worker.exitcode, 0)
                self.assertEqual(maximum.value, 2)
                self.assertEqual(active.value, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(5)

    def test_timeout_exception_release_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = {}
            with ExitStack() as stack:
                stack.enter_context(native_slot(directory, metrics=metrics))
                stack.enter_context(native_slot(directory, metrics=metrics))
                with self.assertRaises(TimeoutError):
                    with native_slot(directory, timeout=0.05):
                        self.fail('Third native slot acquired')
            self.assertEqual(metrics['acquisitions'], 2)
            self.assertGreaterEqual(metrics['wait_seconds'], 0)
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                with native_slot(directory):
                    raise RuntimeError('synthetic')
            with native_slot(directory), native_slot(directory):
                pass

    def test_unsafe_lock_and_invalid_timeout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'target').touch()
            (root / 'slot-0.lock').symlink_to(root / 'target')
            with self.assertRaises(OSError):
                with native_slot(root):
                    self.fail('Symlink accepted')
            for invalid in (0, -1, float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    with native_slot(root, timeout=invalid):
                        self.fail('Invalid timeout')


class NewDefaultsTests(unittest.TestCase):
    def test_single_cli_defaults_and_exact_response_identity(self):
        from run_candidate import parse_args
        from deepseek_provider import Config, response_content, ProviderError
        from test_deepseek_provider import completion
        args = parse_args(['--live', '--case', 'not-read.json'])
        self.assertEqual((args.provider, args.model, args.max_tokens, args.reasoning_effort, args.api_timeout),
                         ('deepseek', 'deepseek-flash', 384000, 'high', 1200))
        self.assertEqual(Config().model, 'deepseek-flash')
        self.assertEqual(response_content(completion('{}')[1], Config())[0], '{}')
        with self.assertRaisesRegex(ProviderError, 'response_model_mismatch'):
            response_content(completion('{}', model='deepseek-v4-flash')[1], Config())

    def test_batch_defaults_and_ten_accepted_eleven_rejected(self):
        import run_agent_batch
        for flags in ([], ['--jobs', '10']):
            output = io.StringIO()
            with patch('sys.argv', ['run_agent_batch', '--plan', *flags]), redirect_stdout(output):
                self.assertEqual(run_agent_batch.main(), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual((plan['service_provider'], plan['model'], plan['api_concurrency'],
                              plan['native_execution_concurrency']), ('deepseek', 'deepseek-flash', 10, 2))
            self.assertEqual(plan['agent_calls'], 0)
        with patch('sys.argv', ['run_agent_batch', '--plan', '--jobs', '11']), self.assertRaises(SystemExit):
            run_agent_batch.main()


if __name__ == '__main__':
    unittest.main()
