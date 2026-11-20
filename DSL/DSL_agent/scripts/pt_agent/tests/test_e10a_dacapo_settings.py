"""Offline checks of the E10a driver: compiler arguments, step guards and the summary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pt_agent.experiments import e10a_dacapo_settings as e10a


class CompileArguments(unittest.TestCase):
    def test_uses_qwen48_settings_and_the_trace_as_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, source = Path(tmp)/'optimized', Path(tmp)/'trace/traced/ResNet.mlir'
            argv = e10a.compile_arguments('optimized', work, source)
        self.assertIn(str(source), argv)
        self.assertIn('--boot-placement=optimized', argv)
        self.assertIn('--waterline=43', argv)
        self.assertIn('--ckks-config=' + str(work.resolve()) + '/boot-compiler-config.json', argv)
        self.assertIn('--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5', argv)
        self.assertFalse(any('/out/' in arg for arg in argv))

    def test_depth_based_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = e10a.compile_arguments('depth-dp', Path(tmp)/'depth-dp', Path(tmp)/'t.mlir')
        self.assertIn('--boot-placement=depth-dp', argv)
        self.assertNotIn('--boot-placement=optimized', argv)

    def test_compiler_output_stays_in_the_work_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/'optimized'
            argv = e10a.compile_arguments('optimized', work, Path(tmp)/'t.mlir')
        self.assertEqual(argv[argv.index('-o') + 1], str(work.resolve()) + '/lowered.mlir')


class Steps(unittest.TestCase):
    def test_a_step_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'trace').mkdir()
            (Path(tmp)/'trace/REPORT.json').write_text('{}')
            with mock.patch.object(e10a.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    e10a.child(Path(tmp), 'trace', 'e10a-trace', ['--worker', 'trace'])
            run.assert_not_called()

    def test_a_ledger_change_is_an_error(self):
        done = dict(complete=True, boots=20)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(e10a, 'child', return_value=done), \
                mock.patch.object(e10a, 'register'), \
                mock.patch.object(e10a, 'ledger_rows', side_effect=[[1] * 430, [1] * 431]):
            code = e10a.run(mock.Mock(root=Path(tmp)))
            data = json.loads((Path(tmp)/'E10a.json').read_text())
        self.assertEqual(code, 1)
        self.assertIn('ledger', data['error'])

    def test_a_failed_trace_stops_the_compiles(self):
        calls = []
        def child(root, step, mode, extra=()):
            calls.append(step)
            return dict(complete=False, error='trace crashed')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(e10a, 'child', side_effect=child), \
                mock.patch.object(e10a, 'register'), mock.patch.object(e10a, 'ledger_rows', return_value=[]):
            self.assertEqual(e10a.run(mock.Mock(root=Path(tmp))), 1)
        self.assertEqual(calls, ['trace'])


class Summary(unittest.TestCase):
    def test_table_lists_both_settings(self):
        data = dict(api_before=430, api_after=430, compile=dict(
            optimized=dict(complete=True, boots=33, seconds=50.0, peak_rss_mib=2400.0),
            **{'depth-dp': dict(complete=True, boots=120, seconds=9.0, peak_rss_mib=2200.0)}))
        with tempfile.TemporaryDirectory() as tmp:
            e10a.summarize(Path(tmp), data)
            text = (Path(tmp)/'E10a.md').read_text()
        self.assertIn('| optimized | 19 | 33 |', text)
        self.assertIn('| depth-dp | 85 | 120 |', text)
        self.assertIn('账本 430 → 430', text)


if __name__ == '__main__':
    unittest.main()
