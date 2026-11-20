import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pt_agent.experiments import s5_tuner_off as s5


class S5ControlTests(unittest.TestCase):
    def test_only_tuner_keyword_is_changed(self):
        def original(*args, **kwargs):
            return args, kwargs
        selected = s5.disable_tuner(original)
        args, kwargs = selected('context', max_iters=10, confirm_rule='c2', numeric_lock=True)
        self.assertEqual(args, ('context',))
        self.assertEqual(kwargs, dict(max_iters=10, confirm_rule='c2', numeric_lock=True, tuner=False))
        with self.assertRaises(ValueError):
            selected('context', tuner=True)

    def test_cached_pure_shell_keeps_original_arguments(self):
        before = ['python', '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '7200', '--python', '/intake',
                  '--', '-B', '-m', 'worker', '--no-tuner']
        after = s5.cached_entry(before)
        self.assertEqual(after[:3], before[:3])
        self.assertEqual(after[3:5], [s5.s4.MODULE, '--enter'])
        self.assertEqual(after[5:], before[4:])

    def test_checkpoint_command_remains_the_original_direct_command(self):
        args = SimpleNamespace(model=Path('/model'), out=Path('/case'), checkpoint_out=Path('/audit'),
                               intake_python=Path('/intake'))
        argv = s5.bert_command(args, 'checkpoint', Path('/work'), 400)
        self.assertIn('pt_agent.experiments.bert_checkpoint_audit', argv)
        self.assertNotIn('--calibration-snapshot', argv)

    def test_cnn_calibration_argument_reaches_the_original_worker(self):
        argv = ['--worker-phase', 'intake', '--model', 'resnet20_silu',
                '--calibration', '/explicit/calibration.json', '--no-tuner']
        with patch.object(s5.cnn, 'main', return_value=0) as original:
            self.assertEqual(s5.main(['--cnn-worker', *argv]), 0)
        original.assert_called_once_with(argv)

    def test_nested_report_does_not_raise_its_inherited_hard_file_limit(self):
        with patch('resource.getrlimit', return_value=(4*1024**3, 4*1024**3)):
            self.assertEqual(s5.inherited_file_limit(dict(mem_gib=6, fsize_gib=8)),
                             dict(mem_gib=6, fsize_gib=4))

    def test_report_continuation_preserves_the_unstarted_claim_and_rejects_reads(self):
        from pt_agent.experiments import r30_frozen_cnn_report as report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claims = root / 'frozen-cnn-report-access'; claims.mkdir()
            old = root / 'old'; old.mkdir()
            marker = dict(source_case=str(root / 'source'), out=str(old), model='resnet20_silu',
                          report_started=False, report_complete=False, report_samples=0)
            prior = claims / 'source.json'; prior.write_text(json.dumps(marker))
            (old / 'result.json').write_text(json.dumps(dict(status='error', report_samples_read=0,
                report_samples_completed=0, error='SubprocessError: Exception occurred in preexec_fn.')))
            gate = dict(source_case=marker['source_case'], model=marker['model'])
            with patch.object(s5.paths, 'RESULTS_DIR', root):
                new = s5.claim_report_access(report, root / 'new', gate)
                self.assertEqual(json.loads(prior.read_text()), marker)
                self.assertEqual(json.loads(new.read_text())['previous_unstarted_access'], str(prior))
                prior.write_text(json.dumps(dict(marker, report_started=True)))
                with self.assertRaisesRegex(ValueError, 'once-only report access'):
                    s5.claim_report_access(report, root / 'another', gate)


if __name__ == '__main__':
    unittest.main()
