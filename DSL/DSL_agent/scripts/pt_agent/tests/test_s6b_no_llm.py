import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import s6_no_llm as driver
from pt_agent.experiments import s6b_serial as serial


class S6bTests(unittest.TestCase):
    def test_existing_rule_is_only_extra_tuning_policy_argument(self):
        def original(ctx, **kwargs):
            return kwargs
        given = dict(max_iters=1, tuner=True, tuner_budget_s=2400, tune_policy='all',
            tune_rule='v8-6.25', confirm_rule='c2', engine_profile='tensor', numeric_lock=True)
        actual = driver.single_seed_state(original, tune_rule='v8-6.23')(object(), **given)
        self.assertEqual(actual, dict(given, tune_rule='v8-6.23'))

    def test_cnn_command_passes_rule_and_separate_case(self):
        settings = dict(source=dict(records='/samples', selections='/select', compiler_metadata_recipe='/metadata'),
            memory_gib='12', case_gib='32', trace_seconds='1800', compile_seconds='7200',
            archive_emit=True, diagnostic_elide=True)
        with patch.object(serial.base.previous, 'settings', return_value=settings):
            for model in serial.base.MODELS[:3]:
                argv = serial.command(SimpleNamespace(), 'S6b', model, 7200)
                self.assertEqual(argv[argv.index('--tune-rule')+1], 'v8-6.23')
                self.assertTrue(Path(argv[argv.index('--out')+1]).name.startswith('s6b-'))
                self.assertEqual(argv[argv.index('--tuner-seconds')+1], '2400')
                self.assertNotIn('--no-tuner', argv)

    def test_bert_continuation_explicitly_skips_unused_scheduler_gate(self):
        settings = dict(source=dict(model='/original/model', inputs='/original/inputs'))
        args = SimpleNamespace(template=Path('/metadata'), calibration_snapshot=Path('/calibration'))
        with patch.object(serial.base.previous, 'settings', return_value=settings):
            argv = serial.command(args, 'S6', 'bert_tiny', 10800)
            self.assertIn('--s6-continue', argv)
            self.assertEqual(argv[argv.index('--tune-rule')+1], 'v8-6.25')
            self.assertTrue(Path(argv[argv.index('--out')+1]).name.startswith('s6-'))

    def test_rejection_counts_come_from_retained_events(self):
        with tempfile.TemporaryDirectory() as root:
            case = Path(root)
            session = case / 'work/agent'
            session.mkdir(parents=True)
            events = [dict(phase='coarse', accepted=False, rejected='refit_floor'),
                      dict(phase='coarse', accepted=False), dict(phase='skip_cost'),
                      dict(phase='align', fallback=[{}, {}], accepted_lines=[3])]
            (session / 'agent_trace.jsonl').write_text(json.dumps(dict(tuned=dict(history=events)))+'\n')
            self.assertEqual(serial.rejection_counts(case, 'resnet20'), dict(refit_floor=1,
                coarse_failed=1, skip_cost=1, prefix_fallback_attempts=2, prefix_fallback_accepted=1))

    def test_deferred_report_claims_no_samples_or_access_marker(self):
        from pt_agent import paths
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / 's6-s6b-ordered-no-llm-20261118').mkdir()
            out = base / 'deferred'
            with patch.object(paths, 'RESULTS_DIR', base):
                self.assertTrue(driver.defer_report(['--out', str(out), '--source-case', '/original']))
            value = json.loads((out / 'result.json').read_text())
            self.assertEqual(value['actual_report_attempts'], 0)
            self.assertEqual(value['report_samples_read'], 0)
            self.assertFalse(value['report_access_claimed'])
            self.assertFalse((base / 'frozen-cnn-report-access').exists())

    def test_optional_admission_reserves_full_three_hour_window(self):
        with patch.object(driver.time, 'time', return_value=driver.off.s4.DEADLINE-10800):
            self.assertFalse(driver.optional_admission([]))

    def test_optional_skip_is_metadata_with_no_scientific_attempt(self):
        from pt_agent.experiments import s2_heir_serial, completion_metrics
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'unused'
            with patch.object(s2_heir_serial, 'record_registry'), patch.object(completion_metrics, 'persist_case_completion'):
                self.assertTrue(driver.optional_admission(['--out', str(out)], now=driver.off.s4.DEADLINE-100))
            value = json.loads((out / 'result.json').read_text())
            self.assertEqual(value['status'], 'not_run')
            self.assertEqual(value['scientific_attempts'], 0)
            self.assertEqual(value['model_reads'], 0)
            self.assertEqual(value['api_calls'], 0)
            self.assertFalse((out / 'model').exists())
            self.assertFalse((out / 'session').exists())


if __name__ == '__main__':
    unittest.main()
