import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import s6_no_llm as driver
from pt_agent.experiments import s6_serial as serial


def original_audit(record, **kwargs):
    session = dict(calls=1)
    if session.get('calls') != record.get('actual_api_calls'):
        raise ValueError('call count')
    if record['bad_quality']:
        raise ValueError('quality rejected')
    return dict(quality=True, actual_api_calls=record['actual_api_calls'])


class S6Tests(unittest.TestCase):
    def test_one_local_reply_and_no_ledger_or_client_call(self):
        from pt_agent.experiments.e9b_cnn_sessions import ReplayClient
        with tempfile.TemporaryDirectory() as root:
            client = driver.RuleClient(lambda: object(), Path(root))
            client.replay = ReplayClient([json.dumps(dict(program='local'))])
            self.assertEqual(json.loads(client.chat([], purpose='first')['content'])['program'], 'local')
            from pt_agent.llm import LLMError
            with self.assertRaises(LLMError):
                client.chat([], purpose='second')
            self.assertEqual(client.calls, 0)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_state_changes_only_declared_seed_and_budget_policy(self):
        seen = {}
        def original(ctx, **kwargs):
            seen.update(kwargs)
            return kwargs
        driver.single_seed_state(original)(object(), max_iters=6, tuner=True, tuner_budget_s=600,
            tune_policy='all', tune_rule='v8-6.25', placement='program-plan', confirm_rule='c2')
        self.assertEqual(seen, dict(max_iters=1, tuner=True, tuner_budget_s=2400,
            tune_policy='all', tune_rule='v8-6.25', placement='program-plan', confirm_rule='c2'))

    def test_audit_distinguishes_local_reply_and_keeps_quality_failure(self):
        adapted = driver.offline_audit_function(original_audit)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'offline-client.json').write_text('{}')
            base = dict(actual_api_calls=0, offline_reply_calls=1, bad_quality=False)
            self.assertTrue(adapted(base, agent_dir=root)['quality'])
            with self.assertRaisesRegex(ValueError, 'quality rejected'):
                adapted(dict(base, bad_quality=True), agent_dir=root)
            with self.assertRaisesRegex(ValueError, 'zero real provider'):
                adapted(dict(base, actual_api_calls=1), agent_dir=root)

    def test_wrapper_passes_full_calibration_argument(self):
        with patch.object(driver, 'cnn_worker', return_value=0) as call:
            self.assertEqual(driver.main(['--cnn-worker', '--calibration', '/input/cal.json']), 0)
            call.assert_called_once_with(['--calibration', '/input/cal.json'])

    def test_rule_seed_has_original_degree_margin_and_depth_header(self):
        from pt_agent.planners import rule
        ctx = SimpleNamespace(graph=SimpleNamespace(intake_stats={}), calibration={'source': 'original'})
        with patch.object(rule, 'plan', return_value='seed') as plan:
            self.assertEqual(driver.rule_seed(ctx), 'seed')
            plan.assert_called_once_with(ctx.graph, ctx.calibration, deg=63, margin=1.5,
                slots=16384, placement='depth-dp', refresh='compiler', act_method='cheb')

    def test_all_cnn_commands_enable_same_tuner_without_provider_options(self):
        chosen = dict(source=dict(records='/samples', selections='/select', compiler_metadata_recipe='/metadata'),
            memory_gib='12', case_gib='32', trace_seconds='1800', compile_seconds='7200',
            archive_emit=True, diagnostic_elide=True)
        with patch.object(serial.previous, 'settings', return_value=chosen):
            for model in serial.MODELS[:3]:
                argv = serial.command(SimpleNamespace(), model, 7200)
                self.assertNotIn('--no-tuner', argv)
                self.assertEqual(argv[argv.index('--calls')+1], '1')
                self.assertEqual(argv[argv.index('--tuner-seconds')+1], '2400')
                self.assertIn(driver.MODULE, argv)

    def test_stage_groups_partition_the_same_exclusive_values(self):
        stages = dict(candidate_tuning=2., candidate_parse_math_and_cost_tools=3.,
            placement=5., trace=7., native_compile_stage=11., calibration=13.,
            independent_freeze_confirmation=17.)
        grouped = serial.grouped_stages(stages)
        self.assertEqual(grouped, dict(tuning=2., simulation_and_counting=3.,
            placement_and_compile=23., setup_and_checks=30.))
        self.assertEqual(sum(grouped.values()), sum(stages.values()))

    def test_activation_details_keep_each_degree_and_domain(self):
        text = ('@fhe.program(slots=8)\ndef x(a):\n'
                ' b = act(a, fn="silu", method="cheb", deg=63, domain=(-4., 6.))\n'
                ' c = softmax(b, exp_domain=(-2., 1.), inv_deg=31, inv_domain=(.1, 1.))\n'
                ' return c\n')
        rows = serial.recipes(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['attrs']['deg'], 63)
        self.assertEqual(rows[0]['attrs']['domain'], (-4., 6.))
        self.assertEqual(rows[1]['attrs']['inv_domain'], (.1, 1.))

    def test_report_inner_worker_keeps_offline_accounting_wrapper(self):
        argv = ['python', '-m', 'pt_agent.tools.nixrun', '--seconds', '1800', '--',
                '-m', 'pt_agent.experiments.r30_frozen_cnn_report', '--worker', '--model', 'original']
        selected = driver.report_worker_command(argv)
        self.assertIn(driver.MODULE, selected)
        self.assertIn('--cnn-report-worker', selected)
        self.assertIn('--worker', selected)
        self.assertNotIn('pt_agent.experiments.r30_frozen_cnn_report', selected)

    def test_zero_read_report_gate_failure_can_continue_once(self):
        from pt_agent import paths
        from pt_agent.experiments import r30_frozen_cnn_report as report
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            old = base / 'old'
            old.mkdir()
            source = str(base / 'source')
            gate = dict(source_case=source, model='original')
            access = base / 'frozen-cnn-report-access'
            access.mkdir()
            marker = access / 'source.json'
            marker.write_text(json.dumps(dict(source_case=source, model='original', out=str(old),
                report_started=False, report_complete=False, report_samples=0)))
            failure = dict(status='error', report_samples_read=0, report_samples_completed=0,
                worker=dict(error='ValueError: Fresh model policy, completed session or actual call accounting differs',
                    report_samples_read=0, report_samples_completed=0))
            (old / 'result.json').write_text(json.dumps(failure))
            with patch.object(paths, 'RESULTS_DIR', base):
                selected = driver.claim_report_access(report, base / 'new', gate)
                self.assertTrue(selected.is_file())
                self.assertEqual(json.loads(marker.read_text())['out'], str(old))
                with self.assertRaises(FileExistsError):
                    driver.claim_report_access(report, base / 'newer', gate)
                (old / 'report-rows.jsonl').write_text('started')
                with self.assertRaisesRegex(ValueError, 'untouched'):
                    driver.claim_report_access(report, base / 'another', gate)

    def test_quality_observer_returns_same_result_and_calls_original_once(self):
        result = dict(top1=92.4, exact_top1=92.4, agree=1000, total=1000, max_u_analytic=float('inf'))
        with tempfile.TemporaryDirectory() as root:
            original = unittest.mock.Mock(return_value=result)
            ctx = SimpleNamespace(val_split='heldout')
            observed = driver.quality_observer(original, lambda: root, lambda p: 'original source')
            self.assertIs(observed('program', ctx, 1000, engine='tensor', cache=False), result)
            original.assert_called_once_with('program', ctx, 1000, engine='tensor', cache=False)
            row = json.loads((Path(root) / 'observed-quality.jsonl').read_text())
            self.assertEqual(row['metrics']['top1'], 92.4)
            self.assertIsNone(row['metrics']['max_u_analytic'])
            self.assertEqual(row['split'], 'heldout')


if __name__ == '__main__':
    unittest.main()
