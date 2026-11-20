"""R15 stdlib protocol checks and one dedicated intake-environment wrapper."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


INTAKE_ARTIFACT = (paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink()
                   else paths.INTAKE_PYTHON)
MODE = 'fixed-rule-heldout-validation'
ENGINES = ('reference_forward', 'tensor_exact', 'tensor_program',
           'mock_exact', 'mock_program')
EXPECTED_COMPLETED = (1000, 3, 1000, 3, 3)
FACTORIES = {
    'resnet20-silu': ('resnet20_silu', 'R14', 'common-model-rule-preparation',
                    'r14-common-resnet20-silu-20261115'),
    'vgg16-silu-avgpool': ('vgg16_silu_avgpool', 'R14a', 'vgg-emit-resume',
                         'r14a-vgg16-emit-resume-20261115'),
    'alexnet-silu-avgpool': ('alexnet_silu_avgpool', 'R14b', 'alex-emit-resume',
                           'r14b-alexnet-emit-resume-20261115'),
}


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


@require(INTAKE_ARTIFACT)
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_round_inner_checks_run_once_without_skips(self):
        from pt_agent.tests import _r15_rule_validation_cases as cases
        expected = unittest.defaultTestLoader.loadTestsFromModule(cases).countTestCases()
        self.assertGreater(expected, 0)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r15_rule_validation_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), expected, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R15_INNER_TESTS=' + count.group(1), flush=True)
        print('R15_REAL_MODELS_AND_HELDOUT_DATA_IN_UNIT_TESTS=0', flush=True)


class ValidationOuterChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r15_rule_validation_preflight as outer
        self.outer = outer
        temporary = TemporaryDirectory(prefix='r15-outer-fixtures-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self, kind='resnet20-silu'):
        suffix, round_name, mode, export_name = FACTORIES[kind]
        model = 'pt_agent.experiments.r13_common_models:' + suffix
        source = self.root / ('r14-common-' + kind + '-20261115')
        work = source / 'worker'
        work.mkdir(parents=True)
        program = work / 'program.ptdsl'
        program.write_text('# deterministic protocol fixture; never evaluated\n')
        count = work / 'count.json'
        self.save(count, {'stats': {'counts': {'boot': 0}}})
        r13 = self.root / ('r13-common-' + kind + '-20261115')
        info = r13 / 'worker/model_info'
        info.mkdir(parents=True)
        self.save(info / 'model_info.json', {})
        r13_result = dict(round='R13', model=model, status='ok',
                          worker=dict(round='R13', model=model, status='ok',
                                      artifacts=dict(model_info=str(info))))
        self.save(r13 / 'result.json', r13_result)
        source_r13 = dict(run=str(r13), result=str(r13 / 'result.json'), model=model,
                          artifacts=dict(r13_result=str(r13 / 'result.json'),
                                         model_info=str(info)),
                          included_in_current_turnaround=False)
        stage_flags = dict(source_checked=True, typecheck_passed=True,
                           signature_passed=True, stats_equal=True,
                           emit_completed=True, lowering_ready=True,
                           reference_passed=True, calibration_completed=True,
                           plan_completed=True, count_completed=True)
        original = dict(round='R14', mode='common-model-rule-preparation', model=model,
                        status='ok' if round_name == 'R14' else 'artifact_budget_exceeded',
                        worker=dict(round='R14', model=model,
                                    status='ok' if round_name == 'R14' else 'artifact_budget_exceeded',
                                    failed_phase=None if round_name == 'R14' else 'emit_write',
                                    artifacts=dict(program=str(program), count=str(count)),
                                    r13_source_result=str(r13 / 'result.json'), **stage_flags),
                        source_r13=source_r13,
                        timing=dict(spans=[dict(event_id='old:rule', stage_id='rule.plan',
                                               inclusive_ns=100, exclusive_ns=100)]))
        self.save(source / 'result.json', original)
        export = self.root / export_name
        exported = original
        if round_name != 'R14':
            resume = dict(source_result=str(source / 'result.json'), program=str(program),
                          count=str(count), r13_result=str(r13 / 'result.json'),
                          model_info=str(info))
            exported = dict(round=round_name, mode=mode, model=model, status='ok',
                            worker=dict(round=round_name, model=model, status='ok',
                                        source_resume=resume, **stage_flags),
                            source_r14=dict(run=str(source), result=str(source / 'result.json'),
                                            artifacts=copy.deepcopy(resume),
                                            prior_timing=copy.deepcopy(original['timing']),
                                            included_in_current_turnaround=False),
                            timing=dict(spans=[dict(event_id='old:emit', stage_id='emit_only',
                                                   inclusive_ns=200, exclusive_ns=200)]))
            self.save(export / 'result.json', exported)
        records = self.root / 'work-results/agent-cifar-offline-records-r1804/cifar10-records'
        records.mkdir(parents=True, exist_ok=True)
        self.save(records / 'dataset.json', {})
        (records / 'test-09000.records').touch()
        interpreter = self.root / 'intake-python'
        interpreter.touch(exist_ok=True)
        args = SimpleNamespace(export_run=export, out=self.root / ('new-' + kind))
        return args, dict(model=model, export=export, source=source, r13=r13, info=info,
                          program=program, count=count, records=records,
                          interpreter=interpreter, exported=exported, original=original)

    def worker(self, model):
        result = dict(round='R15', model=model, status='ok', error=None,
                      evaluation_completed=True, task_passed=True, domain_passed=True,
                      heldout_acceptance=True, all_finite=True, completed_samples=1000,
                      candidate_frozen=False,
                      engine_counts={key: dict(attempted=count, completed=count)
                                     for key, count in zip(ENGINES, EXPECTED_COMPLETED)},
                      validation_metrics=dict(exact_top1=90.0, top1=90.0, acc_drop_pp=0.0,
                                              agree=1000, total=1000),
                      domain_rows=[], max_u=0.25,
                      source_count_stats=dict(counts={'boot': 0}),
                      max_logit_absolute_difference=0.125)
        result.update({key: True for key in self.outer.STAGE_FIELDS})
        result.update({key: 0 for key in self.outer.ZERO_FIELDS})
        result.update({key: None for key in self.outer.UNMEASURED_FIELDS})
        return result

    @staticmethod
    def observation(**values):
        result = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[],
                      started_ns=10, ended_ns=20, peak_rss_kib_by_pid={}, limits={})
        result.update(values)
        return result

    @staticmethod
    def write_worker(args, worker):
        (args.out / 'worker/worker_result.json').write_text(json.dumps(worker))

    def environment(self, fixture, launch):
        stack = ExitStack()
        stack.enter_context(patch.object(self.outer.paths, 'RESULTS_DIR', self.root))
        stack.enter_context(patch.object(self.outer.paths, 'WORK_RESULTS', self.root / 'work-results'))
        stack.enter_context(patch.object(self.outer.paths, 'INTAKE_PYTHON', fixture['interpreter']))
        stack.enter_context(patch.object(self.outer, '_git_state', return_value=dict(branch='fixture')))
        stack.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                        return_value=SimpleNamespace(free=11 * 1024**3)))
        launched = stack.enter_context(patch.object(self.outer, 'run_process', side_effect=launch))
        return stack, launched

    def ledger(self):
        path = self.root / 'registry.jsonl'
        return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []

    def assert_one_ledger(self, result):
        rows = self.ledger()
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                       'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual((rows[0]['round'], rows[0]['mode']), ('R15', MODE))
        self.assertEqual(rows[0]['metrics'], result['metrics'])

    def test_resolve_all_three_export_sources_and_original_paths(self):
        for kind in FACTORIES:
            with self.subTest(kind=kind):
                args, fixture = self.fixture(kind)
                stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
                with stack:
                    previous, selected = self.outer.resolve_source(args.export_run)
                self.assertEqual(previous, fixture['exported'])
                expected = dict(export_result=args.export_run / 'result.json',
                                rule_result=fixture['source'] / 'result.json',
                                program=fixture['program'], count=fixture['count'],
                                r13_result=fixture['r13'] / 'result.json',
                                model_info=fixture['info'], records=fixture['records'])
                self.assertEqual(set(selected), set(expected))
                for key, value in expected.items():
                    self.assertEqual(Path(selected[key]), value, key)
                launched.assert_not_called()
                self.assertFalse(args.out.exists())

    def test_resume_cross_reference_and_source_status_are_enforced(self):
        args, fixture = self.fixture('vgg16-silu-avgpool')
        for invalid in ('resume_program', 'outer_rule', 'worker_status', 'signature'):
            changed = copy.deepcopy(fixture['exported'])
            if invalid == 'resume_program':
                changed['worker']['source_resume']['program'] = str(self.root / 'foreign.ptdsl')
            elif invalid == 'outer_rule':
                changed['source_r14']['result'] = str(self.root / 'foreign/result.json')
            elif invalid == 'worker_status':
                changed['worker']['status'] = 'error'
            else:
                changed['worker']['signature_passed'] = False
            self.save(args.export_run / 'result.json', changed)
            stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
            with self.subTest(invalid=invalid), stack:
                with self.assertRaises(self.outer.PhaseStopped):
                    self.outer.resolve_source(args.export_run)
            launched.assert_not_called()
        self.assertFalse(args.out.exists())

    def test_execute_passes_explicit_paths_limits_and_one_worker(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok')
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index('--seconds') + 1], '1800')
        self.assertEqual(command[command.index('--python') + 1], str(fixture['interpreter']))
        self.assertIn('pt_agent.experiments.r15_rule_validation_worker', command)
        expected = {'--export-result': args.export_run / 'result.json',
                    '--rule-result': fixture['source'] / 'result.json',
                    '--program': fixture['program'], '--count': fixture['count'],
                    '--r13-result': fixture['r13'] / 'result.json',
                    '--model-info': fixture['info'], '--records': fixture['records'],
                    '--work-dir': args.out / 'worker'}
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(command.count(option), 1)
                self.assertEqual(command[command.index(option) + 1], str(value))
        self.assertNotIn('--frontend', command)
        self.assertNotIn('--r13-probes', command)
        limits = launched.call_args.kwargs
        self.assertEqual((limits['timeout'], limits['mem_gib'], limits['fsize_gib']), (1860, 4, 8))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(limits['env'][name], '1')
        self.assertEqual(result['metrics']['engine_counts'], worker['engine_counts'])
        self.assertEqual(result['metrics']['source_count_stats'], {'counts': {'boot': 0}})
        self.assertEqual(result['metrics']['validation_metrics']['acc_drop_pp'], 0.0)
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.UNMEASURED_FIELDS))
        self.assertFalse(result['metrics']['candidate_frozen'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertNotIn('old:rule', result['metrics']['phase_elapsed_ns'])
        self.assertFalse(result['source_export']['included_in_current_turnaround'])
        self.assertEqual(result['source_export']['prior_timing'], fixture['original']['timing'])
        self.assertEqual(result['metrics']['acc_drop_pp_unit'], 'percentage points')
        self.assert_one_ledger(result)

    def test_completed_quality_rejection_is_a_valid_evaluation(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        worker.update(task_passed=False, domain_passed=False, heldout_acceptance=False)
        worker['validation_metrics'].update(top1=89.0, acc_drop_pp=1.0)
        worker['max_u'] = 0.95
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['metrics']['evaluation_completed'])
        self.assertFalse(result['metrics']['heldout_acceptance'])
        self.assertFalse(result['metrics']['task_passed'])
        self.assertFalse(result['metrics']['domain_passed'])
        self.assertEqual(result['metrics']['completed_samples'], 1000)
        self.assert_one_ledger(result)

    def test_worker_failure_preserves_partial_actual_counts_and_null_metrics(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        worker.update(status='math_failed', error='fake strict probe disagreement',
                      math_probes_passed=False, evaluation_completed=False, completed_samples=0,
                      task_passed=None, domain_passed=None, heldout_acceptance=None,
                      validation_metrics=None, max_u=None)
        worker['engine_counts'] = {key: dict(attempted=0, completed=0) for key in ENGINES}
        worker['engine_counts']['reference_forward'] = dict(attempted=1, completed=1)
        worker['engine_counts']['tensor_exact'] = dict(attempted=1, completed=1)
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            (args.out / 'worker/partial_outputs.json').write_text('{"partial": true}')
            return self.observation(exit_code=1)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'math_failed')
        self.assertFalse(result['metrics']['evaluation_completed'])
        self.assertEqual(result['metrics']['engine_counts'], worker['engine_counts'])
        self.assertIsNone(result['metrics']['heldout_acceptance'])
        self.assertIsNone(result['metrics']['validation_metrics'])
        self.assertTrue((args.out / 'worker/partial_outputs.json').is_file())
        self.assert_one_ledger(result)

    def test_timeout_records_one_attempt_and_partial_completion(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        worker.update(status='error', evaluation_completed=False, completed_samples=3,
                      task_passed=None, domain_passed=None, heldout_acceptance=None,
                      validation_metrics=None)
        worker['engine_counts']['reference_forward'] = dict(attempted=4, completed=3)
        worker['engine_counts']['tensor_program'] = dict(attempted=3, completed=3)
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            return self.observation(exit_code=-15, timed_out=True, signal=15)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['metrics']['engine_counts']['reference_forward'],
                         dict(attempted=4, completed=3))
        self.assertIsNone(result['metrics']['heldout_acceptance'])
        self.assert_one_ledger(result)

    def test_unobserved_worker_counters_and_stages_remain_null(self):
        result = self.outer.finalize_result(worker=None, processes=[], timing={},
                                             status='dependency_missing', error='fake missing source',
                                             git_state={}, out=self.root / 'unstarted')
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.STAGE_FIELDS))
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.ZERO_FIELDS))
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.UNMEASURED_FIELDS))
        for key in ('evaluation_completed', 'task_passed', 'domain_passed',
                    'heldout_acceptance', 'all_finite', 'completed_samples',
                    'validation_metrics', 'max_u'):
            self.assertIsNone(result['metrics'][key], key)
        self.assertIsNone(result['metrics']['engine_counts'])

    def test_no_worker_file_does_not_invent_zero_or_target_counts(self):
        args, fixture = self.fixture()
        stack, launched = self.environment(fixture, lambda *a, **kw: self.observation(exit_code=1))
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertNotEqual(result['status'], 'ok')
        self.assertIsNone(result['metrics']['engine_counts'])
        self.assertIsNone(result['metrics']['completed_samples'])
        self.assert_one_ledger(result)

    def test_success_requires_actual_full_counts_and_no_forbidden_actions(self):
        args, fixture = self.fixture()
        for invalid in ('short_count', 'forbidden_action', 'formal_measurement'):
            worker = self.worker(fixture['model'])
            if invalid == 'short_count':
                worker['engine_counts']['tensor_program']['completed'] = 999
            elif invalid == 'forbidden_action':
                worker['compiler_runs'] = 1
            else:
                worker['compiled_bootstraps'] = 0
            args.out = self.root / ('invalid-' + invalid)
            def launch(*command, **kwargs):
                self.write_worker(args, worker)
                return self.observation()
            stack, launched = self.environment(fixture, launch)
            with self.subTest(invalid=invalid), stack:
                result = self.outer.execute(args)
            launched.assert_called_once()
            self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(self.ledger()), 3)

    def test_disk_shortage_stops_before_worker_and_preserves_unknown_counts(self):
        args, fixture = self.fixture()
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
        with stack, patch.object(self.outer.shutil, 'disk_usage',
                                 return_value=SimpleNamespace(free=9 * 1024**3)):
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'dependency_missing')
        launched.assert_not_called()
        self.assertIsNone(result['metrics']['completed_samples'])
        self.assert_one_ledger(result)

    def test_missing_saved_count_prevents_launch_and_records_unknown(self):
        args, fixture = self.fixture()
        fixture['count'].unlink()
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'dependency_missing')
        launched.assert_not_called()
        self.assertIsNone(result['metrics']['engine_counts'])
        self.assertIsNone(result['metrics']['completed_samples'])
        self.assert_one_ledger(result)

    def test_original_artifact_and_r13_model_info_locations_are_enforced(self):
        args, fixture = self.fixture()
        for invalid in ('foreign_program', 'foreign_model_info', 'r13_model'):
            original = copy.deepcopy(fixture['original'])
            r13 = json.loads((fixture['r13'] / 'result.json').read_text())
            if invalid == 'foreign_program':
                original['worker']['artifacts']['program'] = str(self.root / 'foreign/program.ptdsl')
            elif invalid == 'foreign_model_info':
                original['source_r13']['artifacts']['model_info'] = str(self.root / 'foreign/info')
            else:
                r13['worker']['model'] = 'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool'
            self.save(args.export_run / 'result.json', original)
            self.save(fixture['r13'] / 'result.json', r13)
            stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
            with self.subTest(invalid=invalid), stack:
                with self.assertRaises(self.outer.PhaseStopped):
                    self.outer.resolve_source(args.export_run)
            launched.assert_not_called()
            if invalid == 'r13_model':
                r13['worker']['model'] = fixture['model']
                self.save(fixture['r13'] / 'result.json', r13)
        self.assertFalse(args.out.exists())

    def test_existing_output_and_all_source_descendants_are_preserved(self):
        args, fixture = self.fixture('vgg16-silu-avgpool')
        existing = self.root / 'existing-output'
        existing.mkdir()
        marker = existing / 'preserve.txt'
        marker.write_text('saved evidence')
        destinations = (existing, self.root, fixture['export'] / 'new',
                        fixture['source'] / 'new', fixture['r13'] / 'new')
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
        with stack:
            for destination in destinations:
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    self.outer.execute(SimpleNamespace(export_run=args.export_run, out=destination))
        self.assertEqual(marker.read_text(), 'saved evidence')
        self.assertFalse((fixture['export'] / 'new').exists())
        self.assertFalse((fixture['source'] / 'new').exists())
        self.assertFalse((fixture['r13'] / 'new').exists())
        launched.assert_not_called()
        self.assertEqual(self.ledger(), [])

    def test_terminal_result_its_own_size_and_metrics_are_counted(self):
        out = self.root / 'terminal-case'
        out.mkdir()
        (out / 'small.log').write_text('small fake log')
        result = self.outer.finalize_result(worker=self.worker('fixture'), processes=[],
                                             timing={}, status='ok', error=None,
                                             git_state={}, out=out)
        self.assertLess(self.outer._case_size(out)['bytes'], 1024)
        with patch.object(self.outer, 'CASE_BUDGET_BYTES', 1024):
            self.outer._write_terminal_result(out, result, [])
        saved = json.loads((out / 'result.json').read_text())
        self.assertEqual(saved, result)
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertFalse(result['metrics']['case_artifact_budget_passed'])
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(out)['bytes'])

    def test_post_worker_case_budget_stops_and_registry_matches_terminal_state(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker(fixture['model']))
            (args.out / 'worker/partial_outputs.jsonl').write_text('0' * 32768)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack, patch.object(self.outer, 'CASE_BUDGET_BYTES', 16384):
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertTrue((args.out / 'worker/partial_outputs.jsonl').is_file())
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assert_one_ledger(result)

    def test_cli_requires_only_export_run_and_output(self):
        for argv in ([], ['--export-run', '/fixture/export'], ['--out', '/fixture/new']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                    patch.object(self.outer, 'execute') as launched:
                with self.assertRaises(SystemExit) as caught:
                    self.outer.main(argv)
                self.assertEqual(caught.exception.code, 2)
                launched.assert_not_called()
        with patch.object(self.outer, 'execute', return_value={'status': 'ok'}) as launched, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.outer.main(['--export-run', '/fixture/export',
                                              '--out', '/fixture/new']), 0)
        self.assertEqual(vars(launched.call_args.args[0]),
                         dict(export_run=Path('/fixture/export'), out=Path('/fixture/new')))

    def test_fixed_production_limits_and_mode_are_preserved(self):
        self.assertEqual((self.outer.INTERNAL_SECONDS, self.outer.OUTER_SECONDS,
                          self.outer.MEMORY_GIB, self.outer.CASE_BUDGET_BYTES),
                         (1800, 1860, 4, 4 * 1024**3))
        self.assertEqual(self.outer.MODE, MODE)
