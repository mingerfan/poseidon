"""R17 stdlib outer checks and one dedicated intake-environment wrapper."""

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
MODE = 'cached-margin-heldout-validation'
ENGINES = ('reference_forward', 'tensor_exact', 'tensor_program',
           'mock_exact', 'mock_program')
ENGINE_COMPLETIONS = (1000, 3, 1000, 3, 3)


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


@require(INTAKE_ARTIFACT)
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_margin_cases_run_once_without_skips(self):
        from pt_agent.tests import _r17_margin_validation_cases as cases

        expected = unittest.defaultTestLoader.loadTestsFromTestCase(cases.MarginCases).countTestCases()
        self.assertGreater(expected, 0)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r17_margin_validation_cases.MarginCases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), expected, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R17_MARGIN_INNER_TESTS=' + count.group(1), flush=True)
        print('R17_REAL_MODELS_WEIGHTS_AND_DATA_IN_UNIT_TESTS=0', flush=True)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r17_margin_validation_preflight as outer

        self.outer = outer
        temporary = TemporaryDirectory(prefix='r17-outer-cases-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.work_results = self.root / 'work-results'
        self.records = self.work_results / 'agent-cifar-offline-records-r1804/cifar10-records'
        self.records.mkdir(parents=True)
        self._save(self.records / 'dataset.json', {})
        (self.records / 'test-09000.records').touch()
        self.interpreter = self.root / 'intake/bin/python'
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.touch()
        self.fixtures = []
        for name, short, act_count in outer.RUNS:
            model = 'pt_agent.experiments.r13_common_models:' + short
            run = self.results / name
            work = run / 'worker'
            work.mkdir(parents=True)
            artifacts = dict(calibration=work / 'calibration.json',
                             calibration_source=work / 'calibration_source.json',
                             program=work / 'program.ptdsl')
            self._save(artifacts['calibration'], {})
            self._save(artifacts['calibration_source'], {})
            artifacts['program'].write_text('# protocol fixture; never evaluated\n')
            r13_run = self.results / name.replace('r14-', 'r13-', 1)
            info = r13_run / 'worker/model_info'
            info.mkdir(parents=True)
            self._save(info / 'model_info.json', {})
            r13_artifacts = dict(model_info=str(info),
                                 training_probes=str(r13_run / 'worker/training_probes.npz'),
                                 training_probe_source=str(r13_run / 'worker/training_probe_source.json'))
            r13_worker = dict(round='R13', status='ok', model=model,
                              import_completed=True, real_probes_passed=True,
                              graph_math_passed=True, roundtrip_passed=True,
                              static_coverage=True, artifacts=r13_artifacts)
            r13 = dict(round='R13', mode='common-model-intake-preparation',
                       status='ok', model=model, worker=r13_worker,
                       timing=dict(spans=[dict(event_id='old:import', stage_id='import_model',
                                              inclusive_ns=11, exclusive_ns=11)]))
            self._save(r13_run / 'result.json', r13)
            completed = short == 'resnet20_silu'
            source_status = 'ok' if completed else 'artifact_budget_exceeded'
            worker = dict(round='R14', model=model, status=source_status,
                          failed_phase=None if completed else 'emit_write',
                          r13_source_result=str(r13_run / 'result.json'),
                          artifacts={key: str(value) for key, value in artifacts.items()},
                          source_checked=True, reference_passed=True,
                          calibration_completed=True, plan_completed=True,
                          typecheck_passed=True, signature_passed=True,
                          count_completed=True, stats_equal=True)
            original = dict(round='R14', mode='common-model-rule-preparation',
                            model=model, status=source_status, worker=worker,
                            source_r13=dict(run=str(r13_run), result=str(r13_run / 'result.json'),
                                            model=model, artifacts=dict(r13_artifacts,
                                            r13_result=str(r13_run / 'result.json')),
                                            prior_import_spans=copy.deepcopy(r13['timing']['spans']),
                                            included_in_current_turnaround=False),
                            timing=dict(spans=[dict(event_id='old:rule', stage_id='rule.plan',
                                                   inclusive_ns=22, exclusive_ns=22)]))
            self._save(run / 'result.json', original)
            self.fixtures.append(dict(run=run, model=model, short=short, act_count=act_count,
                                      artifacts=artifacts, r13_run=r13_run, info=info,
                                      original=original, r13=r13))
        for target, name, value in ((outer.paths, 'RESULTS_DIR', self.results),
                                    (outer.paths, 'WORK_RESULTS', self.work_results),
                                    (outer.paths, 'INTAKE_PYTHON', self.interpreter),
                                    (outer, '_git_state', lambda: dict(branch='fixture'))):
            context = patch.object(target, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(outer.shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=11 * 1024**3))
        context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def _save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def _args(self, index=0, label='case'):
        fixture = self.fixtures[index]
        return SimpleNamespace(r14_run=fixture['run'],
                               out=self.results / 'queue' / label)

    def _worker(self, index=0):
        worker = dict(round='R17', model=self.fixtures[index]['model'], status='ok', error=None,
                      evaluation_completed=True, task_passed=True, domain_passed=True,
                      heldout_acceptance=True, all_finite=True, completed_samples=1000,
                      heldout_samples=1000, candidate_frozen=False, outputs_partial=False,
                      plan_attempts=1, plan_completions=1,
                      count_attempts=1, count_completions=1,
                      model_restore_attempts=1, model_restore_completions=1,
                      cached_source_reads={name:dict(attempted=1, completed=1) for name in
                          ('r14_result', 'r13_result', 'calibration',
                           'calibration_source', 'original_program')},
                      engine_counts={key: dict(attempted=n, completed=n)
                                     for key, n in zip(ENGINES, ENGINE_COMPLETIONS)},
                      validation_metrics=dict(exact_top1=90.0, top1=90.0, acc_drop_pp=0.0,
                                              agree=1000, total=1000),
                      domain_rows=[], max_u=0.4, max_logit_absolute_difference=0.1,
                      source_count_stats=dict(counts=dict(boot=0), ops_total=1),
                      count_stats=dict(counts=dict(boot=0), ops_total=2))
        worker.update(dict.fromkeys(self.outer.STAGE_FIELDS, True))
        worker.update(dict.fromkeys(self.outer.ZERO_FIELDS, 0))
        worker.update(dict.fromkeys(self.outer.UNMEASURED_FIELDS))
        return worker

    @staticmethod
    def _observation(**updates):
        value = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[],
                     started_ns=10, ended_ns=20, termination_observed_ns=20,
                     peak_rss_kib_by_pid={}, outer_peak_rss_kib=0,
                     limits=dict(memory_gib=4, file_gib=8, wall_seconds=1860), signals_sent=[])
        value.update(updates)
        return value

    def _process(self, worker, *, artifact_bytes=0, **updates):
        def run(command, **kwargs):
            self.assertEqual((kwargs['timeout'], kwargs['mem_gib'], kwargs['fsize_gib']),
                             (1860, 4, 8))
            self.assertEqual(command[command.index('--seconds') + 1], '1800')
            self.assertEqual(command[command.index('--python') + 1], str(self.interpreter))
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                self.assertEqual(kwargs['env'][key], '1')
            work = Path(command[command.index('--work-dir') + 1])
            if worker is not None:
                self._save(work / 'worker_result.json', worker)
            recorder = self.outer.StageRecorder(work / 'events.jsonl')
            # A fake worker shares the test PID; keep its event IDs separate.
            recorder.sequence = 1000
            with recorder.span('worker'):
                with recorder.span('fake_plan'):
                    pass
            kwargs['log'].write_text('fake worker; no numerical engine executed\n')
            if artifact_bytes:
                (work / 'partial.jsonl').write_text('0' * artifact_bytes)
            return self._observation(**updates)
        return run

    def _ledger(self):
        path = self.results / 'registry.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _assert_ledger(self, result, count=1):
        rows = self._ledger()
        self.assertEqual(len(rows), count)
        row = next(item for item in rows if item['artifacts_dir'] == str(result['artifacts_dir'])) \
            if 'artifacts_dir' in result else rows[-1]
        self.assertEqual(set(row), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                    'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual((row['round'], row['mode']), ('R17', MODE))
        self.assertEqual(row['metrics'], result['metrics'])

    def test_all_three_original_sources_restore_only_explicit_cached_paths(self):
        for fixture in self.fixtures:
            with self.subTest(model=fixture['short']):
                previous, selected = self.outer.resolve_source(fixture['run'])
                self.assertEqual(previous, fixture['original'])
                expected = dict(r14_result=fixture['run'] / 'result.json',
                                r13_result=fixture['r13_run'] / 'result.json',
                                model_info=fixture['info'], records=self.records,
                                calibration=fixture['artifacts']['calibration'],
                                calibration_source=fixture['artifacts']['calibration_source'],
                                original_program=fixture['artifacts']['program'])
                self.assertEqual(set(selected), set(expected))
                for key, value in expected.items():
                    self.assertIsInstance(selected[key], Path)
                    self.assertEqual(selected[key], value, key)
                self.assertNotIn('count', selected)
                self.assertNotIn('emit', selected)

    def test_vgg_and_alex_require_the_original_completed_frontend_and_emit_write_failure(self):
        for fixture in self.fixtures[1:]:
            for failure in ('status', 'failed_phase', 'calibration', 'reference', 'stats'):
                with self.subTest(model=fixture['short'], failure=failure):
                    changed = copy.deepcopy(fixture['original'])
                    if failure == 'status':
                        changed['status'] = changed['worker']['status'] = 'ok'
                    elif failure == 'failed_phase':
                        changed['worker']['failed_phase'] = 'count'
                    else:
                        key = {'calibration':'calibration_completed', 'reference':'reference_passed',
                               'stats':'stats_equal'}[failure]
                        changed['worker'][key] = False
                    self._save(fixture['run'] / 'result.json', changed)
                    try:
                        with self.assertRaises(self.outer.PhaseStopped):
                            self.outer.resolve_source(fixture['run'])
                    finally:
                        self._save(fixture['run'] / 'result.json', fixture['original'])

    def test_foreign_cached_artifacts_and_r13_cross_references_are_rejected(self):
        fixture = self.fixtures[0]
        for failure in ('calibration', 'calibration_source', 'program', 'model_info', 'r13_model'):
            changed = copy.deepcopy(fixture['original'])
            r13 = copy.deepcopy(fixture['r13'])
            if failure in ('calibration', 'calibration_source', 'program'):
                changed['worker']['artifacts'][failure] = str(self.root / ('foreign-' + failure))
            elif failure == 'model_info':
                changed['source_r13']['artifacts']['model_info'] = str(self.root / 'foreign-info')
            else:
                r13['worker']['model'] = self.fixtures[2]['model']
            self._save(fixture['run'] / 'result.json', changed)
            self._save(fixture['r13_run'] / 'result.json', r13)
            try:
                with self.subTest(failure=failure), self.assertRaises(self.outer.PhaseStopped):
                    self.outer.resolve_source(fixture['run'])
            finally:
                self._save(fixture['run'] / 'result.json', fixture['original'])
                self._save(fixture['r13_run'] / 'result.json', fixture['r13'])

    def test_success_runs_one_worker_with_fixed_limits_and_one_registry(self):
        args, worker = self._args(), self._worker()
        with patch.object(self.outer, 'run_process', side_effect=self._process(worker)) as launched:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok', result.get('error'))
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertIn('pt_agent.experiments.r17_margin_validation_worker', command)
        fixture = self.fixtures[0]
        expected = {'--r14-result':fixture['run'] / 'result.json',
                    '--r13-result':fixture['r13_run'] / 'result.json',
                    '--model-info':fixture['info'], '--records':self.records,
                    '--calibration':fixture['artifacts']['calibration'],
                    '--calibration-source':fixture['artifacts']['calibration_source'],
                    '--original-program':fixture['artifacts']['program'],
                    '--work-dir':args.out / 'worker'}
        for option, value in expected.items():
            self.assertEqual(command.count(option), 1)
            self.assertEqual(command[command.index(option) + 1], str(value))
        for option in ('--checkpoint', '--count', '--frontend', '--provider'):
            self.assertNotIn(option, command)
        self.assertEqual(result['metrics']['engine_counts'], worker['engine_counts'])
        self.assertEqual(result['metrics']['count_stats'], worker['count_stats'])
        self.assertEqual(result['metrics']['source_count_stats'], worker['source_count_stats'])
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.UNMEASURED_FIELDS))
        self.assertFalse(result['metrics']['candidate_frozen'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self._assert_ledger(result)

    def test_old_preparation_timing_is_separate_and_exclusive_spans_cover_current_root(self):
        args = self._args()
        with patch.object(self.outer, 'run_process', side_effect=self._process(self._worker())):
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok', result.get('error'))
        spans = result['timing']['spans']
        roots = [span for span in spans if span.get('parent_id') is None]
        self.assertEqual(len(roots), 1)
        self.assertTrue(result['timing']['attribution_ready'])
        self.assertEqual(result['timing']['diagnostics'], [])
        self.assertTrue(all(span['exclusive_ns'] >= 0 for span in spans))
        self.assertEqual(sum(span['exclusive_ns'] for span in spans), roots[0]['inclusive_ns'])
        self.assertTrue(any(span['stage_id'] == 'fake_plan' for span in spans))
        self.assertFalse({'old:rule', 'old:import'} & set(result['metrics']['phase_elapsed_ns']))
        self.assertFalse(result['source_r14']['included_in_current_turnaround'])
        self.assertEqual(result['source_r14']['prior_timing'], self.fixtures[0]['original']['timing'])
        self.assertEqual(result['source_r14']['prior_r13_import_spans'],
                         self.fixtures[0]['r13']['timing']['spans'])

    def test_complete_domain_rejection_preserves_a_successful_evaluation(self):
        args, worker = self._args(), self._worker()
        worker.update(domain_passed=False, heldout_acceptance=False, max_u=0.95)
        with patch.object(self.outer, 'run_process', side_effect=self._process(worker)) as launched:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['metrics']['evaluation_completed'])
        self.assertTrue(result['metrics']['task_passed'])
        self.assertFalse(result['metrics']['domain_passed'])
        self.assertFalse(result['metrics']['heldout_acceptance'])
        self.assertEqual(result['metrics']['completed_samples'], 1000)
        self._assert_ledger(result)

    def test_partial_failure_keeps_actual_counts_outputs_and_null_acceptance(self):
        args, worker = self._args(), self._worker()
        worker.update(status='math_failed', error='fake strict mismatch', math_probes_passed=False,
                      evaluation_completed=False, completed_samples=0, heldout_samples=0,
                      task_passed=None, domain_passed=None, heldout_acceptance=None,
                      validation_metrics=None, max_u=None)
        worker['engine_counts'] = {key:dict(attempted=0, completed=0) for key in ENGINES}
        worker['engine_counts']['reference_forward'] = dict(attempted=1, completed=1)
        worker['engine_counts']['tensor_exact'] = dict(attempted=1, completed=1)
        with patch.object(self.outer, 'run_process',
                          side_effect=self._process(worker, artifact_bytes=8, exit_code=1)):
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'math_failed')
        self.assertEqual(result['metrics']['engine_counts'], worker['engine_counts'])
        self.assertEqual(result['metrics']['plan_completions'], 1)
        self.assertEqual(result['metrics']['count_completions'], 1)
        self.assertIsNone(result['metrics']['heldout_acceptance'])
        self.assertIsNone(result['metrics']['validation_metrics'])
        self.assertTrue((args.out / 'worker/partial.jsonl').is_file())
        self._assert_ledger(result)

    def test_timeout_and_termination_failures_preserve_worker_partial_counts(self):
        for label, observation, status in (
                ('timeout', dict(exit_code=-15, timed_out=True, signal=15), 'timeout'),
                ('signal', dict(exit_code=-9, signal=9), 'process_failed'),
                ('remaining', dict(exit_code=1, remaining_pids=[45678]), 'process_failed')):
            args, worker = self._args(label=label), self._worker()
            worker.update(status='error', evaluation_completed=False, completed_samples=3,
                          heldout_samples=3, task_passed=None, domain_passed=None,
                          heldout_acceptance=None, validation_metrics=None)
            worker['engine_counts']['reference_forward'] = dict(attempted=4, completed=3)
            with self.subTest(label=label), patch.object(self.outer, 'run_process',
                    side_effect=self._process(worker, **observation)) as launched:
                result = self.outer.execute(args)
            launched.assert_called_once()
            self.assertEqual(result['status'], status)
            self.assertEqual(result['metrics']['engine_counts']['reference_forward'],
                             dict(attempted=4, completed=3))
            self.assertIsNone(result['metrics']['heldout_acceptance'])
        self.assertEqual(len(self._ledger()), 3)

    def test_missing_worker_observations_keep_stages_counts_and_actions_unknown(self):
        result = self.outer.finalize_result(worker=None, processes=[], timing={},
                                             status='dependency_missing', error='fake unavailable',
                                             git_state={}, out=self.results / 'unstarted')
        for key in self.outer.STAGE_FIELDS + self.outer.ZERO_FIELDS + self.outer.UNMEASURED_FIELDS:
            self.assertIsNone(result['metrics'][key], key)
        for key in ('plan_attempts', 'plan_completions', 'count_attempts', 'count_completions',
                    'model_restore_attempts', 'model_restore_completions', 'cached_source_reads',
                    'engine_counts', 'completed_samples', 'heldout_samples', 'evaluation_completed',
                    'validation_metrics', 'heldout_acceptance'):
            self.assertIsNone(result['metrics'][key], key)
        args = self._args()
        with patch.object(self.outer, 'run_process', side_effect=self._process(None, exit_code=1)):
            actual = self.outer.execute(args)
        self.assertNotEqual(actual['status'], 'ok')
        self.assertIsNone(actual['metrics']['engine_counts'])
        self.assertIsNone(actual['metrics']['completed_samples'])
        self._assert_ledger(actual)

    def test_success_requires_actual_single_plan_count_full_engines_and_zero_forbidden_actions(self):
        invalid_cases = ('plan_count', 'count_count', 'short_engine', 'forbidden', 'formal')
        for invalid in invalid_cases:
            args, worker = self._args(label=invalid), self._worker()
            if invalid == 'plan_count':worker['plan_completions'] = 0
            elif invalid == 'count_count':worker['count_attempts'] = 2
            elif invalid == 'short_engine':worker['engine_counts']['tensor_program']['completed'] = 999
            elif invalid == 'forbidden':worker['compiler_runs'] = 1
            else:worker['compiled_bootstraps'] = 0
            with self.subTest(invalid=invalid), patch.object(self.outer, 'run_process',
                    side_effect=self._process(worker)) as launched:
                result = self.outer.execute(args)
            launched.assert_called_once()
            self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(self._ledger()), len(invalid_cases))

    def test_missing_dependency_and_disk_shortage_stop_before_worker(self):
        for reason in ('disk', 'calibration'):
            args = self._args(label=reason)
            with ExitStack() as contexts:
                launched = contexts.enter_context(patch.object(self.outer, 'run_process'))
                if reason == 'disk':
                    contexts.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                                        return_value=SimpleNamespace(free=9 * 1024**3)))
                else:self.fixtures[0]['artifacts']['calibration'].unlink()
                result = self.outer.execute(args)
            self.assertEqual(result['status'], 'dependency_missing')
            launched.assert_not_called()
            self.assertIsNone(result['metrics']['engine_counts'])
            self.assertIsNone(result['metrics']['completed_samples'])
        self.assertEqual(len(self._ledger()), 2)

    def test_existing_results_and_original_sources_data_and_r15_are_protected(self):
        existing = self.results / 'existing'
        existing.mkdir()
        marker = existing / 'preserved.txt'
        marker.write_text('saved evidence')
        fixture = self.fixtures[0]
        r15 = self.results / 'r15-rule-heldout-resnet20-silu-20261115'
        r15.mkdir()
        destinations = (existing, self.results, fixture['run'] / 'new',
                        fixture['r13_run'] / 'new', fixture['info'] / 'new',
                        self.records / 'new', r15 / 'new')
        with patch.object(self.outer, 'run_process') as launched:
            for destination in destinations:
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    self.outer.execute(SimpleNamespace(r14_run=fixture['run'], out=destination))
        launched.assert_not_called()
        self.assertEqual(marker.read_text(), 'saved evidence')
        self.assertEqual(self._ledger(), [])
        self.assertTrue(all(not path.exists() for path in destinations[2:]))

    def test_terminal_json_and_post_worker_artifacts_count_toward_case_and_queue_budgets(self):
        out = self.results / 'terminal-queue/case'
        out.mkdir(parents=True)
        (out / 'small.log').write_text('saved log')
        result = self.outer.finalize_result(worker=self._worker(), processes=[], timing={},
                                             status='ok', error=None, git_state={}, out=out)
        with patch.object(self.outer, 'CASE_BUDGET_BYTES', 64):
            self.outer._write_terminal_result(out, result, [])
        self.assertEqual(json.loads((out / 'result.json').read_text()), result)
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(out)['bytes'])
        self.assertFalse(result['metrics']['case_artifact_budget_passed'])
        self.assertFalse(result['metrics']['queue_artifact_budget_passed'])
        args = self._args(label='budget')
        with patch.object(self.outer, 'CASE_BUDGET_BYTES', 16384), \
                patch.object(self.outer, 'run_process',
                             side_effect=self._process(self._worker(), artifact_bytes=32768)) as launched:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertTrue((args.out / 'worker/partial.jsonl').is_file())
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self._assert_ledger(result)

    def test_queue_runs_three_in_order_and_continues_completed_quality_rejection(self):
        calls = []
        def execute(args):
            calls.append(args.r14_run.name)
            return dict(status='ok', error=None, metrics=dict(heldout_acceptance=False))
        out = self.results / 'quality-queue'
        with patch.object(self.outer, 'execute', side_effect=execute):
            result = self.outer.run_queue(SimpleNamespace(out=out))
        self.assertEqual(calls, [name for name, _, _ in self.outer.RUNS])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['attempted_cases'], 3)
        self.assertEqual(result['completed_cases'], 3)
        self.assertEqual(result['retry_count'], 0)
        self.assertTrue(all(item['status'] == 'ok' for item in result['cases']))
        self.assertEqual(result['artifact_bytes'], self.outer._case_size(out)['bytes'])
        self.assertEqual(self._ledger(), [])

    def test_queue_stops_first_execution_failure_and_leaves_remaining_unstarted(self):
        calls = []
        def execute(args):
            calls.append(args.r14_run.name)
            return dict(status='ok' if len(calls) == 1 else 'math_failed',
                        error=None if len(calls) == 1 else 'saved strict mismatch')
        with patch.object(self.outer, 'execute', side_effect=execute):
            result = self.outer.run_queue(SimpleNamespace(out=self.results / 'stopped-queue'))
        self.assertEqual(calls, [row[0] for row in self.outer.RUNS[:2]])
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['attempted_cases'], 2)
        self.assertEqual(result['completed_cases'], 1)
        self.assertEqual([item['status'] for item in result['cases']],
                         ['ok', 'math_failed', 'not_started'])
        self.assertEqual(result['retry_count'], 0)
        self.assertEqual(self._ledger(), [])

    def test_fixed_limits_mode_and_external_cli_only_accept_queue_output(self):
        self.assertEqual((self.outer.INTERNAL_SECONDS, self.outer.OUTER_SECONDS,
                          self.outer.MEMORY_GIB, self.outer.CASE_BUDGET_BYTES),
                         (1800, 1860, 4, 4 * 1024**3))
        self.assertEqual(self.outer.MODE, MODE)
        self.assertEqual([row[2] for row in self.outer.RUNS], [19, 15, 7])
        with patch.object(self.outer, 'run_queue') as launched, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:self.outer.main([])
        self.assertEqual(caught.exception.code, 2)
        launched.assert_not_called()
        output = self.results / 'cli'
        with patch.object(self.outer, 'run_queue', return_value=dict(status='ok',
                attempted_cases=3, completed_cases=3)) as launched, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.outer.main(['--out', str(output)]), 0)
        self.assertEqual(vars(launched.call_args.args[0]), dict(out=output))
