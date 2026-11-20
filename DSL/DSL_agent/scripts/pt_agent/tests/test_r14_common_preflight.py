"""R14 outer protocol checks; real three-model preparation is never repeated."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
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


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


@require(INTAKE_ARTIFACT)
class CommonEnvironmentChecks(unittest.TestCase):
    def test_only_current_round_inner_checks_run_without_skips(self):
        from pt_agent.tests import _r14_common_cases as cases
        expected = unittest.defaultTestLoader.loadTestsFromModule(cases).countTestCases()
        self.assertGreater(expected, 0)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r14_common_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), expected, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R14_INNER_TESTS=' + count.group(1), flush=True)
        print('R14_FULL_MODEL_PREPARATIONS_IN_UNIT_TESTS=0', flush=True)


class CommonResultChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r14_common_preflight as outer
        self.outer = outer
        temporary = TemporaryDirectory(prefix='r14-outer-fixtures-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def finalize(self, worker=None, timing=None):
        return self.outer.finalize_result(
            worker=worker, processes=[], timing=timing or {}, status='error',
            error='fixture failure', git_state=dict(branch='fixture', revision='fixture'),
            out=self.root / 'run')

    def fixture(self):
        outer = self.outer
        model = next(iter(outer.FACTORIES))
        metadata = outer.FACTORIES[model]
        source_run = self.root / 'r13-source'
        work = source_run / 'worker'
        saved = work / 'model_info'
        saved.mkdir(parents=True)
        (saved / 'model_info.json').write_text('{}')
        probes, probe_source = work / 'training_probes.npz', work / 'training_probe_source.json'
        probes.touch()
        probe_source.write_text('{}')
        source = dict(factory=model, checkpoint=str(metadata['checkpoint']),
                      original_source=str(metadata['source']))
        record = dict(round='R13', mode='common-model-intake-preparation', status='ok',
                      model=model, worker=dict(model=model, status='ok', model_source=source,
                      import_completed=True, real_probes_passed=True, graph_math_passed=True,
                      roundtrip_passed=True, static_coverage=True,
                      artifacts=dict(model_info=str(saved), training_probes=str(probes),
                                     training_probe_source=str(probe_source))),
                      timing=dict(spans=[dict(event_id='old:1', stage_id='import_model',
                                             inclusive_ns=123, exclusive_ns=123)]))
        (source_run / 'result.json').write_text(json.dumps(record))
        records = self.root / 'work/agent-cifar-offline-records-r1804/cifar10-records'
        records.mkdir(parents=True)
        (records / 'dataset.json').write_text('{}')
        selections = self.root / 'work/agent-resnet20-original-accuracy-r1805/frozen-selections.json'
        selections.parent.mkdir(parents=True)
        selections.write_text('{}')
        frontend, interpreter = self.root / 'expr.py', self.root / 'intake-python'
        frontend.touch()
        interpreter.touch()
        return SimpleNamespace(r13_run=source_run, out=self.root / 'r14-run'), dict(
            model=model, records=records, selections=selections, frontend=frontend,
            interpreter=interpreter, source=record)

    def environment(self, fixture, launch):
        stack = ExitStack()
        stack.enter_context(patch.object(self.outer.paths, 'RESULTS_DIR', self.root))
        stack.enter_context(patch.object(self.outer.paths, 'WORK_RESULTS', self.root / 'work'))
        stack.enter_context(patch.object(self.outer.paths, 'INTAKE_PYTHON', fixture['interpreter']))
        stack.enter_context(patch.object(self.outer.toolchain, 'host_paths',
                                        return_value=dict(frontend=fixture['frontend'])))
        stack.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                        return_value=SimpleNamespace(free=20 * 1024**3)))
        stack.enter_context(patch.object(self.outer, '_git_state',
                                        return_value=dict(branch='fixture', revision='fixture',
                                                          working_tree_changed=True)))
        mock = stack.enter_context(patch.object(self.outer, 'run_process', side_effect=launch))
        return stack, mock

    def worker(self, model):
        outer = self.outer
        record = dict(round='R14', model=model, status='ok', error=None, artifacts={},
                      count_stats=dict(counts=dict(boot=0, mul_cc=2), ops_total=4),
                      emit_stats=dict(counts=dict(boot=0, mul_cc=2), ops_total=4))
        record.update({key: True for key in outer.STAGE_FIELDS})
        record.update({key: 1 for key in outer.ATTEMPT_FIELDS})
        record.update({key: 0 for key in outer.ZERO_FIELDS})
        record.update({key: None for key in outer.UNMEASURED_FIELDS})
        return record

    def observation(self, **changes):
        return dict(exit_code=changes.pop('exit_code', 0),
                    timed_out=changes.pop('timed_out', False),
                    signal=changes.pop('signal', None),
                    remaining_pids=changes.pop('remaining_pids', []), **changes)

    def write_worker(self, args, worker):
        (args.out / 'worker/worker_result.json').write_text(json.dumps(worker))

    def ledger(self):
        return [json.loads(line) for line in (self.root / 'registry.jsonl').read_text().splitlines()]

    def test_unstarted_stages_stay_null_and_unmeasured_fields_are_separate(self):
        result = self.finalize()
        self.assertEqual(result['round'], 'R14')
        self.assertEqual(result['mode'], 'common-model-rule-preparation')
        for key in self.outer.STAGE_FIELDS + self.outer.UNMEASURED_FIELDS:
            with self.subTest(key=key):
                self.assertIsNone(result['metrics'][key])
        for key in self.outer.ATTEMPT_FIELDS + self.outer.ZERO_FIELDS:
            with self.subTest(key=key):
                self.assertEqual(result['metrics'][key], 0)
        self.assertIsNone(result['metrics']['count_stats'])

    def test_partial_worker_fields_and_count_emit_stats_are_retained(self):
        worker = dict(model='fixture', source_checked=True, reference_passed=False,
                      count_stats=dict(counts=dict(boot=0)),
                      emit_stats=dict(counts=dict(boot=2)), calibration_attempts=1)
        metrics = self.finalize(worker)['metrics']
        self.assertIs(metrics['source_checked'], True)
        self.assertIs(metrics['reference_passed'], False)
        self.assertIsNone(metrics['calibration_completed'])
        self.assertEqual(metrics['count_stats'], worker['count_stats'])
        self.assertEqual(metrics['emit_stats'], worker['emit_stats'])
        self.assertEqual(metrics['calibration_attempts'], 1)
        self.assertIsNone(metrics['compiled_bootstraps'])
        self.assertIsNone(metrics['accuracy_acceptance'])

    def test_cli_requires_saved_run_and_new_output_only(self):
        for argv in ([], ['--out', '/fixture/run'], ['--r13-run', '/fixture/old']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 patch.object(self.outer, 'execute') as launch:
                with self.assertRaises(SystemExit) as caught:
                    self.outer.main(argv)
                self.assertEqual(caught.exception.code, 2)
                launch.assert_not_called()
        with patch.object(self.outer, 'execute', return_value=dict(status='ok')) as launch, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(self.outer.main(['--r13-run', '/fixture/old',
                                              '--out', '/fixture/new']), 0)
        args = launch.call_args.args[0]
        self.assertEqual(vars(args), dict(r13_run=Path('/fixture/old'), out=Path('/fixture/new')))

    def test_one_worker_receives_all_paths_and_finite_limits(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        def launch(command, **kwargs):
            self.write_worker(args, worker)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok')
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index('--seconds')+1], '1800')
        self.assertEqual(command[command.index('--python')+1], str(fixture['interpreter']))
        self.assertIn('pt_agent.experiments.r14_common_worker', command)
        for option, expected in {
            '--model-info': args.r13_run / 'worker/model_info',
            '--r13-result': args.r13_run / 'result.json',
            '--r13-probes': args.r13_run / 'worker/training_probes.npz',
            '--r13-probe-source': args.r13_run / 'worker/training_probe_source.json',
            '--inputs': fixture['records'], '--selections': fixture['selections'],
            '--frontend': fixture['frontend'], '--work-dir': args.out / 'worker'
        }.items():
            with self.subTest(option=option):
                self.assertEqual(command[command.index(option)+1], str(expected))
        limits = launched.call_args.kwargs
        self.assertEqual(limits['timeout'], 1860)
        self.assertEqual(limits['mem_gib'], 4)
        self.assertEqual(limits['fsize_gib'], 8)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(limits['env'][name], '1')
        self.assertFalse(result['source_r13']['included_in_current_turnaround'])
        self.assertEqual(result['source_r13']['prior_import_spans'][0]['event_id'], 'old:1')
        self.assertNotIn('old:1', result['metrics']['phase_elapsed_ns'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        rows = self.ledger()
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                      'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual(rows[0]['round'], 'R14')
        self.assertEqual(rows[0]['metrics'], result['metrics'])

    def test_preflight_failure_preserves_a_result_and_one_registry_record(self):
        args, fixture = self.fixture()
        fixture['frontend'].unlink()
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected process'))
        with stack:
            result = self.outer.execute(args)
        launched.assert_not_called()
        self.assertEqual(result['status'], 'dependency_missing')
        self.assertTrue((args.out / 'result.json').is_file())
        self.assertEqual(len(self.ledger()), 1)
        self.assertIsNone(result['metrics']['source_checked'])

    def test_cached_nix_interpreter_symlink_is_passed_without_resolving_it(self):
        args, fixture = self.fixture()
        interpreter = fixture['interpreter']
        interpreter.unlink()
        interpreter.symlink_to('/nix/store/r14-fixture-python/bin/python')
        self.assertFalse(interpreter.exists())
        self.assertTrue(interpreter.is_symlink())
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker(fixture['model']))
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok')
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index('--python')+1], str(interpreter))

    def test_failed_r13_source_never_starts_a_worker(self):
        args, fixture = self.fixture()
        fixture['source']['worker']['graph_math_passed'] = False
        (args.r13_run / 'result.json').write_text(json.dumps(fixture['source']))
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected process'))
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'source_failed')
        launched.assert_not_called()

    def test_worker_failure_retains_partial_outputs_and_never_retries(self):
        args, fixture = self.fixture()
        worker = dict(round='R14', model=fixture['model'], status='math_failed',
                      error='fixture reference mismatch', source_checked=True,
                      reference_passed=False, calibration_completed=None)
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            (args.out / 'worker/partial.json').write_text('{"partial": true}')
            return self.observation(exit_code=1)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'math_failed')
        self.assertIsNone(result['metrics']['calibration_completed'])
        self.assertIs(result['metrics']['reference_passed'], False)
        self.assertTrue((args.out / 'worker/partial.json').exists())
        self.assertEqual(len(self.ledger()), 1)

    def test_timeout_preserves_partial_worker_without_retry(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, dict(model=fixture['model'], status='error',
                                         source_checked=True, calibration_attempts=1))
            return self.observation(exit_code=-15, timed_out=True)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['metrics']['calibration_attempts'], 1)
        self.assertIsNone(result['metrics']['calibration_completed'])
        self.assertEqual(len(self.ledger()), 1)

    def test_incomplete_success_is_rejected_without_replacing_null_stages(self):
        args, fixture = self.fixture()
        worker = self.worker(fixture['model'])
        worker['emit_completed'] = None
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['metrics']['emit_completed'])

    def test_post_worker_case_budget_is_checked_by_file_metadata(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker(fixture['model']))
            (args.out / 'worker/partial.constants').write_bytes(b'0' * 32768)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack, patch.object(self.outer, 'CASE_BUDGET_BYTES', 16384):
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertFalse(result['metrics']['case_artifact_budget_passed'])
        self.assertTrue((args.out / 'worker/partial.constants').exists())
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertEqual(len(self.ledger()), 1)

    def test_existing_output_is_preserved_without_a_second_registry_row(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker(fixture['model']))
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            self.outer.execute(args)
            with self.assertRaisesRegex(ValueError, 'Existing run directories'):
                self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(len(self.ledger()), 1)

    def test_two_same_named_phases_keep_both_event_ids_and_exclusive_attribution(self):
        def event(kind, identity, stage, start, parent=None, end=None):
            result = dict(type=kind, event_id=identity, stage_id=stage, parent_id=parent,
                          start_ns=start, pid=1, ppid=0)
            if end is not None:
                result.update(end_ns=end, status='ok')
            return result
        timing = self.outer.summarize_events([
            event('begin', 'root', 'turnaround', 0),
            event('begin', 'first', 'reference_sample', 10, 'root'),
            event('end', 'first', 'reference_sample', 10, 'root', 30),
            event('begin', 'second', 'reference_sample', 40, 'root'),
            event('end', 'second', 'reference_sample', 40, 'root', 70),
            event('end', 'root', 'turnaround', 0, end=100)])
        phases = self.finalize(timing=timing)['metrics']['phase_elapsed_ns']
        self.assertEqual(set(phases), {'root', 'first', 'second'})
        self.assertEqual(phases['first']['stage_id'], phases['second']['stage_id'])
        self.assertEqual(sum(value['exclusive_ns'] for value in phases.values()), 100)
        self.assertTrue(timing['attribution_ready'])
