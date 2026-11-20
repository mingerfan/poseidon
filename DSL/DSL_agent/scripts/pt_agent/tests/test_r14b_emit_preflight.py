"""R14b protocol tests; full Alex preparation is absent from unit tests."""

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
class EmitEnvironmentChecks(unittest.TestCase):
    def test_only_current_round_inner_checks_run_without_skips(self):
        from pt_agent.tests import _r14b_emit_cases as cases
        expected = unittest.defaultTestLoader.loadTestsFromModule(cases).countTestCases()
        self.assertGreater(expected, 0)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r14b_emit_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), expected, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R14B_INNER_TESTS=' + count.group(1), flush=True)
        print('R14B_FULL_MODEL_PREPARATIONS_IN_UNIT_TESTS=0', flush=True)


class EmitOuterChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r14b_emit_preflight as outer
        self.outer = outer
        temporary = TemporaryDirectory(prefix='r14b-outer-fixtures-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self):
        outer = self.outer
        source = self.root / 'r14-source'
        work = source / 'worker'
        work.mkdir(parents=True)
        artifacts = {}
        for key, filename in [('program', 'program.ptdsl'), ('count', 'count.json'),
                              ('emit_stats', 'emit_stats.json')]:
            path = work / filename
            path.write_text('{}')
            artifacts[key] = str(path)
        r13 = self.root / 'r13-source'
        info = r13 / 'worker/model_info'
        info.mkdir(parents=True)
        (r13 / 'result.json').write_text('{}')
        (info / 'model_info.json').write_text('{}')
        frontend = self.root / 'original_expr.py'
        frontend.write_text('# fake original frontend\n')
        interpreter = self.root / 'intake-python'
        interpreter.touch()
        previous = dict(round='R14', model=outer.MODEL,
                        mode='common-model-rule-preparation', status='artifact_budget_exceeded',
                        worker=dict(round='R14', model=outer.MODEL,
                                    status='artifact_budget_exceeded', failed_phase='emit_write',
                                    artifacts=artifacts, r13_source_result=str(r13 / 'result.json')),
                        processes=[dict(command=['python', '--frontend', str(frontend)])],
                        source_r13=dict(run=str(r13), result=str(r13 / 'result.json'),
                                        artifacts=dict(model_info=str(info),
                                                       training_probes=str(r13 / 'unused-missing.npz'))),
                        timing=dict(spans=[dict(event_id='old:1', stage_id='calibrate',
                                               inclusive_ns=100, exclusive_ns=100)]))
        (source / 'result.json').write_text(json.dumps(previous))
        return SimpleNamespace(source_run=source, out=self.root / 'new-run'), dict(
            source=previous, frontend=frontend, interpreter=interpreter, info=info, r13=r13)

    def worker(self):
        outer = self.outer
        result = dict(round='R14b', model=outer.MODEL, status='ok', error=None,
                      emit_attempts=1, count_stats={'counts': {'boot': 0}},
                      emit_stats={'counts': {'boot': 0}})
        result.update({key: True for key in outer.STAGE_FIELDS})
        result.update({key: None for key in outer.PRIOR_STAGE_FIELDS + outer.UNMEASURED_FIELDS})
        result.update({key: 0 for key in outer.ZERO_FIELDS})
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
        stack.enter_context(patch.object(self.outer.paths, 'INTAKE_PYTHON', fixture['interpreter']))
        stack.enter_context(patch.object(self.outer, '_git_state', return_value=dict(branch='fixture')))
        stack.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                        return_value=SimpleNamespace(free=11 * 1024**3)))
        launched = stack.enter_context(patch.object(self.outer, 'run_process', side_effect=launch))
        return stack, launched

    def ledger(self):
        path = self.root / 'registry.jsonl'
        return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []

    def test_one_emit_worker_paths_limits_and_current_time_only(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker())
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok')
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index('--seconds') + 1], '1800')
        self.assertEqual(command[command.index('--python') + 1], str(fixture['interpreter']))
        self.assertIn('pt_agent.experiments.r14b_emit_worker', command)
        expected = {'--source-result': args.source_run / 'result.json',
                    '--program': args.source_run / 'worker/program.ptdsl',
                    '--count': args.source_run / 'worker/count.json',
                    '--emit-stats': args.source_run / 'worker/emit_stats.json',
                    '--r13-result': fixture['r13'] / 'result.json',
                    '--model-info': fixture['info'], '--frontend': fixture['frontend'],
                    '--work-dir': args.out / 'worker'}
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(command[command.index(option) + 1], str(value))
        self.assertNotIn('--inputs', command)
        self.assertNotIn('--r13-probes', command)
        limits = launched.call_args.kwargs
        self.assertEqual((limits['timeout'], limits['mem_gib'], limits['fsize_gib']), (1860, 4, 8))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(limits['env'][name], '1')
        self.assertFalse(result['source_r14']['included_in_current_turnaround'])
        self.assertEqual(result['source_r14']['prior_timing']['spans'][0]['event_id'], 'old:1')
        self.assertNotIn('old:1', result['metrics']['phase_elapsed_ns'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertEqual(len(self.ledger()), 1)
        row = self.ledger()[0]
        self.assertEqual(set(row), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual((row['round'], row['mode']), ('R14b', 'alex-emit-resume'))
        self.assertEqual(row['metrics'], result['metrics'])
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.UNMEASURED_FIELDS))

    def test_original_frontend_and_interpreter_symlink_are_reused(self):
        args, fixture = self.fixture()
        fixture['interpreter'].unlink()
        fixture['interpreter'].symlink_to('/nix/store/r14b-fixture-python/bin/python')
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker())
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'ok')
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index('--frontend') + 1], str(fixture['frontend']))

    def test_missing_saved_file_stops_before_launch_and_records_once(self):
        args, fixture = self.fixture()
        (args.source_run / 'worker/count.json').unlink()
        stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'dependency_missing')
        launched.assert_not_called()
        self.assertEqual(len(self.ledger()), 1)
        self.assertEqual(result['metrics']['emit_attempts'], 0)

    def test_wrong_original_stop_or_ambiguous_frontend_never_launches(self):
        for kind in ('failed_phase', 'duplicate_frontend'):
            with self.subTest(kind=kind), TemporaryDirectory(dir=self.root) as tmp:
                previous_root = self.root
                self.root = Path(tmp)
                try:
                    args, fixture = self.fixture()
                    if kind == 'failed_phase':
                        fixture['source']['worker']['failed_phase'] = 'calibrate'
                    else:
                        fixture['source']['processes'][0]['command'] += ['--frontend', str(fixture['frontend'])]
                    (args.source_run / 'result.json').write_text(json.dumps(fixture['source']))
                    stack, launched = self.environment(fixture, lambda *a, **kw: self.fail('unexpected launch'))
                    with stack:
                        result = self.outer.execute(args)
                    self.assertEqual(result['status'], 'source_failed')
                    launched.assert_not_called()
                finally:
                    self.root = previous_root

    def test_worker_error_and_partial_artifacts_are_preserved_without_retry(self):
        args, fixture = self.fixture()
        worker = self.worker()
        worker.update(status='artifact_budget_exceeded', error='fake prewrite failure', emit_completed=None)
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            (args.out / 'worker/partial.json').write_text('{"partial": true}')
            return self.observation(exit_code=1)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertTrue((args.out / 'worker/partial.json').exists())
        self.assertEqual(len(self.ledger()), 1)
        self.assertIsNone(result['metrics']['emit_completed'])

    def test_timeout_is_retained_with_one_attempt(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, dict(model=self.outer.MODEL, status='error', emit_attempts=1))
            return self.observation(exit_code=-15, timed_out=True)
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['metrics']['emit_attempts'], 1)
        self.assertEqual(len(self.ledger()), 1)

    def test_success_rejects_a_repeated_prior_stage(self):
        args, fixture = self.fixture()
        worker = self.worker()
        worker['count_attempts'] = 1
        def launch(*command, **kwargs):
            self.write_worker(args, worker)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'incomplete')
        launched.assert_called_once()

    def test_case_limit_counts_terminal_result_itself(self):
        out = self.root / 'terminal-case'
        out.mkdir()
        (out / 'small.log').write_text('small fake log')
        result = self.outer.finalize_result(worker=self.worker(), processes=[], timing={},
                                             status='ok', error=None, git_state={}, out=out)
        self.assertLess(self.outer._case_size(out)['bytes'], 1024)
        with patch.object(self.outer, 'CASE_BUDGET_BYTES', 1024):
            self.outer._write_terminal_result(out, result, [])
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertFalse(result['metrics']['case_artifact_budget_passed'])
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(out)['bytes'])

    def test_post_worker_case_budget_retains_partial_file_and_one_registry_row(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker())
            (args.out / 'worker/partial.constants').write_bytes(b'0' * 32768)
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack, patch.object(self.outer, 'CASE_BUDGET_BYTES', 16384):
            result = self.outer.execute(args)
        launched.assert_called_once()
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertTrue((args.out / 'worker/partial.constants').is_file())
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertEqual(len(self.ledger()), 1)
        self.assertEqual(self.ledger()[0]['metrics'], result['metrics'])

    def test_existing_output_is_preserved_and_r13_r14_subdirectories_are_rejected(self):
        args, fixture = self.fixture()
        def launch(*command, **kwargs):
            self.write_worker(args, self.worker())
            return self.observation()
        stack, launched = self.environment(fixture, launch)
        with stack:
            self.outer.execute(args)
            for destination in (args.out, args.source_run / 'new', fixture['r13'] / 'new'):
                with self.subTest(destination=destination):
                    with self.assertRaises(ValueError):
                        self.outer.execute(SimpleNamespace(source_run=args.source_run, out=destination))
        launched.assert_called_once()
        self.assertEqual(len(self.ledger()), 1)
        self.assertFalse((fixture['r13'] / 'new').exists())

    def test_finalize_keeps_unstarted_and_formal_fields_null(self):
        result = self.outer.finalize_result(worker=None, processes=[], timing={},
                                             status='dependency_missing', error='fake missing file',
                                             git_state={}, out=self.root / 'run')
        self.assertTrue(all(result['metrics'][key] is None
                            for key in self.outer.STAGE_FIELDS + self.outer.UNMEASURED_FIELDS))
        self.assertTrue(all(result['metrics'][key] == 0 for key in self.outer.ZERO_FIELDS))
        self.assertEqual(result['metrics']['emit_attempts'], 0)

    def test_cli_accepts_source_run_and_new_output_without_sample_arguments(self):
        for argv in ([], ['--source-run', '/fixture/source'], ['--out', '/fixture/new']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                    patch.object(self.outer, 'execute') as launched:
                with self.assertRaises(SystemExit) as caught:
                    self.outer.main(argv)
                self.assertEqual(caught.exception.code, 2)
                launched.assert_not_called()
        with patch.object(self.outer, 'execute', return_value={'status': 'ok'}) as launched, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.outer.main(['--source-run', '/fixture/source', '--out', '/fixture/new']), 0)
        self.assertEqual(vars(launched.call_args.args[0]),
                         dict(source_run=Path('/fixture/source'), out=Path('/fixture/new')))
