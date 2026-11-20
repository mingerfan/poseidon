"""R23 stdlib protocol fixtures and one supervised Orbit-environment wrapper."""

import ast
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


ORBIT_PYTHON = Path(os.environ.get('R23_ORBIT_PYTHON',
                                  str(paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python')))
BOOTSTRAP = r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
work = Path(sys.argv[1]).resolve()
os.environ['R23_TEST_WORK'] = str(work)
os.environ['TMPDIR'] = str(work)
tempfile.tempdir = str(work)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
signal.alarm(240)
suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._r23_orbit_optimizer_cases.R23OrbitOptimizerCases')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = dict(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skips=len(result.skipped), success=result.wasSuccessful())
(work / 'inner-unit-result.json').write_text(json.dumps(record, allow_nan=False) + '\n')
raise SystemExit(0 if record['success'] and record['testsRun'] > 0 and record['skips'] == 0 else 1)
'''


@require(ORBIT_PYTHON)
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_fake_optimizer_cases_run_once_without_skips(self):
        from pt_agent.experiments.r10_paper_metrics import run_process

        tree = ast.parse(Path(__file__).with_name('_r23_orbit_optimizer_cases.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'R23OrbitOptimizerCases']
        self.assertEqual(len(classes), 1)
        expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                       for node in classes[0].body)
        self.assertGreater(expected, 0)
        work = Path(os.environ['R23_TEST_WORK']).resolve()
        work.mkdir(parents=True, exist_ok=True)
        unit, summary = work / 'inner-unit-result.json', work / 'inner-wrapper-result.json'
        self.assertFalse(unit.exists(), 'This inner suite has one approved attempt')
        self.assertFalse(summary.exists(), 'This wrapper has one approved attempt')
        command = [str(ORBIT_PYTHON), '-B', '-c', BOOTSTRAP, str(work)]
        log = work / 'inner-tests.log'
        observation = run_process(command, cwd=Path(__file__).resolve().parents[2],
                                  log=log, resources=work / 'inner-resources.jsonl',
                                  timeout=300, mem_gib=4, fsize_gib=8,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        output = log.read_text()
        print(output, end='', flush=True)
        self.assertEqual(observation['exit_code'], 0, output)
        self.assertFalse(observation['timed_out'], observation)
        self.assertIsNone(observation['signal'], observation)
        self.assertEqual(observation['remaining_pids'], [], observation)
        counts = re.findall(r'Ran (\d+) tests? in ', output)
        self.assertTrue(counts, output)
        self.assertEqual(int(counts[-1]), expected, output)
        self.assertNotRegex(output, r'skipped=\d+|\.\.\. skipped ')
        record = json.loads(unit.read_text())
        self.assertEqual(record, dict(testsRun=expected, failures=0, errors=0, skips=0, success=True))
        summary.write_text(json.dumps(dict(status='ok', wrapper_runs=1, inner_tests=expected,
            inner_skips=0, observation=observation), allow_nan=False) + '\n')
        print('R23_INNER_TESTS=' + str(expected), flush=True)
        print('R23_REAL_SOLVER_LICENSE_MODEL_AND_OPTIMIZER_CALLS_IN_TESTS=0', flush=True)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r23_orbit_optimizer_preflight as outer

        self.outer = outer
        parent = os.environ.get('R23_TEST_WORK')
        temporary = TemporaryDirectory(prefix='r23-outer-', dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'; self.results.mkdir()
        python = self.root / 'orbit-venv/bin/python'
        self.locations = dict(results=self.results, repo=self.root / 'repo',
            scripts=self.root / 'repo/scripts', orbit=self.root / 'orbit', orbit_python=python,
            profile=self.root / 'orbit' / outer.PROFILE,
            license_file=python.parent.parent / 'lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic',
            r22_result=self.results / 'r22-original/result.json')
        for key in ('orbit_python', 'profile', 'license_file'):
            path = self.locations[key]; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('metadata fixture only; never loaded\n')
        self.locations['scripts'].mkdir(parents=True)
        self.mlir = self.locations['r22_result'].parent / 'work/trace-worker/trace/original.mlir'
        self.mlir.parent.mkdir(parents=True); self.mlir.write_text('protocol fixture; never parsed\n')
        self.original = dict(round='R22', status='ok', model=outer.MODEL,
            mode='orbit-resnet20-trace-tdag-preflight', orbit_model_trace_passed=True,
            tdag_parse_passed=True, attempted=dict(trace=1, parse=1), completed=dict(trace=1, parse=1),
            phase_results=dict(trace=dict(status='ok', trace_runs=1, activation_calls=19,
                activation_names=['act_' + str(i) for i in range(19)],
                mlir=dict(path=str(self.mlir), size_bytes=self.mlir.stat().st_size)),
                parse=dict(status='ok', parse_runs=1,
                    graph_check=dict(passed=True, node_count=10113, edge_count=13342),
                    parser_metadata=dict(profile=str(self.locations['profile']), poly_deg=131072,
                                         max_slot=65536))))
        self._save(self.locations['r22_result'], self.original)

    @staticmethod
    def _save(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, allow_nan=False) + '\n')

    def _args(self):
        return SimpleNamespace(out=self.results / self.outer.OUT_NAME)

    def _registry(self):
        path = self.results / 'registry.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _fresh_args(self, label):
        self.results = self.root / label / 'results'
        self.results.mkdir(parents=True)
        self.locations['results'] = self.results
        return self._args()

    def _process(self, *, fail=None, missing_worker=False, fatal=False, empty_output=False,
                 test_count=15, skip=False, **overrides):
        calls = []
        def run(command, **options):
            stage = Path(options['log']).stem
            calls.append((stage, command, options))
            observation = dict(exit_code=0, timed_out=False, signal=None,
                started_ns=1, ended_ns=2, termination_observed_ns=3,
                peak_rss_kib_by_pid={}, outer_peak_rss_kib=0,
                signals_sent=[], remaining_pids=[])
            log = Path(options['log'])
            if stage.startswith('git-'):
                log.write_text('fake-revision\n' if stage == 'git-revision' else 'lhy-agent-dsl\n')
            elif stage == 'r23-tests':
                log.write_text('Ran ' + str(test_count) + ' tests in 0.001s\n\nOK' +
                               (' (skipped=1)' if skip else '') + '\n')
                self._save(Path(options['env']['R23_TEST_WORK']) / 'inner-wrapper-result.json',
                    dict(status='ok', wrapper_runs=1, inner_tests=11, inner_skips=0,
                         observation=observation))
            else:
                phase = stage.removeprefix('model-')
                work = Path(command[command.index('--work-dir') + 1]); work.mkdir(parents=True)
                (work / 'events.jsonl').touch()
                worker = dict(status='ok', native_optimizer_completed=phase=='optimize',
                    placement_artifact_checked=phase=='validate',
                    optimize_attempts=1 if phase=='optimize' else 0,
                    optimize_completed=1 if phase=='optimize' else 0,
                    solve_ilp_attempts=3 if phase=='optimize' else 0,
                    solve_ilp_completed=3 if phase=='optimize' else 0,
                    solve_ilp_feasible=2 if phase=='optimize' else 0,
                    solve_ilp_infeasible=1 if phase=='optimize' else 0, solve_ilp_errors=0,
                    placement_bootstrap_nodes=2 if phase=='validate' else None,
                    placement_bootstrap_weight_sum=9 if phase=='validate' else None)
                if phase == 'optimize':
                    if not empty_output:
                        Path(command[command.index('--output-mlir') + 1]).write_text('fake native output; never parsed\n')
                    log.write_text('Final assignment latency: 0.125 sec.\nFinal tdag latency: 0.25 sec.\n')
                    if fatal:
                        self._save(work / 'first-fatal.json', dict(
                            termination_origin='worker_caught_exception',
                            exception_type='FakeLicenseError', message='fixed restricted-license fixture'))
                else:log.write_text('fake validation, no parser execution\n')
                if not missing_worker:self._save(work / 'phase_result.json', worker)
            if stage == fail:
                observation.update(exit_code=1)
                observation.update(overrides)
            return observation
        run.calls = calls
        return run

    def _execute(self, args, process):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.outer, '_locations', return_value=self.locations))
            stack.enter_context(patch.object(self.outer, 'run_process', side_effect=process))
            stack.enter_context(patch.object(self.outer.resource, 'setrlimit'))
            stack.enter_context(patch.object(self.outer.signal, 'signal'))
            stack.enter_context(patch.object(self.outer.signal, 'setitimer'))
            stack.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                             return_value=SimpleNamespace(free=11 * 1024**3)))
            stack.enter_context(patch.object(self.outer.threading, 'Thread',
                return_value=SimpleNamespace(start=Mock(), is_alive=lambda:False)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.outer.execute(args)

    def test_source_requires_the_exact_completed_r22_layout_without_reading_constants(self):
        source = self.outer._source_preflight(self.locations)
        self.assertEqual(Path(source['input_mlir']), self.mlir)
        self.assertEqual(self.mlir.read_text(), 'protocol fixture; never parsed\n')

    def test_source_rejects_identity_counts_activation_graph_profile_and_foreign_mlir(self):
        for variant in ('identity', 'count', 'activation', 'nodes', 'edges', 'degree', 'slots', 'path', 'size'):
            record = copy.deepcopy(self.original)
            trace, parse = record['phase_results']['trace'], record['phase_results']['parse']
            if variant == 'identity':record['model'] = 'foreign-model'
            elif variant == 'count':record['completed']['trace'] = 0
            elif variant == 'activation':trace['activation_names'].pop()
            elif variant == 'nodes':parse['graph_check']['node_count'] = 10112
            elif variant == 'edges':parse['graph_check']['edge_count'] = 13341
            elif variant == 'degree':parse['parser_metadata']['poly_deg'] = 32768
            elif variant == 'slots':parse['parser_metadata']['max_slot'] = 16384
            elif variant == 'path':trace['mlir']['path'] = str(self.root / 'foreign.mlir')
            else:trace['mlir']['size_bytes'] += 1
            self._save(self.locations['r22_result'], record)
            with self.subTest(variant=variant), self.assertRaises((ValueError, RuntimeError)):
                self.outer._source_preflight(self.locations)

    def test_license_path_is_fixed_and_missing_license_stops_source_preflight(self):
        original = self.locations['license_file']
        foreign = self.root / 'foreign.lic'; foreign.write_text('metadata fixture\n')
        with self.assertRaises((ValueError, RuntimeError)):
            self.outer._source_preflight(dict(self.locations, license_file=foreign))
        original.unlink()
        with self.assertRaises((ValueError, RuntimeError)):
            self.outer._source_preflight(self.locations)

    def test_worker_argv_preserves_saved_input_output_profile_and_conditional_source(self):
        work = self.root / 'case/work'
        for phase in ('optimize', 'validate'):
            command = self.outer.phase_command(self.locations, work, phase, self.mlir)
            self.assertEqual(command[0], str(self.locations['orbit_python']))
            self.assertNotIn('pt_agent.tools.nixrun', command)
            for option, value in (('--phase', phase), ('--input-mlir', self.mlir),
                    ('--output-mlir', work / 'optimized.mlir'), ('--profile', self.locations['profile']),
                    ('--license-file', self.locations['license_file']),
                    ('--work-dir', work / (phase + '-worker'))):
                self.assertEqual(command.count(option), 1)
                self.assertEqual(command[command.index(option) + 1], str(value))
            self.assertIn('--source-record', command)
            if phase == 'optimize':
                self.assertEqual(command[command.index('--source-record') + 1], str(self.locations['r22_result']))

    def test_child_environment_clears_inherited_license_values_and_keeps_home(self):
        with patch.dict(os.environ, GRB_LICENSE_FILE='never-record-this', GRB_TEST_SECRET='never-record-this',
                        GUROBI_HOME='never-record-this', HOME='/fixture-home', PYTHONPATH='old'):
            env = self.outer.child_environment(self.root / 'work', self.locations)
        self.assertEqual(env['GRB_LICENSE_FILE'], str(self.locations['license_file']))
        self.assertNotIn('GRB_TEST_SECRET', env)
        self.assertNotIn('GUROBI_HOME', env)
        self.assertEqual(env['HOME'], '/fixture-home')
        self.assertEqual(env['R23_TEST_WORK'], str(self.root / 'work/test-worker'))
        self.assertEqual(env['R23_ORBIT_PYTHON'], str(self.locations['orbit_python']))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name], '1')

    def test_fixed_overall_memory_and_artifact_budgets(self):
        self.assertEqual(self.outer.CATEGORIES, ('tests', 'optimize', 'validate'))
        self.assertEqual((self.outer.OVERALL_SECONDS, self.outer.CASE_BYTES, self.outer.FREE_BYTES),
                         (8400, 4 * 1024**3, 10 * 1024**3))

    def test_success_launches_tests_optimizer_and_validation_once_with_native_budgets(self):
        args, process = self._args(), self._process()
        result = self._execute(args, process)
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertEqual([row[0] for row in process.calls],
            ['git-revision', 'git-branch', 'r23-tests', 'model-optimize', 'model-validate'])
        self.assertEqual(result['attempted'], dict.fromkeys(self.outer.CATEGORIES, 1))
        self.assertEqual(result['completed'], result['attempted'])
        self.assertEqual((result['git_metadata_attempted'], result['git_metadata_completed']), (2, 2))
        self.assertEqual([row[2]['timeout'] for row in process.calls], [30, 30, 600, 7200, 300])
        self.assertTrue(all((row[2]['mem_gib'], row[2]['fsize_gib']) == (4, 8) for row in process.calls))
        self.assertTrue(result['native_optimizer_completed'])
        self.assertTrue(result['placement_artifact_checked'])
        self.assertEqual((result['placement_bootstrap_nodes'], result['placement_bootstrap_weight_sum']), (2, 9))
        self.assertTrue(all(result['metrics'][name] is None for name in self.outer.FORMAL))
        self.assertIsNone(result['metrics']['model_optimize_calls'])
        self.assertFalse(result['formal_matrix_complete'])
        self.assertFalse(result['backend_ready'])
        self.assertEqual(result['retry_count'], 0)
        self.assertEqual(result['case_artifact_bytes'], self.outer.case_size(args.out))
        rows = self._registry(); self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual((rows[0]['round'], rows[0]['mode']), ('R23', self.outer.MODE))
        self.assertEqual(rows[0]['metrics'], result['metrics'])
        timing = json.loads((args.out / 'timing.json').read_text())
        self.assertTrue(timing['attribution_ready'])
        self.assertEqual(timing['diagnostics'], [])
        roots = [row for row in timing['spans'] if row['parent_id'] is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(sum(row['exclusive_ns'] for row in timing['spans']), roots[0]['inclusive_ns'])

    def test_optimizer_failure_stops_validation_with_native_and_outer_signals_separate(self):
        variants = [('native', dict(exit_code=-15, signal=None)),
                    ('outer', dict(exit_code=-15, signal=15)),
                    ('timeout', dict(timed_out=True)), ('residual', dict(remaining_pids=[98765]))]
        for label, observation in variants:
            args, process = self._fresh_args(label), self._process(fail='model-optimize', **observation)
            with self.subTest(label=label):
                result = self._execute(args, process)
                self.assertEqual(result['status'], 'stopped_on_error')
                self.assertEqual(result['attempted'], dict(tests=1, optimize=1, validate=0))
                self.assertEqual(result['completed'], dict(tests=1, optimize=0, validate=0))
                command = result['processes'][-1]
                self.assertEqual(command['native_termination_signal'], 15 if label in ('native','outer') else None)
                self.assertEqual(command['observation']['signal'], observation.get('signal'))
                self.assertEqual(process.calls[-1][0], 'model-optimize')

    def test_first_fatal_notification_stops_even_if_process_returns_zero(self):
        result = self._execute(self._args(), self._process(fatal=True))
        self.assertEqual(result['status'], 'stopped_on_error')
        self.assertEqual(result['termination_origin'], 'worker_caught_exception')
        self.assertEqual(result['first_fatal']['exception_type'], 'FakeLicenseError')
        self.assertEqual(result['attempted']['validate'], 0)

    def test_missing_worker_and_missing_output_keep_unknown_or_unstarted_metrics(self):
        for label, changes in [('missing', dict(missing_worker=True)), ('empty', dict(empty_output=True))]:
            args, process = self._fresh_args(label), self._process(**changes)
            with self.subTest(label=label):
                result = self._execute(args, process)
                self.assertEqual(result['status'], 'stopped_on_error')
                self.assertEqual(result['attempted']['validate'], 0)
                self.assertIsNone(result['placement_bootstrap_nodes'])
                self.assertIsNone(result['placement_bootstrap_weight_sum'])
                if label == 'missing':self.assertIsNone(result['metrics']['optimizer_attempts'])

    def test_empty_and_skipped_tests_stop_before_optimizer(self):
        for label, changes in [('empty', dict(test_count=0)), ('skip', dict(skip=True))]:
            args, process = self._fresh_args(label), self._process(**changes)
            with self.subTest(label=label):
                result = self._execute(args, process)
                self.assertEqual(result['status'], 'stopped_on_error')
                self.assertEqual(result['attempted']['optimize'], 0)
                self.assertEqual(process.calls[-1][0], 'r23-tests')

    def test_terminal_json_fixed_point_and_late_budget_failure_preserve_unique_registry(self):
        args, process = self._args(), self._process()
        original_write = self.outer.write_json
        def write(path, record):
            original_write(path, record)
            if Path(path).name == 'timing.json':
                (args.out / 'late-terminal.log').write_bytes(b'x' * 1024**2)
        with patch.object(self.outer, 'CASE_BYTES', 1024**2), patch.object(self.outer, 'write_json', side_effect=write):
            result = self._execute(args, process)
        self.assertEqual(result['status'], 'stopped_on_error')
        self.assertTrue(result['terminal_budget_error'])
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer.case_size(args.out))
        self.assertEqual(json.loads((args.out / 'result.json').read_text()), result)
        self.assertEqual(len(self._registry()), 1)
        self.assertEqual(self._registry()[0]['metrics'], result['metrics'])

    def test_existing_foreign_and_source_directories_remain_untouched(self):
        existing = self._args().out; existing.mkdir()
        marker = existing / 'saved.txt'; marker.write_text('preserved\n')
        for out in (existing, self.results / 'unapproved-name', self.locations['r22_result'].parent):
            with self.subTest(out=out), patch.object(self.outer, '_locations', return_value=self.locations), \
                    patch.object(self.outer, 'run_process') as launch, self.assertRaises(ValueError):
                self.outer.execute(SimpleNamespace(out=out))
            launch.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserved\n')
        self.assertEqual(self._registry(), [])


if __name__ == '__main__':
    unittest.main()
