"""R22 stdlib protocol checks and one supervised intake-environment run."""

import ast
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import re
import signal
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


INTAKE_ARTIFACT = (paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink()
                   else paths.INTAKE_PYTHON)
DEPS_ARTIFACT = Path(os.environ.get('R22_DEPS_DIR', '/r22-approved-dependencies-not-provided'))
INNER_MODULE = 'pt_agent.tests._r22_orbit_model_cases'
BOOTSTRAP = r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
deps, work = (Path(value).resolve() for value in sys.argv[1:3])
os.environ['R22_DEPS_DIR'] = str(deps)
os.environ['R22_TEST_WORK'] = str(work)
os.environ['TMPDIR'] = str(work)
tempfile.tempdir = str(work)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
signal.alarm(300)
suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._r22_orbit_model_cases.OrbitModelCases')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = dict(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skips=len(result.skipped), success=result.wasSuccessful())
(work / 'inner-unit-result.json').write_text(json.dumps(record, allow_nan=False) + '\n')
raise SystemExit(0 if record['success'] and record['testsRun'] > 0 and record['skips'] == 0 else 1)
'''


@require(INTAKE_ARTIFACT, DEPS_ARTIFACT / 'einops/__init__.py')
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_fake_orbit_cases_run_once_without_skips(self):
        from pt_agent.experiments.r10_paper_metrics import run_process

        helper = Path(__file__).with_name('_r22_orbit_model_cases.py')
        tree = ast.parse(helper.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'OrbitModelCases']
        self.assertEqual(len(selected), 1)
        expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                       for node in selected[0].body)
        self.assertGreater(expected, 0)
        work = Path(os.environ['R22_TEST_WORK']).resolve()
        work.mkdir(parents=True, exist_ok=True)
        unit_path, summary_path = work / 'inner-unit-result.json', work / 'inner-wrapper-result.json'
        self.assertFalse(unit_path.exists(), 'The approved inner suite must run once')
        self.assertFalse(summary_path.exists(), 'The approved wrapper must run once')
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '300',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-c', BOOTSTRAP,
                   str(DEPS_ARTIFACT.resolve()), str(work)]
        log = work / 'inner-tests.log'
        observation = run_process(command, cwd=Path(__file__).resolve().parents[2],
                                  log=log, resources=work / 'inner-resources.jsonl',
                                  timeout=480, mem_gib=4, fsize_gib=8,
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
        record = json.loads(unit_path.read_text())
        self.assertEqual(record, dict(testsRun=expected, failures=0, errors=0, skips=0, success=True))
        summary = dict(status='ok', wrapper_runs=1, inner_tests=expected, inner_skips=0,
                       observation=observation)
        summary_path.write_text(json.dumps(summary, allow_nan=False) + '\n')
        print('R22_INNER_TESTS=' + str(expected), flush=True)
        print('R22_REAL_MODELS_WEIGHTS_DATA_AND_FRONTEND_TRACES_IN_TESTS=0', flush=True)


class OuterProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_handler = signal.getsignal(signal.SIGALRM)
        cls.saved_timer = signal.getitimer(signal.ITIMER_REAL)
        def expired(signum, frame):
            raise TimeoutError('R22 fake checks exhausted 180 seconds')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 180)

    @classmethod
    def tearDownClass(cls):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, cls.saved_handler)
        if cls.saved_timer[0] or cls.saved_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *cls.saved_timer)

    def setUp(self):
        from pt_agent.experiments import r22_orbit_model_preflight as outer

        self.outer = outer
        temporary = TemporaryDirectory(prefix='r22-outer-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.locations = dict(results=self.results, repo=self.root / 'repo',
            scripts=self.root / 'repo/scripts', orbit=self.root / 'orbit',
            frontend=self.root / 'orbit/frontend/dacapo', intake_python=self.root / 'intake/python',
            orbit_python=self.root / 'orbit-venv/bin/python',
            profile=self.root / 'orbit' / outer.PROFILE,
            benchmark=self.root / 'orbit/frontend/dacapo_patch/benchmarks/ResNetSiLU64k.py',
            model_source=self.root / 'orbit/frontend/dacapo/python/poly/poly/models/ResNetSiLU.py',
            checkpoint=self.root / 'original-model.th', poly_data=self.root / 'upstream/data',
            r13_result=self.results / 'r13-original/result.json',
            r16_result=self.results / 'r16-original/result.json')
        required = [self.locations[name] for name in ('intake_python', 'orbit_python', 'benchmark',
                    'model_source', 'checkpoint', 'profile')]
        required += [self.locations['frontend'] / 'python/hecate/hecate/expr.py',
                     self.locations['frontend'] / 'build/lib/libHecateFrontend.so']
        required += [self.locations['frontend'] / 'python/poly/poly' / name for name in outer.POLY_FILES]
        required += [self.locations['poly_data'] / name for name in outer.DATA_FILES]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('tiny metadata fixture; never executed or loaded\n')
        self.locations['scripts'].mkdir(parents=True)
        self.r13 = dict(round='R13', status='ok', model=outer.MODEL,
                       worker=dict(round='R13', status='ok', model=outer.MODEL,
                                   **dict.fromkeys(('import_completed', 'real_probes_passed',
                                      'graph_math_passed', 'roundtrip_passed', 'static_coverage'), True)))
        w16 = dict(round='R16', status='ok', model=outer.MODEL, baseline='orbit',
            **dict.fromkeys(('source_checked', 'checkpoint_checked', 'model_built', 'strict_load_passed',
              'structure_passed', 'probes_checked', 'math_probes_passed', 'compatibility_passed', 'all_finite'), True),
            source_summary={name:str(self.locations[name]) for name in ('r13_result', 'model_source', 'checkpoint')},
            forward_attempts=3, forward_completed=3,
            math_probes=[dict(probe_index=index, passed=True, finite=True, shape_ok=True, dtype_ok=True,
                              criterion='1e-12+1e-12*abs(reference)') for index in range(3)],
            state_report=dict(passed=True, strict_load=True, structure=dict(passed=True),
                              original_state=[dict(name='tiny.weight', finite=True, numerically_equal=True)]))
        self.r16 = dict(round='R16', baseline='orbit', status='ok', model=outer.MODEL, worker=w16,
                        source_r13=dict(result=str(self.locations['r13_result'])))
        self._save(self.locations['r13_result'], self.r13)
        self._save(self.locations['r16_result'], self.r16)

    @staticmethod
    def _save(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, allow_nan=False) + '\n')

    def _metadata(self):
        return dict(info=dict(name='einops', version='0.8.1'), urls=[dict(
            filename=self.outer.WHEEL_NAME, packagetype='bdist_wheel', size=11,
            url='https://files.pythonhosted.org/packages/fixed/' + self.outer.WHEEL_NAME)])

    def _args(self, label='case'):
        return SimpleNamespace(out=self.results / label)

    def _registry(self):
        path = self.results / 'registry.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _fake_dependencies(self, locations, work, result, recorder, launch, checkpoint):
        for category in ('metadata_download', 'wheel_download', 'dependency_install'):
            result['attempted'][category] += 1
            result['completed'][category] += 1
        target = work / 'python-deps/einops/__init__.py'
        target.parent.mkdir(parents=True)
        target.write_text("__version__ = '0.8.1'\n")
        return target.parent.parent

    def _process(self, *, fail=None, missing_worker=False, test_count=14, skip=False, **overrides):
        calls = []
        def run(command, **kwargs):
            name = Path(kwargs['log']).stem
            calls.append((name, command, kwargs))
            obs = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[], signals_sent=[])
            log = Path(kwargs['log'])
            if name.startswith('git-'):
                log.write_text('fake-revision\n' if name == 'git-revision' else 'lhy-agent-dsl\n')
            elif name == 'r22-tests':
                log.write_text('Ran ' + str(test_count) + ' tests in 0.001s\n\nOK' +
                               (' (skipped=1)' if skip else '') + '\n')
                work = Path(kwargs['env']['R22_TEST_WORK'])
                self._save(work / 'inner-unit-result.json',
                           dict(testsRun=14, failures=0, errors=0, skips=0, success=True))
                self._save(work / 'inner-wrapper-result.json',
                           dict(status='ok', wrapper_runs=1, inner_tests=14, inner_skips=0, observation=obs))
            else:
                log.write_text('fake supervised phase, no toolchain execution\n')
                phase = name.removeprefix('model-')
                work = Path(command[command.index('--work-dir') + 1])
                work.mkdir(parents=True, exist_ok=True)
                (work / 'events.jsonl').touch()
                worker = dict(status='ok', phase=phase, **{phase + '_runs':1})
                if phase == 'trace':
                    mlir = work / 'trace/model.mlir'
                    mlir.parent.mkdir()
                    mlir.write_text('fake protocol MLIR, never parsed\n')
                    worker.update(mlir=dict(path=str(mlir), size_bytes=mlir.stat().st_size),
                                  orbit_model_trace_passed=True)
                else:
                    worker.update(tdag_parse_passed=True, graph_check=dict(passed=True))
                if not missing_worker:
                    self._save(work / 'phase_result.json', worker)
            if name == fail:
                obs.update(exit_code=1, **overrides)
            return obs
        run.calls = calls
        return run

    def _execute(self, args, process, *, preparation=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.outer, '_locations', return_value=self.locations))
            stack.enter_context(patch.object(self.outer, 'prepare_dependencies',
                                             side_effect=preparation or self._fake_dependencies))
            stack.enter_context(patch.object(self.outer, 'run_process', side_effect=process))
            stack.enter_context(patch.object(self.outer.shutil, 'disk_usage',
                                             return_value=SimpleNamespace(free=11 * 1024**3)))
            stack.enter_context(patch.object(self.outer.resource, 'setrlimit'))
            stack.enter_context(patch.object(self.outer.signal, 'signal'))
            stack.enter_context(patch.object(self.outer.signal, 'setitimer'))
            stack.enter_context(patch.object(self.outer.threading, 'Thread',
                                             return_value=SimpleNamespace(start=Mock(), is_alive=lambda:False)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.outer.execute(args)

    def test_original_source_evidence_is_checked_without_loading_models_or_probes(self):
        record = self.outer._source_preflight(self.locations)
        self.assertEqual(record['original_reference_probes'], 3)
        self.assertEqual(record['references_repeated'], 0)
        self.assertFalse(record['coefficient_source_equivalence_verified'])
        self.assertEqual((record['r13_recorded_mux_slots'], record['proposed_native_template_slots']),
                         (16384, 65536))
        self.assertTrue(record['inputs'])
        self.assertTrue(all(row['bytes'] > 0 for row in record['inputs']))

    def test_source_identity_strict_math_and_original_state_fail_before_children(self):
        variants = ('baseline', 'source', 'forwards', 'criterion', 'finite', 'state', 'missing')
        for variant in variants:
            changed = copy.deepcopy(self.r16)
            if variant == 'baseline':changed['baseline'] = 'orion'
            elif variant == 'source':changed['worker']['source_summary']['checkpoint'] = str(self.root / 'foreign.th')
            elif variant == 'forwards':changed['worker']['forward_completed'] = 2
            elif variant == 'criterion':changed['worker']['math_probes'][0]['criterion'] = 'looser'
            elif variant == 'finite':changed['worker']['math_probes'][0]['finite'] = False
            elif variant == 'state':changed['worker']['state_report']['original_state'][0]['numerically_equal'] = False
            else:changed['worker']['math_probes'].pop()
            self._save(self.locations['r16_result'], changed)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.outer._source_preflight(self.locations)
        self._save(self.locations['r16_result'], self.r16)

    def test_only_fixed_universal_wheel_and_approved_host_are_selected(self):
        metadata = self._metadata()
        selected = self.outer.select_wheel(metadata)
        self.assertEqual(selected, dict(filename=self.outer.WHEEL_NAME,
                         url=metadata['urls'][0]['url'], bytes=11))
        for variant in ('version', 'duplicate', 'http', 'host', 'query', 'empty', 'large'):
            changed = copy.deepcopy(metadata)
            if variant == 'version':changed['info']['version'] = '0.8.0'
            elif variant == 'duplicate':changed['urls'].append(copy.deepcopy(changed['urls'][0]))
            elif variant == 'http':changed['urls'][0]['url'] = changed['urls'][0]['url'].replace('https:', 'http:')
            elif variant == 'host':changed['urls'][0]['url'] = changed['urls'][0]['url'].replace('files.pythonhosted.org', 'example.org')
            elif variant == 'query':changed['urls'][0]['url'] += '?unapproved=1'
            else:changed['urls'][0]['size'] = 0 if variant == 'empty' else 1024**2 + 1
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.outer.select_wheel(changed)

    def test_install_and_phase_commands_keep_explicit_paths_native_profile_and_limits(self):
        work = self.root / 'work'
        wheel, target = work / self.outer.WHEEL_NAME, work / 'python-deps'
        self.assertEqual(self.outer.dependency_install_command(self.locations, wheel, target),
            [str(self.locations['orbit_python']), '-B', '-m', 'pip', '--isolated',
             '--disable-pip-version-check', '--no-cache-dir', 'install', '--no-index',
             '--no-deps', '--no-compile', '--target', str(target), str(wheel)])
        trace = self.outer.phase_commands(self.locations, work, 'trace')
        self.assertEqual(trace[trace.index('--seconds') + 1], '1860')
        self.assertEqual(trace[trace.index('--python') + 1], str(self.locations['intake_python']))
        for option, key in (('--frontend', 'frontend'), ('--checkpoint', 'checkpoint'),
                            ('--benchmark', 'benchmark'), ('--r13-result', 'r13_result'),
                            ('--r16-result', 'r16_result')):
            self.assertEqual(trace[trace.index(option) + 1], str(self.locations[key]))
        mlir = work / 'trace-worker/trace/model.mlir'
        parse = self.outer.phase_commands(self.locations, work, 'parse', mlir)
        self.assertEqual(parse[0], str(self.locations['orbit_python']))
        self.assertEqual(parse[parse.index('--profile') + 1], str(self.locations['profile']))
        self.assertEqual(parse[parse.index('--mlir') + 1], str(mlir))
        self.assertNotIn('--checkpoint', parse)
        with self.assertRaises(ValueError):self.outer.phase_commands(self.locations, work, 'parse')

    def test_preparation_downloads_two_bounded_files_and_installs_once(self):
        work = self.root / 'preparation'; work.mkdir()
        result = dict(attempted=dict.fromkeys(self.outer.CATEGORIES, 0),
                      completed=dict.fromkeys(self.outer.CATEGORIES, 0))
        recorder = self.outer.StageRecorder(work / 'events.jsonl')
        calls = []
        def download(url, destination, seconds, record, checkpoint):
            calls.append((url, seconds))
            destination.write_text(json.dumps(self._metadata())) if len(calls) == 1 else destination.write_bytes(b'x' * 11)
            return dict(bytes=destination.stat().st_size)
        def install(name, command, seconds, category):
            self.assertEqual((name, seconds, category), ('dependency-install', 180, 'dependency_install'))
            result['attempted'][category] += 1
            target = Path(command[command.index('--target') + 1]) / 'einops/__init__.py'
            target.parent.mkdir(parents=True); target.write_text('fixture\n')
            result['completed'][category] += 1
        with patch.object(self.outer, '_download', side_effect=download):
            target = self.outer.prepare_dependencies(self.locations, work, result, recorder, install, lambda:None)
        self.assertEqual(calls, [(self.outer.METADATA_URL, 60), (self._metadata()['urls'][0]['url'], 120)])
        self.assertEqual(target, work / 'python-deps')
        for category in self.outer.CATEGORIES[:3]:
            self.assertEqual((result['attempted'][category], result['completed'][category]), (1, 1))

    def test_dependency_error_stops_before_install_tests_and_model(self):
        def failed(locations, work, result, recorder, launch, checkpoint):
            result['attempted']['metadata_download'] += 1
            raise RuntimeError('fixed metadata error')
        process = self._process()
        result = self._execute(self._args(), process, preparation=failed)
        self.assertEqual(result['status'], 'stopped_on_error')
        self.assertIn('fixed metadata error', result['error'])
        self.assertEqual([row[0] for row in process.calls], ['git-revision', 'git-branch'])
        self.assertEqual(result['attempted']['metadata_download'], 1)
        self.assertEqual(result['completed']['metadata_download'], 0)
        self.assertEqual(result['attempted']['trace'], 0)

    def test_success_records_one_trace_parse_unique_registry_and_exclusive_timing(self):
        args, process = self._args(), self._process()
        result = self._execute(args, process)
        self.assertEqual(result['status'], 'ok', result.get('error'))
        self.assertEqual([row[0] for row in process.calls],
                         ['git-revision', 'git-branch', 'r22-tests', 'model-trace', 'model-parse'])
        self.assertEqual(result['attempted'], dict.fromkeys(self.outer.CATEGORIES, 1))
        self.assertEqual(result['completed'], result['attempted'])
        self.assertEqual((result['git_metadata_attempted'], result['git_metadata_completed']), (2, 2))
        for _, _, options in process.calls:
            self.assertEqual((options['mem_gib'], options['fsize_gib']), (4, 8))
        self.assertTrue(result['orbit_model_trace_passed'])
        self.assertTrue(result['tdag_parse_passed'])
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.FORMAL))
        self.assertEqual((result['metrics']['trace_runs'], result['metrics']['parse_runs']), (1, 1))
        self.assertEqual(result['case_artifact_bytes'], self.outer.case_size(args.out))
        self.assertEqual(result['artifact_bytes'], result['case_artifact_bytes'])
        timing = json.loads((args.out / 'timing.json').read_text())
        self.assertTrue(timing['attribution_ready'])
        self.assertEqual(timing['diagnostics'], [])
        roots = [row for row in timing['spans'] if row['parent_id'] is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(sum(row['exclusive_ns'] for row in timing['spans']), roots[0]['inclusive_ns'])
        rows = self._registry(); self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual((rows[0]['round'], rows[0]['mode']), ('R22', self.outer.MODE))
        self.assertEqual(rows[0]['metrics'], result['metrics'])

    def test_trace_failure_and_supervisor_termination_stop_before_parse(self):
        for label, overrides in [('exit', {}), ('timeout', dict(timed_out=True)),
                                 ('signal', dict(signal=15)), ('remaining', dict(remaining_pids=[98765]))]:
            process = self._process(fail='model-trace', **overrides)
            with self.subTest(label=label):
                result = self._execute(self._args(label), process)
                self.assertEqual(result['status'], 'stopped_on_error')
                self.assertEqual(result['attempted']['trace'], 1)
                self.assertEqual(result['completed']['trace'], 0)
                self.assertEqual(result['attempted']['parse'], 0)
                self.assertEqual(process.calls[-1][0], 'model-trace')

    def test_empty_or_skipped_discovery_stops_before_trace(self):
        for label, changes in [('empty', dict(test_count=0)), ('skip', dict(skip=True))]:
            process = self._process(**changes)
            with self.subTest(label=label):
                result = self._execute(self._args(label), process)
                self.assertEqual(result['status'], 'stopped_on_error')
                self.assertEqual(result['attempted']['trace'], 0)
                self.assertEqual(process.calls[-1][0], 'r22-tests')

    def test_missing_worker_keeps_unobserved_trace_count_null(self):
        result = self._execute(self._args(), self._process(missing_worker=True))
        self.assertEqual(result['status'], 'stopped_on_error')
        self.assertIsNone(result['metrics']['trace_runs'])
        self.assertEqual(result['metrics']['parse_runs'], 0)
        self.assertTrue(all(result['metrics'][name] is None for name in self.outer.FORMAL))

    def test_existing_and_foreign_outputs_are_preserved_before_any_launch(self):
        existing = self.results / 'existing'; existing.mkdir()
        marker = existing / 'saved.txt'; marker.write_text('preserved evidence\n')
        for out in (existing, self.root / 'foreign', self.results / 'nested/child', self.results):
            with self.subTest(out=out), patch.object(self.outer, '_locations', return_value=self.locations), \
                    patch.object(self.outer, 'run_process') as launched, self.assertRaises(ValueError):
                self.outer.execute(SimpleNamespace(out=out))
            launched.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserved evidence\n')
        self.assertEqual(self._registry(), [])

    def test_final_json_is_included_and_terminal_budget_error_updates_registry(self):
        original_write = self.outer.write_json
        args, process = self._args(), self._process()
        def write(path, record):
            original_write(path, record)
            if Path(path).name == 'timing.json':
                (args.out / 'late-terminal.log').write_bytes(b'x' * (1024**2))
        with patch.object(self.outer, 'CASE_BYTES', 1024**2), patch.object(self.outer, 'write_json', side_effect=write):
            result = self._execute(args, process)
        self.assertEqual(result['status'], 'stopped_on_error')
        self.assertTrue(result['terminal_budget_error'])
        self.assertEqual(result['metrics']['status'], result['status'])
        self.assertEqual(result['case_artifact_bytes'], self.outer.case_size(args.out))
        self.assertEqual(json.loads((args.out / 'result.json').read_text()), result)
        self.assertEqual(self._registry()[0]['metrics'], result['metrics'])
        self.assertEqual([row[0] for row in process.calls][-1], 'model-parse')

    def test_child_environment_restores_controlled_paths_and_four_threads(self):
        with patch.dict(os.environ, HECATE='old', HECATE_SDK='old', PYTHONPATH='old',
                        LD_LIBRARY_PATH='old', NIX_PATH='old', MKL_NUM_THREADS='4'):
            env = self.outer._child_environment(self.root / 'work')
        for name in ('HECATE', 'HECATE_SDK', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'NIX_PATH'):
            self.assertNotIn(name, env)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name], '1')
        self.assertEqual(env['R22_DEPS_DIR'], str(self.root / 'work/python-deps'))
        self.assertEqual(env['R22_TEST_WORK'], str(self.root / 'work/test-worker'))

    def test_fixed_budgets_and_external_cli_keep_only_new_output_argument(self):
        self.assertEqual((self.outer.MEMORY_GIB, self.outer.FILE_GIB, self.outer.OVERALL_SECONDS,
                          self.outer.CASE_BYTES, self.outer.FREE_BYTES),
                         (4, 8, 3600, 4 * 1024**3, 10 * 1024**3))
        with patch.object(self.outer, 'execute', return_value=dict(status='ok')) as launch:
            self.assertEqual(self.outer.main(['--out', str(self.results / 'cli')]), 0)
        self.assertEqual(vars(launch.call_args.args[0]), dict(out=self.results / 'cli'))


if __name__ == '__main__':
    unittest.main()
