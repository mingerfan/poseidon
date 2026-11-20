"""R16 standard-library protocol checks and two dedicated environment wrappers."""

from pathlib import Path
import os
import re
import resource
import subprocess
import sys
import unittest

from pt_agent import paths
from pt_agent.tests._artifacts import require


ORION_PYTHON = paths.ORION_ROOT.parent / 'orion-venv/bin/python'
ORION_ARTIFACT = ORION_PYTHON.parent if ORION_PYTHON.is_symlink() else ORION_PYTHON
INTAKE_ARTIFACT = paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink() else paths.INTAKE_PYTHON


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


def _run_inner(case, interpreter, class_name):
    from pt_agent.tests import _r16_baseline_validation_cases as checks

    test_class = getattr(checks, class_name)
    expected = unittest.defaultTestLoader.loadTestsFromTestCase(test_class).countTestCases()
    case.assertGreater(expected, 0)
    test_name = 'pt_agent.tests._r16_baseline_validation_cases.' + class_name
    if class_name == 'OrionCases':
        from pt_agent.experiments.r16_baseline_validation_preflight import _native_orion_env
        bootstrap = ("import runpy, signal, sys; signal.alarm(180); "
                     "sys.argv = ['unittest', *sys.argv[1:]]; "
                     "runpy.run_module('unittest', run_name='__main__')")
        command = [str(interpreter), '-B', '-c', bootstrap, test_name, '-v']
        environment = _native_orion_env()
    else:
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(interpreter), '--', '-m', 'unittest', test_name, '-v']
        environment = None
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=300, preexec_fn=limited, env=environment)
    print(completed.stdout, end='', flush=True)
    case.assertEqual(completed.returncode, 0, completed.stdout)
    count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
    case.assertIsNotNone(count, completed.stdout)
    case.assertEqual(int(count.group(1)), expected, completed.stdout)
    case.assertNotRegex(completed.stdout, r'skipped=\d+')
    print('R16_' + class_name.upper() + '_TESTS=' + count.group(1), flush=True)
    print('R16_REAL_MODELS_WEIGHTS_AND_DATA_IN_UNIT_TESTS=0', flush=True)


@require(ORION_ARTIFACT, paths.ORION_ROOT / 'orion')
class OrionEnvironmentChecks(unittest.TestCase):
    def test_current_round_orion_checks_run_once_without_skips(self):
        _run_inner(self, ORION_PYTHON, 'OrionCases')


@require(INTAKE_ARTIFACT)
class OrbitEnvironmentChecks(unittest.TestCase):
    def test_current_round_orbit_checks_run_once_without_skips(self):
        _run_inner(self, paths.INTAKE_PYTHON, 'OrbitCases')


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from pt_agent.experiments import r16_baseline_validation_preflight as outer

        self.json, self.Namespace, self.patch, self.outer = json, SimpleNamespace, patch, outer
        temporary = tempfile.TemporaryDirectory(prefix='r16-outer-cases-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.orion = self.root / 'baselines/orion'
        self.orbit = self.root / 'baselines/orbit'
        self.orion.mkdir(parents=True)
        self.orbit.mkdir()
        self.interpreter = self.root / 'intake/bin/python'
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.touch()
        orion_python = self.orion.parent / 'orion-venv/bin/python'
        orion_python.parent.mkdir(parents=True)
        orion_python.touch()
        self.factories = {}
        for name, short, orbit_source in outer.RUNS:
            model = 'pt_agent.experiments.r13_common_models:' + short
            source = self.root / 'sources' / short
            source.mkdir(parents=True)
            original = dict(source=source/'model.py', checkpoint=source/'weights.pt',
                            inference_preprocessing_source=source/'preprocessing.py')
            for path in original.values():
                path.touch()
            self.factories[model] = original
            orbit_definition = self.orbit / 'frontend/dacapo/python/poly/poly/models' / orbit_source
            orbit_definition.parent.mkdir(parents=True, exist_ok=True)
            orbit_definition.touch()
            run = self.results / name
            info = run / 'worker/model_info'
            (info / 'graph').mkdir(parents=True)
            (info / 'model_info.json').write_text('{}')
            (info / 'graph/graph.json').write_text('{}')
            probes, provenance = run/'worker/training_probes.npz', run/'worker/training_probe_source.json'
            probes.touch()
            provenance.write_text('{}')
            worker = dict(round='R13', status='ok', model=model,
                          import_completed=True, real_probes_passed=True, graph_math_passed=True,
                          roundtrip_passed=True, static_coverage=True,
                          artifacts=dict(model_info=str(info), training_probes=str(probes),
                                         training_probe_source=str(provenance)))
            self._write(run/'result.json', dict(round='R13', status='ok', model=model,
                                               mode='common-model-intake-preparation', worker=worker,
                                               timing=dict(saved_prior_ns=123)))
        for obj, key, value in ((outer.paths, 'RESULTS_DIR', self.results),
                                (outer.paths, 'ORION_ROOT', self.orion),
                                (outer.paths, 'ORBIT_ROOT', self.orbit),
                                (outer.paths, 'INTAKE_PYTHON', self.interpreter),
                                (outer, 'FACTORIES', self.factories)):
            context = patch.object(obj, key, value)
            context.start()
            self.addCleanup(context.stop)

    def _write(self, path, value):
        path.write_text(self.json.dumps(value))

    def _worker(self, baseline='orion', short='resnet20_silu'):
        outer = self.outer
        worker = dict(round='R16', baseline=baseline,
                      model='pt_agent.experiments.r13_common_models:' + short,
                      status='ok', compatibility_passed=True, all_finite=True, candidate_frozen=False)
        worker.update(dict.fromkeys(outer.STAGE_FIELDS, True))
        worker.update(dict.fromkeys(outer.ZERO_FIELDS, 0))
        worker.update(dict.fromkeys(outer.UNMEASURED_FIELDS))
        for name, count in (('forward', 3), ('checkpoint_load', 1), ('probe_load', 1), ('model_build', 1)):
            for suffix in ('attempts', 'completed'):
                worker[name+'_'+suffix] = count
        return worker

    def _args(self, baseline='orion', short=None):
        name = self.outer.RUNS[0][0]
        return self.Namespace(baseline=baseline, r13_run=self.results/name,
                              out=self.results/'queue'/(baseline+'-case'))

    def _process(self, worker, **updates):
        def run(argv, **kwargs):
            self.assertEqual(kwargs['timeout'], 360)
            self.assertEqual(kwargs['mem_gib'], 4)
            self.assertEqual(kwargs['fsize_gib'], 8)
            baseline = argv[argv.index('--baseline')+1]
            if baseline == 'orion':
                self.assertEqual(argv[:2], [str(self.orion.parent/'orion-venv/bin/python'), '-B'])
                self.assertNotIn('pt_agent.tools.nixrun', argv)
                self.assertNotIn('--seconds', argv)
            else:
                self.assertIn('pt_agent.tools.nixrun', argv)
                self.assertEqual(argv[argv.index('--seconds')+1], '300')
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                self.assertEqual(kwargs['env'][key], '1')
            work = Path(argv[argv.index('--work-dir')+1])
            if worker is not None:
                self._write(work/'worker_result.json', worker)
            kwargs['log'].write_text('fake child; no model or checkpoint was run\n')
            observed = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[])
            observed.update(updates)
            return observed
        return run

    def test_all_six_sources_resolve_original_files_and_existing_interpreters(self):
        for name, short, source_name in self.outer.RUNS:
            for baseline in ('orion', 'orbit'):
                with self.subTest(model=short, baseline=baseline):
                    previous, selected = self.outer.resolve_source(baseline, self.results/name)
                    self.assertEqual(previous['model'], 'pt_agent.experiments.r13_common_models:'+short)
                    self.assertEqual(selected['model_info'].name, 'model_info.json')
                    self.assertEqual(selected['graph_record'], selected['model_info'].parent/'graph/graph.json')
                    expected = self.factories[previous['model']]['source'] if baseline=='orion' else (
                        self.orbit/'frontend/dacapo/python/poly/poly/models'/source_name)
                    self.assertEqual(selected['model_source'], expected)
                    self.assertEqual(selected['checkpoint'], self.factories[previous['model']]['checkpoint'])
                    self.assertEqual(selected['interpreter'], self.orion.parent/'orion-venv/bin/python'
                                     if baseline=='orion' else self.interpreter)

    def test_failed_source_or_artifact_outside_original_run_is_rejected(self):
        args = self._args()
        original = self.json.loads((args.r13_run/'result.json').read_text())
        for changed in ('status', 'artifact'):
            record = self.json.loads(self.json.dumps(original))
            if changed=='status':
                record['worker']['real_probes_passed'] = False
            else:
                record['worker']['artifacts']['training_probes'] = str(self.root/'outside.npz')
            self._write(args.r13_run/'result.json', record)
            with self.assertRaises(self.outer.PhaseStopped):
                self.outer.resolve_source('orion', args.r13_run)

    def test_execute_one_worker_and_one_eight_field_registry_with_self_counted_json(self):
        args, worker = self._args(), self._worker()
        with self.patch.object(self.outer, 'run_process', side_effect=self._process(worker)) as run, \
             self.patch.object(self.outer, '_git_state', return_value={'branch':'fixture'}):
            result = self.outer.execute(args)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['metrics']['forward_completed'], 3)
        self.assertFalse(result['source_r13']['included_in_current_turnaround'])
        self.assertEqual(result['source_r13']['prior_timing'], {'saved_prior_ns':123})
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertEqual(result['queue_artifact_bytes'], self.outer._case_size(args.out.parent)['bytes'])
        rows = [self.json.loads(line) for line in (self.results/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(rows[0]['metrics'], result['metrics'])
        self.assertTrue(result['timing']['attribution_ready'])
        roots = [span for span in result['timing']['spans'] if span['parent_id'] is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(sum(span['exclusive_ns'] for span in result['timing']['spans']), roots[0]['inclusive_ns'])

    def test_missing_worker_keeps_unobserved_counts_null(self):
        args = self._args()
        with self.patch.object(self.outer, 'run_process', side_effect=self._process(None)), \
             self.patch.object(self.outer, '_git_state', return_value={}):
            result = self.outer.execute(args)
        self.assertEqual(result['status'], 'result_missing')
        for field in self.outer.STAGE_FIELDS+self.outer.ZERO_FIELDS+('forward_attempts','forward_completed'):
            self.assertIsNone(result['metrics'][field], field)
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.UNMEASURED_FIELDS))

    def test_partial_failure_preserves_actual_calls_and_timeout_stops(self):
        for variant in ('partial', 'timeout'):
            args = self._args()
            args.out = args.out.with_name(variant)
            worker = self._worker()
            worker.update(status='math_failed', error='saved first probe mismatch',
                          forward_attempts=1, forward_completed=1, math_probes_passed=False,
                          compatibility_passed=False)
            extra = {} if variant=='partial' else {'timed_out':True}
            with self.patch.object(self.outer, 'run_process', side_effect=self._process(worker, **extra)), \
                 self.patch.object(self.outer, '_git_state', return_value={}):
                result = self.outer.execute(args)
            self.assertEqual(result['status'], 'math_failed' if variant=='partial' else 'timeout')
            self.assertEqual(result['metrics']['forward_attempts'], 1)
            self.assertEqual(result['metrics']['forward_completed'], 1)

    def test_success_requires_all_counts_and_forbidden_actions_zero(self):
        for name, value in (('forward_completed',2), ('checkpoint_load_attempts',2),
                            ('solver_runs',1), ('trace_runs',None), ('keygen_runs',True)):
            worker = self._worker()
            worker[name] = value
            with self.assertRaises(self.outer.PhaseStopped):
                self.outer._check_success(worker, worker['model'], 'orion')

    def test_existing_and_source_containing_output_directories_are_protected(self):
        existing = self.results/'existing'
        existing.mkdir()
        for output in (existing, self.results, self.results/self.outer.RUNS[0][0]/'new-output',
                       self.root/'outside-results'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.outer._check_out(output)
        sentinel = self.results/self.outer.RUNS[0][0]/'worker/training_probes.npz'
        self.assertTrue(sentinel.exists())

    def test_disk_or_missing_dependency_prevents_worker_and_keeps_unknown_counts(self):
        for variant in ('disk','dependency'):
            args = self._args()
            args.out = args.out.with_name(variant)
            if variant=='dependency':
                (args.r13_run/'worker/training_probes.npz').unlink()
            with self.patch.object(self.outer, 'run_process') as run, \
                 self.patch.object(self.outer, '_git_state', return_value={}), \
                 self.patch.object(self.outer.shutil, 'disk_usage', return_value=self.Namespace(free=0)):
                result = self.outer.execute(args)
            self.assertEqual(result['status'], 'dependency_missing')
            self.assertEqual(run.call_count, 0)
            self.assertIsNone(result['metrics']['forward_attempts'])

    def test_case_and_queue_terminal_budget_include_final_json(self):
        args = self._args()
        args.out.mkdir(parents=True)
        (args.out/'saved.log').write_text('retained prior evidence')
        result = self.outer.finalize_result(worker=None, processes=[], timing={'spans':[]},
            status='ok', error=None, git_state={}, out=args.out)
        with self.patch.object(self.outer, 'CASE_BUDGET_BYTES', 64):
            self.outer._write_terminal_result(args.out, result, [])
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertEqual(result['metrics']['status'], 'artifact_budget_exceeded')
        self.assertEqual(result['case_artifact_bytes'], self.outer._case_size(args.out)['bytes'])
        self.assertFalse(result['metrics']['case_artifact_budget_passed'])
        self.assertFalse(result['metrics']['queue_artifact_budget_passed'])

    def test_queue_fixed_six_order_and_first_failure_retains_not_started(self):
        for failure in (None, 1, 3):
            calls=[]
            def execute(args):
                calls.append((args.baseline, args.r13_run.name))
                failed = len(calls)==failure
                return {'status':'math_failed' if failed else 'ok', 'error':'retained mismatch' if failed else None}
            output = self.results/('queue-'+str(failure))
            with self.patch.object(self.outer, 'execute', side_effect=execute):
                result = self.outer.run_queue(self.Namespace(out=output))
            expected=[(baseline,name) for name,_,_ in self.outer.RUNS for baseline in ('orion','orbit')]
            attempted=6 if failure is None else failure
            self.assertEqual(calls, expected[:attempted])
            self.assertEqual(result['attempted_cases'], attempted)
            self.assertEqual(result['completed_cases'], 6 if failure is None else failure-1)
            self.assertEqual(result['status'], 'ok' if failure is None else 'stopped')
            self.assertTrue(all(item['status']=='not_started' for item in result['cases'][attempted:]))
            self.assertEqual(result['artifact_bytes'], self.outer._case_size(output)['bytes'])
            self.assertFalse((self.results/'registry.jsonl').exists())

    def test_native_environment_and_both_fixed_launches_preserve_parent_and_scope(self):
        cleared = ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG',
                   'NIX_PATH', '__NIX_SETUP', 'HECATE_PYTHON_LIBRARY_PATH', 'IN_NIX_SHELL',
                   'PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE',
                   'PYTHONPLATLIBDIR', 'PYTHONSAFEPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                   'TORCH_USE_RTLD_GLOBAL', 'TORCH_COMPILE_STROBELIGHT']
        original = {key: 'fixture inherited override' for key in cleared}
        original.update(KEEP_R16A='retained fixture', PT_AGENT_RESULTS_ROOT=str(self.results))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            original[key] = '7'
        before = dict(original)
        cleaned = self.outer._native_orion_env(original)
        self.assertEqual(original, before)
        self.assertIsNot(cleaned, original)
        for key in cleared:
            if key != 'PYTHONPATH':
                self.assertNotIn(key, cleaned)
        controlled = [str(self.orion), str(self.outer.paths.REPO/'DSL/DSL_agent/scripts')]
        self.assertEqual(cleaned['PYTHONPATH'].split(os.pathsep), controlled)
        self.assertEqual(cleaned['KEEP_R16A'], 'retained fixture')
        self.assertEqual(cleaned['PT_AGENT_RESULTS_ROOT'], str(self.results))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE'):
            self.assertEqual(cleaned[key], '1')
        for baseline in ('orion', 'orbit'):
            previous, selected = self.outer.resolve_source(baseline, self.results/self.outer.RUNS[0][0])
            with self.patch.dict(os.environ, original, clear=True):
                argv, environment, metadata = self.outer._launch(selected, self.results/'unused-worker',
                                                                baseline, previous['model'])
                self.assertEqual(dict(os.environ), original)
            self.assertEqual(metadata['supplemental_spec'], 'R16a')
            self.assertEqual(argv[argv.index('--baseline')+1], baseline)
            if baseline == 'orion':
                self.assertEqual(argv[:2], [str(selected['interpreter']), '-B'])
                self.assertNotIn('pt_agent.tools.nixrun', argv)
                self.assertEqual(environment, cleaned)
                self.assertEqual(metadata['launch_mode'], 'orion-native-existing-venv')
                self.assertEqual(set(metadata['removed_environment_keys']), set(cleared))
                self.assertEqual(metadata['controlled_pythonpath'], controlled)
            else:
                self.assertIn('pt_agent.tools.nixrun', argv)
                self.assertEqual(argv[argv.index('--seconds')+1], '300')
                self.assertEqual(metadata['launch_mode'], 'orbit-existing-nix')
                self.assertEqual(environment['KEEP_R16A'], 'retained fixture')

    def test_production_limits_and_only_external_output_argument(self):
        self.assertEqual((self.outer.INTERNAL_SECONDS,self.outer.OUTER_SECONDS,self.outer.MEMORY_GIB), (300,360,4))
        self.assertEqual(self.outer.CASE_BUDGET_BYTES, 4*1024**3)
        self.assertEqual(self.outer.MODE, 'baseline-plaintext-compatibility')
        with self.patch.object(self.outer, 'run_queue', return_value=dict(status='ok',attempted_cases=6,completed_cases=6)) as run:
            self.assertEqual(self.outer.main(['--out',str(self.results/'cli')]), 0)
        self.assertEqual(vars(run.call_args.args[0]), {'out':self.results/'cli'})
