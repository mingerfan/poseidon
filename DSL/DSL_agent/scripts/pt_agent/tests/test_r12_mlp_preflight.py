"""R12 bounded arithmetic wrapper and outer evidence checks; no CKKS runs."""

from contextlib import redirect_stderr, redirect_stdout
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
from pt_agent.tests._r12_mlp_cases import MATH_CASE_COUNT


INTAKE_ARTIFACT = (paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink()
                   else paths.INTAKE_PYTHON)
OFFICIAL_MODEL = paths.DACAPO_UPSTREAM / 'examples/data/mlp.model'
OFFICIAL_SOURCE = paths.DACAPO_UPSTREAM / 'examples/benchmarks/MLP.py'


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


@require(INTAKE_ARTIFACT, OFFICIAL_MODEL, OFFICIAL_SOURCE)
class MlpMathEnvironmentChecks(unittest.TestCase):
    def test_real_intake_environment_math_only(self):
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
                   '--seconds', '180', '--python', str(paths.INTAKE_PYTHON), '--',
                   '-m', 'unittest', 'pt_agent.tests._r12_mlp_cases.MathCases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), MATH_CASE_COUNT, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R12_MATH_INNER_TESTS=' + count.group(1), flush=True)
        print('R12_CPP_GATE=pending_build_and_real_check_only', flush=True)


class MlpResultChecks(unittest.TestCase):
    def finalize(self, *, worker=None, processes=None, timing=None, status='failed'):
        from pt_agent.experiments.r12_mlp_preflight import finalize_result
        return finalize_result(phase='execute', worker=worker, processes=processes or [],
                               timing=timing or {}, status=status, error='fixture failure',
                               git_state=dict(branch='fixture', working_tree_changed=True),
                               out=Path('/fixture/r12-test'))

    def test_unstarted_measurements_are_unknown_and_skipped(self):
        result = self.finalize()
        self.assertEqual(result['round'], 'R12')
        metrics = result['metrics']
        self.assertEqual(metrics['llm_calls'], 0)
        self.assertEqual(metrics['real_ckks_attempts'], 0)
        self.assertEqual(metrics['real_ckks_completed'], 0)
        self.assertIs(metrics['execution_skipped'], True)
        for key in ('ckks_tolerance_passed', 'compiled_bootstraps',
                    'calibrated_predictions', 'compiler_profile_prediction'):
            with self.subTest(key=key):
                self.assertIsNone(metrics[key])

    def test_interrupted_execution_does_not_fabricate_zero_counts(self):
        result = self.finalize(worker={'execute': {'status': 'result_missing'}},
                               processes=[dict(label='execute', exit_code=-9)])
        self.assertIs(result['metrics']['execution_skipped'], False)
        self.assertIsNone(result['metrics']['real_ckks_attempts'])
        self.assertIsNone(result['metrics']['real_ckks_completed'])
        self.assertIsNone(result['metrics']['ckks_tolerance_passed'])

    def test_observed_counts_and_prediction_completeness_are_preserved(self):
        predictions = [dict(sample_index=1, prediction_complete=True, missing_keys=[],
                            predicted_compute_ns=250, measured_compute_ns=200,
                            relative_abs_error=.25, runtime_run_ns=310),
                       dict(sample_index=2, prediction_complete=False,
                            missing_keys=[dict(op='rotate', rotation_step=7)],
                            predicted_compute_ns=None, measured_compute_ns=220,
                            relative_abs_error=None, runtime_run_ns=340)]
        worker = dict(compile=dict(plan_stats=dict(boots=0),
                                   compiler_profile_prediction=dict(time_us=123)),
                      execute=dict(client_result=dict(encrypted_execution_attempts=3,
                                                      encrypted_execution_completed=3,
                                                      ckks_tolerance_passed=False,
                                                      predictions=predictions)))
        result = self.finalize(worker=worker, status='failed')
        metrics = result['metrics']
        self.assertEqual(metrics['real_ckks_attempts'], 3)
        self.assertEqual(metrics['real_ckks_completed'], 3)
        self.assertIs(metrics['ckks_tolerance_passed'], False)
        self.assertEqual(metrics['compiled_bootstraps'], 0)
        self.assertEqual(metrics['calibrated_predictions'], predictions)
        self.assertIsNone(metrics['calibrated_predictions'][1]['predicted_compute_ns'])
        self.assertEqual(metrics['compiler_profile_prediction'], dict(time_us=123))

    def test_parent_and_child_timing_are_not_added_twice(self):
        from pt_agent.experiments.r10_paper_metrics import summarize_events

        def event(kind, identifier, stage, start, parent=None, end=None):
            value = dict(type=kind, event_id=identifier, stage_id=stage,
                         parent_id=parent, start_ns=start, pid=17, ppid=1)
            if end is not None:
                value.update(end_ns=end, status='ok')
            return value

        events = [event('begin', 'run', 'execute', 0),
                  event('begin', 'runtime', 'runtime.run', 10, 'run'),
                  event('begin', 'compute', 'compute', 20, 'runtime'),
                  event('end', 'compute', 'compute', 20, 'runtime', 60),
                  event('end', 'runtime', 'runtime.run', 10, 'run', 80),
                  event('end', 'run', 'execute', 0, end=100)]
        timing = summarize_events(events)
        self.assertEqual(timing['status'], 'ok')
        self.assertIs(timing['attribution_ready'], True)
        by_stage = {span['stage_id']: span for span in timing['spans']}
        self.assertEqual(by_stage['runtime.run']['inclusive_ns'], 70)
        self.assertEqual(by_stage['runtime.run']['exclusive_ns'], 30)
        self.assertEqual(sum(span['exclusive_ns'] for span in timing['spans']), 100)
        result = self.finalize(timing=timing)
        self.assertEqual(result['metrics']['phase_elapsed_ns'], {'execute': 100})

    def test_preflight_failure_writes_r12_result_and_registry_without_launch(self):
        from pt_agent.experiments import r12_mlp_preflight as preflight
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(model=OFFICIAL_MODEL, mnist=root / 'fixture-images.gz',
                                   out=root / 'run')
            with patch.object(preflight.paths, 'RESULTS_DIR', root), \
                 patch.object(preflight, '_git_state', return_value=dict(branch='fixture')), \
                 patch.object(preflight.Path, 'exists',
                              new=lambda path: path != args.out), \
                 patch.object(preflight.shutil, 'disk_usage',
                              return_value=SimpleNamespace(free=9 * 1024**3)), \
                 patch.object(preflight, 'run_process') as launch:
                result = preflight.execute(args)
            launch.assert_not_called()
            saved = json.loads((args.out / 'result.json').read_text())
            records = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(result['status'], 'dependency_missing')
            self.assertEqual(result['round'], 'R12')
            self.assertEqual(saved['metrics'], result['metrics'])
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(set(record), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                           'metrics', 'artifacts_dir', 'ts'})
            self.assertEqual(record['round'], saved['round'])
            self.assertEqual(record['metrics'], saved['metrics'])
            self.assertEqual(record['artifacts_dir'], str(args.out))

    def test_cli_requires_model_and_mnist_without_launch(self):
        from pt_agent.experiments import r12_mlp_preflight as preflight
        for arguments in (['--out', '/fixture/run'],
                          ['--model', str(OFFICIAL_MODEL), '--out', '/fixture/run'],
                          ['--mnist', '/fixture/train-images.gz', '--out', '/fixture/run']):
            with self.subTest(arguments=arguments), \
                 patch.object(sys, 'argv', ['r12', *arguments]), \
                 patch.object(preflight, 'execute') as launch, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    preflight.main()
                self.assertEqual(caught.exception.code, 2)
                launch.assert_not_called()

    def test_default_and_eva_mode_are_recorded_in_final_result(self):
        from pt_agent.experiments.r12_mlp_preflight import finalize_result
        values = dict(phase='compile', worker={}, processes=[], timing={}, status='failed',
                      error='fixture', git_state={}, out=Path('/fixture/run'))
        for extra, expected in (({}, 'dacapo-optimized'), ({'compile_mode': 'eva'}, 'eva')):
            with self.subTest(expected=expected):
                result = finalize_result(**values, **extra)
                self.assertEqual(result['round'], 'R12')
                self.assertEqual(result['spec_revision'], 'R12b')
                self.assertEqual(result['compile_mode'], expected)
                self.assertEqual(result['metrics']['compile_mode'], expected)
                self.assertIsNone(result['metrics']['calibrated_predictions'])

    def test_mode_in_outer_cli_result_and_registry(self):
        from pt_agent.experiments import r12_mlp_preflight as preflight
        for flags, expected in (([], 'dacapo-optimized'), (['--compile-mode', 'eva'], 'eva')):
            with self.subTest(expected=expected), TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['r12', '--model', str(OFFICIAL_MODEL), '--mnist', str(root / 'images.gz'),
                        '--out', str(root / 'run'), *flags]
                with patch.object(sys, 'argv', argv), \
                     patch.object(preflight, 'execute', return_value=dict(status='ok', phase='preflight')) as execute, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(preflight.main(), 0)
                args = execute.call_args.args[0]
                self.assertIsInstance(args.compile_mode, str)
                self.assertEqual(args.compile_mode, expected)
                with patch.object(preflight.paths, 'RESULTS_DIR', root), \
                     patch.object(preflight, '_git_state', return_value=dict(branch='fixture')), \
                     patch.object(preflight.Path, 'exists', new=lambda path: path != args.out), \
                     patch.object(preflight.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 1024**3)), \
                     patch.object(preflight, 'run_process') as launch:
                    result = preflight.execute(args)
                launch.assert_not_called()
                saved = json.loads((args.out / 'result.json').read_text())
                records = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
                self.assertEqual(result['status'], 'dependency_missing')
                self.assertEqual(saved['compile_mode'], expected)
                self.assertEqual(saved['spec_revision'], 'R12b')
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(set(record), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                               'metrics', 'artifacts_dir', 'ts'})
                self.assertEqual(record['round'], 'R12')
                self.assertEqual(record['metrics']['compile_mode'], expected)
                suffix = '-eva' if expected == 'eva' else ''
                self.assertEqual(record['mode'], 'official-artifact-mlp-native-ckks' + suffix)




if __name__ == '__main__':
    unittest.main()
