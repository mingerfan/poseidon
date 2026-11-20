"""R13 environment wrapper and outer failure evidence; no full-model inference."""

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


INTAKE_ARTIFACT = (paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink()
                   else paths.INTAKE_PYTHON)


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


@require(INTAKE_ARTIFACT)
class CommonEnvironmentChecks(unittest.TestCase):
    def test_only_current_round_inner_checks_run_without_skips(self):
        from pt_agent.tests import _r13_common_cases as cases
        expected = unittest.defaultTestLoader.loadTestsFromModule(cases).countTestCases()
        self.assertGreater(expected, 0)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(paths.INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r13_common_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), expected, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R13_INNER_TESTS=' + count.group(1), flush=True)
        print('R13_FULL_MODEL_IMPORTS_IN_UNIT_TESTS=0', flush=True)


class CommonResultChecks(unittest.TestCase):
    def finalize(self, worker=None, timing=None):
        from pt_agent.experiments.r13_common_preflight import finalize_result
        return finalize_result(model='fixture:factory', worker=worker, processes=[],
                               timing=timing or {}, status='error', error='fixture failure',
                               git_state=dict(branch='fixture'), out=Path('/fixture/r13-run'))

    def test_unstarted_results_do_not_invent_compilation_or_encrypted_metrics(self):
        result = self.finalize()
        self.assertEqual(result['round'], 'R13')
        metrics = result['metrics']
        self.assertEqual(metrics['llm_calls'], 0)
        self.assertEqual(metrics['real_ckks_attempts'], 0)
        for key in ('lowering_ready', 'compiled_bootstraps', 'predicted_latency', 'CKKS_latency'):
            with self.subTest(key=key):
                self.assertIsNone(metrics[key])
        for key in ('import_completed', 'real_probes_passed', 'graph_math_passed', 'roundtrip_passed'):
            with self.subTest(key=key):
                self.assertIn(metrics[key], (None, False))

    def test_completed_semantic_observation_keeps_static_coverage_separate(self):
        coverage = True
        record = dict(status='ok', import_completed=True, real_probes_passed=True,
                      graph_math_passed=True, roundtrip_passed=True, static_coverage=coverage)
        result = self.finalize(record)
        self.assertEqual(result['metrics']['static_coverage'], coverage)
        for key in ('import_completed', 'real_probes_passed', 'graph_math_passed', 'roundtrip_passed'):
            self.assertIs(result['metrics'][key], True)
        for key in ('lowering_ready', 'compiled_bootstraps', 'predicted_latency', 'CKKS_latency'):
            self.assertIsNone(result['metrics'][key])

    def test_failure_result_and_eight_field_registry_are_retained_without_worker(self):
        from pt_agent.experiments import r13_common_preflight as outer
        from pt_agent.experiments.r13_common_models import FACTORIES
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(model=next(iter(FACTORIES)), inputs=root / 'missing-records',
                                   out=root / 'run')
            with patch.object(outer.paths, 'RESULTS_DIR', root), \
                 patch.object(outer, '_git_state', return_value=dict(branch='fixture')), \
                 patch.object(outer, 'run_process') as launch:
                result = outer.execute(args)
            launch.assert_not_called()
            saved = json.loads((args.out / 'result.json').read_text())
            rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertNotEqual(saved['status'], 'ok')
            self.assertEqual(result['metrics'], saved['metrics'])
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                           'metrics', 'artifacts_dir', 'ts'})
            self.assertEqual(rows[0]['round'], 'R13')
            self.assertEqual(rows[0]['mode'], 'common-model-intake-preparation')
            self.assertEqual(rows[0]['metrics'], saved['metrics'])
            self.assertEqual(rows[0]['artifacts_dir'], str(args.out))

    def test_cli_requires_original_model_and_inputs(self):
        from pt_agent.experiments import r13_common_preflight as outer
        for argv in (['--out', '/fixture/run'],
                     ['--model', 'fixture:factory', '--out', '/fixture/run'],
                     ['--inputs', '/fixture/records', '--out', '/fixture/run']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 patch.object(outer, 'execute') as launch:
                with self.assertRaises(SystemExit) as caught:
                    outer.main(argv)
                self.assertEqual(caught.exception.code, 2)
                launch.assert_not_called()

    def test_cli_passes_only_original_entry_and_input_location(self):
        from pt_agent.experiments import r13_common_preflight as outer
        from pt_agent.experiments.r13_common_models import FACTORIES
        source = next(iter(FACTORIES))
        with patch.object(outer, 'execute', return_value=dict(status='ok', phase='complete')) as execute, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(outer.main(['--model', source, '--inputs', '/fixture/records',
                                         '--out', '/fixture/run']), 0)
        args = execute.call_args.args[0]
        self.assertEqual(args.model, source)
        self.assertEqual(args.inputs, Path('/fixture/records'))
        self.assertEqual(args.out, Path('/fixture/run'))
        self.assertFalse(hasattr(args, 'shape'))
        self.assertFalse(hasattr(args, 'normalization'))

    def test_timing_attribution_remains_exclusive(self):
        from pt_agent.experiments.r10_paper_metrics import summarize_events
        def event(kind, name, stage, start, parent=None, end=None):
            record = dict(type=kind, event_id=name, stage_id=stage, parent_id=parent,
                          start_ns=start, pid=1, ppid=0)
            if end is not None:
                record.update(end_ns=end, status='ok')
            return record
        timing = summarize_events([event('begin', 'a', 'turnaround', 0),
            event('begin', 'b', 'import_model', 10, 'a'),
            event('end', 'b', 'import_model', 10, 'a', 60),
            event('end', 'a', 'turnaround', 0, end=100)])
        self.assertEqual(sum(s['exclusive_ns'] for s in timing['spans']), 100)
        result = self.finalize(timing=timing)
        self.assertEqual(result['timing'], timing)
