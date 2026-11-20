"""Fresh BERT/SST-2 outer protocol and a single synthetic intake suite."""

import ast
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


INTAKE = Path(os.environ.get('BERT_SST2_INTAKE_PYTHON', str(paths.INTAKE_PYTHON)))
DATA_PYTHON = Path(os.environ.get('BERT_SST2_DATA_PYTHON', str(paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent / 'paper-data-venv/bin/python')))
BOOTSTRAP = r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
work = Path(sys.argv[1]).resolve()
os.environ['BERT_SST2_TEST_WORK'] = str(work)
os.environ['BERT_SST2_DATA_PYTHON'] = sys.argv[2]
os.environ['TMPDIR'] = str(work)
tempfile.tempdir = str(work)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
signal.alarm(180)
suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._bert_sst2_cases.SST2Cases')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = dict(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skips=len(result.skipped), success=result.wasSuccessful())
(work / 'inner-unit-result.json').write_text(json.dumps(record, allow_nan=False) + '\n')
raise SystemExit(0 if record['success'] and record['testsRun'] > 0 and record['skips'] == 0 else 1)
'''


@require(INTAKE.parent if INTAKE.is_symlink() else INTAKE,
         DATA_PYTHON.parent if DATA_PYTHON.is_symlink() else DATA_PYTHON)
class IntakeEnvironmentChecks(unittest.TestCase):
    def test_only_new_synthetic_sst2_cases_run_once(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        tree = ast.parse(Path(__file__).with_name('_bert_sst2_cases.py').read_text())
        count = sum(isinstance(case, ast.FunctionDef) and case.name.startswith('test_')
                    for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SST2Cases'
                    for case in node.body)
        if os.environ.get('BERT_SST2_TEST_WORK'):
            work = Path(os.environ['BERT_SST2_TEST_WORK']) / 'intake-inner'
            work.mkdir(parents=True)
        else:
            temporary = TemporaryDirectory(prefix='bert-sst2-inner-')
            self.addCleanup(temporary.cleanup)
            work = Path(temporary.name)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '240', '--python', str(INTAKE),
                   '--', '-B', '-c', BOOTSTRAP, str(work), str(DATA_PYTHON)]
        observation = run_process(command, cwd=Path(__file__).resolve().parents[2], log=work / 'inner.log',
            resources=work / 'resources.jsonl', timeout=300, mem_gib=2, fsize_gib=8,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        output = (work / 'inner.log').read_text()
        print(output, end='', flush=True)
        self.assertEqual(observation['exit_code'], 0, output)
        self.assertFalse(observation['timed_out'])
        self.assertIsNone(observation['signal'])
        self.assertEqual(observation['remaining_pids'], [])
        self.assertEqual(json.loads((work / 'inner-unit-result.json').read_text()),
                         dict(testsRun=count, failures=0, errors=0, skips=0, success=True))
        self.assertGreater(count, 0)
        (work / 'inner-wrapper-result.json').write_text(json.dumps(dict(status='ok', wrapper_runs=1,
            inner_tests=count, inner_skips=0, observation=observation), allow_nan=False) + '\n')
        print('BERT_SST2_INNER_TESTS=' + str(count), flush=True)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import bert_sst2
        self.outer = bert_sst2
        temporary = TemporaryDirectory(prefix='bert-sst2-outer-', dir=os.environ.get('BERT_SST2_TEST_WORK'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.model, self.inputs = self.root / 'model', self.root / 'sst2'
        self.model.mkdir()
        (self.inputs / 'data').mkdir(parents=True)
        (self.model / 'config.json').write_text(json.dumps(dict(model_type='bert', architectures=['BertForSequenceClassification'])))
        for name in ('train', 'validation'):
            (self.inputs / 'data' / (name + '-00000-of-00001.parquet')).write_text('metadata fixture, not loaded')
        self.interpreter = self.root / 'intake-python'
        self.interpreter.write_text('metadata fixture')
        self.args = SimpleNamespace(model=self.model, inputs=self.inputs, out=self.results / 'fresh',
            intake_python=self.interpreter, ledger=self.results / 'llm_calls.jsonl',
            allowance=self.results / 'allowance.json', mem_gib=4, phase='prepare', data_python=self.interpreter)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.outer.paths, 'RESULTS_DIR', self.results))

    def worker(self, phase='prepare'):
        from pt_agent.experiments.bert_sst2_worker import _initial_result
        worker = _initial_result(phase)
        worker.update(status='ok', import_completed=True,
                      split_counts=dict(calibration=1000, heldout=1000, confirm=1000, report=872),
                      reference_checks=[dict(passed=True) for _ in range(3)])
        return worker

    def observation(self):
        return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[], signals_sent=[],
                    wall_seconds=0.01, resource_usage={}, pid=42)

    def test_source_uses_train_validation_only_with_recorded_formal_partition(self):
        (self.inputs / 'data/test-00000-of-00001.parquet').write_text('unreadable hidden-label fixture')
        record = self.outer.source_preflight(self.args)
        self.assertEqual(set(record['original_splits']), {'train', 'validation'})
        self.assertFalse(record['sample_contents_read'])
        self.assertEqual(record['partition_policy']['report'], 872)
        self.assertEqual(record['partition_policy']['confirm'], 1000)

    def test_command_preserves_explicit_paths_and_finite_limits(self):
        command = self.outer.command(self.args, 'step', self.root / 'phase')
        self.assertIn('7260', command)
        self.assertEqual(command[command.index('--ledger') + 1], str(self.args.ledger))
        self.assertEqual(command[command.index('--allowance') + 1], str(self.args.allowance))
        self.assertEqual(command[command.index('--model') + 1], str(self.model))
        self.assertEqual(command.count(self.outer.WORKER), 1)
        self.assertNotIn('test-00000-of-00001.parquet', ' '.join(command))

    def test_original_source_and_results_root_are_protected(self):
        self.args.out = self.results
        with self.assertRaises(ValueError):
            self.outer.source_preflight(self.args)
        self.args.out = self.inputs
        with self.assertRaises(ValueError):
            self.outer.source_preflight(self.args)

    def test_prepare_contract_requires_three_strict_checks_and_full_training_splits(self):
        worker = self.worker()
        self.outer.check_worker('prepare', worker)
        for key, value in (('import_completed', False), ('reference_checks', [dict(passed=True)]),
                           ('split_counts', dict(calibration=23, heldout=20, confirm=20, report=872)),
                           ('report_samples', 1), ('CKKS_latency', 0.0)):
            changed = copy.deepcopy(worker)
            changed[key] = value
            with self.assertRaises(ValueError):
                self.outer.check_worker('prepare', changed)

    def test_frozen_quality_rejection_is_retained_as_completed_report(self):
        worker = self.worker('report')
        worker.update(frozen=True, report_completed=True, report_samples=872, task_policy_passed=False)
        self.outer.check_worker('report', worker)
        worker['report_completed'] = False
        with self.assertRaises(ValueError):
            self.outer.check_worker('report', worker)

    def test_success_saves_exactly_one_registry_record_and_terminal_size(self):
        worker = self.worker()
        def run(command, **kwargs):
            work = Path(command[command.index('--work-dir') + 1])
            (work / 'worker_result.json').write_text(json.dumps(worker))
            return self.observation()
        with patch.object(self.outer, 'run_process', side_effect=run), \
                patch.object(self.outer, 'check_budget', return_value=0), \
                patch('pt_agent.registry.git_rev', return_value='synthetic'):
            result = self.outer.execute_phase(self.args, 'prepare')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['metrics']['llm_calls'], 0)
        self.assertEqual(result['case_artifact_bytes'], self.outer.case_size(self.args.out))
        rows = [json.loads(line) for line in (self.results / 'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        self.assertTrue(all(result['metrics'][name] is None for name in self.outer.FORMAL))

    def test_child_failure_stops_run_before_any_following_stage(self):
        with patch.object(self.outer, 'run_process', return_value=dict(self.observation(), exit_code=1)), \
                patch.object(self.outer, 'check_budget', return_value=0), \
                patch('pt_agent.registry.git_rev', return_value='synthetic'):
            result = self.outer.execute_phase(self.args, 'prepare')
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['worker'])
        self.assertIsNone(result['metrics']['real_ckks_attempts'])
        self.assertEqual(len((self.results / 'registry.jsonl').read_text().splitlines()), 1)

    def test_budget_failure_does_not_launch_child(self):
        with patch.object(self.outer, 'run_process') as run, \
                patch.object(self.outer, 'check_budget', side_effect=RuntimeError('budget')), \
                patch('pt_agent.registry.git_rev', return_value='synthetic'):
            result = self.outer.execute_phase(self.args, 'prepare')
        run.assert_not_called()
        self.assertEqual(result['status'], 'error')


if __name__ == '__main__':
    unittest.main()
