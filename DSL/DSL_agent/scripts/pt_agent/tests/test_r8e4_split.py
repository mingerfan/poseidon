import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.data import cifar
from pt_agent.experiments import r8b_dev
from pt_agent.intake import Graph
from pt_agent.planners.agent_tools import build_context
from pt_agent.run import evaluate_batches
from pt_agent.paths import RESULTS_DIR


ROOT = RESULTS_DIR
R3 = ROOT/'r3-resnet20-20261020T131658'


class HeldoutTests(unittest.TestCase):
    def test_01_disjoint_complete_and_first_record(self):
        self.assertFalse(set(cifar.HELDOUT) & set(cifar.REPORT))
        self.assertEqual(set(cifar.HELDOUT) | set(cifar.REPORT), set(range(10000)))
        images, labels = cifar.heldout_set()
        pixels, expected = cifar.read_rows('test', [9000])
        np.testing.assert_array_equal(images[:1], cifar.normalize(pixels))
        np.testing.assert_array_equal(labels[:1], expected)
        report, report_labels = cifar.report_test_set()
        self.assertEqual(len(report), 9000)
        pixels, expected = cifar.read_rows('test', [0, 8999])
        np.testing.assert_array_equal(report[[0, -1]], cifar.normalize(pixels))
        np.testing.assert_array_equal(report_labels[[0, -1]], expected)
        with self.assertRaises(ValueError):
            cifar.heldout_set(1001)

    def test_02_encrypted_selection_is_report_only(self):
        entries = json.loads(cifar.SELECTIONS.read_text())['encrypted_test']
        self.assertEqual(len(entries), 40)
        self.assertTrue(all(row['index'] in cifar.REPORT for row in entries))

    def test_03_context_uses_heldout(self):
        graph = Graph.load(R3/'graph')
        calibration = json.loads((R3/'calibration.json').read_text())
        context = build_context(graph, calibration, n_val=50)
        images, _ = cifar.heldout_set(50)
        np.testing.assert_array_equal(context.val_images, images)
        self.assertEqual(context.val_split, 'test[9000:10000]')

    def test_04_checkpoint_resumes_and_keeps_correct_count(self):
        images = np.array([[0.0], [1.0]])
        labels = np.array([0, 1])
        seen = []

        def first(batch):
            index = int(batch[0, 0])
            if index == 1:
                raise KeyboardInterrupt('simulated interruption')
            seen.append(index)
            return np.array([[1.0, 0.0]])

        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'test_progress.json'
            with self.assertRaises(KeyboardInterrupt):
                evaluate_batches(first, images, labels, 1, progress_path=path, program_text='frozen')
            saved = json.loads(path.read_text())
            self.assertEqual((saved['images_done'], saved['correct']), (1, 1))

            def second(batch):
                index = int(batch[0, 0])
                seen.append(index)
                return np.array([[0.0, 1.0]])

            result = evaluate_batches(second, images, labels, 1, progress_path=path, program_text='frozen')
            self.assertEqual(seen, [0, 1])
            self.assertEqual(result['correct'], 2)
            self.assertEqual(result['top1_test'], 100.0)
            self.assertEqual(result['test_images_done'], 2)
            evaluate_batches(lambda batch: np.array([[1.0, 0.0]]), images, labels, 1,
                             progress_path=path, program_text='different source')
            self.assertTrue((path.parent/'test_progress.stale.json').exists())
            self.assertEqual(json.loads(path.read_text())['program_text'], 'different source')

    def test_05_resume_skips_only_complete_models_and_keeps_budget(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'run'
            run.mkdir()
            models = []
            for name, complete in [('resnet20', True), ('resnet20_silu', False)]:
                case = run/name
                case.mkdir()
                (case/'REPORT.json').write_text(json.dumps(dict(complete=complete, stage='complete' if complete else 'plan')))
                models.append(dict(model=name, report_path=str(case/'REPORT.json'), complete=complete,
                                   call_indices=[], attempts=[]))
            baseline = root/'baseline.json'
            baseline.write_text(json.dumps(dict(models=[dict(model=row['model'], rule_report_path='unused') for row in models])))
            original = dict(round='R8e4', tuner='on', models_dir='unused', baseline_report=str(baseline),
                            val_split='test[9000:10000]', test_split='test[0:9000]', api_before=0, api_limit=24,
                            api_budget=24, models=models, git_rev='offline')
            (run/'REPORT.json').write_text(json.dumps(original))
            calls = []

            def fake_run(command, **kwargs):
                name = command[command.index('--model')+1]
                calls.append(name)
                case = Path(command[command.index('--out')+1])
                (case/'REPORT.json').write_text(json.dumps(dict(complete=True, stage='complete', top1_test=90,
                                                                torch_top1_report=91, n_diverged=0)))
                return SimpleNamespace(returncode=0)

            args = SimpleNamespace(resume=run, out=None, round='R8e4', tuner='on')
            with patch.dict(os.environ, PT_RESULTS=str(root)), patch.object(r8b_dev.subprocess, 'run', side_effect=fake_run):
                self.assertEqual(r8b_dev._outer(args), 0)
            self.assertEqual(calls, ['resnet20_silu'])
            final = json.loads((run/'REPORT.json').read_text())
            self.assertEqual(final['api_limit'], 24)
            self.assertEqual(final['api_before'], 0)
            self.assertTrue((run/'resnet20_silu'/'REPORT.json').exists())


if __name__ == '__main__':
    unittest.main()
