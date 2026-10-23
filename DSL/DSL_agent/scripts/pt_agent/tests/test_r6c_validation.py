import contextlib
import io
import json
from unittest import mock
import unittest

import numpy as np

from pt_agent.backends.base import BackendError
from pt_agent.data import cifar
from pt_agent.run import evaluate_batches, main, select
from pt_agent.tests._artifacts import DATA, require


@require(DATA)
class ValidationTests(unittest.TestCase):
    def test_validation_set(self):
        with mock.patch.object(cifar, 'read_rows', wraps=cifar.read_rows) as read:
            images, labels = cifar.validation_set(2000)
        split, indices = read.call_args.args
        calibration = {entry['index'] for entry in json.loads(cifar.SELECTIONS.read_text())['calibration']}
        expected = [index for index in range(10000, 50000) if index not in calibration][:2000]
        self.assertEqual(split, 'train')
        self.assertEqual(indices, expected)
        self.assertFalse(calibration.intersection(indices))
        self.assertTrue(all(index >= 10000 for index in indices))
        self.assertEqual(images.shape, (2000, 3, 32, 32))
        self.assertEqual(labels.shape, (2000,))
        self.assertEqual(images.dtype, np.float64)

    def test_select(self):
        def row(deg, margin, agree, max_u=.95, diverged=False):
            return dict(deg=deg, margin=margin, agree=agree, max_u=max_u, diverged=diverged)

        rows = [row(63, 1.4, 2000, diverged=True), row(63, 1.5, 2000, max_u=.951),
                row(127, 1.4, 1990), row(63, 1.75, 1990), row(63, 1.5, 1990)]
        self.assertEqual(select(rows), dict(deg=63, margin=1.5))
        self.assertEqual(select(rows + [row(127, 1.75, 1991)]), dict(deg=127, margin=1.75))
        with self.assertRaisesRegex(ValueError, 'no nondivergent candidate'):
            select(rows[:2])
        with self.assertRaises(ValueError):
            select([])

    def test_evaluate_batches_isolates_image_seven(self):
        images = np.arange(12).reshape(-1, 1)
        labels = np.arange(12) % 3
        calls = []

        def run_batch(batch):
            indices = batch[:, 0]
            calls.append(indices.tolist())
            if 7 in indices:
                raise BackendError('fake divergence at image 7')
            return np.eye(3)[indices % 3]

        result = evaluate_batches(run_batch, images, labels, 5)
        self.assertEqual(result['n_diverged'], 1)
        self.assertEqual(result['diverged_indices'], [7])
        self.assertEqual(result['test_images_done'], 12)
        self.assertAlmostEqual(result['top1_test'], 100*11/12)
        self.assertEqual(calls, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5], [6], [7], [8], [9], [10, 11]])

    def test_validate_rejects_explicit_parameters(self):
        for option, value in (('--deg', '63'), ('--margin', '1.5')):
            with self.subTest(option=option), mock.patch('sys.argv', ['pt_agent.run', '--model', 'resnet20',
                    '--planner', 'rule', '--stages', 'validate,plan', option, value]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
