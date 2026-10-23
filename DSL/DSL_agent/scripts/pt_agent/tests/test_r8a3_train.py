import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pt_agent.models.train_small import (CLIP_NORM, DIVERGENCE_FACTOR, RECIPE,
                                         _write, check_divergence, run_epochs)
from pt_agent.models.zoo import Mlp
from pt_agent.tests._artifacts import DATA, require


@require(DATA)
class TrainTests(unittest.TestCase):
    def test_divergence(self):
        check_divergence([1.7, 1.5, 1.4])
        with self.assertRaisesRegex(RuntimeError, 'epoch 2'):
            check_divergence([3.3, 1488.0])
        for value in [float('nan'), float('inf')]:
            with self.subTest(loss=value):
                with self.assertRaises(RuntimeError) as caught:
                    check_divergence([1.0, value])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'failed.json'
                    _write(path, dict(error=str(caught.exception), complete=False,
                                      epochs=[dict(loss=1.0), dict(loss=value)]))
                    saved = json.loads(path.read_text())
                self.assertIn('epoch 2', saved['error'])
                self.assertFalse(saved['complete'])
                self.assertEqual(saved['epochs'][-1]['loss'], str(value))

    def test_recipe_constants(self):
        self.assertEqual(RECIPE, 'v2-clip1.0')
        self.assertEqual(CLIP_NORM, 1.0)
        self.assertEqual(DIVERGENCE_FACTOR, 1.5)

    def test_small_training_clips_every_batch(self):
        torch.manual_seed(20261022)
        torch.set_num_threads(2)
        rng = np.random.default_rng(20261022)
        images = rng.normal(size=(512, 3072)).astype(np.float32)
        labels = rng.integers(0, 10, size=512, dtype=np.int64)
        with mock.patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            report = run_epochs(Mlp(), images, labels, 2, rng)
        self.assertEqual(clip.call_count, 8)
        self.assertTrue(all(call.args[1] == CLIP_NORM for call in clip.call_args_list))
        self.assertEqual(len(report['epochs']), 2)
