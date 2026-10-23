from collections import Counter
import unittest

import numpy as np
import torch

from pt_agent.analyzer import reference_forward
from pt_agent.data.cifar import calibration_set, encrypted_test_set, read_rows, test_set
from pt_agent.intake.coverage import report
from pt_agent.models.resnet20 import load_graph, load_torch
from pt_agent.tests._artifacts import DATA, WEIGHTS, require


@require(DATA, WEIGHTS)
class ResNet20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph()
        cls.model = load_torch().double()
        cls.calibration, cls.labels = calibration_set()

    def test_graph(self):
        self.assertEqual(len(self.graph.nodes), 74)
        self.assertEqual(Counter(node.op for node in self.graph.nodes),
                         dict(conv2d=19, batchnorm=19, act=19, add=9, slice=4, pad=2, mean=1, linear=1))
        self.assertTrue(all(node.attrs["fn"] == "relu" for node in self.graph.nodes if node.op == "act"))
        self.assertEqual(self.graph.tensors[self.graph.outputs[0]].shape, (1, 10))
        self.assertEqual(report(self.graph)["needs_composition"], 0)

    def test_shortcut_pad_and_slice(self):
        self.assertEqual([node.attrs["pads"] for node in self.graph.nodes if node.op == "pad"],
                         [(0, 0, 0, 0, 8, 8), (0, 0, 0, 0, 16, 16)])
        for node in self.graph.nodes:
            if node.op == "slice":
                self.assertEqual(node.attrs["step"], 2)
                self.assertIn(node.attrs["axis"], (2, 3))

    def test_calibration_logits(self):
        x = self.calibration[:8]
        actual = reference_forward(self.graph, {"x": x})[self.graph.outputs[0]]
        with torch.no_grad():
            expected = self.model(torch.from_numpy(x)).numpy()
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_first_500_argmax(self):
        images, _ = test_set(500)
        for start in range(0, len(images), 250):
            x = images[start:start+250]
            actual = reference_forward(self.graph, {"x": x})[self.graph.outputs[0]].argmax(axis=1)
            with torch.no_grad():
                expected = self.model(torch.from_numpy(x)).numpy().argmax(axis=1)
            np.testing.assert_array_equal(actual, expected)

    def test_frozen_data(self):
        self.assertEqual(self.calibration.shape, (1000, 3, 32, 32))
        np.testing.assert_array_equal(np.bincount(self.labels, minlength=10), [100]*10)
        images, labels = encrypted_test_set()
        self.assertEqual(images.shape, (40, 3, 32, 32))
        np.testing.assert_array_equal(np.bincount(labels, minlength=10), [4]*10)
        pixels, labels = read_rows("test", [0])
        self.assertEqual(int(labels[0]), 3)
        normalized, _ = test_set(1)
        self.assertEqual(normalized[0, 0, 0, 0], (int(pixels[0, 0, 0, 0])/255-0.485)/0.229)
