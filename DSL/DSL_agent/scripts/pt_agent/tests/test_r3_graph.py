from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from pt_agent.intake import Graph, import_module


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(4*4*4, 5)

    def forward(self, x):
        return self.linear(self.flatten(self.pool(self.relu(self.bn(self.conv(x))))))


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.graph = import_module(CNN(), (torch.zeros(1, 3, 8, 8),), "cnn")

    def test_save_load(self):
        with TemporaryDirectory() as directory:
            self.graph.save(directory)
            loaded = Graph.load(directory)
            self.assertTrue((Path(directory) / "graph.json").is_file())
            self.assertEqual(loaded.name, self.graph.name)
            self.assertEqual(loaded.nodes, self.graph.nodes)
            self.assertEqual(loaded.tensors, self.graph.tensors)
            self.assertEqual(loaded.inputs, self.graph.inputs)
            self.assertEqual(loaded.outputs, self.graph.outputs)
            self.assertEqual(set(loaded.weights), set(self.graph.weights))
            for name in loaded.weights:
                np.testing.assert_array_equal(loaded.weights[name], self.graph.weights[name])
                self.assertEqual(loaded.weights[name].dtype, np.float64)
                self.assertFalse(loaded.weights[name].flags.writeable)

    def test_producer_consumers(self):
        conv, bn = self.graph.nodes[:2]
        self.assertEqual(self.graph.producer(conv.outputs[0]), conv)
        self.assertEqual(self.graph.consumers(conv.outputs[0]), [bn])
        self.assertIsNone(self.graph.producer(self.graph.inputs[0]))
        self.assertEqual(self.graph.consumers(self.graph.outputs[0]), [])
