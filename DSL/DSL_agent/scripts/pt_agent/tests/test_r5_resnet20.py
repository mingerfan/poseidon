import copy
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import numpy as np

from pt_agent.analyzer import reference_forward
from pt_agent.backends import CountBackend, MockBackend
from pt_agent.data.cifar import calibration_set
from pt_agent.intake import Graph
from pt_agent.kernels import KernelError
from pt_agent.models.resnet20_fhe import build, decode_logits, pack_input
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, require


GRAPH = pt_result('r3-resnet20-20261020T131658/graph')


@require(DATA, GRAPH)
class ResNet20FheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(GRAPH)
        cls.images = calibration_set()[0][:2].copy()

    def tearDown(self):
        gc.collect()

    def test_exact(self):
        cfg = {node.inputs[0]: ("exact",) for node in self.graph.nodes if node.op == "act"}
        for batch in (1, 2):
            images = self.images[:batch]
            backend = MockBackend(16384, batch=batch)
            backend.bind([np.stack([pack_input(image) for image in images])])
            started = time.monotonic()
            value, layout = build(backend, self.graph, cfg)
            seconds = time.monotonic()-started
            expected = reference_forward(self.graph, {"x": images})[self.graph.outputs[0]]
            self.assertLessEqual(float(np.max(np.abs(decode_logits(value.value, layout)-expected))), 1e-9)
            if batch == 1:
                self.assertLessEqual(seconds, 60)
            print(f"R5 exact batch={batch} build_seconds={seconds:.6f}", flush=True)
            del backend, value
            gc.collect()

    def test_count(self):
        cfg = {node.inputs[0]: ("cheb", 31, -10, 10) for node in self.graph.nodes if node.op == "act"}
        backend = CountBackend(16384)
        value, _ = build(backend, self.graph, cfg)
        backend.output([value])
        self.assertGreater(backend.stats()["counts"]["mul_cc"], 0)
        self.assertGreaterEqual(backend.stats()["max_depth"], 100)

    def test_unsupported_activation(self):
        graph = copy.deepcopy(self.graph)
        node = next(node for node in graph.nodes if node.op == "act")
        node.attrs["fn"] = "hardswish"
        cfg = {op.inputs[0]: ("exact",) for op in graph.nodes if op.op == "act"}
        backend = MockBackend(16384, batch=2)
        backend.bind([np.stack([pack_input(image) for image in self.images])])
        with self.assertRaisesRegex(KernelError, node.id):
            build(backend, graph, cfg)

    def test_count_calibration_memory(self):
        calibration = json.loads((GRAPH.parent / 'calibration.json').read_text())
        cfg = {node.inputs[0]: ("cheb", 63, calibration[node.inputs[0]]["min"],
                                calibration[node.inputs[0]]["max"])
               for node in self.graph.nodes if node.op == "act"}
        backend = CountBackend(16384)
        value, _ = build(backend, self.graph, cfg)
        backend.output([value])
        stats = backend.stats()
        self.assertEqual(stats["counts"], dict(add=6487, sub=171, neg=0, mul_cc=304,
                                              mul_cp=4800, rot=2624, boot=0))
        self.assertEqual(stats["ops_total"], 14386)
        self.assertEqual(stats["distinct_consts"], 4992)
        self.assertEqual(len(stats["rot_steps"]), 182)
        self.assertEqual(stats["max_depth"], 193)
        self.assertTrue(all(plain.data is None for plain in backend._constants.values()))
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.assertLessEqual(peak_kib, 1024**2)
        print(f"R5b Count peak_rss_kib={peak_kib} stats={stats}", flush=True)

    def test_grid_divergence(self):
        from pt_agent.experiments.r5_resnet20 import _grid_case

        calibration = json.loads((GRAPH.parent / 'calibration.json').read_text())
        cfg = {node.inputs[0]: ("cheb", 63, calibration[node.inputs[0]]["min"],
                                calibration[node.inputs[0]]["max"])
               for node in self.graph.nodes if node.op == "act"}
        name = next(iter(cfg))
        kind, degree, lo, hi = cfg[name]
        cfg[name] = (kind, degree, lo*0.5, hi*0.5)
        references = reference_forward(self.graph, {"x": self.images})[self.graph.outputs[0]]
        labels = calibration_set()[1][:2]
        row = _grid_case(self.graph, cfg, self.images, labels, references, degree=63, margin=1.0)
        self.assertTrue(row["diverged"])
        self.assertEqual(row["images_done"], 0)
        self.assertEqual(row["diverged_batch_start"], 0)
        self.assertIsNone(row["top1"])
        self.assertIsNone(row["logit_rmse"])
        self.assertGreater(row["max_u"], 1)
        self.assertIn(row["max_u_act"], cfg)
