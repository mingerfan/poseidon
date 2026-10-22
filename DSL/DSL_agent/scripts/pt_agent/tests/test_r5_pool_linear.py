import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pt_agent.backends import CountBackend, EmitBackend, MockBackend
from pt_agent.kernels import (FlatLayout, KernelError, linear_flat, mean_mux, mux_layout,
                              pack_flat, pack_mux, unpack_flat)


class PoolLinearTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261029)

    def test_mean(self):
        for dims in ((16384, 64, 8, 8, 4), (256, 4, 4, 4, 1), (256, 8, 2, 2, 2)):
            with self.subTest(layout=dims):
                lin = mux_layout(*dims)
                image = self.rng.uniform(-1, 1, (lin.C, lin.H, lin.W))
                mock = MockBackend(lin.n)
                mock.bind([pack_mux(image, lin)])
                value, layout = mean_mux(mock, mock.input(0), lin)
                self.assertEqual(layout, FlatLayout(lin.n, lin.C))
                self.assertLessEqual(float(np.max(np.abs(unpack_flat(value.value, layout)-image.mean(axis=(1, 2))))), 1e-12)
                self.assertTrue(np.all(value.value[lin.C:] == 0))
                if lin.C == 64:
                    counter = CountBackend(lin.n)
                    value, _ = mean_mux(counter, counter.input(0), lin)
                    counter.output([value])
                    self.assertEqual(counter.stats()["counts"], dict(add=21, sub=0, neg=0, mul_cc=0, mul_cp=16, rot=21, boot=0))
                    self.assertEqual(counter.stats()["max_depth"], 1)

    def test_linear_algorithms(self):
        for I, O in ((64, 10), (16, 16), (10, 3)):
            x = self.rng.uniform(-1, 1, I)
            weight = self.rng.normal(0, 0.3, (O, I))
            bias = self.rng.normal(0, 0.3, O)
            lin = FlatLayout(16384, I)
            for algo in ("diag", "bsgs"):
                with self.subTest(shape=(I, O), algo=algo):
                    mock = MockBackend(lin.n)
                    mock.bind([pack_flat(x, lin)])
                    value, layout = linear_flat(mock, mock.input(0), lin, weight, bias, algo=algo)
                    self.assertLessEqual(float(np.max(np.abs(unpack_flat(value.value, layout)-(weight@x+bias)))), 1e-12)
                    self.assertTrue(np.all(value.value[O:] == 0))
                    counter = CountBackend(lin.n)
                    value, _ = linear_flat(counter, counter.input(0), lin, weight, bias, algo=algo)
                    counter.output([value])
                    self.assertEqual(counter.stats()["max_depth"], 2)
                    if I == 64:
                        self.assertEqual(counter.stats()["counts"], dict(add=19, sub=0, neg=0, mul_cc=0, mul_cp=17,
                                                                        rot=18 if algo == "diag" else 9, boot=0))
        with self.assertRaises(KernelError):
            counter = CountBackend(16384)
            linear_flat(counter, counter.input(0), FlatLayout(16384, 32), np.ones((64, 32)), None)

    def test_linear_emit(self):
        lin = FlatLayout(16384, 64)
        vector = pack_flat(self.rng.uniform(-1, 1, 64), lin)
        weight = self.rng.normal(0, 0.3, (10, 64))
        bias = self.rng.normal(0, 0.3, 10)
        mock = MockBackend(lin.n)
        mock.bind([vector])
        value, _ = linear_flat(mock, mock.input(0), lin, weight, bias)
        mock.output([value])
        counter = CountBackend(lin.n)
        value, _ = linear_flat(counter, counter.input(0), lin, weight, bias)
        counter.output([value])
        fixture = Path(__file__).resolve().parent / "fixtures/fake_hecate_expr.py"
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            emitter = EmitBackend(lin.n, frontend=str(fixture), constants=str(directory/"constants.npz"), save_dir=str(directory))
            value, _ = linear_flat(emitter, emitter.input(0), lin, weight, bias)
            emitter.output([value])
            self.assertEqual(emitter.stats()["counts"], counter.stats()["counts"])
            emitter.write(directory)
            np.savez(directory/"inputs.npz", x0=vector)
            completed = subprocess.run([sys.executable, "-B", str(directory/"candidate_trace.py")],
                                       env={**os.environ, "FAKE_HECATE_INPUTS": str(directory/"inputs.npz")},
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            with np.load(directory/"fake_outputs.npz", allow_pickle=False) as output:
                np.testing.assert_array_equal(output["o0"], mock.outputs[0])
