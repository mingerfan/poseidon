import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np

from pt_agent.backends import CountBackend, EmitBackend, MockBackend, run_program
from pt_agent.kernels import KernelError, add_mux, conv2d_mux, downsample_pad_mux, mux_layout, pack_mux, unpack_mux
from pt_agent.kernels.conv import fold_bn
from pt_agent.kernels.layout import channel_slots


def conv_ref(x, w, stride, pad):
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, w.shape[2:], axis=(1, 2))[:, ::stride, ::stride]
    return np.einsum("cyxij,ocij->oyx", windows, w, optimize=True)


class ConvTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261027)
        self.lin = mux_layout(256, 4, 4, 4, 1)
        self.x = self.rng.uniform(-1, 1, (4, 4, 4))

    def weights(self, Co, Ci):
        return self.rng.normal(0, 0.3, (Co, Ci, 3, 3))

    def check_vector(self, vector, layout, expected):
        unused = np.ones(layout.n, dtype=bool)
        for copy in range(layout.p):
            actual = unpack_mux(vector, layout, copy)
            self.assertEqual(actual.shape, expected.shape)
            self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)
            for c in range(layout.C):
                unused[channel_slots(layout, c)+copy*layout.block] = False
        if unused.any():
            self.assertLessEqual(float(np.max(np.abs(vector[unused]))), 1e-12)

    def evaluate(self, fn, expected, lin=None, x=None):
        lin = self.lin if lin is None else lin
        x = self.x if x is None else x
        mock = MockBackend(lin.n)
        mock.bind([pack_mux(x, lin)])
        value, layout = fn(mock, mock.input(0))
        mock.output([value])
        self.check_vector(mock.outputs[0], layout, expected)
        counter = CountBackend(lin.n)
        counted, counted_layout = fn(counter, counter.input(0))
        counter.output([counted])
        self.assertEqual(layout, counted_layout)
        return layout, counter.stats(), mock.outputs[0]

    def test_t1(self):
        w = self.weights(4, 4)
        _, stats, _ = self.evaluate(lambda b, x: conv2d_mux(b, x, self.lin, w, stride=1), conv_ref(self.x, w, 1, 1))
        # Step 1: 8 rotations (the center translation emits no operation).
        # Step 2: 9 mul_cp and 8 adds. Step 3: T=4 gives 2 rotations and 2 adds.
        # Step 4: 4 mul_cp, 3 rotations and 3 adds. Step 5: 2 rotations and 2 adds.
        # Total: rot=15, mul_cp=13, add=15; multiplication depth=2.
        self.assertEqual(stats["counts"], dict(add=15, sub=0, neg=0, mul_cc=0, mul_cp=13, rot=15, boot=0))
        self.assertEqual(stats["max_depth"], 2)

    def test_t2(self):
        w = self.weights(4, 4)
        layout, _, _ = self.evaluate(lambda b, x: conv2d_mux(b, x, self.lin, w, stride=2), conv_ref(self.x, w, 2, 1))
        self.assertEqual(layout, mux_layout(256, 4, 2, 2, 2))

    def test_t3(self):
        lin = mux_layout(128, 2, 4, 4, 1)
        x = self.rng.uniform(-1, 1, (2, 4, 4))
        w = self.weights(6, 2)
        self.assertEqual(lin.p, 4)
        self.evaluate(lambda b, value: conv2d_mux(b, value, lin, w, stride=1), conv_ref(x, w, 1, 1), lin, x)

    def test_t4(self):
        weights = [self.weights(4, 4), self.weights(8, 4), self.weights(8, 8)]

        def fn(b, value):
            layout = self.lin
            for w, stride in zip(weights, (1, 2, 1)):
                value, layout = conv2d_mux(b, value, layout, w, stride=stride)
            return value, layout

        expected = self.x
        for w, stride in zip(weights, (1, 2, 1)):
            expected = conv_ref(expected, w, stride, 1)
        _, stats, _ = self.evaluate(fn, expected)
        self.assertEqual(stats["max_depth"], 6)

    def test_t5(self):
        w = self.weights(4, 4)
        bias, gamma, beta, mean = self.rng.normal(0, 0.3, (4, 4))
        var = self.rng.uniform(0.5, 2, 4)
        scale, shift = fold_bn(gamma, beta, mean, var, 1e-5, conv_bias=bias)
        convolved = conv_ref(self.x, w, 1, 1)+bias[:, None, None]
        expected = (convolved-mean[:, None, None])/np.sqrt(var[:, None, None]+1e-5)*gamma[:, None, None]+beta[:, None, None]
        self.evaluate(lambda b, x: conv2d_mux(b, x, self.lin, w, stride=1, scale=scale, shift=shift), expected)

    def test_t6(self):
        expected = np.pad(self.x[:, ::2, ::2], ((2, 2), (0, 0), (0, 0)))
        layout, stats, _ = self.evaluate(lambda b, x: downsample_pad_mux(b, x, self.lin, 2, 2), expected)
        self.assertEqual(layout, mux_layout(256, 8, 2, 2, 2))
        w = self.weights(4, 4)
        counter = CountBackend(256)
        value, _ = conv2d_mux(counter, counter.input(0), self.lin, w, stride=2)
        counter.output([value])
        self.assertLess(stats["counts"]["mul_cp"], counter.stats()["counts"]["mul_cp"])

    def test_t7(self):
        w = self.weights(8, 4)

        def fn(b, x):
            main, lm = conv2d_mux(b, x, self.lin, w, stride=2)
            shortcut, ls = downsample_pad_mux(b, x, self.lin, 2, 2)
            return add_mux(b, main, lm, shortcut, ls)

        expected = conv_ref(self.x, w, 2, 1)+np.pad(self.x[:, ::2, ::2], ((2, 2), (0, 0), (0, 0)))
        self.evaluate(fn, expected)
        counter = CountBackend(256)
        x = counter.input(0)
        with self.assertRaises(KernelError):
            add_mux(counter, x, self.lin, x, mux_layout(256, 4, 2, 2, 2))

    def test_t8(self):
        layout = mux_layout(16384, 3, 32, 32, 1)
        x = self.rng.uniform(-1, 1, (3, 32, 32))
        mock = MockBackend(16384)
        mock.bind([pack_mux(x, layout)])
        value = mock.input(0)
        elapsed = 0.0
        for Co, stride, target in ((16, 1, (16384, 16, 32, 32, 1)), (32, 2, (16384, 32, 16, 16, 2)), (64, 2, (16384, 64, 8, 8, 4))):
            w = self.weights(Co, layout.C)
            started = time.monotonic()
            value, layout = conv2d_mux(mock, value, layout, w, stride=stride)
            seconds = time.monotonic()-started
            elapsed += seconds
            x = conv_ref(x, w, stride, 1)
            self.assertEqual(layout, mux_layout(*target))
            self.check_vector(value.value, layout, x)
            print(f"T8 Mock Co={Co} stride={stride} seconds={seconds:.6f}", flush=True)
        mock.output([value])
        print(f"T8 Mock total_seconds={elapsed:.6f}", flush=True)

    def test_t9(self):
        w = self.weights(4, 4)
        fn = lambda b, x: conv2d_mux(b, x, self.lin, w, stride=1)[0]
        mock = MockBackend(256)
        mock.bind([pack_mux(self.x, self.lin)])
        run_program(mock, fn, 1)
        counter = CountBackend(256)
        run_program(counter, fn, 1)
        fixture = Path(__file__).resolve().parent / "fixtures/fake_hecate_expr.py"
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            emitter = EmitBackend(256, frontend=str(fixture), constants=str(directory / "constants.npz"), save_dir=str(directory))
            run_program(emitter, fn, 1)
            self.assertEqual(emitter.stats()["counts"], counter.stats()["counts"])
            emitter.write(directory)
            np.savez(directory / "inputs.npz", x0=pack_mux(self.x, self.lin))
            completed = subprocess.run([sys.executable, "-B", str(directory / "candidate_trace.py")],
                                       env={**os.environ, "FAKE_HECATE_INPUTS": str(directory / "inputs.npz")},
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            with np.load(directory / "fake_outputs.npz", allow_pickle=False) as outputs:
                np.testing.assert_array_equal(outputs["o0"], mock.outputs[0])

    def test_t10(self):
        for lin, weight, stride in ((self.lin, np.ones((4, 4, 2, 2)), 1),
                                    (self.lin, np.ones((4, 4, 3, 3)), 3),
                                    (self.lin, np.ones((4, 3, 3, 3)), 1),
                                    (mux_layout(256, 4, 3, 4, 1), np.ones((4, 4, 3, 3)), 2)):
            with self.subTest(layout=lin, weight=weight.shape, stride=stride), self.assertRaises(KernelError):
                b = CountBackend(lin.n)
                conv2d_mux(b, b.input(0), lin, weight, stride=stride)
