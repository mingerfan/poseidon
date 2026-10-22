from pathlib import Path
import sys
import unittest

import numpy as np

from pt_agent.backends import BackendError, CountBackend, LevelError, MockBackend, run_program


SLOTS = 64


def inputs(n):
    rng = np.random.default_rng(20261022)
    return [rng.uniform(-1, 1, SLOTS) for _ in range(n)]


def p1(backend, x0, x1):
    c1 = backend.const(np.arange(SLOTS)/SLOTS)
    return (x0 * c1 + x0.rotate(3)) * x1 - 0.5


def p2(backend, x0):
    y = x0 * 0.6 + 0.5
    for coefficient in (0.4, 0.3, 0.2, 0.1):
        y = y * x0 + coefficient
    return y


def p3(backend, x0):
    y = x0 * x0
    y = y.bootstrap()
    return y * y


def p4(backend, x0):
    for step in (1, 2, 4, 8, 16, 32):
        x0 = x0 + x0.rotate(step)
    return x0


def p5(backend, x0, x1):
    return [x0 + x1, x0 * x1]


def reverse_program(backend, x0):
    plain = backend.const([0.25])
    return [-((plain + x0) * 2.0 - (1.0 - x0)), plain - x0, plain * x0, 2 + x0]


PROGRAMS = [("P1", p1, 2), ("P2", p2, 1), ("P3", p3, 1), ("P4", p4, 1), ("P5", p5, 2)]


def mock_run(fn, values):
    backend = MockBackend(SLOTS)
    backend.bind(values)
    run_program(backend, fn, len(values))
    return backend


def count_run(fn, n):
    backend = CountBackend(SLOTS, input_level=2, boot_out_level=3) if fn is p3 else CountBackend(SLOTS)
    run_program(backend, fn, n)
    return backend


class BackendTests(unittest.TestCase):
    def test_p1(self):
        x0, x1 = inputs(2)
        mock = mock_run(p1, [x0, x1])
        expected = ((x0 * (np.arange(SLOTS)/SLOTS) + np.roll(x0, -3)) * x1) - 0.5
        self.assertTrue(np.array_equal(mock.outputs[0], expected))
        self.assertFalse(mock.outputs[0].flags.writeable)
        stats = count_run(p1, 2).stats()
        self.assertEqual(stats["counts"], dict(add=1, sub=1, neg=0, mul_cc=1, mul_cp=1, rot=1, boot=0))
        self.assertEqual(stats["max_depth"], 2)
        self.assertEqual(stats["distinct_consts"], 2)
        self.assertEqual(stats["rot_steps"], [3])

    def test_p2_horner(self):
        x, = inputs(1)
        expected = np.polyval([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], x)
        self.assertLessEqual(np.max(np.abs(mock_run(p2, [x]).outputs[0] - expected)), 1e-12)
        stats = count_run(p2, 1).stats()
        self.assertEqual(stats["counts"], dict(add=5, sub=0, neg=0, mul_cc=4, mul_cp=1, rot=0, boot=0))
        self.assertEqual(stats["max_depth"], 5)
        self.assertEqual(stats["distinct_consts"], 6)

    def test_p3_levels(self):
        stats = count_run(p3, 1).stats()
        self.assertEqual(stats["min_level"], 1)
        self.assertEqual(stats["max_depth"], 1)
        self.assertEqual(stats["max_total_depth"], 2)
        self.assertEqual(stats["counts"]["boot"], 1)
        backend = CountBackend(SLOTS, input_level=1)
        x = backend.input(0)
        y = x*x
        with self.assertRaises(LevelError) as caught:
            y*y
        self.assertEqual(caught.exception.op_index, 1)

    def test_p4_sum(self):
        x, = inputs(1)
        self.assertLessEqual(np.max(np.abs(mock_run(p4, [x]).outputs[0] - x.sum())), 1e-12)
        stats = count_run(p4, 1).stats()
        self.assertEqual(stats["counts"], dict(add=6, sub=0, neg=0, mul_cc=0, mul_cp=0, rot=6, boot=0))
        self.assertEqual(stats["rot_steps"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(stats["max_depth"], 0)

    def test_p5_outputs(self):
        x0, x1 = inputs(2)
        outputs = mock_run(p5, [x0, x1]).outputs
        self.assertTrue(np.array_equal(outputs[0], x0+x1))
        self.assertTrue(np.array_equal(outputs[1], x0*x1))
        self.assertEqual(count_run(p5, 2).stats()["outputs"], 2)

    def test_rotate_matches_existing_math(self):
        baseline = str(Path(__file__).resolve().parents[2] / "baseline")
        sys.path.insert(0, baseline)
        try:
            from first_output_dsl_math import _Value, _Work
        finally:
            sys.path.remove(baseline)
        x, = inputs(1)
        backend = MockBackend(SLOTS)
        backend.bind([x])
        value = backend.input(0)
        for step in (3, -5):
            with self.subTest(step=step):
                self.assertTrue(np.array_equal(value.rotate(step).value, _Value(x, _Work(10**6)).rotate(step).value))

    def test_boundaries(self):
        backend = MockBackend(SLOTS)
        with self.assertRaises(BackendError):
            backend.input(0)
        backend.bind(inputs(1))
        x = backend.input(0)
        counter = CountBackend(SLOTS)
        cx = counter.input(0)
        for step in (0, 64):
            self.assertIs(x.rotate(step), x)
            self.assertIs(cx.rotate(step), cx)
        self.assertEqual(counter.stats()["ops_total"], 0)
        with self.assertRaises(BackendError):
            x.rotate(True)
        with self.assertRaises(BackendError):
            x + True
        with self.assertRaises(BackendError):
            backend.const([1]) + backend.const([2])
        plain = backend.const([1])
        for operation in (lambda: plain.rotate(1), lambda: plain.bootstrap(), lambda: -plain):
            with self.assertRaises(BackendError):
                operation()
        other = MockBackend(SLOTS)
        other.bind(inputs(1))
        with self.assertRaises(BackendError):
            x + other.input(0)
        with self.assertRaises(BackendError):
            backend.const([1, 2, 3])
        with self.assertRaises(BackendError):
            backend.const([np.inf])
        with self.assertRaises(BackendError):
            backend.output([x])
        backend.output([x+1])
        with self.assertRaises(BackendError):
            backend.output([x+2])
        with self.assertRaises(BackendError):
            cx.bootstrap()
        self.assertIs(counter.const([2]), counter.const(np.full(SLOTS, 2.0)))
        overflow = MockBackend(SLOTS)
        overflow.bind([np.ones(SLOTS)])
        with self.assertRaises(BackendError):
            overflow.input(0) * 1e308 * 1e308

    def test_reverse_arithmetic_and_negation(self):
        x, = inputs(1)
        expected = [-((0.25+x)*2.0 - (1.0-x)), 0.25-x, 0.25*x, 2+x]
        for actual, reference in zip(mock_run(reverse_program, [x]).outputs, expected):
            self.assertTrue(np.array_equal(actual, reference))
        counts = count_run(reverse_program, 1).stats()["counts"]
        self.assertEqual(counts, dict(add=2, sub=3, neg=1, mul_cc=0, mul_cp=2, rot=0, boot=0))
