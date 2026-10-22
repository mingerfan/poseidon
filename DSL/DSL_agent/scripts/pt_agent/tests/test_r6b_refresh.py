import unittest

import numpy as np

from pt_agent.backends import CountBackend, EmitBackend, MockBackend, RefreshBackend
from pt_agent.cost import load_latency


LEVELS = dict(input_level=5, boot_in_level=1, boot_out_level=5)


class RefreshTests(unittest.TestCase):
    def test_eight_squares(self):
        data = np.linspace(-.9, .9, 64)
        plain = MockBackend(64)
        plain.bind([data])
        expected = plain.input(0)
        for _ in range(8):
            expected = expected*expected
        inner = MockBackend(64)
        inner.bind([data])
        wrapped = RefreshBackend(inner, **LEVELS)
        value = wrapped.input(0)
        for _ in range(8):
            value = value*value
        np.testing.assert_array_equal(value.value.value, expected.value)
        self.assertEqual(wrapped.stats()['boots'], 1)
        self.assertIsNone(wrapped.stats()['latency_us'])
        self.assertEqual(wrapped.stats()['min_level'], 1)
        for inner in (CountBackend(64, boot_out_level=5), EmitBackend(64)):
            wrapped = RefreshBackend(inner, **LEVELS)
            value = wrapped.input(0)
            for _ in range(8):
                value = value*value
            wrapped.output([value])
            self.assertEqual(inner.stats()['counts']['boot'], 1)
            self.assertEqual(wrapped.stats()['boots'], 1)
            if isinstance(inner, EmitBackend):
                self.assertEqual(inner.render()[0].count('hc.bootstrap('), 1)

    def test_refresh_cache(self):
        inner = CountBackend(64, boot_out_level=5)
        backend = RefreshBackend(inner, **LEVELS)
        value = backend.input(0)
        for _ in range(4):
            value = value*value
        self.assertEqual(value.level, 1)
        plain = backend.const([2.0])
        first = value*plain
        second = plain*value
        self.assertEqual((first.level, second.level), (4, 4))
        self.assertEqual(inner.stats()['counts']['boot'], 1)
        self.assertEqual(backend._cache[id(value)][0], value)

    def test_two_low_values(self):
        backend = RefreshBackend(CountBackend(64, boot_out_level=5), **LEVELS)
        left, right = backend.input(0), backend.input(1)
        for _ in range(4):
            left, right = left*left, right*right
        result = left*right
        self.assertEqual(result.level, 4)
        self.assertEqual(backend.stats()['boots'], 2)

    def test_explicit_bootstrap_rotate_and_interface(self):
        inner = MockBackend(64)
        data = np.linspace(-.5, .5, 64)
        inner.bind([data])
        backend = RefreshBackend(inner, **LEVELS)
        x = backend.input(0)
        refreshed = x.bootstrap()
        self.assertEqual(refreshed.level, 5)
        self.assertIs(refreshed.rotate(0), refreshed)
        self.assertIs(refreshed.rotate(64), refreshed)
        self.assertEqual(refreshed.rotate(1).level, 5)
        np.testing.assert_array_equal((-x).value.value, -data)
        np.testing.assert_array_equal(backend.apply_exact(x, 'relu').value.value, np.maximum(data, 0))
        plain = backend.const([2.0])
        for constant in (2.0, plain):
            np.testing.assert_array_equal((constant+x).value.value, 2+data)
            np.testing.assert_array_equal((constant-x).value.value, 2-data)
            np.testing.assert_array_equal((constant*x).value.value, 2*data)
        self.assertEqual(backend.stats()['boots'], 1)

    def test_latency(self):
        table = load_latency('qwen48')
        backend = RefreshBackend(CountBackend(64, boot_out_level=5), **LEVELS, latency_table=table)
        x = backend.input(0)
        result = (x*x).rotate(3)
        self.assertEqual(result.level, 4)
        expected = table['mul_double'][5]+table['rescale_single'][5]+table['rotate_single'][4]
        self.assertEqual(backend.stats()['latency_us'], expected)
        self.assertEqual(backend.stats()['level_hist'], {5: 1, 4: 1})
