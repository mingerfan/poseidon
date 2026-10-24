"""U3a streaming constant lifetime and linear workspace regressions."""

import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from pt_agent.backends import CountBackend, MockBackend
from pt_agent.backends.base import _Plain
from pt_agent.kernels import cnn, multi
from pt_agent.kernels.layout import FlatLayout, multi_flat_layout, multi_mux_layout
from pt_agent.kernels.linear import linear_flat


class MemoryTests(unittest.TestCase):
    def test_constants_release_data_and_keep_distinct_names(self):
        backend = MockBackend(16)
        value = backend.const([2.])
        duplicate = backend.const(np.full(16, 2.))
        other = backend.const([3.])
        self.assertIsNot(value, duplicate)
        self.assertEqual((value.name, duplicate.name, other.name), ('k0', 'k0', 'k1'))
        self.assertEqual(len(backend._constants), 2)
        self.assertTrue(all(isinstance(key, bytes) and isinstance(index, int)
                            for key, index in backend._constants.items()))
        self.assertFalse(value.data.flags.writeable)
        reference = weakref.ref(value.data)
        del value
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsInstance(duplicate, _Plain)
        np.testing.assert_array_equal(duplicate.data, np.full(16, 2.))

    def test_streaming_and_retained_arithmetic_identical(self):
        data = np.linspace(-3, 3, 64).reshape(2, 32)
        outputs = []
        for streaming in (True, False):
            backend = MockBackend(32, batch=2, stream_constants=streaming)
            backend.bind([data])
            x = backend.input(0)
            a = backend.const(np.arange(32, dtype=np.float64))
            z = ((x*a).rotate(3)+backend.const([.25]))*backend.const([2.])
            backend.output([z])
            outputs.append(backend.outputs[0])
            self.assertEqual(len(backend._constants), 3)
            if not streaming:
                self.assertIs(a, backend.const(np.arange(32, dtype=np.float64)))
        np.testing.assert_array_equal(*outputs)

    def test_linear_kernels_use_vector_workspace(self):
        zeros = np.zeros

        def vector_only(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) > 1:
                raise AssertionError('linear kernel allocated a dense workspace')
            return zeros(shape, *args, **kwargs)

        flat = FlatLayout(64, 7)
        blocked = multi_flat_layout(64, 45)
        mux = multi_mux_layout(64, 5, 4, 4, 1)
        weights = [np.arange(35, dtype=np.float64).reshape(5, 7),
                   np.arange(31*45, dtype=np.float64).reshape(31, 45),
                   np.arange(11*80, dtype=np.float64).reshape(11, 80)]
        for algo in ('diag', 'bsgs'):
            for kernel, layout, weight in ((linear_flat, flat, weights[0]),
                                          (multi.linear_blocks, blocked, weights[1]),
                                          (cnn.flatten_linear, mux, weights[2])):
                with self.subTest(algo=algo, kernel=kernel.__name__):
                    from pt_agent.kernels.layout import layout_blocks
                    backend = CountBackend(64)
                    values = [backend.input(i) for i in range(len(layout_blocks(layout)))]
                    value = values if len(values) > 1 else values[0]
                    with patch('numpy.zeros', side_effect=vector_only):
                        kernel(backend, value, layout, weight, None, algo=algo)
                    self.assertGreater(backend.stats()['counts']['mul_cp'], 0)


if __name__ == '__main__':
    unittest.main()
