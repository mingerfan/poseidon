import unittest

import numpy as np

from pt_agent.backends import BackendError, CountBackend, EmitBackend, MockBackend, run_program
from pt_agent.kernels.poly import EXACT
from pt_agent.tests.test_r2_backends import p1


class MockBatchTests(unittest.TestCase):
    def test_p1_batch(self):
        rng = np.random.default_rng(20261029)
        inputs = [rng.uniform(-1, 1, (5, 64)) for _ in range(2)]
        batch = MockBackend(64, batch=5)
        batch.bind(inputs)
        run_program(batch, p1, 2)
        self.assertEqual(batch.outputs[0].shape, (5, 64))
        self.assertFalse(batch.outputs[0].flags.writeable)
        for row in range(5):
            single = MockBackend(64)
            single.bind([values[row] for values in inputs])
            run_program(single, p1, 2)
            np.testing.assert_array_equal(batch.outputs[0][row], single.outputs[0])

    def test_exact_activations(self):
        rng = np.random.default_rng(20261029)
        for batch in (None, 5):
            data = rng.uniform(-1, 1, (64,) if batch is None else (batch, 64))
            backend = MockBackend(64, batch=batch)
            backend.bind([data])
            value = backend.input(0)
            for fn in EXACT:
                with self.subTest(batch=batch, fn=fn):
                    np.testing.assert_array_equal(backend.apply_exact(value, fn).value, EXACT[fn](data))
        for backend in (CountBackend(64), EmitBackend(64)):
            with self.assertRaises(AttributeError):
                backend.apply_exact(backend.input(0), "relu")

    def test_batch_validation(self):
        backend = MockBackend(64, batch=5)
        for value in (np.ones(64), np.full((5, 64), np.nan), np.ones((4, 64))):
            with self.assertRaises(BackendError):
                backend.bind([value])
