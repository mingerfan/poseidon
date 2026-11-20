"""Synthetic streaming projection checks; no model weights, fitting or CKKS."""

import gc
import os
import unittest
import weakref


class StreamingEmbeddingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        from pt_agent.backends import CountBackend, MockBackend, EmitBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels import encrypted_generation, streaming_linear, transformer, slots
        from pt_agent.kernels.layout import layout_blocks, multi_flat_layout
        cls.np, cls.Count, cls.Mock, cls.Emit = np, CountBackend, MockBackend, EmitBackend
        cls.StreamMock, cls.streaming, cls.original = StreamingMockBackend, streaming_linear, transformer
        cls.generation, cls.scopes = encrypted_generation, slots
        cls.blocks, cls.layout = staticmethod(layout_blocks), staticmethod(multi_flat_layout)

    def bind(self, backend, source):
        layout = self.layout(backend.slots, len(source))
        packed = []
        for start, stop, _ in self.blocks(layout):
            vector = self.np.zeros(backend.slots)
            vector[:stop-start] = source[start:stop]
            packed.append(vector)
        if isinstance(backend, self.Mock):
            backend.bind(packed)
        return [backend.input(index) for index in range(len(packed))], layout

    def unpack(self, result, layout):
        values = result if isinstance(result, list) else self.original.values(result, layout)
        return self.np.concatenate([value.value[:stop-start] for value, (start, stop, _) in zip(values, self.blocks(layout))])

    def assert_math(self, reference, observed):
        self.assertEqual(reference.dtype, self.np.float64)
        self.assertEqual(reference.shape, observed.shape)
        self.assertTrue(self.np.isfinite(observed).all())
        self.assertTrue(self.np.all(self.np.abs(reference-observed) <= 1e-12+1e-12*self.np.abs(reference)))

    def fixture(self, size, hidden):
        source = self.np.sin(self.np.arange(size, dtype=self.np.float64)*.71)
        matrix = self.np.cos(self.np.arange(size*hidden, dtype=self.np.float64).reshape(hidden, size)*.13)
        bias = self.np.linspace(-.1, .2, hidden, dtype=self.np.float64)
        matrix.flags.writeable = False
        return source, matrix, bias

    def test_original_algorithm_and_independent_matmul_match_across_flat_and_multiblock(self):
        for slots, size, hidden in ((8, 3, 2), (8, 9, 3), (16, 19, 11), (32, 71, 13)):
            for algorithm in ('diag', 'bsgs'):
                with self.subTest(slots=slots, size=size, hidden=hidden, algorithm=algorithm):
                    source, weight, bias = self.fixture(size, hidden)
                    original, stream = self.Mock(slots), self.StreamMock(slots)
                    values0, layout = self.bind(original, source)
                    values1, _ = self.bind(stream, source)
                    reference0, output0 = self.original.linear(original, values0, layout, weight, bias, algo=algorithm)
                    observed, output1 = self.streaming.streaming_linear(stream, values1, layout, weight, bias, algo=algorithm)
                    self.assertEqual(output0, output1)
                    decoded = self.unpack(observed, output1)
                    self.assert_math(self.unpack(reference0, output0), decoded)
                    self.assert_math(weight@source+bias, decoded)
                    self.assertTrue(all(value.data is None for value in stream._constants.values()))
                    self.assertFalse(stream._constant_objects)

    def test_count_operation_and_constant_inventory_matches_original_algorithm(self):
        source, weight, bias = self.fixture(39, 7)
        original, stream = self.Count(16), self.Count(16)
        original_values, layout = self.bind(original, source)
        stream_values, _ = self.bind(stream, source)
        self.original.linear(original, original_values, layout, weight, bias)
        self.streaming.streaming_linear(stream, stream_values, layout, weight, bias)
        self.assertEqual(original.stats(), stream.stats())
        self.assertGreater(len(original._constant_objects), 0)
        self.assertFalse(stream._constant_objects)
        self.assertTrue(all(value.data is None for value in stream._constants.values()))

    def test_existing_outer_scope_is_suspended_and_restored_without_cached_vectors(self):
        source, weight, bias = self.fixture(19, 3)
        backend = self.StreamMock(16)
        values, layout = self.bind(backend, source)
        sentinel = object()
        previous = (backend, {'untouched': sentinel}, {})
        token = self.scopes._current.set(previous)
        try:
            result, output = self.streaming.streaming_linear(backend, values, layout, weight, bias)
            self.assertIs(self.scopes._current.get(), previous)
            self.assertEqual(previous[1], {'untouched': sentinel})
            self.assertEqual(previous[2], {})
            self.assert_math(weight@source+bias, self.unpack(result, output))
        finally:
            self.scopes._current.reset(token)

    def test_exception_restores_existing_scope(self):
        from unittest.mock import patch
        source, weight, bias = self.fixture(5, 3)
        backend = self.Count(16)
        values, layout = self.bind(backend, source)
        previous = (backend, {}, {})
        token = self.scopes._current.set(previous)
        try:
            with patch.object(self.original.linear, '__wrapped__', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    self.streaming.streaming_linear(backend, values, layout, weight, bias)
            self.assertIs(self.scopes._current.get(), previous)
        finally:
            self.scopes._current.reset(token)

    def test_plain_payload_lifetime_ends_after_arithmetic_without_retained_weight_views(self):
        class Observed(self.StreamMock):
            def __init__(self):
                super().__init__(16)
                self.payloads, self.active, self.peak = [], weakref.WeakSet(), 0
            def const(self, vector):
                value = super().const(vector)
                self.payloads.append(weakref.ref(value.data))
                self.active.add(value)
                self.peak = max(self.peak, len(self.active))
                return value
        source, weight, bias = self.fixture(39, 7)
        backend = Observed()
        values, layout = self.bind(backend, source)
        matrix_ref = weakref.ref(weight)
        result, output = self.streaming.streaming_linear(backend, values, layout, weight, bias)
        self.assert_math(weight@source+bias, self.unpack(result, output))
        del weight
        gc.collect()
        self.assertIsNone(matrix_ref())
        self.assertTrue(backend.payloads)
        self.assertTrue(all(reference() is None for reference in backend.payloads))
        self.assertLessEqual(backend.peak, 2)
        self.assertFalse(backend._constant_objects)

    def test_readonly_direct_plain_is_ephemeral_and_keeps_source_unchanged(self):
        backend = self.StreamMock(8)
        array = self.np.arange(8, dtype=self.np.float64)
        array.flags.writeable = False
        source_ref = weakref.ref(array)
        plain = backend.const(array)
        payload_ref = weakref.ref(plain.data)
        self.np.testing.assert_array_equal(array, plain.data)
        self.assertFalse(self.np.shares_memory(array, plain.data))
        del array, plain
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(payload_ref())
        self.assertFalse(backend._constant_objects)
        self.assertTrue(all(value.data is None for value in backend._constants.values()))

    def test_encrypted_embedding_interface_uses_streaming_backend_without_changing_matrix(self):
        logits = self.np.asarray([.7, -.8, .4, -.2, .1, .3, -.5, .2, -.1], dtype=self.np.float64)
        embedding = self.np.arange(27, dtype=self.np.float64).reshape(9, 3)/30.
        original = embedding.copy()
        coefficients = (self.np.asarray([0., .5]),)
        backend = self.StreamMock(8)
        values, _ = self.bind(backend, logits)
        selected = self.generation.select_maximum(backend, values, len(logits), bound=2., normalized_gap=.01,
                                                  coefficients=coefficients)
        result, layout = self.generation.embedding_from_selection(backend, selected, embedding)
        one_hot, _ = self.generation.numpy_selection(logits, bound=2., normalized_gap=.01, coefficients=coefficients)
        self.assert_math(one_hot@embedding, self.unpack(result, layout))
        self.np.testing.assert_array_equal(embedding, original)
        self.assertFalse(backend._constant_objects)

    def test_emit_and_uncovered_tall_projection_fail_before_any_operations(self):
        backend = self.Emit(8)
        values, layout = self.bind(backend, self.np.zeros(5))
        with self.assertRaisesRegex(ValueError, 'CountBackend or StreamingMockBackend'):
            self.streaming.streaming_linear(backend, values, layout, self.np.ones((3, 5)))
        selection = dict(one_hot=values, layout=layout, metadata=dict(vocabulary=5))
        with self.assertRaisesRegex(ValueError, 'Embedding Emit'):
            self.generation.embedding_from_selection(backend, selection, self.np.ones((5, 3)))
        self.assertEqual(backend.stats()['ops_total'], 0)
        count = self.Count(8)
        values, layout = self.bind(count, self.np.zeros(3))
        with self.assertRaisesRegex(ValueError, 'shrinking flat'):
            self.streaming.streaming_linear(count, values, layout, self.np.ones((5, 3)))
        self.assertEqual(count.stats()['ops_total'], 0)
        self.assertFalse(self.streaming.POLICY['emit_supported'])


if __name__ == '__main__':
    unittest.main()
