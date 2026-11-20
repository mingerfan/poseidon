"""Small deterministic rectangular schedules; no original model/data reads."""

import os
import unittest


class CompactLinearCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        from pt_agent.backends import MockBackend, CountBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.backends.streaming_count import StreamingCountBackend
        from pt_agent.kernels import multi, transformer, slots, compact_linear
        from pt_agent.kernels.layout import multi_flat_layout, layout_blocks, TokenLayout
        cls.np, cls.Mock, cls.StreamMock = np, MockBackend, StreamingMockBackend
        cls.Count, cls.StreamCount = CountBackend, StreamingCountBackend
        cls.multi, cls.transformer, cls.slots, cls.compact = multi, transformer, slots, compact_linear
        cls.layout, cls.blocks, cls.Token = staticmethod(multi_flat_layout), staticmethod(layout_blocks), TokenLayout

    def bind(self, backend, data, layout):
        packed = []
        for start, stop, _ in self.blocks(layout):
            vector = self.np.zeros(backend.slots, dtype=self.np.float64)
            vector[:stop-start] = data.reshape(-1)[start:stop]
            packed.append(vector)
        if isinstance(backend, self.Mock):
            backend.bind(packed)
        return [backend.input(index) for index in range(len(packed))]

    def unpack(self, value, layout):
        values = value if isinstance(value,list) else self.transformer.values(value, layout)
        return self.np.concatenate([term.value[:stop-start]
            for term, (start, stop, _) in zip(values, self.blocks(layout))])

    def check_shape(self, inputs, outputs, *, tokens=None, zero=False):
        np = self.np
        data = np.sin(np.arange((tokens or 1)*inputs, dtype=np.float64)*.37)/3
        if tokens:
            data = data.reshape(tokens, inputs)
        weight = np.cos(np.arange(outputs*inputs, dtype=np.float64).reshape(outputs, inputs)*.19)/7
        if zero:
            weight.fill(0)
            weight[:, -1] = np.arange(outputs, dtype=np.float64)/17
        bias = np.arange(outputs, dtype=np.float64)/23-.1
        layout = self.Token(32, tokens, inputs) if tokens else self.layout(32, inputs)
        for algo in ('diag', 'bsgs'):
            with self.subTest(shape=(outputs, inputs), tokens=tokens, algo=algo):
                old, new = self.Mock(32), self.StreamMock(32)
                old_value, old_layout = self.transformer.linear(old, self.bind(old, data, layout),
                    layout, weight, bias, algo=algo)
                new_value, new_layout = self.compact.compact_projection(new, self.bind(new, data, layout),
                    layout, weight, bias, algo=algo)
                expected, observed = self.unpack(old_value, old_layout), self.unpack(new_value, new_layout)
                direct = (data@weight.T+bias).reshape(-1)
                self.assertEqual(old_layout, new_layout)
                self.assertTrue(np.all(np.abs(observed-expected) <= 1e-12+1e-12*np.abs(expected)))
                self.assertTrue(np.all(np.abs(observed-direct) <= 1e-12+1e-12*np.abs(direct)))
                self.assertFalse(new._constant_objects)
                self.assertTrue(all(value.data is None for value in new._constants.values()))

    def test_single_block_rectangle(self):
        self.check_shape(9, 3)

    def test_input_blocks_and_non_power_of_two_tail(self):
        self.check_shape(29, 5)

    def test_one_element_tail_splits_output_without_padding_weights(self):
        self.check_shape(17, 7)

    def test_input_and_output_blocks(self):
        self.check_shape(37, 21)

    def test_token_layout_and_bias_once(self):
        self.check_shape(29, 5, tokens=3)

    def test_embedding_route_preserves_onehot_and_dense_selection_mathematics(self):
        np = self.np
        matrix = np.cos(np.arange(29*5,dtype=np.float64).reshape(29,5)*.13)/7
        for onehot in (True,False):
            selected = np.zeros(29,dtype=np.float64) if onehot else np.sin(np.arange(29,dtype=np.float64)*.17)/13
            if onehot:
                selected[13] = 1.
            backend = self.StreamMock(32)
            layout = self.layout(32,29)
            values = self.bind(backend,selected,layout)
            output, actual_layout = self.compact.embedding_from_selection(backend,
                dict(one_hot=values,layout=layout,metadata=dict(vocabulary=29)),matrix)
            reference = selected@matrix
            observed = self.unpack(output,actual_layout)
            self.assertTrue(np.all(np.abs(observed-reference) <= 1e-12+1e-12*np.abs(reference)))
            self.assertFalse(backend._constant_objects)

    def test_zero_blocks_inventory_and_scope_lifetime(self):
        self.check_shape(17, 7, zero=True)
        import gc
        import weakref
        from pt_agent.kernels.layout import KernelError
        layout = self.layout(32, 29)
        old, new = self.Count(32), self.StreamCount(32)
        weight = self.np.arange(3*29, dtype=self.np.float64).reshape(3, 29)+.1
        original_values, values = self.bind(old, self.np.ones(29), layout), self.bind(new, self.np.ones(29), layout)
        self.multi.linear_blocks(old, original_values, layout, weight, None)
        previous = (new, {}, {})
        token = self.slots._current.set(previous)
        try:
            self.compact.compact_linear_blocks(new, values, layout, weight)
            self.assertIs(self.slots._current.get(), previous)
            self.assertEqual(previous[1:], ({}, {}))
            original = self.transformer.linear
            with self.assertRaises(KernelError):
                with self.compact.compact_transformer_projections():
                    self.transformer.linear(new, values, layout, weight, algo='invalid')
            self.assertIs(self.transformer.linear, original)
        finally:
            self.slots._current.reset(token)
        self.assertLess(new.stats()['counts']['mul_cp'], old.stats()['counts']['mul_cp'])
        self.assertLess(new.stats()['distinct_consts'], old.stats()['distinct_consts'])
        reference = weakref.ref(weight)
        del weight
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(new._constant_objects)
        row = self.compact.inventory(16384, 151936, 896)
        self.assertEqual(row['compact_diagonal_sites'], 19*1024)
        self.assertEqual(row['compact_diagonal_dense_bytes'], 2550136832)
        self.assertEqual(row['original_square_geometrically_nonempty_diagonal_sites'], 152831)
        self.assertFalse(row['native_bytes_measured'])


if __name__ == '__main__':
    unittest.main()
