"""Small arithmetic/quality/state checks; no real model, fit, API or CKKS."""

from collections import Counter
import os
import unittest


class EncryptedGenerationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        from pt_agent.backends import MockBackend, CountBackend
        from pt_agent.kernels import encrypted_generation
        cls.np, cls.MockBackend, cls.CountBackend = np, MockBackend, CountBackend
        cls.module = encrypted_generation
        # Explicit synthetic smoothing polynomials; no fitting or model data.
        cls.coefficients = (np.asarray([0.0, 1.125, 0.0, -0.125]),
                            np.asarray([0.0, 1.125, 0.0, -0.125]),
                            np.asarray([0.0, 0.5625, 0.0, -0.0625]))

    def inputs(self, backend, logits):
        from pt_agent.kernels.layout import layout_blocks, multi_flat_layout
        layout = multi_flat_layout(backend.slots, len(logits))
        vectors = []
        for start, stop, _ in layout_blocks(layout):
            value = self.np.zeros(backend.slots, dtype=self.np.float64)
            value[:stop-start] = logits[start:stop]
            vectors.append(value)
        if isinstance(backend, self.MockBackend):
            backend.bind(vectors)
        return [backend.input(index) for index in range(len(vectors))]

    def unpack(self, values, layout):
        from pt_agent.kernels.layout import layout_blocks
        return self.np.concatenate([value.value[:stop-start] for value, (start, stop, _) in zip(values, layout_blocks(layout))])

    def test_packed_tournament_matches_independent_math_for_odd_multiblock_vocabulary(self):
        logits = self.np.asarray([0.9, -0.8, 0.4, -0.5, 0.2, -0.1, 0.6, -0.7, 0.3], dtype=self.np.float64)
        backend = self.MockBackend(8)
        selection = self.module.select_maximum(backend, self.inputs(backend, logits), len(logits),
            bound=2.0, normalized_gap=0.01, coefficients=self.coefficients)
        observed = self.unpack(selection['one_hot'], selection['layout'])
        quality = self.module.selection_quality(logits, observed, bound=2.0, normalized_gap=0.01,
                                                coefficients=self.coefficients)
        self.assertTrue(quality['arithmetic_passed'], quality)
        self.assertTrue(quality['token_agreement'], quality)
        self.assertFalse(quality['exact_discrete_one_hot_claimed'])
        self.assertLess(abs(quality['approximate_one_hot_sum']-1.0), 1e-12)
        self.assertEqual(selection['metadata']['tournament_rounds'], 4)

    def test_parity_moves_preserve_all_widths_at_block_boundaries_without_general_gather(self):
        from unittest.mock import patch
        for slots in (4, 8, 16, 32):
            capacity = slots//2
            for width in range(2, 2*capacity+4):
                with self.subTest(slots=slots, width=width):
                    logits = self.np.sin(self.np.arange(width, dtype=self.np.float64)*1.31)*.8
                    backend = self.MockBackend(slots)
                    with patch('pt_agent.kernels.transformer.gather', side_effect=AssertionError('general gather')):
                        selected = self.module.select_maximum(backend, self.inputs(backend, logits), width,
                            bound=2., normalized_gap=.01, coefficients=self.coefficients)
                    observed = self.unpack(selected['one_hot'], selected['layout'])
                    quality = self.module.selection_quality(logits, observed, bound=2., normalized_gap=.01,
                                                           coefficients=self.coefficients)
                    self.assertTrue(quality['arithmetic_passed'], quality)
                    self.assertEqual(selected['metadata']['general_gather_calls'], 0)
                    self.assertLessEqual(selected['metadata']['pattern_mask_count'], 6*slots.bit_length()+12)
                    for value, (_, _, layout) in zip(selected['one_hot'], self._blocks(selected['layout'])):
                        self.np.testing.assert_array_equal(value.value[layout.size:], 0.)

    @staticmethod
    def _blocks(layout):
        from pt_agent.kernels.layout import layout_blocks
        return layout_blocks(layout)

    def test_near_tie_math_passes_while_gap_contract_fails(self):
        logits = self.np.asarray([1e-7, 0.0], dtype=self.np.float64)
        backend = self.MockBackend(8)
        selection = self.module.select_maximum(backend, self.inputs(backend, logits), 2,
            bound=2.0, normalized_gap=0.1, coefficients=self.coefficients)
        quality = self.module.selection_quality(logits, self.unpack(selection['one_hot'], selection['layout']),
            bound=2.0, normalized_gap=0.1, coefficients=self.coefficients)
        self.assertTrue(quality['arithmetic_passed'])
        self.assertFalse(quality['comparison_gap_contract_passed'])
        self.assertGreater(quality['approximate_one_hot_max_error'], 0.49)

    def test_exact_tie_is_explicitly_outside_positive_gap_contract(self):
        logits = self.np.asarray([0.5, 0.5], dtype=self.np.float64)
        expected, metadata = self.module.numpy_selection(logits, bound=2.0, normalized_gap=0.1, coefficients=self.coefficients)
        self.np.testing.assert_array_equal(expected, [0.5, 0.5])
        self.assertFalse(metadata['comparison_gap_contract_passed'])

    def test_declared_difference_bound_is_checked_by_quality_diagnostic(self):
        logits = self.np.asarray([2.1, 0.0], dtype=self.np.float64)
        _, metadata = self.module.numpy_selection(logits, bound=2.0, normalized_gap=0.1, coefficients=(self.np.asarray([0., .5]),))
        self.assertFalse(metadata['comparison_gap_contract_passed'])
        self.assertGreater(metadata['max_normalized_comparison_difference'], 1.0)

    def test_original_embedding_linear_selection_preserves_values(self):
        logits = self.np.asarray([0.9, -0.8, 0.4, -0.5, 0.2], dtype=self.np.float64)
        embedding = self.np.arange(15, dtype=self.np.float64).reshape(5, 3) / 10.0
        original = embedding.copy()
        backend = self.MockBackend(8)
        selection = self.module.select_maximum(backend, self.inputs(backend, logits), 5,
            bound=2.0, normalized_gap=0.01, coefficients=self.coefficients)
        value, layout = self.module.embedding_from_selection(backend, selection, embedding)
        observed = self.unpack(value, layout)
        one_hot, _ = self.module.numpy_selection(logits, bound=2.0, normalized_gap=0.01, coefficients=self.coefficients)
        reference = one_hot @ embedding
        self.assertTrue(self.np.all(self.np.abs(observed-reference) <= 1e-12+1e-12*self.np.abs(reference)))
        self.np.testing.assert_array_equal(embedding, original)

    def test_count_uses_same_operation_kinds_without_access_to_slot_values(self):
        outer = self
        class ObservedMock(self.MockBackend):
            def __init__(self):
                super().__init__(8)
                self.operations = []
            def _apply(self, kind, *args, **kwargs):
                self.operations.append(kind)
                return super()._apply(kind, *args, **kwargs)
        logits = self.np.asarray([.9, -.8, .4, -.5, .2], dtype=self.np.float64)
        mock, count = ObservedMock(), self.CountBackend(8)
        for backend in (mock, count):
            selection = self.module.select_maximum(backend, self.inputs(backend, logits), 5,
                bound=2.0, normalized_gap=.01, coefficients=self.coefficients)
            embedding = self.np.arange(15, dtype=self.np.float64).reshape(5, 3) / 10.0
            self.module.embedding_from_selection(backend, selection, embedding)
        self.assertEqual(dict(Counter(mock.operations)), {key: value for key, value in count.stats()['counts'].items() if value})
        self.assertGreater(count.stats()['max_depth'], 0)
        self.assertGreater(count.stats()['ops_total'], 0)

    def test_two_token_prefix_feeds_back_ciphertext_embedding_before_final_decode(self):
        backend = self.MockBackend(8)
        vectors = [self.np.zeros(8, dtype=self.np.float64) for _ in range(32)]
        vectors[0][0] = 1.0
        backend.bind(vectors)
        blocks = [[backend.input(index)] for index in range(32)]
        state = self.module.EncryptedPrefix(backend, blocks, prompt_length=1, hidden_size=2)
        embedding = self.np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=self.np.float64)
        calls = []
        def forward(prefix):
            calls.append(prefix.length)
            masks = prefix.public_masks()
            self.assertEqual(int(masks['attention_mask'].sum()), prefix.length)
            self.assertEqual(int(masks['sel'].argmax()), prefix.length-1)
            value = prefix.token_blocks[prefix.length-1][0]
            first = self.np.zeros(8); first[0] = 1.0
            second = self.np.zeros(8); second[1] = 1.0
            if prefix.selections:
                return [-value*backend.const(first) + value.rotate(-1)*backend.const(second)]
            return [value*backend.const(first) - value.rotate(-1)*backend.const(second)]
        result = self.module.generate_two(state, forward, embedding, bound=2.0, normalized_gap=.1,
                                         coefficients=(self.np.asarray([0.0, 0.5]),))
        self.assertEqual(calls, [1, 2])
        self.assertEqual(state.length, 3)
        self.assertEqual(result['metadata']['intermediate_decryptions'], 0)
        self.assertEqual(result['metadata']['intermediate_reencryptions'], 0)
        self.assertFalse(result['metadata']['kv_cache_used'])
        self.assertFalse(result['metadata']['original_model_binding_verified'])
        decoded = self.np.stack([self.unpack(row['one_hot'], row['layout']) for row in state.selections])
        self.np.testing.assert_allclose(decoded, [[1.0, 0.0], [0.0, 1.0]], atol=1e-12, rtol=1e-12)
        self.assertEqual(self.module.decode_final_tokens(decoded, eos_token_id=1), [0, 1])
        self.assertEqual(self.module.decode_final_tokens(decoded, eos_token_id=0), [0])
        with self.assertRaises(ValueError):
            self.module.generate_two(state, forward, embedding, bound=2.0, normalized_gap=.1,
                                      coefficients=self.coefficients)

    def test_original_prompt_lengths_are_inferred_without_truncation(self):
        calls = []
        def tokenizer(texts, **kwargs):
            calls.append(kwargs)
            return dict(input_ids=[list(range(len(text.split())+2)) for text in texts])
        texts = ['x '*29, 'x '*28, 'x '*3, 'x '*2]
        original = list(texts)
        chosen = self.module.eligible_prompts(tokenizer, texts, ['a','b','c','d'])
        self.assertEqual(chosen['indices'], [1, 2, 3])
        self.assertEqual(chosen['token_lengths'], [30, 5, 4])
        self.assertEqual(calls, [dict(padding=False, truncation=False)])
        self.assertEqual(texts, original)
        with self.assertRaises(ValueError):
            self.module.eligible_prompts(tokenizer, [texts[0]], ['a'])

    def test_bad_coefficients_contract_or_nonlocal_values_reject_before_arithmetic(self):
        backend = self.MockBackend(8)
        logits = self.np.asarray([.8, -.8], dtype=self.np.float64)
        values = self.inputs(backend, logits)
        for coefficients in ((), ([0.5, 1.0],), ([0., self.np.nan],), ([0., 0.],)):
            with self.assertRaises(ValueError):
                self.module.select_maximum(backend, values, 2, bound=2., normalized_gap=.1, coefficients=coefficients)
        with self.assertRaises(ValueError):
            self.module.select_maximum(backend, values, 2, bound=2., normalized_gap=0., coefficients=self.coefficients)
        with self.assertRaises(ValueError):
            self.module.select_maximum(self.CountBackend(8), values, 2, bound=2., normalized_gap=.1, coefficients=self.coefficients)


if __name__ == '__main__':
    unittest.main()
