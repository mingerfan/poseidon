"""A2's public masked seed; not integrated AV or GPU correctness evidence."""
from collections import Counter
from dataclasses import replace
import unittest
import numpy as np

from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic
from equivalent_rewrite import rewrite_compact_graph
from packing_invariants import ideal_head_gaps_are_zero
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class MaskedReciprocalTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        self.live = np.arange(32) % 4 < 2

    def test_three_updates_preserve_original_live_polynomial_and_prove_gaps(self):
        graph = CompactGraph(32)
        d = graph.input('d')
        arithmetic = _Arithmetic(graph, self.config)
        old = arithmetic.reciprocal(d)
        new = arithmetic.reciprocal(d, zero_head_gaps=True)
        rng = np.random.default_rng(157)
        expected, actual = graph.evaluate([old, new], {'d': rng.uniform(3, 5, 32)})
        np.testing.assert_allclose(actual, expected*self.live, rtol=1e-15, atol=1e-16)
        self.assertTrue(ideal_head_gaps_are_zero(new, self.config))
        self.assertFalse(ideal_head_gaps_are_zero(new.args[1], self.config))  # correction has gap2
        self.assertEqual(sum(n.op == 'mul' and all(x.op != 'plain' for x in n.args)
                             for n in graph.reachable([new])), 4)

    def test_same_real_prime_depth_and_arithmetic_as_folded_baseline(self):
        results = []
        for masked in (False, True):
            graph = CompactGraph(32)
            output = _Arithmetic(graph, self.config).reciprocal(graph.input('d'), zero_head_gaps=masked)
            graph, outputs, stats = rewrite_compact_graph(graph, [output], tokens=4)
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=18)
            final = lowerer.lower(outputs)[0]
            check_arithmetic(self, lowerer)
            counts = Counter(row.get('op') for row in lowerer.execution if row['kind'] == 'compute')
            self.assertEqual(counts['boot'], 0)
            self.assertEqual(counts['mul_cc'], 4)
            self.assertEqual(final.level, 8)
            results.append((counts, final.level, final.scale))
        self.assertEqual(results[0], results[1])

    def test_constant_partial_av_counterexample_is_fixed_before_merge(self):
        graph = CompactGraph(32)
        arithmetic = _Arithmetic(graph, self.config)
        d, partial = graph.input('d'), graph.input('partial')
        ordinary_inverse = arithmetic.reciprocal(d)
        masked_inverse = arithmetic.reciprocal(d, zero_head_gaps=True)
        bad_context = partial*ordinary_inverse
        context = partial*masked_inverse
        bad = arithmetic.merge_heads([bad_context, bad_context])[0]
        good = arithmetic.merge_heads([context, context])[0]
        wrong, actual = graph.evaluate([bad, good], {'d': np.full(32, 4.), 'partial': np.full(32, 4.)})
        np.testing.assert_array_equal(wrong, np.full(32, 2.))
        np.testing.assert_array_equal(actual, np.ones(32))
        self.assertFalse(ideal_head_gaps_are_zero(partial, self.config))
        self.assertTrue(ideal_head_gaps_are_zero(context, self.config))

    def test_masked_seed_is_not_an_offset_attention_rewrite(self):
        graph = CompactGraph(32)
        arithmetic = _Arithmetic(graph, replace(self.config, attention_layout='offset'))
        with self.assertRaisesRegex(ValueError, 'cyclic two-head gap'):
            arithmetic.reciprocal(graph.input('d'), zero_head_gaps=True)


if __name__ == '__main__':
    unittest.main()
