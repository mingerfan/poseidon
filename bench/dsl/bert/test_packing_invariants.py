"""Ideal layout proofs are CPU-only and never inspect encrypted values."""
import unittest
import numpy as np

from cerium_lowering import CompactGraph
from cerium_packing import Config, _Arithmetic, build_model, create_weights
from equivalent_rewrite import rewrite_compact_graph
from packing_invariants import ideal_head_gaps_are_zero
from polynomial_rewrite import reassociate_polynomials


class PackingInvariantTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        self.graph = CompactGraph(32)
        self.raw = self.graph.input('x')
        self.even, self.odd = _Arithmetic(self.graph, self.config).expand_heads([self.raw])

    def test_masks_establish_but_raw_inputs_do_not_claim_zero_gaps(self):
        self.assertFalse(ideal_head_gaps_are_zero(self.raw, self.config))
        for expr in (self.even, self.odd):
            self.assertTrue(ideal_head_gaps_are_zero(expr, self.config))
        self.assertTrue(ideal_head_gaps_are_zero(self.graph.plain(np.ones(2)), self.config))
        self.assertFalse(ideal_head_gaps_are_zero(self.graph.plain(np.ones(3)), self.config))

    def test_slotwise_operations_and_only_gap_preserving_rotations(self):
        for expr in (-self.even, self.even + self.odd, self.even - self.odd,
                     self.raw * self.even, self.even.rotate(4), self.even.rotate(8)):
            self.assertTrue(ideal_head_gaps_are_zero(expr, self.config))
        self.assertFalse(ideal_head_gaps_are_zero(self.even + self.graph.plain(np.ones(32)), self.config))
        self.assertFalse(ideal_head_gaps_are_zero(self.even.rotate(2), self.config))

    def test_boot_is_only_an_ideal_identity_and_proof_does_not_mutate_graph(self):
        expr = self.graph.bootstrap(self.even, label='semantic-identity-not-noise-zero')
        before = len(self.graph.nodes), set(self.graph.constants)
        self.assertTrue(ideal_head_gaps_are_zero(expr, self.config))
        self.assertEqual(before, (len(self.graph.nodes), set(self.graph.constants)))

    def test_actual_model_qk_inputs_have_provable_gaps_after_rewrites(self):
        graph = CompactGraph(32, prepared_qk=True, prepared_av=True)
        outputs = build_model(graph, [graph.input('x')], create_weights(self.config, 72), self.config)
        graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        graph, outputs, _ = reassociate_polynomials(graph, outputs)
        pairs = [node for node in graph.reachable(outputs) if node.op == 'compact_qk']
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            for argument in pair.args:
                self.assertTrue(ideal_head_gaps_are_zero(argument, self.config))
            self.assertFalse(ideal_head_gaps_are_zero(pair, self.config))
        av = [node for node in graph.reachable(outputs) if node.op == 'compact_av']
        self.assertTrue(all(ideal_head_gaps_are_zero(node, self.config) for node in av))

    def test_foreign_child_is_rejected(self):
        foreign = CompactGraph(32).input('y')
        malformed = self.graph.node('neg', (foreign,))
        with self.assertRaisesRegex(ValueError, 'foreign graph'):
            ideal_head_gaps_are_zero(malformed, self.config)


if __name__ == '__main__':
    unittest.main()
