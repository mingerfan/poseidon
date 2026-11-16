"""CPU mathematical and physical regression tests; never invoke a GPU."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from backend import Graph
from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class EquivalentRewriteTests(unittest.TestCase):
    def test_full_slot_units_but_not_prefix_masks(self):
        graph = Graph(32)
        x = graph.input('x')
        minus = graph.plain(-np.ones(32))
        prefix = graph.plain(np.ones(8))
        outputs = [x * graph.plain(np.ones(32)), x * minus, x * prefix]
        target, rewritten, stats = rewrite_compact_graph(graph, outputs, tokens=4)
        self.assertEqual([x.op for x in rewritten], ['input', 'neg', 'mul'])
        self.assertEqual(stats['rewrite_counts']['full_slot_mul_one_removed'], 1)
        data = {'x': np.arange(32, dtype=np.float64) - 7}
        np.testing.assert_array_equal(graph.evaluate(outputs, data), target.evaluate(rewritten, data))
        self.assertTrue(np.all(target.evaluate([rewritten[2]], data)[0][8:] == 0))

    def test_minus_one_is_negate_without_consuming_level(self):
        graph = Graph(32)
        x = graph.input('x')
        outputs = [x * graph.plain(-np.ones(32))]
        target, rewritten, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        lowerer = MultiPrimeLowering(target, CONTEXT, PROFILE, initial_level=13)
        result = lowerer.lower(rewritten)[0]
        self.assertEqual(result.level, 13)
        self.assertEqual([r['op'] for r in lowerer.execution if r['kind'] == 'compute'], ['negate'])
        check_arithmetic(self, lowerer)

    def test_pending_product_is_still_really_rescaled(self):
        graph = Graph(32)
        x = graph.input('x')
        outputs = [(x * graph.plain(np.full(32, .25))) * graph.plain(-np.ones(32))]
        target, rewritten, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        lowerer = MultiPrimeLowering(target, CONTEXT, PROFILE, initial_level=13)
        result = lowerer.lower(rewritten)[0]
        ops = [r['op'] for r in lowerer.execution if r['kind'] == 'compute']
        self.assertEqual(result.level, 11)
        self.assertEqual(ops.count('mul_cp'), 1)
        self.assertEqual(ops.count('rescale'), 1)
        self.assertIn('negate', ops)
        check_arithmetic(self, lowerer)

    def test_reciprocal_has_same_three_newton_steps(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        graph = Graph(32)
        x = graph.input('denominator')
        output = _Arithmetic(graph, config).reciprocal(x)
        target, rewritten, stats = rewrite_compact_graph(graph, [output], tokens=4)
        self.assertEqual(stats['rewrite_counts']['reciprocal_first_steps_folded'], 1)
        self.assertEqual(stats['rewrite_counts']['full_slot_mul_minus_one_to_negate'], 2)
        # Folding the PUBLIC first step must preserve all four CC products
        # in the remaining two Newton iterations.
        def cc_products(g, ys):
            return sum(n.op == 'mul' and all(a.op != 'plain' for a in n.args)
                       for n in g.reachable(ys))
        self.assertEqual(cc_products(graph, [output]), 4)
        self.assertEqual(cc_products(target, rewritten), 4)
        data = {'denominator': np.linspace(2.5, 5.5, 32)}
        np.testing.assert_allclose(graph.evaluate([output], data), target.evaluate(rewritten, data),
                                   rtol=1e-14, atol=1e-14)

    def test_non_newton_coefficient_tree_is_not_distributed(self):
        graph = Graph(32)
        x = graph.input('x')
        output = (x * graph.plain(np.full(32, -.3)) + graph.plain(np.full(32, 2.))) * graph.plain(np.full(32, .25))
        target, rewritten, stats = rewrite_compact_graph(graph, [output], tokens=4)
        self.assertEqual(stats['rewrite_counts'].get('reciprocal_first_steps_folded', 0), 0)
        self.assertEqual(len(target.reachable(rewritten)), len(graph.reachable([output])))

    def test_two_layers_independent_weights_and_polynomial_oracle(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24, layers=2)
        graph = CompactGraph(config.slots)
        inputs = [graph.input(str(i)) for i in range(config.hidden_tiles)]
        weights = create_weights(config, 77)
        outputs = build_model(graph, inputs, weights, config, boot_policy='stage')
        target, rewritten, stats = rewrite_compact_graph(graph, outputs, tokens=config.tokens)
        values = pack_inputs(np.random.default_rng(4).normal(0, .2, (4, 8)), config)
        data = {str(i): v for i, v in enumerate(values)}
        np.testing.assert_allclose(graph.evaluate(outputs, data), target.evaluate(rewritten, data),
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(len(target.matrices), 12)
        for label in graph.matrices:
            for key in ('weight', 'bias'):
                np.testing.assert_array_equal(graph.matrices[label][key], target.matrices[label][key])
                self.assertFalse(np.shares_memory(graph.matrices[label][key], target.matrices[label][key]))
        self.assertFalse(stats['approximation_changed'])
        lowerer = MultiPrimeLowering(target, CONTEXT, PROFILE, initial_level=12)
        with tempfile.TemporaryDirectory() as temp:
            schedule = lowerer.export(rewritten, Path(temp) / 'plan', resident=True)
        self.assertEqual(schedule['final_levels'], [12])
        self.assertEqual(schedule['prepared_linear_calls'], 16)
        check_arithmetic(self, lowerer)

    def test_original_graph_is_not_mutated_and_disabled_is_equivalent(self):
        graph = Graph(32)
        x = graph.input('x')
        output = x * graph.plain(-np.ones(32))
        before = len(graph.nodes)
        target, rewritten, stats = rewrite_compact_graph(graph, [output], tokens=4,
                                                        scalar_units=False, reciprocal_seed=False)
        self.assertEqual(len(graph.nodes), before)
        self.assertEqual(rewritten[0].op, 'mul')
        self.assertEqual(stats['rewrite_counts'], {})
        self.assertIsNot(target, graph)

    def test_invalid_tokens_and_foreign_outputs_are_rejected(self):
        graph = Graph(32)
        for value in (0, 3, True):
            with self.assertRaises(ValueError):
                rewrite_compact_graph(graph, [], tokens=value)
        with self.assertRaises(ValueError):
            rewrite_compact_graph(graph, [Graph(32).input('foreign')], tokens=4)


if __name__ == '__main__':
    unittest.main()
