"""CPU regression tests for honest multi-prime physical scheduling."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from backend import Graph, coefficient_moduli
from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, build_model, create_weights, pack_inputs


CONTEXT = dict(poly_degree=64, q_moduli_bits=[28] * 49,
               p_moduli_bits=[28] * 25, default_scale_log2=28)
PROFILE = dict(profile_id='compact-test', input_level_min=2,
    input_level_max=48, input_scale_log2_min=40, input_scale_log2_max=120,
    output_level=20, output_scale_log2=56)


def check_arithmetic(test, lowerer):
    descriptions = {x['id']: x for x in lowerer.values}
    for row in lowerer.execution:
        if row['kind'] != 'compute':
            continue
        args = [descriptions[x] for x in row['inputs']]
        out = descriptions[row['output']]
        op = row['op']
        if op in ('mul_cp', 'mul_cc'):
            test.assertEqual(args[0]['level'], args[1]['level'])
            test.assertAlmostEqual(out['scale_log2'], sum(a['scale_log2'] for a in args), places=9)
            test.assertEqual(out['level'], args[0]['level'])
        elif op == 'rescale':
            src = args[0]
            drop = sum(math.log2(q) for q in lowerer.q[out['level'] + 1:src['level'] + 1])
            test.assertLess(out['level'], src['level'])
            test.assertAlmostEqual(out['scale_log2'], src['scale_log2'] - drop, places=9)
        elif op in ('add_cc', 'sub_cc', 'add_cp', 'sub_cp'):
            test.assertEqual(args[0]['level'], args[1]['level'])
            test.assertAlmostEqual(args[0]['scale_log2'], args[1]['scale_log2'], places=9)
        elif op in ('rotate', 'negate', 'relinearize'):
            test.assertEqual(out['level'], args[0]['level'])
            test.assertEqual(out['scale_log2'], args[0]['scale_log2'])
        elif op == 'boot':
            test.assertGreaterEqual(args[0]['level'], 2)
            test.assertGreaterEqual(args[0]['scale_log2'], 40)
            test.assertEqual(out['level'], PROFILE['output_level'])


class CompactLoweringTests(unittest.TestCase):
    def test_graph_ring_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'physical CKKS ring'):
            MultiPrimeLowering(Graph(16), CONTEXT, PROFILE)

    def test_two_prime_product_uses_actual_primes(self):
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=13)
        result = lowerer.lower([x * x])[0]
        self.assertEqual(result.level, 11)
        self.assertAlmostEqual(result.scale, 2 * lowerer.scales[13]
                               - math.log2(lowerer.q[13]) - math.log2(lowerer.q[12]))
        check_arithmetic(self, lowerer)

    def test_alignment_is_real_arithmetic(self):
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=13)
        lowerer.lower([(x * x + x) * x])
        self.assertGreater(sum(row.get('op') == 'mul_cp' for row in lowerer.execution), 0)
        check_arithmetic(self, lowerer)

    def test_budget_refresh_precedes_illegal_two_prime_product(self):
        graph = Graph(32)
        x = graph.input('x')
        y = x
        for _ in range(10):
            y = y * x
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=5)
        lowerer.lower([y])
        self.assertGreater(lowerer.boot_labels['budget-exhausted'], 0)
        check_arithmetic(self, lowerer)

    def test_explicit_q_chain(self):
        context = dict(CONTEXT)
        q = coefficient_moduli(64, [28] * 80)[:49]
        context['q_moduli'] = q
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, context, PROFILE)
        lowerer.lower([x * x])
        self.assertEqual(lowerer.q, q)
        check_arithmetic(self, lowerer)

    def test_prepared_graph_matches_primitive_graph(self):
        config = Config(tokens=4, hidden=12, heads=3, tile=4, slots=16,
                        intermediate=20, bsgs_baby_step=3, attention_layout='offset')
        weights = create_weights(config, 33)
        data = np.random.default_rng(2).normal(size=(4, 12)) * 0.1
        packed = pack_inputs(data, config)
        results = []
        for kind in (Graph, CompactGraph):
            graph = kind(config.slots)
            inputs = [graph.input(str(i)) for i in range(config.hidden_tiles)]
            outputs = build_model(graph, inputs, weights, config, boot_policy='stage')
            results.append(graph.evaluate(outputs, {str(i): p for i, p in enumerate(packed)}))
        np.testing.assert_allclose(results[0], results[1], rtol=1e-10, atol=1e-10)

    def test_full_tiny_physical_schedule_and_registry(self):
        config = Config(tokens=4, hidden=12, heads=3, tile=4, slots=16,
                        intermediate=20, bsgs_baby_step=3, attention_layout='offset')
        graph = CompactGraph(config.slots)
        inputs = [graph.input(str(i)) for i in range(config.hidden_tiles)]
        outputs = build_model(graph, inputs, create_weights(config, 33), config)
        context = dict(CONTEXT, poly_degree=2 * config.slots)
        lowerer = MultiPrimeLowering(graph, context, PROFILE, initial_level=13)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'plan'
            stats = lowerer.export(outputs, target, resident=True)
            plan = json.loads((target / 'model.runtime-plan.json').read_text())
            manifest = json.loads((target / 'compact-linear.json').read_text())
            self.assertEqual(plan['format_version'], 7)
            self.assertEqual(len(manifest['matrices']), 6)
            self.assertEqual(len(manifest['calls']), 20)
            self.assertEqual(stats['physical_rescale_count'], 2)
            self.assertEqual(len(plan['external_inputs']), 3)
            self.assertEqual(len(plan['final_outputs']), 3)
            self.assertEqual(stats['encoder_boundary_level'], 13)
            self.assertEqual(stats['final_levels'], [13, 13, 13])
            self.assertTrue(all(c['input_level'] - c['output_level'] == 2 for c in manifest['calls']))
        check_arithmetic(self, lowerer)

    def test_prepared_only_linear_does_not_declare_unused_bundle(self):
        config = Config(tokens=4, hidden=12, heads=3, tile=4, slots=16,
                        intermediate=20, attention_layout='offset')
        graph = CompactGraph(16)
        inputs = [graph.input(str(i)) for i in range(3)]
        outputs = graph.compact_linear(inputs, np.ones((12, 12)), np.zeros(12),
                                       config=config, label='q')
        context = dict(CONTEXT, poly_degree=2 * config.slots)
        lowerer = MultiPrimeLowering(graph, context, PROFILE, initial_level=13)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'plan'
            stats = lowerer.export(outputs, target)
            plan = json.loads((target / 'model.runtime-plan.json').read_text())
            self.assertNotIn('plaintext_bundle', plan)
            self.assertEqual(stats['prepared_linear_arithmetic']['rescale_operations'], 3)
            self.assertEqual(stats['prepared_linear_arithmetic']['rescale_prime_drops'], 6)

    def test_plain_graph_export_is_supported(self):
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=13)
        with tempfile.TemporaryDirectory() as temp:
            lowerer.export([x * x], Path(temp) / 'plan')

    def test_matrix_label_reuse_rejected(self):
        config = Config(tokens=4, hidden=12, heads=3, tile=4, slots=16, intermediate=20, attention_layout='offset')
        graph = CompactGraph(16)
        inputs = [graph.input(str(i)) for i in range(3)]
        weight = np.ones((12, 12), dtype=np.float32)
        graph.compact_linear(inputs, weight, np.zeros(12), config=config, label='q')
        with self.assertRaises(ValueError):
            graph.compact_linear(inputs, weight, np.zeros(12), config=config, label='q')

    def test_single_prime_remains_supported(self):
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, rescale_count=1,
                                    working_scale_log2=28, initial_level=13)
        self.assertEqual(lowerer.lower([x * x])[0].level, 12)
        check_arithmetic(self, lowerer)


if __name__ == '__main__':
    unittest.main()
