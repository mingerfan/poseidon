"""CPU-only integration of opt-in pre-CC QK packing, not GPU evidence."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from generate import dense_reference
from generate_aligned import main, make_parser
from packing_invariants import ideal_head_gaps_are_zero
from polynomial_rewrite import reassociate_polynomials
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class PackedInputQkTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)

    def masked_pair(self, graph):
        mask = np.tile([1., 1., 0., 0.], 8)
        return tuple(graph.input(name) * graph.plain(mask) for name in ('q', 'k'))

    def test_defaults_dependency_and_unproven_raw_operands_fail_closed(self):
        self.assertFalse(CompactGraph(32).qk_pack_inputs)
        self.assertFalse(make_parser().parse_args(['--output-dir', 'unused']).qk_pack_inputs)
        with self.assertRaisesRegex(ValueError, 'requires prepared_qk'):
            CompactGraph(32, qk_pack_inputs=True)
        graph = CompactGraph(32, prepared_qk=True, qk_pack_inputs=True)
        q, k = self.masked_pair(graph)
        # Raw inputs remain unknown even if a caller happens to supply zeros.
        for query, key in ((graph.input('raw_q'), k), (q, graph.input('raw_k'))):
            with self.assertRaisesRegex(ValueError, 'public proof.*query and key'):
                graph.compact_qk(query, key, config=self.config)
        self.assertEqual(graph.qk_specs, {})
        output = graph.compact_qk(q, k, config=self.config)
        self.assertTrue(graph.qk_specs[output.data]['pack_inputs'])
        result = graph.evaluate([output], dict(q=np.zeros(32), k=np.zeros(32)))[0]
        np.testing.assert_array_equal(result, np.zeros(32))

    def test_cli_rejects_missing_dependency_and_wrong_physical_contract_early(self):
        cases = [([], 'requires --prepared-qk'),
                 (['--prepared-qk', '--scope', 'ffn-up'], 'cyclic selective'),
                 (['--prepared-qk', '--attention-layout', 'offset'], 'cyclic selective'),
                 (['--prepared-qk', '--strategy', 'budget'], 'cyclic selective'),
                 (['--prepared-qk', '--candidate', 'q49-p13'], 'cyclic selective'),
                 (['--prepared-qk', '--input-level', '10'], 'input L12')]
        with tempfile.TemporaryDirectory() as directory:
            for flags, error in cases:
                with self.subTest(flags=flags), patch('generate_aligned.create_weights') as weights:
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                        main(['--output-dir', str(Path(directory) / 'absent'), '--qk-pack-inputs'] + flags)
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn(error, stderr.getvalue())
                    weights.assert_not_called()

    def test_masked_pair_matches_primitive_baseline_and_independent_index_oracle(self):
        rng = np.random.default_rng(830)
        data = {name: rng.normal(size=32) for name in ('q', 'k')}
        oracle = np.empty((4, 2, 4))
        q, k = (data[name].reshape(4, 2, 4) for name in ('q', 'k'))
        for token in range(4):
            for head in range(2):
                for offset in range(4):
                    oracle[token, head, offset] = sum(
                        q[token, head, feature] * k[(token + offset) % 4, head, feature]
                        for feature in range(2)) / np.sqrt(2)
        for prepared, packed in ((False, False), (True, False), (True, True)):
            graph = CompactGraph(32, prepared_qk=prepared, qk_pack_inputs=packed)
            query, key = self.masked_pair(graph)
            output = _Arithmetic(graph, self.config).cyclic_qk(query, key)
            np.testing.assert_allclose(graph.evaluate([output], data)[0], oracle.reshape(-1),
                                       rtol=1e-13, atol=1e-13)

    def test_two_real_ssa_layers_and_both_rewrites_preserve_every_mode_and_math(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24, layers=2)
        weights = create_weights(config, 83)
        hidden = np.random.default_rng(84).normal(0, .2, (4, 8))
        data = dict(x=pack_inputs(hidden, config)[0])
        oracle = pack_inputs(dense_reference(hidden, weights, config, 'encoder'), config)
        for packed in (False, True):
            graph = CompactGraph(32, prepared_qk=True, qk_pack_inputs=packed)
            outputs = build_model(graph, [graph.input('x')], weights, config, boot_policy='stage')
            original = {name: dict(spec) for name, spec in graph.qk_specs.items()}
            for rewrite in (lambda g, o: rewrite_compact_graph(g, o, tokens=4), reassociate_polynomials):
                graph, outputs, _ = rewrite(graph, outputs)
                self.assertEqual(graph.qk_pack_inputs, packed)
                self.assertEqual(graph.qk_specs, original)
                self.assertIsNot(graph.qk_specs, original)
            pairs = [n for n in graph.reachable(outputs) if n.op == 'compact_qk']
            self.assertEqual(len(pairs), 4)
            self.assertTrue(all(ideal_head_gaps_are_zero(a, config) for n in pairs for a in n.args))
            self.assertTrue(all(spec.get('pack_inputs', False) == packed for spec in graph.qk_specs.values()))
            self.assertTrue(all(('pack_inputs' in spec) == packed for spec in graph.qk_specs.values()))
            np.testing.assert_allclose(graph.evaluate(outputs, data), oracle, rtol=2e-12, atol=2e-12)
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                         normalize_shared_rotations=True)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'plan'
                stats = lowerer.export(outputs, target, resident=True)
                qk = json.loads((target / 'prepared-qk.json').read_bytes())
                boundaries = json.loads((target / 'layer-boundaries.json').read_bytes())
            self.assertTrue(boundaries['layers'][1]['consumes_previous_boundary_directly'])
            self.assertTrue(all(row['boundary_matches_input'] for row in boundaries['layers']))
            self.assertEqual(stats['final_levels'], [12])
            self.assertEqual(stats['prepared_qk_calls'], 4)
            self.assertEqual(qk['format_version'], 2 if packed else 1)
            self.assertTrue(all(c.get('pack_inputs', False) == packed for c in qk['calls']))
            self.assertTrue(all(c['input_level'] - c['output_level'] == 4 for c in qk['calls']))
            self.assertEqual(stats['prepared_qk_arithmetic']['cc_products'], 8 if packed else 16)
            self.assertEqual(stats['prepared_qk_arithmetic']['plaintext_products'], 16)
            self.assertEqual(stats['prepared_qk_arithmetic']['rescale2_operations'], 12 if packed else 20)
            self.assertEqual('qk_pack_inputs' in stats, packed)
            check_arithmetic(self, lowerer)

    def test_manifest_version_is_a_nonbaseline_capability_gate(self):
        for fields, version in (({}, 1), ({'pack_inputs': False, 'pair_products': False}, 1),
                                ({'pack_inputs': True}, 2), ({'pair_products': True}, 2)):
            with self.subTest(fields=fields):
                graph = CompactGraph(32, prepared_qk=True, qk_pack_inputs=fields.get('pack_inputs', False))
                output = graph.compact_qk(*self.masked_pair(graph), config=self.config)
                graph.qk_specs[output.data].update(fields)
                lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12)
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / 'plan'
                    lowerer.export([output], target)
                    manifest = json.loads((target / 'prepared-qk.json').read_bytes())
                self.assertEqual(manifest['format_version'], version)

    def test_full_shape_single_pair_key_union_counters_and_unchanged_four_prime_depth(self):
        from generate_aligned import candidate
        config = Config()
        context, profile, _ = candidate('q52-p11')
        graph = CompactGraph(config.slots, prepared_qk=True, qk_pack_inputs=True)
        mask = graph.plain(np.tile(np.r_[np.ones(64), np.zeros(64)], 256))
        output = graph.compact_qk(graph.input('q') * mask, graph.input('k') * mask, config=config)
        lowerer = MultiPrimeLowering(graph, context, profile, initial_level=10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            stats = lowerer.export([output], target)
            manifest = json.loads((target / 'prepared-qk.json').read_bytes())
        self.assertEqual(manifest['format_version'], 2)
        self.assertEqual(len(manifest['rotation_steps']), 53)
        self.assertTrue({-64, 63, 192} <= set(manifest['rotation_steps']))
        self.assertEqual(stats['prepared_qk_arithmetic'], dict(cc_products=64, relinearizations=64,
            plaintext_products=128, rescale2_operations=65, rescale_prime_drops=130,
            rotations=531, additions=513, hoisted_rotation_batches=2))
        call, = manifest['calls']
        self.assertEqual((call['input_level'], call['output_level']), (8, 4))
        self.assertAlmostEqual(call['intermediate_scale_log2'], lowerer.scales[6])
        self.assertTrue(call['pack_inputs'])
        with self.assertRaisesRegex(ValueError, 'two-prime'):
            MultiPrimeLowering(graph, context, profile, initial_level=12, rescale_count=1).lower([output])


if __name__ == '__main__':
    unittest.main()
