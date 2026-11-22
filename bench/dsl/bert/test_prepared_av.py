"""CPU math and physical-spec contract tests; AV GPU remains separately gated."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from polynomial_rewrite import reassociate_polynomials
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class PreparedAvTests(unittest.TestCase):
    def test_pair_matches_direct_all_offsets_all_lanes_and_original_bsgs(self):
        config = Config(tokens=16, hidden=32, heads=4, tile=32, slots=512, intermediate=64)
        rng = np.random.default_rng(28)
        scores = rng.normal(size=(16, 2, 16))
        values = rng.normal(size=(16, 2, 16))  # dirty gaps test full arithmetic, not input validation
        expected = np.empty_like(values)
        for token in range(16):
            for head in range(2):
                for feature in range(16):
                    expected[token, head, feature] = sum(scores[token, head, offset] *
                        values[(token + offset) % 16, head, feature] for offset in range(16))
        for prepared in (False, True):
            graph = CompactGraph(512, prepared_av=prepared)
            output = _Arithmetic(graph, config).cyclic_av(graph.input('s'), graph.input('v'), label='pair')
            actual = graph.evaluate([output], dict(s=scores.reshape(-1), v=values.reshape(-1)))[0]
            np.testing.assert_allclose(actual, expected.reshape(-1), rtol=1e-12, atol=1e-12)
            self.assertEqual(output.op == 'compact_av', prepared)

    def test_two_layer_qk_av_and_polynomial_hooks_preserve_math_and_boundaries(self):
        config = Config(tokens=4, hidden=24, heads=12, tile=8, slots=32, intermediate=48, layers=2)
        weights = create_weights(config, 81)
        packed = pack_inputs(np.random.default_rng(48).normal(0, .2, (4, 24)), config)
        data = {str(i): value for i, value in enumerate(packed)}
        actual = []
        for av in (False, True):
            graph = CompactGraph(32, prepared_qk=True, prepared_av=av)
            outputs = build_model(graph, [graph.input(str(i)) for i in range(3)], weights, config)
            graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
            graph, outputs, _ = reassociate_polynomials(graph, outputs)
            actual.append(graph.evaluate(outputs, data))
            if not av:
                continue
            self.assertTrue(graph.use_prepared_av)
            self.assertEqual(len(graph.av_specs), 12)
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                         normalize_shared_rotations=True)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'plan'
                schedule = lowerer.export(outputs, target, resident=True)
                manifest = json.loads((target / 'prepared-av.json').read_bytes())
                boundaries = json.loads((target / 'layer-boundaries.json').read_bytes())
            self.assertEqual(manifest['format_version'], 1)
            self.assertFalse(graph.av_pack_inputs)
            self.assertEqual(schedule['prepared_av_calls'], 12)
            self.assertEqual(schedule['prepared_av_arithmetic']['cc_products'], 48)
            self.assertEqual(len(manifest['calls']), 12)
            for row in manifest['calls']:
                self.assertTrue(all(key not in row for key in ('algorithm', 'callee',
                    'output_semantics', 'partial_output', 'client_value_gap_zero_required')))
                self.assertEqual((row['score_input_level'], row['value_input_level'],
                                  row['common_level'], row['output_level']), (10, 8, 6, 4))
            self.assertEqual(schedule['final_levels'], [12, 12, 12])
            self.assertTrue(boundaries['layers'][1]['consumes_previous_boundary_directly'])
            check_arithmetic(self, lowerer)
        np.testing.assert_allclose(actual[0], actual[1], rtol=1e-11, atol=1e-12)

    def test_fixed_physical_interval_is_not_metadata_coerced(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        graph = CompactGraph(32, prepared_av=True)
        scores, value = graph.input('s'), graph.input('v')
        output = graph.compact_av(scores, value, config=config, label='invalid-level')
        with self.assertRaisesRegex(ValueError, 'scoreL10/valueL8'):
            MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10).lower([output])
        graph = CompactGraph(32, prepared_av=True)
        scores, value = graph.input('s'), graph.input('v')
        output = graph.compact_av(scores, value * graph.plain(np.full(32, .5)),
                                  config=config, label='valid')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
        result = lowerer.lower([output])
        self.assertEqual(result[0].level, 4)
        self.assertEqual(lowerer.prepared_av_calls[0]['value_input_level'], 8)
        check_arithmetic(self, lowerer)

    def test_foreign_inputs_rejected_and_disabled_hook_is_noop(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        graph = CompactGraph(32, prepared_av=True)
        with self.assertRaises(ValueError):
            graph.compact_av(graph.input('s'), CompactGraph(32).input('v'), config=config)
        disabled = CompactGraph(32)
        self.assertIsNone(disabled.compact_av(disabled.input('s'), disabled.input('v'), config=config))


if __name__ == '__main__':
    unittest.main()
