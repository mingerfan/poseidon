"""Boundary auditing must not change final outputs, liveness or math."""
import json
from pathlib import Path
import tempfile
import unittest

from backend import Graph
from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, build_model, create_weights
from equivalent_rewrite import rewrite_compact_graph
from test_cerium_lowering import CONTEXT, PROFILE


class LayerObservabilityTests(unittest.TestCase):
    def test_two_layer_physical_boundaries_and_cross_layer_inputs(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24, layers=2)
        graph = CompactGraph(32)
        outputs = build_model(graph, [graph.input('x')], create_weights(config, 5), config)
        graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                     normalize_shared_rotations=True)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            lowerer.export(outputs, target)
            plan = json.loads((target / 'model.runtime-plan.json').read_text())
            boundaries = json.loads((target / 'layer-boundaries.json').read_text())
        self.assertEqual(len(boundaries['layers']), 2)
        self.assertEqual(len(plan['external_inputs']), 1)
        self.assertEqual(len(plan['final_outputs']), 1)
        self.assertFalse(boundaries['timing_measured'])
        self.assertFalse(boundaries['ciphertext_snapshots_retained'])
        layers = boundaries['layers']
        self.assertIsNone(layers[0]['consumes_previous_boundary_directly'])
        self.assertTrue(layers[1]['consumes_previous_boundary_directly'])
        self.assertTrue(all(r['boundary_matches_input'] for r in layers))
        self.assertEqual(layers[1]['values'][0]['value_id'], plan['final_outputs'][0])
        self.assertLess(layers[0]['end_ordinal'], layers[1]['end_ordinal'])
        self.assertEqual(sum(r['planned_native_boot_calls'] for r in layers),
                         sum(r.get('op') == 'boot' for r in plan['execution']))
        released = {r['value'] for r in plan['execution'] if r['kind'] == 'release'}
        self.assertIn(layers[0]['values'][0]['value_id'], released)

    def test_three_ct_groups_preserve_cross_layer_order_and_release(self):
        config = Config(tokens=4, hidden=24, heads=12, tile=8, slots=32, intermediate=48, layers=2)
        graph = CompactGraph(32)
        inputs = [graph.input('x' + str(group)) for group in range(3)]
        outputs = build_model(graph, inputs, create_weights(config, 11), config)
        graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                     normalize_shared_rotations=True)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            lowerer.export(outputs, target)
            plan = json.loads((target / 'model.runtime-plan.json').read_bytes())
            boundaries = json.loads((target / 'layer-boundaries.json').read_bytes())
        previous, last = boundaries['layers']
        previous_values = [v['value_id'] for v in previous['values']]
        self.assertEqual([v['group'] for v in previous['values']], [0, 1, 2])
        self.assertEqual(last['input_device_values'], previous_values)
        self.assertTrue(last['consumes_previous_boundary_directly'])
        self.assertEqual(plan['final_outputs'], [v['value_id'] for v in last['values']])
        released = {r['value'] for r in plan['execution'] if r['kind'] == 'release'}
        self.assertTrue(all(value in released for value in previous_values))
        self.assertTrue(previous['boundary_matches_input'] and last['boundary_matches_input'])

    def test_non_encoder_graph_has_no_fake_layer_boundaries(self):
        graph = Graph(32)
        x = graph.input('x')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            lowerer.export([x * x], target)
            boundaries = json.loads((target / 'layer-boundaries.json').read_text())
        self.assertEqual(boundaries['layers'], [])


if __name__ == '__main__':
    unittest.main()
