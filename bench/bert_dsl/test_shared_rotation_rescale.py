"""Physical schedule checks for optional shared rotation-frontier rescales."""
import unittest
import numpy as np

from backend import Graph
from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, build_model, create_weights
from equivalent_rewrite import rewrite_compact_graph
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class SharedRotationRescaleTests(unittest.TestCase):
    def lower(self, graph, outputs, enabled, initial=13):
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=initial,
                                     normalize_shared_rotations=enabled)
        result = lowerer.lower(outputs)
        check_arithmetic(self, lowerer)
        return lowerer, result

    def rotate_input_levels(self, lowerer):
        desc = {v['id']: v for v in lowerer.values}
        return [desc[r['inputs'][0]]['level'] for r in lowerer.execution
                if r.get('op') == 'rotate']

    def test_one_rescale_is_moved_before_reused_rotations(self):
        graph = Graph(32)
        x = graph.input('x')
        source = x * graph.plain(np.full(32, .25))
        outputs = [source.rotate(1) + source.rotate(2)]
        old, old_outputs = self.lower(graph, outputs, False)
        new, new_outputs = self.lower(graph, outputs, True)
        self.assertEqual(self.rotate_input_levels(old), [13, 13])
        self.assertEqual(self.rotate_input_levels(new), [11, 11])
        self.assertEqual(old_outputs[0].level, new_outputs[0].level)
        self.assertEqual(old_outputs[0].scale, new_outputs[0].scale)
        self.assertEqual(sum(r.get('op') == 'rescale' for r in old.execution), 1)
        self.assertEqual(sum(r.get('op') == 'rescale' for r in new.execution), 1)
        self.assertEqual(len(new.shared_rotation_frontiers), 1)

    def test_dead_uses_do_not_create_a_shared_frontier(self):
        graph = Graph(32)
        source = graph.input('x') * graph.plain(np.full(32, .25))
        output = source.rotate(1)
        source.rotate(2)  # unreachable: must not affect scheduling decisions
        new, _ = self.lower(graph, [output], True)
        self.assertEqual(self.rotate_input_levels(new), [13])
        self.assertEqual(new.shared_rotation_frontiers, [])

    def test_existing_binary_mask_rescale_is_not_recredited(self):
        graph = Graph(32)
        source = graph.input('x') * graph.plain(np.ones(8))
        outputs = [source.rotate(1)]
        for enabled in (False, True):
            lowerer, _ = self.lower(graph, outputs, enabled)
            self.assertEqual(self.rotate_input_levels(lowerer), [11])
            self.assertEqual(lowerer.shared_rotation_frontiers, [])

    def test_tiny_two_layer_boundaries_and_real_scale_equations(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32,
                        intermediate=24, layers=2)
        graph = CompactGraph(32)
        outputs = build_model(graph, [graph.input('x')], create_weights(config, 88), config)
        graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
        for enabled in (False, True):
            lowerer, result = self.lower(graph, outputs, enabled, initial=12)
            self.assertEqual([r.level for r in result], [12])
            self.assertEqual(len(lowerer.prepared_calls), 16)
            self.assertTrue(all(v['level'] >= lowerer.floor for v in lowerer.values
                                if v['kind'] == 'ciphertext'))


if __name__ == '__main__':
    unittest.main()
