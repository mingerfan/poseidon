"""CPU contract/math/physical checks for opt-in prepared cyclic QK."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class PreparedQkTests(unittest.TestCase):
    def test_pair_hook_matches_bsgs_and_direct_cyclic_oracle(self):
        config = Config(tokens=16, hidden=32, heads=4, tile=32, slots=512, intermediate=64)
        rng = np.random.default_rng(15)
        # Dirty gap lanes deliberately stress that only the first head_dim
        # features in each T-lane block participate in the dot product.
        q = rng.normal(size=(16, 2, 16))
        k = rng.normal(size=(16, 2, 16))
        oracle = np.empty((16, 2, 16))
        for t in range(16):
            for head in range(2):
                for offset in range(16):
                    oracle[t, head, offset] = np.dot(q[t, head, :8], k[(t + offset) % 16, head, :8]) / np.sqrt(8)
        results = []
        for prepared in (False, True):
            graph = CompactGraph(config.slots, prepared_qk=prepared)
            query, key = graph.input('q'), graph.input('k')
            output = _Arithmetic(graph, config).cyclic_qk(query, key, label='pair')
            results.append(graph.evaluate([output], dict(q=q.reshape(-1), k=k.reshape(-1)))[0])
            self.assertEqual(output.op == 'compact_qk', prepared)
        np.testing.assert_allclose(results[0], oracle.reshape(-1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(results[1], oracle.reshape(-1), rtol=1e-12, atol=1e-12)

    def test_two_layers_hooks_rewrites_and_physical_contracts(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24, layers=2)
        weights = create_weights(config, 41)
        data = pack_inputs(np.random.default_rng(7).normal(0, .2, (4, 8)), config)
        results = []
        for prepared in (False, True):
            graph = CompactGraph(32, prepared_qk=prepared)
            outputs = build_model(graph, [graph.input('x')], weights, config)
            graph, outputs, _ = rewrite_compact_graph(graph, outputs, tokens=4)
            results.append(graph.evaluate(outputs, dict(x=data[0])))
            if not prepared:
                continue
            self.assertTrue(graph.use_prepared_qk)
            self.assertEqual(len(graph.qk_specs), 4)
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                         normalize_shared_rotations=True)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'plan'
                stats = lowerer.export(outputs, target, resident=True)
                qk = json.loads((target / 'prepared-qk.json').read_text())
                plan = json.loads((target / 'model.runtime-plan.json').read_text())
            self.assertEqual(stats['prepared_qk_calls'], 4)
            self.assertEqual(stats['prepared_qk_arithmetic']['cc_products'], 16)
            self.assertEqual(len(qk['calls']), 4)
            self.assertTrue(all(c['input_level'] - c['output_level'] == 4 for c in qk['calls']))
            self.assertEqual(sum(r.get('attrs', {}).get('callee') == 'bert.compact_qk'
                                 for r in plan['execution']), 4)
            self.assertEqual(stats['final_levels'], [12])
            self.assertTrue(all(-16 < step <= 16 for step in qk['rotation_steps']))
            check_arithmetic(self, lowerer)
        np.testing.assert_allclose(results[0], results[1], rtol=1e-12, atol=1e-12)

    def test_short_budget_is_refreshed_before_opaque_four_prime_interval(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        graph = CompactGraph(32, prepared_qk=True)
        output = graph.compact_qk(graph.input('q'), graph.input('k'), config=config, label='pair')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=5)
        result = lowerer.lower([output])[0]
        self.assertEqual(lowerer.boot_labels['qk-budget-frontier'], 2)
        self.assertGreaterEqual(result.level, lowerer.floor)
        self.assertEqual(lowerer.prepared_qk_calls[0]['input_level'], 18)
        check_arithmetic(self, lowerer)

    def test_wrong_layout_and_foreign_inputs_are_rejected(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        graph = CompactGraph(32, prepared_qk=True)
        with self.assertRaises(ValueError):
            graph.compact_qk(graph.input('q'), CompactGraph(32).input('k'), config=config)
        with self.assertRaises(ValueError):
            MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12, rescale_count=1).lower([
                graph.compact_qk(graph.input('q2'), graph.input('k2'), config=config, label='pair')])


if __name__ == '__main__':
    unittest.main()
