"""CPU-only A2 integration: defined partial slots, real arithmetic, leak guard.

Ideal equivalence is NOT HE precision evidence: the production seed coefficient
2**-14 has only about 41.43 amplitude bits at PTscale55.43, and encrypted gaps
carry encoding/rotation/rescale noise multiplied by the nonzero partial gaps.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from cerium_lowering import (AV_PARTIAL_ALGORITHM, AV_PARTIAL_CALLEE,
    AV_PARTIAL_OUTPUT_SEMANTICS, CompactGraph, MultiPrimeLowering,
    compact_av_partial_reference)
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from generate import dense_reference
from packing_invariants import ideal_head_gaps_are_zero
from polynomial_rewrite import reassociate_polynomials
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


def partial_oracle(scores, values, config):
    """Independent scalar indexing, including every gap and final head/row wrap."""
    t, d = config.tokens, config.head_dim
    scores = scores.reshape(t, 2, t)
    values = values.reshape(t, 2, t)
    even, odd = np.zeros((t, 2, d)), np.zeros((t, 2, d))
    for row in range(t):
        for head in range(2):
            for feature in range(d):
                a, b = np.longdouble(0), np.longdouble(0)
                for key in range(t):
                    term = (np.longdouble(scores[row, head, key]) *
                            np.longdouble(values[(row + key) % t, head, feature]))
                    if (key - feature) % 2:
                        b += term
                    else:
                        a += term
                even[row, head, feature], odd[row, head, feature] = a, b
    result = np.empty((t, 2, t))
    for row in range(t):
        for head in range(2):
            next_row, next_head = (row, 1) if head == 0 else ((row + 1) % t, 0)
            for feature in range(d):
                result[row, head, feature] = even[row, head, feature] + odd[row, head, feature]
                result[row, head, d + feature] = (odd[row, head, feature] +
                                                   even[next_row, next_head, feature])
    return result.reshape(-1)


def full_av_oracle(scores, values, config):
    t = config.tokens
    scores, values = (a.reshape(t, 2, t) for a in (scores, values))
    result = np.zeros_like(values)
    for row in range(t):
        for head in range(2):
            for feature in range(t):
                result[row, head, feature] = sum(
                    scores[row, head, key] * values[(row + key) % t, head, feature]
                    for key in range(t))
    return result.reshape(-1)


class PackedInputAvTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)

    def mask(self, config=None):
        config = config or self.config
        return (np.arange(config.slots) % config.tokens < config.head_dim).astype(float)

    def pair(self, graph=None, label='pair'):
        graph = graph or CompactGraph(32, prepared_av=True, av_pack_inputs=True)
        value = graph.input('v') * graph.plain(self.mask())
        partial = graph.compact_av(graph.input('s'), value, config=self.config, label=label)
        return graph, partial

    def test_default_dependency_and_unknown_values_fail_even_for_numeric_zero(self):
        self.assertFalse(CompactGraph(32).av_pack_inputs)
        with self.assertRaisesRegex(ValueError, 'requires prepared_av'):
            CompactGraph(32, av_pack_inputs=True)
        graph = CompactGraph(32, prepared_av=True, av_pack_inputs=True)
        raw = graph.input('v')
        np.testing.assert_array_equal(graph.evaluate([raw], {'v': np.zeros(32)})[0], np.zeros(32))
        with self.assertRaisesRegex(ValueError, 'public proof.*zero value head gaps'):
            graph.compact_av(graph.input('s'), raw, config=self.config)
        self.assertEqual(graph.av_specs, {})
        masked = raw * graph.plain(self.mask())
        partial = graph.compact_av(graph.input('s'), masked, config=self.config)
        self.assertEqual(partial.op, 'compact_av_partial')
        self.assertFalse(ideal_head_gaps_are_zero(partial, self.config))
        self.assertTrue(ideal_head_gaps_are_zero(masked, self.config))
        self.assertTrue(ideal_head_gaps_are_zero(partial * graph.plain(self.mask()), self.config))
        self.assertEqual(graph.av_specs[partial.data], dict(tokens=4, tile=8, head_dim=2, baby=4,
            algorithm='PackedInput2Partial', callee='bert.compact_av_packed_partial',
            output_semantics='live_full_av_gap_B_current_plus_A_next_head_row',
            partial_output=True, client_value_gap_zero_required=True))

    def test_foreign_values_and_scores_still_reject_and_disabled_hook_is_noop(self):
        graph = CompactGraph(32, prepared_av=True, av_pack_inputs=True)
        foreign = CompactGraph(32)
        for scores, value in ((foreign.input('s'), graph.input('v')),
                              (graph.input('s'), foreign.input('v'))):
            with self.assertRaisesRegex(ValueError, 'another graph'):
                graph.compact_av(scores, value, config=self.config)
        disabled = CompactGraph(32)
        self.assertIsNone(disabled.compact_av(disabled.input('s'), disabled.input('v'), config=self.config))

    def test_value_proof_is_rechecked_at_graph_and_export_admission(self):
        graph, partial = self.pair()
        # A low-level IR edit cannot bypass the public hook's proof obligation.
        partial.args = (partial.args[0], graph.input('raw-zero-valued-v'))
        output = partial * graph.plain(self.mask())
        with self.assertRaisesRegex(ValueError, 'public proof.*zero value head gaps'):
            graph.evaluate([output], {'s': np.ones(32), 'raw-zero-valued-v': np.zeros(32)})
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'public proof.*zero value head gaps'):
                lowerer.export([output], target)
            self.assertFalse(target.exists())

    def test_new_mode_keeps_fixed_physical_interval_and_two_prime_products(self):
        for initial, rescales in ((12, 2), (8, 2), (10, 1)):
            graph, partial = self.pair()
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE,
                                         initial_level=initial, rescale_count=rescales)
            with self.assertRaisesRegex(ValueError, 'scoreL10/valueL8'):
                lowerer.lower([partial * graph.plain(self.mask())])
            self.assertEqual(lowerer.prepared_av_calls, [])

    def test_only_new_cyclic_partial_opts_into_masked_reciprocal(self):
        from dataclasses import replace

        class ObserveArithmetic(_Arithmetic):
            def reciprocal(self, denominator, **kwargs):
                calls.append(kwargs.get('zero_head_gaps', False))
                return super().reciprocal(denominator, **kwargs)

        for layout, packed in (('cyclic', False), ('cyclic', True), ('offset', True)):
            config = replace(self.config, attention_layout=layout)
            graph = CompactGraph(32, prepared_av=True, av_pack_inputs=packed)
            calls = []
            arithmetic = ObserveArithmetic(graph, config)
            outputs = arithmetic.attention([graph.input('q')], [graph.input('k')],
                                           [graph.input('v')], 'attention')
            self.assertTrue(calls)
            self.assertEqual(set(calls), {layout == 'cyclic' and packed})
            partials = [n for n in graph.reachable(outputs) if n.op == 'compact_av_partial']
            self.assertEqual(bool(partials), layout == 'cyclic' and packed)
            graph.validate_partial_av_consumers(outputs)

    def test_t16_and_production_every_partial_slot_against_independent_oracle(self):
        rng = np.random.default_rng(901)
        for t in (16, 128):
            with self.subTest(tokens=t):
                config = Config(tokens=t, hidden=2*t, heads=4, tile=2*t, slots=2*t*t,
                                intermediate=4*t)
                scores = rng.normal(size=config.slots)
                values = rng.normal(size=config.slots) * self.mask(config)
                expected = partial_oracle(scores, values, config)
                graph = CompactGraph(config.slots, prepared_av=True, av_pack_inputs=True)
                partial = graph.compact_av(graph.input('s'), graph.input('v') *
                    graph.plain(self.mask(config)), config=config)
                # Capture the actual graph evaluator's full partial, but expose
                # only a cleared context as a graph final (raw finals must fail).
                observed = []

                def capture(*args):
                    result = compact_av_partial_reference(*args)
                    observed.append(result.copy())
                    return result

                with patch('cerium_lowering.compact_av_partial_reference', side_effect=capture):
                    actual, = graph.evaluate([partial * graph.plain(self.mask(config))],
                                             dict(s=scores, v=values))
                np.testing.assert_allclose(observed[0], expected, rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(actual, full_av_oracle(scores, values, config),
                                           rtol=2e-12, atol=2e-12)
                self.assertTrue(np.any(expected[self.mask(config) == 0] != 0))

    def test_upper_half_offsets_end_features_and_both_wraps_are_not_discarded(self):
        config = Config()
        t, d = config.tokens, config.head_dim
        spec = dict(tokens=t, tile=config.tile, head_dim=d)
        for key in (0, 64, 65, 127):
            for feature in (0, 63):
                with self.subTest(key=key, feature=feature):
                    scores = np.zeros((t, 2, t))
                    values = np.zeros_like(scores)
                    # Distinct signed heads on the last and first rows expose
                    # h0->h1, h1->next-row, and final-token wrap independently.
                    for row, head, score, value in ((0, 0, 2., 3.), (0, 1, -5., 7.),
                                                   (127, 0, 11., -13.), (127, 1, -17., -19.)):
                        scores[row, head, key] = score
                        values[(row + key) % t, head, feature] = value
                    actual = compact_av_partial_reference(scores, values, spec).reshape(t, 2, t)
                    live = np.zeros((t, 2))
                    for row in range(t):
                        for head in range(2):
                            live[row, head] = scores[row, head, key] * values[(row + key) % t, head, feature]
                    np.testing.assert_array_equal(actual[:, :, feature], live)
                    expected_gap = live if (key - feature) % 2 else np.roll(live.reshape(-1), -1).reshape(t, 2)
                    np.testing.assert_array_equal(actual[:, :, d + feature], expected_gap)
                    self.assertTrue(np.any(actual[:, :, feature] != 0))
                    actual[:, :, feature] = 0
                    actual[:, :, d + feature] = 0
                    np.testing.assert_array_equal(actual, np.zeros_like(actual))
        dirty = np.zeros(config.slots)
        dirty[d] = 1
        with self.assertRaisesRegex(ValueError, 'zero value head gaps'):
            compact_av_partial_reference(np.ones(config.slots), dirty, spec)

    def test_all_score_denominator_three_masked_updates_context_and_merge(self):
        config = Config(tokens=16, hidden=32, heads=4, tile=32, slots=512, intermediate=64)
        graph = CompactGraph(config.slots, prepared_av=True, av_pack_inputs=True)
        arithmetic = _Arithmetic(graph, config)
        rng = np.random.default_rng(902)
        outputs, contexts, expected_contexts, data = [], [], [], {}
        for group in range(2):
            scores = graph.input('s' + str(group))
            value = graph.input('v' + str(group)) * graph.plain(self.mask(config))
            numeric_scores = rng.uniform(.8, 1., config.slots).reshape(16, 2, 16)
            numeric_scores[:, :, 8:] += .25  # high-half keys must remain in d
            numeric_values = rng.normal(size=config.slots) * self.mask(config)
            data['s' + str(group)] = numeric_scores.reshape(-1)
            data['v' + str(group)] = numeric_values
            denominator = arithmetic.group_row_sum(scores)
            inverse = arithmetic.reciprocal(denominator, zero_head_gaps=True)
            r2, r1 = inverse.args[0], inverse.args[0].args[0]
            partial = arithmetic.cyclic_av(scores, value, label='pair' + str(group))
            context = partial * inverse
            contexts.append(context)
            outputs.extend((denominator, r1, r2, inverse, context))
            d = np.repeat(numeric_scores.sum(axis=2), config.tokens).reshape(-1)
            r = 2/config.tokens - d/config.tokens**2
            expected = [d, r*self.mask(config)]
            for _ in range(2):
                r = r * (2 - d*r)
                expected.append(r*self.mask(config))
            expected_contexts.append(full_av_oracle(numeric_scores.reshape(-1), numeric_values, config)*r)
            expected.append(expected_contexts[-1])
            for node in (r1, r2, inverse, context):
                self.assertTrue(ideal_head_gaps_are_zero(node, config))
            self.assertFalse(ideal_head_gaps_are_zero(partial, config))
            self.assertFalse(ideal_head_gaps_are_zero(inverse.args[1], config))  # gap correction=2
            actual = graph.evaluate(outputs[-5:], data)
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-14)
        merged = arithmetic.merge_heads(contexts)[0]
        expected = expected_contexts[0] + np.roll(expected_contexts[1], config.head_dim)
        np.testing.assert_allclose(graph.evaluate([merged], data)[0], expected, rtol=1e-13, atol=1e-14)
        # The public folded seed is real CP+add arithmetic, no metadata-only mask.
        self.assertEqual(r1.op, 'add')
        self.assertEqual(r1.args[0].op, 'mul')
        self.assertEqual(sum(n.op == 'mul' and all(a.op != 'plain' for a in n.args)
                             for n in graph.reachable([inverse])), 4)

    def test_actual_partial_constant_doubling_negative_and_masked_fix(self):
        graph, partial = self.pair()
        arithmetic = _Arithmetic(graph, self.config)
        denominator = arithmetic.group_row_sum(partial.args[0])
        old_context = partial * arithmetic.reciprocal(denominator)
        good_context = partial * arithmetic.reciprocal(denominator, zero_head_gaps=True)
        data = dict(s=np.ones(32), v=self.mask())
        partial_values = compact_av_partial_reference(data['s'], data['v'], graph.av_specs[partial.data])
        np.testing.assert_array_equal(partial_values, np.full(32, 4.))
        wrong_context = partial_values / 4
        np.testing.assert_array_equal(wrong_context + np.roll(wrong_context, 2), np.full(32, 2.))
        with self.assertRaisesRegex(ValueError, 'unsafe before.*gap clearing'):
            graph.evaluate([arithmetic.merge_heads([old_context, old_context])[0]], data)
        np.testing.assert_array_equal(graph.evaluate([
            arithmetic.merge_heads([good_context, good_context])[0]], data)[0], np.ones(32))

    def test_every_unsafe_fanout_rejects_graph_lowering_and_export_before_writes(self):
        cases = ('raw-final', 'boot-final', 'unmasked', 'shared', 'rotate', 'dense',
                 'neg', 'add', 'sub', 'self-dependent', 'boot-shared')
        for case in cases:
            with self.subTest(case=case):
                graph, partial = self.pair()
                safe = partial * graph.plain(self.mask())
                if case == 'raw-final':
                    outputs = [partial]
                elif case == 'boot-final':
                    outputs = [graph.bootstrap(partial, label='identity-hint')]
                elif case == 'unmasked':
                    outputs = [partial * graph.input('unknown-inverse')]
                elif case == 'shared':
                    outputs = [safe, partial * graph.input('unknown-inverse')]
                elif case == 'rotate':
                    outputs = [partial.rotate(1) * graph.plain(self.mask())]
                elif case == 'dense':
                    outputs = graph.compact_linear([partial], np.eye(8), np.zeros(8),
                                                    config=self.config, label='dense')
                elif case == 'neg':
                    outputs = [-partial * graph.plain(self.mask())]
                elif case == 'add':
                    outputs = [(partial + 0) * graph.plain(self.mask())]
                elif case == 'sub':
                    outputs = [(partial - 0) * graph.plain(self.mask())]
                elif case == 'self-dependent':
                    outputs = [partial * safe]  # proof true, but NOT separate!
                else:
                    hint = graph.bootstrap(partial, label='identity-hint')
                    outputs = [hint * graph.plain(self.mask()), hint.rotate(1)]
                with self.assertRaisesRegex(ValueError, 'partial AV'):
                    graph.evaluate(outputs, dict(s=np.ones(32), v=np.ones(32)))
                lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
                with self.assertRaisesRegex(ValueError, 'partial AV'):
                    lowerer.lower(outputs)
                self.assertEqual(lowerer.execution, [])
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / 'must-not-exist'
                    with self.assertRaisesRegex(ValueError, 'partial AV'):
                        lowerer.export(outputs, target)
                    self.assertFalse(target.exists())

    def test_safe_all_fanouts_boot_hints_dead_uses_and_postclear_mixing(self):
        graph, partial = self.pair()
        mask = graph.plain(self.mask())
        hint = graph.bootstrap(partial, label='identity-hint')
        first, second = partial * mask, hint * graph.plain(self.mask()*.5)
        outputs = [first.rotate(1) + second]  # Mixing AFTER the clear is allowed.
        graph.bootstrap(partial, label='dead-final')
        partial.rotate(1) * graph.input('dead-inverse')
        count = len(graph.nodes)
        reachable = [n.index for n in graph.reachable(outputs)]
        graph.validate_partial_av_consumers(outputs)
        self.assertEqual(count, len(graph.nodes))
        self.assertEqual(reachable, [n.index for n in graph.reachable(outputs)])
        graph.evaluate(outputs, dict(s=np.ones(32), v=np.ones(32)))
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            stats = lowerer.export(outputs, target)
            plan = json.loads((target / 'model.runtime-plan.json').read_bytes())
        self.assertEqual(stats['prepared_av_calls'], 1)
        self.assertEqual(len(plan['final_outputs']), 1)
        self.assertEqual(len(plan['external_inputs']), 2)
        self.assertNotIn('dead-inverse', [n.data for n in graph.reachable(outputs) if n.op == 'input'])
        self.assertTrue(all(lowerer.lowered[n.index].id not in plan['final_outputs']
                            for n in (partial, hint)))
        check_arithmetic(self, lowerer)

    def test_entirely_dead_partial_does_not_add_calls_or_finals(self):
        graph, partial = self.pair()
        partial.rotate(1)  # Unsafe but unreachable, as is the partial itself.
        output = graph.input('only-live-input')
        lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'plan'
            stats = lowerer.export([output], target)
            self.assertFalse((target / 'prepared-av.json').exists())
        self.assertNotIn(partial.index, lowerer.lowered)
        self.assertEqual((stats['input_count'], stats['output_count']), (1, 1))

    def test_both_rewrites_clone_flags_registry_partial_and_consumer_logic(self):
        for safe in (False, True):
            graph, partial = self.pair()
            outputs = [graph.bootstrap(partial, label='identity-hint') * graph.plain(self.mask())]
            if not safe:
                outputs.append(partial.rotate(1))
            for rewrite in (lambda g, o: rewrite_compact_graph(g, o, tokens=4), reassociate_polynomials):
                source = graph
                graph, outputs, _ = rewrite(graph, outputs)
                self.assertTrue(graph.av_pack_inputs)
                self.assertTrue(graph.use_prepared_av)
                self.assertEqual(graph.av_specs, source.av_specs)
                self.assertIsNot(graph.av_specs, source.av_specs)
                self.assertIsNot(graph.av_specs['pair'], source.av_specs['pair'])
                self.assertEqual(sum(n.op == 'compact_av_partial' for n in graph.reachable(outputs)), 1)
                if safe:
                    graph.validate_partial_av_consumers(outputs)
                else:
                    with self.assertRaisesRegex(ValueError, 'partial AV'):
                        graph.validate_partial_av_consumers(outputs)

    def test_two_full_ssa_layers_dense_oracle_both_rewrites_and_qk_versions(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24, layers=2)
        weights = create_weights(config, 904)
        hidden = np.random.default_rng(905).normal(0, .2, (4, 8))
        data = dict(x=pack_inputs(hidden, config)[0])
        oracle = pack_inputs(dense_reference(hidden, weights, config, 'encoder'), config)
        for qk_packed in (False, True):
            ordinary_counts = []
            for av_packed in (False, True):
                graph = CompactGraph(32, prepared_qk=True, prepared_av=True,
                                     qk_pack_inputs=qk_packed, av_pack_inputs=av_packed)
                outputs = build_model(graph, [graph.input('x')], weights, config, boot_policy='stage')
                for rewrite in (lambda g, o: rewrite_compact_graph(g, o, tokens=4), reassociate_polynomials):
                    graph, outputs, _ = rewrite(graph, outputs)
                    self.assertEqual(graph.av_pack_inputs, av_packed)
                    self.assertEqual(graph.qk_pack_inputs, qk_packed)
                    np.testing.assert_allclose(graph.evaluate(outputs, data), oracle, rtol=2e-12, atol=2e-12)
                partials = [n for n in graph.reachable(outputs) if n.op == 'compact_av_partial']
                self.assertEqual(len(partials), 4 if av_packed else 0)
                for partial in partials:
                    self.assertFalse(ideal_head_gaps_are_zero(partial, config))
                    self.assertTrue(ideal_head_gaps_are_zero(partial.args[1], config))
                lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=12,
                                             normalize_shared_rotations=True)
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / 'plan'
                    stats = lowerer.export(outputs, target, resident=True)
                    manifest = json.loads((target / 'prepared-av.json').read_bytes())
                    qk = json.loads((target / 'prepared-qk.json').read_bytes())
                    plan = json.loads((target / 'model.runtime-plan.json').read_bytes())
                    boundaries = json.loads((target / 'layer-boundaries.json').read_bytes())
                ordinary_counts.append(Counter(row['op'] for row in lowerer.execution
                    if row['kind'] == 'compute'))
                self.assertEqual(manifest['format_version'], 2 if av_packed else 1)
                self.assertEqual(qk['format_version'], 2 if qk_packed else 1)
                self.assertEqual(stats['prepared_av_calls'], 4)
                self.assertEqual(stats['prepared_av_arithmetic']['cc_products'], 8 if av_packed else 16)
                self.assertEqual(stats['final_levels'], [12])
                self.assertTrue(boundaries['layers'][1]['consumes_previous_boundary_directly'])
                self.assertTrue(all(row['boundary_matches_input'] for row in boundaries['layers']))
                self.assertEqual(len(plan['external_inputs']), 1)
                self.assertEqual(len(plan['final_outputs']), 1)
                for call in manifest['calls']:
                    self.assertEqual((call['score_input_level'], call['value_input_level'],
                                      call['common_level'], call['output_level']), (10, 8, 6, 4))
                av_callees = [row['attrs']['callee'] for row in lowerer.execution
                    if row.get('op') == 'ApiCall' and 'compact_av' in row['attrs']['callee']]
                self.assertEqual(av_callees, [AV_PARTIAL_CALLEE if av_packed else 'bert.compact_av']*4)
                check_arithmetic(self, lowerer)
            # The masked seed does not create fictitious depth/work savings or extra Boot.
            self.assertEqual(ordinary_counts[0], ordinary_counts[1])

    def test_production_55_real_keys_counts_canonical_levels_and_v2_metadata(self):
        from generate_aligned import candidate
        config = Config()
        context, profile, _ = candidate('q52-p11')
        manifests = []
        for packed in (False, True):
            graph = CompactGraph(config.slots, prepared_av=True, av_pack_inputs=packed)
            mask = graph.plain(self.mask(config))
            partial = graph.compact_av(graph.input('s'), graph.input('v') * mask, config=config)
            output = partial * mask if packed else partial
            lowerer = MultiPrimeLowering(graph, context, profile, initial_level=10)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'plan'
                stats = lowerer.export([output], target)
                manifests.append(json.loads((target / 'prepared-av.json').read_bytes()))
                plan = json.loads((target / 'model.runtime-plan.json').read_bytes())
            check_arithmetic(self, lowerer)
            call, = manifests[-1]['calls']
            self.assertEqual(plan['format_version'], 7)
            self.assertEqual((call['score_input_level'], call['value_input_level'],
                              call['common_level'], call['output_level']), (10, 8, 6, 4))
            self.assertAlmostEqual(call['common_scale_log2'], lowerer.scales[6])
            self.assertAlmostEqual(call['output_scale_log2'], lowerer.scales[4])
            if packed:
                self.assertEqual(stats['prepared_av_arithmetic'], dict(cc_products=64,
                    relinearizations=64, plaintext_products=319, rescale_operations=193,
                    rescale2_operations=65, rescale4_operations=128, rescale_prime_drops=642,
                    rotations=296, additions=319, hoisted_rotation_batches=19,
                    removed_score_alignment_products=127))
                self.assertEqual(call['algorithm'], AV_PARTIAL_ALGORITHM)
                self.assertEqual(call['callee'], AV_PARTIAL_CALLEE)
                self.assertEqual(call['output_semantics'], AV_PARTIAL_OUTPUT_SEMANTICS)
                self.assertIs(call['partial_output'], True)
                self.assertIs(call['client_value_gap_zero_required'], True)
                # Clearing is a real multiplication/rescale, not output metadata.
                self.assertEqual(stats['final_levels'], [2])
                self.assertEqual(stats['instruction_counts']['mul_cp'], 2)  # V mask, output clear
                self.assertEqual(stats['instruction_counts']['rescale'], 2)
            else:
                self.assertEqual(stats['prepared_av_arithmetic'], dict(cc_products=128,
                    relinearizations=128, plaintext_products=383, rescale_operations=257,
                    rescale2_operations=129, rescale4_operations=128, rescale_prime_drops=770,
                    removed_score_alignment_products=127))
        baseline, packed = manifests
        self.assertEqual((baseline['format_version'], packed['format_version']), (1, 2))
        old_steps = set()
        for b in range(1, 8):
            old_steps.update((256*b, b))
        old_steps.update(b-128 for b in range(8))
        for g in range(1, 16):
            old_steps.update((2048*g, -2040*g))
        normalize = lambda step: (step % 32768) - (32768 if step % 32768 > 16384 else 0)
        old_steps = {normalize(step) for step in old_steps}
        self.assertEqual(set(baseline['rotation_steps']), old_steps)
        self.assertEqual(len(old_steps), 52)
        self.assertEqual(set(packed['rotation_steps']), old_steps | {-64, 64, 192})
        self.assertEqual(len(packed['rotation_steps']), 55)

    def test_mixed_manifest_v2_only_when_a_partial_is_reachable(self):
        graph = CompactGraph(32, prepared_av=True)
        mask = graph.plain(self.mask())
        baseline = graph.compact_av(graph.input('s0'), graph.input('v0') * mask,
                                    config=self.config, label='baseline')
        graph.av_pack_inputs = True
        partial = graph.compact_av(graph.input('s1'), graph.input('v1') * mask,
                                   config=self.config, label='partial')
        partial.rotate(1)  # Dead illegal debug use must not leak into either export.
        for outputs, version, count in (([baseline], 1, 1), ([baseline, partial*mask], 2, 2)):
            lowerer = MultiPrimeLowering(graph, CONTEXT, PROFILE, initial_level=10)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'plan'
                lowerer.export(outputs, target)
                manifest = json.loads((target / 'prepared-av.json').read_bytes())
            self.assertEqual(manifest['format_version'], version)
            self.assertEqual(len(manifest['calls']), count)
            self.assertTrue(all(key not in manifest['calls'][0] for key in
                ('algorithm', 'callee', 'output_semantics', 'partial_output', 'client_value_gap_zero_required')))


if __name__ == '__main__':
    unittest.main()
