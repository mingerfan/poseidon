"""Small synthetic regressions for the calibration/analytical bridge only."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.margin import minimum_domain
from pt_agent.planners import stable_softmax_bridge as bridge


def graph():
    return SimpleNamespace(name='attention', nodes=[SimpleNamespace(id='s', op='softmax',
        inputs=['x', 'mask'], outputs=['p'], attrs={'keep': 'causal', 'mask': 'mask'})],
        tensors={'x': SimpleNamespace(shape=(1, 3, 3))})


class SoftmaxBridgeProtocolCases(unittest.TestCase):
    def test_positive_negative_and_crossing_intervals_retain_minimum_domain(self):
        for lo, hi in ((1., 3.), (-3., -1.), (-10., 1010.), (1000., 1001.), (-1001., -1000.)):
            with self.subTest(interval=(lo, hi)):
                policy = bridge.derive_policy({'min': lo, 'max': hi}, 32)
                self.assertEqual(tuple(policy['minimum_raw_domain']), minimum_domain(lo, hi))
                transformed = tuple((x-policy['shift'])*2.**-policy['steps'] for x in minimum_domain(lo, hi))
                self.assertEqual(tuple(policy['exp_domain']), transformed)
                actual = tuple((x-policy['shift'])*2.**-policy['steps'] for x in (lo, hi))
                self.assertLessEqual(policy['exp_domain'][0], actual[0])
                self.assertGreaterEqual(policy['exp_domain'][1], actual[1])

    def test_inverse_interval_includes_exponential_error_contract(self):
        policy = bridge.derive_policy({'min': -10., 'max': 1010.}, 32)
        low, high = policy['proof']['relative_exp_error_scaled_initial_sum']
        self.assertLess(policy['inv_domain'][0], low)
        self.assertGreater(policy['inv_domain'][1], high)
        bound = policy['proof']['relative_inverse_error_square_sum']
        self.assertLessEqual(policy['step_inv_domain'][0], bound[0])
        self.assertGreaterEqual(policy['step_inv_domain'][1], bound[1])
        self.assertFalse(policy['proof']['continuous_polynomial_error_certificate'])

    def test_unused_raw_inverse_is_preserved_without_blocking_required_points(self):
        issue = {'point': 'p::inv', 'reason': 'raw overflow'}
        record = bridge.build_bridge(graph(), {'p::exp': {'min': -2., 'max': 3.}}, raw_issues=[issue])
        self.assertTrue(record['required_original_points_ready'])
        self.assertEqual(record['required_original_points'], ['p::exp'])
        self.assertEqual(record['raw_issues'], [issue])
        self.assertFalse(record['raw_unshifted_inverse_used'])
        self.assertFalse(record['candidate_frozen'])

    def test_missing_required_point_and_constant_domain_fail_explicitly(self):
        with self.assertRaises(ValueError):
            bridge.build_bridge(graph(), {})
        with self.assertRaises(ValueError):
            bridge.derive_policy({'min': 0., 'max': 0.}, 3)

    def test_existing_planner_parameter_binding_restored_after_failure(self):
        calibration = {'p::exp': {'min': -2., 'max': 3.}}
        record = bridge.build_bridge(graph(), calibration)
        record['actual_score_coverage_measured'] = True
        original = bridge.transformer_rule.squared_softmax_adjustments

        def fail(*args, **kwargs):
            parameters = bridge.transformer_rule.squared_softmax_adjustments(calibration['p::exp'], 3)
            self.assertEqual(parameters['inv_domain'], record['rows'][0]['policy']['inv_domain'])
            raise RuntimeError('synthetic planner failure')

        with mock.patch.object(bridge.transformer_rule, 'plan', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic planner failure'):
                bridge.plan_with_bridge(graph(), calibration, record)
        self.assertIs(bridge.transformer_rule.squared_softmax_adjustments, original)

    def test_saved_bridge_roundtrip_retains_source_policy(self):
        calibration = {'p::exp': {'min': -2., 'max': 3.}}
        record = bridge.build_bridge(graph(), calibration)
        record['actual_score_coverage_measured'] = True
        saved = json.loads(json.dumps(record))

        def plan(*args, **kwargs):
            policy = bridge.transformer_rule.squared_softmax_adjustments(calibration['p::exp'], 3)
            self.assertEqual(tuple(policy['inv_domain']), tuple(record['rows'][0]['policy']['inv_domain']))
            return 'source-matched program'

        with mock.patch.object(bridge.transformer_rule, 'plan', side_effect=plan) as planned:
            self.assertEqual(bridge.plan_with_bridge(graph(), calibration, saved), 'source-matched program')
            planned.assert_called_once()
            saved['rows'][0]['policy']['inv_domain'][0] *= .5
            with self.assertRaises(ValueError):
                bridge.plan_with_bridge(graph(), calibration, saved)
            self.assertEqual(planned.call_count, 1)


class SoftmaxCalibrationProtocolCases(unittest.TestCase):
    def run_calibration(self, *, fail_at=None):
        from pt_agent.experiments import qwen_stable_calibration as worker
        from pt_agent.experiments import qwen_reference_worker as reference
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        args = SimpleNamespace(phase='calibration', work_dir=root, model_info=root/'model',
            reference=root/'original-reference.json', framework_result=root/'framework-result.json')
        previous = {'calibration': str(root/'old-calibration.json'), 'calibration_issues': [{'point': 'p::inv'}]}
        source = ({}, previous, {'original_hf_source': str(root/'original-hf')},
                  {'p::exp': {'min': -2., 'max': 3.}})
        result = dict(mapped_forward_completed=0, stable_softmax_attempts=0,
                      stable_softmax_completed=0, candidate_frozen=False)
        recorder = SimpleNamespace(span=lambda *a, **k: nullcontext())

        def observed(scores, mask, row):
            if result['stable_softmax_attempts'] == fail_at:
                raise ValueError('synthetic coverage escape')
            return None, dict(node_id=row['node_id'], passed=True)

        def mapped(selected, stage_recorder, output, np, torch):
            self.assertEqual(selected.phase, 'framework-reference')
            self.assertEqual(selected.local_model, root/'original-hf')
            for index in range(6):
                selected._observation_callback(graph(), {'x': 'same saved scores', 'mask': 'same saved mask'},
                    {'sample_id': 'train:'+str(index//2)}, {'step': index%2}, {'passed': True})
                output['mapped_forward_completed'] += 1
            output['calibration_complete'] = False

        self.enterContext(mock.patch.object(worker, '_source', return_value=source))
        self.enterContext(mock.patch('pt_agent.models.mapped_info.load_mapped_info',
                                    return_value=SimpleNamespace(graph=graph())))
        self.enterContext(mock.patch.object(bridge, 'observe_softmax', side_effect=observed))
        self.enterContext(mock.patch.object(reference, 'mapped_reference', side_effect=mapped))
        return worker, args, recorder, result

    def test_six_state_callback_and_selected_scope_survive_raw_issue(self):
        worker, args, recorder, result = self.run_calibration()
        worker.calibrate(args, recorder, result, None, None)
        self.assertEqual((result['mapped_forward_completed'], result['stable_softmax_attempts'],
                          result['stable_softmax_completed']), (6, 6, 6))
        self.assertTrue(result['selected_decomposition_calibration_complete'])
        self.assertFalse(result['raw_calibration_complete'])
        saved = json.loads(Path(result['bridge_path']).read_text())
        self.assertTrue(saved['actual_score_coverage_measured'])
        self.assertFalse(saved['exp_fit_error_validated'])
        self.assertFalse(saved['candidate_frozen'])
        self.assertEqual(saved['raw_issues'], [{'point': 'p::inv'}])

    def test_coverage_failure_preserves_partial_counts_and_active_state(self):
        worker, args, recorder, result = self.run_calibration(fail_at=3)
        with self.assertRaisesRegex(ValueError, 'synthetic coverage escape'):
            worker.calibrate(args, recorder, result, None, None)
        self.assertEqual((result['mapped_forward_completed'], result['stable_softmax_attempts'],
                          result['stable_softmax_completed']), (2, 3, 2))
        self.assertEqual(result['active_stable_softmax'],
            dict(node_id='s', sample_id='train:1', step=0, completed=False))
        self.assertNotIn('selected_decomposition_calibration_complete', result)
        self.assertFalse((args.work_dir/'stable-bridge.json').exists())


class SoftmaxBridgeMathCases(unittest.TestCase):
    def test_existing_packed_kernel_uses_the_same_stable_arithmetic(self):
        import numpy as np
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels.layout import ScoreLayout, FlatLayout, pack_sequence, pack_flat, unpack_sequence
        from pt_agent.kernels.transformer import softmax
        g = graph()
        g.tensors['x'].shape = (1, 2, 3, 3)
        record = bridge.build_bridge(g, {'p::exp': {'min': -10., 'max': 1010.}})
        x = np.array([[[[1010., 1009., 1008.], [-10., -9., -8.], [1., 2., 3.]],
                        [[-.25, .5, 1.], [3., 2., 1.], [.1, .2, .3]]]], dtype=np.float64)
        mask = np.array([[1., 1., 0.]], dtype=np.float64)
        expected, _ = bridge.observe_softmax(x, mask, record['rows'][0])
        policy = record['rows'][0]['policy']
        layout, mask_layout = ScoreLayout(16, 3, 2, 3), FlatLayout(16, 3)
        packed = pack_sequence(x[0], layout)
        backend = StreamingMockBackend(16)
        backend.bind(packed+[pack_flat(mask[0], mask_layout)])
        inputs = [backend.input(i) for i in range(4)]
        output, out_layout = softmax(backend, inputs[:3], layout,
            {'fn': 'exp', 'domain': policy['exp_domain']}, {'fn': 'inv', 'domain': policy['inv_domain']},
            keep='causal', mask=[inputs[3]], mask_layout=mask_layout, shift=policy['shift'],
            inv_scale=policy['inv_scale'], steps=policy['steps'],
            step_inv={'fn': 'inv', 'domain': policy['step_inv_domain']}, exact=True)
        observed = unpack_sequence([item.value for item in output], out_layout)[None, ...]
        self.assertTrue(np.all(np.abs(observed-expected) <= 1e-12+1e-12*np.abs(expected)))

    def test_real_operator_shape_mask_and_independent_float64_arithmetic(self):
        import numpy as np
        calibration = {'p::exp': {'min': -10., 'max': 1010.}}
        record = bridge.build_bridge(graph(), calibration)
        x = np.array([[[1010., 1009., 1008.], [-10., -9., -8.], [1., 2., 3.]]], dtype=np.float64)
        before = x.copy()
        _, result = bridge.observe_softmax(x, np.array([[1., 1., 0.]], dtype=np.float64), record['rows'][0])
        self.assertTrue(result['passed'])
        self.assertTrue(all(row['passed'] for row in result['domains']))
        self.assertTrue(np.array_equal(x, before))
        self.assertEqual((result['effective_key_min'], result['effective_key_max']), (1, 2))
        self.assertFalse(result['original_framework_float32_reference_replaced'])
        self.assertFalse(result['approximation_fit_or_quality_measured'])

    def test_domain_escape_and_all_masked_rows_remain_failures(self):
        import numpy as np
        record = bridge.build_bridge(graph(), {'p::exp': {'min': -1., 'max': 1.}})
        with self.assertRaises((ValueError, FloatingPointError)):
            bridge.observe_softmax(np.full((1, 3, 3), 1000., dtype=np.float64),
                                   np.ones((1, 3), dtype=np.float64), record['rows'][0])
        with self.assertRaises(ValueError):
            bridge.observe_softmax(np.zeros((1, 3, 3), dtype=np.float64),
                                   np.zeros((1, 3), dtype=np.float64), record['rows'][0])


if __name__ == '__main__':
    unittest.main()
