import unittest
import warnings
import numpy as np

from pt_agent.analyzer import calibrate, approximation_values, reference_forward
from pt_agent.intake.graph import Graph, Node, TensorInfo


def graph():
    return Graph('masked_attention', [Node('s', 'softmax', ['scores'], ['out'],
        dict(axis=-1, keep='causal', mask='mask'), '')],
        dict(scores=TensorInfo('scores', (1, 2, 2), 'act'),
             mask=TensorInfo('mask', (1, 2), 'mask'),
             out=TensorInfo('out', (1, 2, 2), 'act')), {}, ['scores', 'mask'], ['out'])


class StableCalibrationTests(unittest.TestCase):
    def test_large_scores_and_masked_large_entries_are_finite(self):
        model = graph()
        arrays = dict(scores=np.array([[[1000., 1e10], [998., 999.]]]), mask=np.array([[1., 0.]]))
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = calibrate(model, arrays, batch=1)
        inv = result['out::inv']
        cfg = inv['normalization']
        expected = np.exp((np.array([1000., 998.])-cfg['shift'])*2.**-cfg['steps'])/cfg['inv_scale']
        np.testing.assert_allclose([inv['min'], inv['max']], [expected.min(), expected.max()], rtol=1e-12, atol=1e-12)
        self.assertEqual(result['out::exp']['max'], 1000.)

    def test_global_kernel_normalization_does_not_depend_on_batch_partition(self):
        model = graph()
        arrays = dict(scores=np.array([[[1000., 998.], [990., 999.]],
                                      [[2., 1.], [0., 3.]]]), mask=np.ones((2, 2)))
        first, second = (calibrate(model, arrays, batch=n) for n in (1, 2))
        self.assertEqual(first['out::inv'], second['out::inv'])
        self.assertEqual(first['out::exp'], second['out::exp'])

    def test_existing_exact_reference_observation_keeps_its_raw_semantics(self):
        model = graph()
        arrays = dict(scores=np.array([[[1., 2.], [3., 4.]]]), mask=np.ones((1, 2)))
        values = reference_forward(model, arrays, keep=True)
        measured = approximation_values(model, values)['out::inv']
        np.testing.assert_allclose(measured, [[np.exp(1.), np.exp(3.)+np.exp(4.)]], rtol=1e-12, atol=1e-12)

    def test_calibrated_inverse_matches_real_packed_kernel_observer(self):
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels import transformer as tr
        from pt_agent.kernels.layout import TokenLayout, FlatLayout
        model = graph()
        arrays = dict(scores=np.array([[[1000., 998.], [990., 999.]]]), mask=np.ones((1, 2)))
        inv = calibrate(model, arrays, batch=1)['out::inv']
        cfg = inv['normalization']
        backend = StreamingMockBackend(16, batch=1)
        bound = []
        for row in (arrays['scores'][0, 0], arrays['scores'][0, 1], arrays['mask'][0]):
            data = np.zeros((1, 16)); data[0, :2] = row; bound.append(data)
        backend.bind(bound)
        seen = []
        def observe(role, value, layout, attrs, mask):
            if role == 'inv':
                seen.extend(value.value[:, 0].tolist())
        try:
            tr.softmax(backend, [backend.input(0), backend.input(1)], TokenLayout(16, 2, 2),
                dict(fn='exp', method='cheb', deg=7, domain=(-8., 8.)),
                dict(fn='inv', method='cheb', deg=7, domain=(1e-8, 32.)),
                keep='causal', mask=[backend.input(2)], mask_layout=FlatLayout(16, 2),
                shift=cfg['shift'], steps=cfg['steps'], inv_scale=cfg['inv_scale'],
                step_inv=dict(fn='inv', method='cheb', deg=7, domain=(1e-8, 32.)),
                exact=True, observe=observe)
        finally:
            backend.release_batch()
        self.assertEqual(len(seen), 2)
        np.testing.assert_allclose([min(seen), max(seen)], [inv['min'], inv['max']], rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
