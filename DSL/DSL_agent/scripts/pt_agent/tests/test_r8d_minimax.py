import copy
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial import Chebyshev
import torch

from pt_agent.analyzer import calibrate, reference_forward
from pt_agent.backends import CountBackend, MockBackend
from pt_agent.capabilities import render_capabilities
from pt_agent.data.cifar import calibration_set
from pt_agent.dsl.errors import PTDSLError
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.kernels import KernelError
from pt_agent.kernels.poly import MINIMAX_DATA, act_minimax, minimax_coeffs
from pt_agent.lower import decode_outputs, lower, pack_inputs
from pt_agent.models.zoo import get
from pt_agent.planners.agent_tools import build_context, evaluate
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, RESULTS_DIR, require


DEGREES = (15, 15, 27)


@require(DATA, RESULTS_DIR)
class MinimaxTests(unittest.TestCase):
    def test_coefficients(self):
        self.assertTrue(MINIMAX_DATA.is_file())
        self.assertEqual(len(MINIMAX_DATA.read_text().splitlines()), 60)
        coefficients = minimax_coeffs(DEGREES)
        self.assertEqual([len(c) for c in coefficients], [16, 16, 28])
        self.assertEqual(coefficients[0][1], 1.27424444144022223/2.0)
        self.assertIs(minimax_coeffs(DEGREES), coefficients)

    def test_mock_matches_numpy_composition(self):
        x = np.linspace(-10, 10, 16)
        c1, c2, c3 = minimax_coeffs(DEGREES)
        expected = x*(0.5+Chebyshev(c3)(Chebyshev(c2)(Chebyshev(c1)(x/10))))
        backend = MockBackend(16)
        backend.bind([x])
        result = act_minimax(backend, backend.input(0), DEGREES, (-10.0, 10.0))
        self.assertLessEqual(float(np.max(np.abs(result.value-expected))), 1e-12)

    def test_relu_approximation(self):
        x = np.linspace(-10, 10, 2001)
        c1, c2, c3 = minimax_coeffs(DEGREES)
        actual = x*(0.5+Chebyshev(c3)(Chebyshev(c2)(Chebyshev(c1)(x/10))))
        self.assertLessEqual(float(np.max(np.abs(actual-np.maximum(x, 0)))), 2e-3)

    def test_operation_counts(self):
        backend = CountBackend(16)
        value = act_minimax(backend, backend.input(0), DEGREES, (-10.0, 10.0))
        backend.output([value])
        stats = backend.stats()
        self.assertEqual(stats['max_depth'], 17)
        self.assertEqual(stats['counts']['mul_cc'], 26)
        self.assertEqual(stats['counts']['mul_cp'], 46)

    def test_kernel_rejects_degrees_and_domain(self):
        with self.assertRaisesRegex(KernelError, 'minimax_comp supports degrees'):
            minimax_coeffs((7, 7, 7))
        backend = CountBackend(16)
        with self.assertRaisesRegex(KernelError, 'minimax_comp requires a symmetric domain'):
            act_minimax(backend, backend.input(0), DEGREES, (-4.0, 5.0))

    def source(self, attributes):
        return ('@fhe.program(slots=16384, refresh="compiler")\n'
                'def net(v: Cipher[1, 16]):\n'
                '    x = pack(v, layout="flat")\n'
                '    y = act(x, '+attributes+')\n'
                '    return y\n')

    def test_valid_attributes(self):
        source = self.source('fn="relu", method="minimax_comp", degrees=(15,15,27), domain=(-5.0,5.0)')
        self.assertTrue(check(parse(source), {}).ok)

    def test_invalid_attributes(self):
        valid = 'fn="relu", method="minimax_comp", degrees=(15,15,27), domain=(-5.0,5.0)'
        for changed in (valid.replace('(15,15,27)', '(7,7,7)'),
                        valid.replace('(-5.0,5.0)', '(-4.0,5.0)'),
                        valid.replace('fn="relu"', 'fn="silu"'), valid+', deg=127'):
            with self.subTest(attributes=changed), self.assertRaises(PTDSLError) as caught:
                parse(self.source(changed))
            self.assertTrue(caught.exception.code.startswith('E_ATTR'))
            self.assertIn('minimax_comp requires', str(caught.exception))

    def test_capability_document(self):
        text = render_capabilities()
        self.assertIn('minimax_comp', text)
        self.assertIn('17 multiplicative levels', text)


@require(DATA, RESULTS_DIR)
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        model = get('cnn_plain', pt_result('models-v2'))
        cls.graph = model.import_graph()
        cls.images, _ = calibration_set(normalization=model.normalization)
        cls.images = cls.images[:4]
        cls.calibration = calibrate(cls.graph, {cls.graph.inputs[0]: cls.images})
        cls.source = plan(cls.graph, cls.calibration, deg=127, margin=1.5, act_method='minimax_comp')

    def run_model(self, *, exact=False, observer=None):
        program = parse(self.source)
        backend = MockBackend(16384, batch=4)
        backend.bind(pack_inputs(program, [self.images]))
        outputs, layouts = lower(program, self.graph.weights, backend,
                                 act_mode='exact' if exact else 'program', observer=observer)
        return decode_outputs([value.value for value in outputs], layouts)[0]

    def test_rule_typecheck_and_symmetric_domains(self):
        program = parse(self.source)
        checked = check(program, {name: weight.shape for name, weight in self.graph.weights.items()})
        self.assertTrue(checked.ok, checked.diags)
        for node, op in zip((node for node in self.graph.nodes if node.op == 'act'),
                            (op for op in program.ops if op.kind == 'act')):
            self.assertEqual(op.attrs['method'], 'minimax_comp')
            self.assertEqual(op.attrs['degrees'], DEGREES)
            self.assertIsNone(op.attrs['deg'])
            limits = self.calibration[node.inputs[0]]
            bound = 1.5*max(abs(limits['min']), abs(limits['max']))
            self.assertEqual(op.attrs['domain'], (-bound, bound))

    def test_exact_structure(self):
        actual = self.run_model(exact=True)
        expected = reference_forward(self.graph, {self.graph.inputs[0]: self.images})[self.graph.outputs[0]]
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_four_images_and_observer_config(self):
        configs = []
        actual = self.run_model(observer=lambda name, value, layout, cfg: configs.append(cfg))
        self.assertEqual(actual.shape, (4, 10))
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(len(configs), 3)
        for cfg in configs:
            self.assertEqual(len(cfg), 4)
            self.assertEqual(cfg[:2], ('minimax_comp', DEGREES))
            self.assertEqual(cfg[2], -cfg[3])

    def test_feedback_activation_method(self):
        ctx = build_context(self.graph, self.calibration, n_val=4)
        feedback = evaluate(self.source, ctx)
        self.assertEqual(feedback['stage'], 'complete')
        self.assertTrue(feedback['exact']['ok'])
        for row in feedback['validation']['worst_acts']:
            self.assertEqual(row['method'], 'minimax_comp')
            self.assertEqual(row['degrees'], DEGREES)
            self.assertIsNone(row['deg'])

    def test_non_relu_keeps_chebyshev(self):
        graph = copy.deepcopy(self.graph)
        for node in graph.nodes:
            if node.op == 'act':
                node.attrs['fn'] = 'silu'
        default = plan(graph, self.calibration, deg=127, margin=1.5)
        requested = plan(graph, self.calibration, deg=127, margin=1.5, act_method='minimax_comp')
        self.assertEqual(default, requested)


if __name__ == '__main__':
    unittest.main()
