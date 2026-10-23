import gc
import json
from pathlib import Path
import unittest

import numpy as np

from pt_agent.analyzer import reference_forward
from pt_agent.backends import CountBackend, MockBackend
from pt_agent.data.cifar import calibration_set
from pt_agent.dsl.parser import parse
from pt_agent.examples.shapes import mlp_weight_shapes
from pt_agent.intake import Graph
from pt_agent.lower import LoweringError, decode_outputs, lower, pack_inputs
from pt_agent.models.resnet20_fhe import build, pack_input
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, require


R3 = pt_result('r3-resnet20-20261020T131658')
EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@require(DATA, R3)
class LowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(R3 / 'graph')
        cls.calibration = json.loads((R3 / 'calibration.json').read_text())
        cls.images = calibration_set()[0][:2].copy()
        cls.program = parse(plan(cls.graph, cls.calibration, deg=31, margin=1.25))

    def tearDown(self):
        gc.collect()

    def test_count_matches_r5(self):
        backend = CountBackend(16384)
        lower(self.program, self.graph.weights, backend)
        reference = CountBackend(16384)
        cfg = {node.inputs[0]: ('cheb', 31, self.calibration[node.inputs[0]]['min']*1.25,
                                self.calibration[node.inputs[0]]['max']*1.25)
               for node in self.graph.nodes if node.op == 'act'}
        value, _ = build(reference, self.graph, cfg)
        reference.output([value])
        self.assertEqual(backend.stats(), reference.stats())
        print('R6 Count=' + json.dumps(backend.stats()), flush=True)

    def test_exact(self):
        backend = MockBackend(16384, batch=2)
        backend.bind(pack_inputs(self.program, [self.images]))
        values, layouts = lower(self.program, self.graph.weights, backend, act_mode='exact')
        actual = decode_outputs([value.value for value in values], layouts)[0]
        expected = reference_forward(self.graph, {'x': self.images})[self.graph.outputs[0]]
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_flat_mlp(self):
        rng = np.random.default_rng(20261031)
        weights = {name: rng.normal(size=shape) for name, shape in mlp_weight_shapes().items()}
        images = rng.normal(size=(3, 64))
        program = parse((EXAMPLES/'mlp.ptdsl').read_text())
        backend = MockBackend(8192, batch=3)
        backend.bind(pack_inputs(program, [images]))
        values, layouts = lower(program, weights, backend)
        actual = decode_outputs([value.value for value in values], layouts)[0]
        expected = (images@weights['fc1.weight'].T + weights['fc1.bias'])**2
        expected = expected@weights['fc2.weight'].T + weights['fc2.bias']
        self.assertEqual(actual.shape, (3, 10))
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_errors_include_op_id(self):
        act_program = parse((EXAMPLES/'resnet20.ptdsl').read_text())
        first_act = next(op for op in act_program.ops if op.kind == 'act')
        first_act.attrs['domain'] = (-1.0, 2.0)  # minimax_comp requires a symmetric domain
        programs = [
            (act_program, self.graph.weights, 'act'),
            (parse('@fhe.program(slots=16384)\ndef bad(x: Cipher[1,3,32,32]):\n'
                   ' a = pack(x, layout="mux")\n b = relayout(a, layout="flat")\n return b\n'), {}, 'relayout'),
            (parse('@fhe.program(slots=16384)\ndef bad(x: Cipher[1,3,32,32]):\n'
                   ' a = pack(x, layout="mux")\n b = pad(a, pads=(0,0))\n return b\n'), {}, 'pad'),
            (parse((EXAMPLES/'mlp.ptdsl').read_text()), {}, 'linear'),
        ]
        for program, weights, kind in programs:
            with self.subTest(kind=kind):
                op = next(op for op in program.ops if op.kind == kind)
                with self.assertRaises(LoweringError) as error:
                    lower(program, weights, CountBackend(program.header.slots))
                self.assertEqual(error.exception.op_id, op.id)
                self.assertEqual(error.exception.lineno, op.lineno)
                self.assertIn(f'op {op.id}', str(error.exception))
                if kind == 'act':
                    self.assertIn('symmetric', str(error.exception))

    def test_refresh(self):
        for mode, expected in (('agent', 1), ('compiler', 0)):
            text = (f'@fhe.program(slots=8192, refresh="{mode}", l_max=10, boot_out_level=8)\n'
                    'def small(x: Cipher[1,4]):\n a = pack(x, layout="flat")\n'
                    ' b = act(a, fn="square", method="exact")\n c = refresh(b)\n'
                    ' d = act(c, fn="square", method="exact")\n return d\n')
            backend = CountBackend(8192, input_level=2, boot_out_level=8)
            lower(parse(text), {}, backend)
            self.assertEqual(backend.stats()['counts']['boot'], expected)

    def test_pack_inputs(self):
        actual = pack_inputs(self.program, [self.images])[0]
        expected = np.stack([pack_input(image) for image in self.images])
        np.testing.assert_array_equal(actual, expected)
