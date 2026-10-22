import json
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import MockBackend
from pt_agent.capabilities import render_capabilities
from pt_agent.cost import PROFILES, estimate
from pt_agent.dsl.parser import parse
from pt_agent.intake.graph import Graph, Node, TensorInfo
from pt_agent.lower import decode_outputs, lower, pack_inputs
from pt_agent.planners.agent_tools import AgentContext, FEASIBLE_MAX_U, evaluate
from pt_agent.planners.prompts import PROMPT_VERSION, TEMPLATE, build_system_prompt, feedback_message, initial_message


SOURCE = '''@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def toy(v: Cipher[1, 2]):
    x = pack(v, layout="flat")
    h = linear(x, w=W["identity"], algo="bsgs")
    h = activate(h)
    out = linear(h, w=W["identity"], algo="bsgs")
    return out

def activate(x):
    y = act(x, fn="relu", method="cheb", deg=15, domain=(-8.0, 8.0))
    return y
'''


def context():
    images = np.asarray([[-2., 1.], [1., -2.], [-1., 2.], [2., -1.]])
    weights = {'identity': np.eye(2)}
    graph = Graph('toy', [Node('l0', 'linear', ['input'], ['hidden'], dict(weight='identity', bias=None), ''),
                          Node('a0', 'act', ['hidden'], ['activated'], dict(fn='relu'), ''),
                          Node('l1', 'linear', ['activated'], ['output'], dict(weight='identity', bias=None), '')],
                  {name: TensorInfo(name, (1, 2), 'cipher') for name in ('input', 'hidden', 'activated', 'output')},
                  weights, ['input'], ['output'])
    val_exact_logits = np.maximum(images, 0)
    return AgentContext(graph=graph, weights=weights, calibration={'hidden': dict(min=-2., max=2.)},
                        weight_shapes={'identity': (2, 2)}, val_images=images,
                        val_exact_logits=val_exact_logits, val_labels=val_exact_logits.argmax(axis=1),
                        val_exact_top1=100.0, probe_images=images,
                        probe_act_inputs=[images.copy()], needed_margin={'hidden': 1.},
                        act_tensors=['hidden'], model_name='toy')


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()

    def test_resources_const_bytes(self):
        with patch.dict(PROFILES['qwen48']['limits'], const_bytes=1):
            result = evaluate(SOURCE, self.ctx)
        self.assertEqual(result['stage'], 'resources')
        self.assertFalse(result['feasible'])
        self.assertIsNone(result['validation'])
        self.assertEqual(result['errors'][0]['code'], 'E_RESOURCE')
        self.assertEqual(result['errors'][0]['limit'], 'const_bytes')
        self.assertEqual(result['errors'][0]['max'], 1)
        self.assertGreater(result['errors'][0]['value'], 1)

    def test_resources_rot_keys(self):
        with patch.dict(PROFILES['qwen48']['limits'], rot_keys=0):
            result = evaluate(SOURCE, self.ctx)
        self.assertEqual(result['stage'], 'resources')
        self.assertEqual(result['errors'][0]['limit'], 'rot_keys')

    def test_boot_range_reports_without_default_limit(self):
        result = evaluate(SOURCE, self.ctx)
        self.assertTrue(result['feasible'])
        self.assertIsNone(result['validation']['boot_input_bound'])
        self.assertGreater(result['validation']['boot_input_max_abs'], 0)

    def test_boot_range_violation(self):
        with patch.dict(PROFILES['qwen48'], boot_input_bound=1e-6):
            result = evaluate(SOURCE, self.ctx)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['validation']['boot_input_bound'], 1e-6)
        self.assertGreater(result['validation']['boot_input_max_abs'], 1e-6)

    def test_noisy_mock_still_computes_agreement(self):
        program = parse(SOURCE)
        cost = estimate(program, self.ctx.weights)
        self.assertTrue(cost['boot_nodes'])
        backend = MockBackend(16384, batch=4, watch=set(cost['boot_nodes']), noise=(1e-3, 0))
        backend.bind(pack_inputs(program, [self.ctx.val_images]))
        outputs, layouts = lower(program, self.ctx.weights, backend)
        actual = decode_outputs([value.value for value in outputs], layouts)[0]
        agree = int(np.sum(actual.argmax(axis=1) == self.ctx.val_exact_logits.argmax(axis=1)))
        self.assertGreaterEqual(agree, 0)
        self.assertLessEqual(agree, 4)
        with patch.dict(PROFILES['qwen48'], boot_abs_error=1e-3):
            result = evaluate(SOURCE, self.ctx)
        self.assertNotIn('error', result)
        self.assertEqual(result['validation']['images_done'], 4)
        self.assertIsInstance(result['validation']['agree'], int)

    def test_hotspots_source_text_and_target(self):
        result = evaluate(SOURCE, self.ctx)
        self.assertEqual(result['cost']['placement_model'], 'depth-dp')
        self.assertTrue(result['cost']['placement_exact'])
        self.assertTrue(any(row['call_lines'] for row in result['cost']['hotspots']))
        lines = SOURCE.splitlines()
        for row in result['cost']['hotspots']:
            self.assertEqual(row['line_text'], lines[row['line']-1].strip())
            self.assertEqual(row['call_texts'], [lines[line-1].strip() for line in row['call_lines']])
        target = json.loads(initial_message(self.ctx))['target']
        self.assertEqual(target, dict(profile='qwen48', placement='depth-dp',
                                      limits=dict(const_bytes=4*2**30), boot_input_bound=None))

    def test_r8_v2_prompt_and_feedback(self):
        self.assertEqual(PROMPT_VERSION, 'r8-v4')
        self.assertIn('hotspots', TEMPLATE)
        self.assertIn('stage "resources"', TEMPLATE)
        generic = TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U,
                                  n_val=50, examples='').lower()
        for word in ('resnet', 'layer1.', 'conv1.weight', 'linear.weight', 'fc1.weight', 'fc2.weight', 'cifar'):
            self.assertNotIn(word, generic)
        full = build_system_prompt(self.ctx).lower()
        for word in ('resnet', 'cifar', 'layer1.'):
            self.assertNotIn(word, full)
        instruction = json.loads(feedback_message(dict(feasible=True), 1, None))['next']
        self.assertIn('structural changes', instruction)


if __name__ == '__main__':
    unittest.main()
