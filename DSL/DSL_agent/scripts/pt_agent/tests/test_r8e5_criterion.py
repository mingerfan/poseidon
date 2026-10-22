from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.planners import agent_tools
from pt_agent.planners.agent_tools import validate
from pt_agent.planners.prompts import PROMPT_VERSION, TEMPLATE
from pt_agent.planners.tuner import _strict


class CriterionTests(unittest.TestCase):
    def validation(self, correct, max_u=.8, n=None):
        count = 500
        labels = np.zeros(count, dtype=np.int64)
        exact = np.zeros((count, 2))
        exact[:460, 0] = 1
        exact[460:, 1] = 1
        predictions = np.ones(count, dtype=np.int64)
        predictions[:correct-40] = 0
        predictions[460:] = 0
        context = SimpleNamespace(val_images=np.arange(count).reshape(count, 1), val_exact_logits=exact,
                                  val_labels=labels, val_exact_top1=92.0, act_tensors=['hidden'],
                                  profile='qwen48', weights={})
        program = parse('@fhe.program(slots=16384)\ndef net(v: Cipher[1, 1]):\n'
                        '    x = pack(v, layout="flat")\n'
                        '    x = act(x, fn="relu", method="cheb", deg=15, domain=(-1.0, 1.0))\n'
                        '    return x\n')

        class Backend:
            def __init__(self, *args, **kwargs):
                self.watch_max = {}

            def bind(self, images):
                self.images = images

        def fake_lower(program, weights, backend, observer):
            indices = backend.images[:, 0].astype(np.int64)
            observer('hidden', SimpleNamespace(value=np.full((len(indices), 1), max_u)),
                     SimpleNamespace(size=1), ('relu', 15, -1.0, 1.0))
            logits = np.zeros((len(indices), 2))
            logits[np.arange(len(indices)), predictions[indices]] = 1
            return [SimpleNamespace(value=logits)], [None]

        with patch('pt_agent.planners.agent_tools.MockBackend', Backend), \
                patch('pt_agent.planners.agent_tools.pack_inputs', side_effect=lambda program, inputs: inputs[0]), \
                patch('pt_agent.planners.agent_tools.lower', side_effect=fake_lower), \
                patch('pt_agent.planners.agent_tools.decode_outputs', side_effect=lambda values, layouts: values):
            return validate(program, context, n=n, boot_nodes=[])

    def test_01_accuracy_drop_replaces_agreement(self):
        result = self.validation(458)
        self.assertAlmostEqual(result['exact_top1'], 92.0)
        self.assertAlmostEqual(result['top1'], 91.6)
        self.assertAlmostEqual(result['acc_drop_pp'], .4)
        self.assertLess(result['agree'], .99*result['total'])
        self.assertTrue(result['feasible'])
        result = self.validation(457)
        self.assertAlmostEqual(result['top1'], 91.4)
        self.assertAlmostEqual(result['acc_drop_pp'], .6)
        self.assertFalse(result['feasible'])

    def test_02_domain_safety_still_applies(self):
        self.assertFalse(self.validation(458, max_u=.95)['feasible'])

    def test_03_exact_accuracy_uses_the_same_prefix(self):
        result = self.validation(458, n=100)
        self.assertEqual(result['total'], 100)
        self.assertEqual(result['top1'], 100.0)
        self.assertEqual(result['exact_top1'], 100.0)
        self.assertEqual(result['acc_drop_pp'], 0.0)

    def test_04_strict_confirmation(self):
        self.assertTrue(_strict(dict(feasible=True, acc_drop_pp=.3)))
        self.assertFalse(_strict(dict(feasible=True, acc_drop_pp=.4)))
        self.assertFalse(_strict(dict(feasible=False, acc_drop_pp=0.0)))

    def test_05_batch_and_prompt(self):
        self.assertEqual(agent_tools.VAL_BATCH, 20)
        self.assertEqual(PROMPT_VERSION, 'r8-v4')
        self.assertIn('0.5 percentage points', TEMPLATE)
        self.assertNotIn('at least 99%', TEMPLATE)


if __name__ == '__main__':
    unittest.main()
