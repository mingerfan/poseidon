from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import BackendError
from pt_agent.dsl.parser import parse
from pt_agent.planners.agent_tools import validate
from pt_agent.planners.confirm import Candidate, confirm_select


SOURCE = ('@fhe.program(slots=16384)\ndef net(v: Cipher[1, 2]):\n'
          '    x = pack(v, layout="flat")\n'
          '    x = act(x, fn="relu", method="cheb", deg=7, domain=(-100., 100.))\n'
          '    return x\n')


class CriterionTests(unittest.TestCase):
    def synthetic_validation(self, *, stop_on_divergence=True):
        images = np.arange(60, dtype=np.float64).reshape(60, 1)
        logits = np.tile([1., 0.], (60, 1))
        context = SimpleNamespace(
            val_images=images, val_labels=np.zeros(60, dtype=np.int64),
            val_exact_logits=logits, val_exact_top1=100., weights={},
            act_tensors=['activation'], profile='synthetic',
        )
        attempts = []

        class Backend:
            def __init__(self, slots, *, batch, watch, noise):
                self.batch_size = batch
                self.watch_max = {}

            def bind(self, inputs):
                self.images = inputs[0]

        def lower(program, weights, backend, *, observer):
            indices = backend.images[:, 0].astype(int)
            attempts.append(tuple(indices))
            backend.watch_max['watched'] = float(indices.max())
            if 22 in indices:
                raise BackendError('non-finite value in synthetic activation')
            observer('activation', SimpleNamespace(value=indices.reshape(-1, 1).astype(float)),
                     SimpleNamespace(size=1), ('relu', 'cheb', -100., 100.))
            return [SimpleNamespace(value=logits[indices])], [SimpleNamespace(size=2)]

        with ExitStack() as stack:
            stack.enter_context(patch('pt_agent.planners.agent_tools.PROFILES',
                                      {'synthetic': dict(boot_input_bound=None, boot_abs_error=0.)}))
            stack.enter_context(patch('pt_agent.planners.agent_tools.MockBackend', Backend))
            stack.enter_context(patch('pt_agent.planners.agent_tools.pack_inputs',
                                      side_effect=lambda program, inputs: inputs))
            stack.enter_context(patch('pt_agent.planners.agent_tools.lower', side_effect=lower))
            stack.enter_context(patch('pt_agent.planners.agent_tools.decode_outputs',
                                      side_effect=lambda values, layouts: values))
            kwargs = {} if stop_on_divergence else dict(stop_on_divergence=False)
            result = validate(parse(SOURCE), context, boot_nodes=[], **kwargs)
        return result, attempts

    def test_01_batch_failure_retries_each_image_and_finishes_all_batches(self):
        result, attempts = self.synthetic_validation(stop_on_divergence=False)
        self.assertEqual(result['diverged_indices'], [22])
        self.assertEqual(result['n_diverged'], 1)
        self.assertTrue(result['diverged'])
        self.assertEqual(result['total'], 60)
        self.assertEqual(result['images_done'], 60)
        self.assertEqual(result['agree'], 59)
        self.assertAlmostEqual(result['top1'], 100.*59/60)
        self.assertEqual(result['exact_top1'], 100.)
        self.assertAlmostEqual(result['acc_drop_pp'], 100./60)
        self.assertFalse(result['feasible'])
        self.assertEqual(attempts[:2], [tuple(range(20)), tuple(range(20, 40))])
        self.assertEqual([attempt for attempt in attempts if len(attempt) == 1],
                         [(index,) for index in range(20, 40)])
        self.assertEqual(attempts[-1], tuple(range(40, 60)))
        row = result['rows'][0]
        self.assertEqual(row['obs_min'], 0.)
        self.assertEqual(row['obs_max'], 59.)
        self.assertAlmostEqual(row['max_u'], .59)
        self.assertAlmostEqual(result['max_u'], .59)
        self.assertEqual(result['boot_input_max_abs'], 59.)

    def test_02_default_stops_at_first_divergent_image_without_accuracy(self):
        result, attempts = self.synthetic_validation()
        self.assertEqual(result['diverged_indices'], [22])
        self.assertEqual(result['n_diverged'], 1)
        self.assertTrue(result['diverged'])
        self.assertIsNone(result['top1'])
        self.assertIsNone(result['acc_drop_pp'])
        self.assertFalse(result['feasible'])
        self.assertEqual(result['total'], 60)
        self.assertEqual(result['agree'], 22)
        self.assertEqual(attempts,
                         [tuple(range(20)), tuple(range(20, 40)), (20,), (21,), (22,)])

    def test_03_confirmation_uses_accuracy_and_boot_input_bound(self):
        candidate = Candidate(SOURCE, 'llm', 1, 100., 1)
        context = SimpleNamespace(val_split='test[8000:9000]')
        cases = [
            dict(drop=.3, bound=None, maximum=100., passed=True),
            dict(drop=.6, bound=None, maximum=100., passed=False),
            dict(drop=None, bound=None, maximum=100., passed=False),
            dict(drop=.3, bound=10., maximum=10., passed=True),
            dict(drop=.3, bound=10., maximum=10.01, passed=False),
        ]
        for case in cases:
            with self.subTest(case=case):
                validation = dict(
                    feasible=False, top1=89.7 if case['drop'] is not None else None,
                    exact_top1=90., acc_drop_pp=case['drop'], max_u=.95,
                    n_diverged=1, diverged_indices=[22], diverged=True,
                    boot_input_bound=case['bound'], boot_input_max_abs=case['maximum'],
                )
                with patch('pt_agent.planners.confirm.validate', return_value=validation) as checked:
                    result = confirm_select([candidate], context)
                self.assertEqual(checked.call_count, 1)
                self.assertIs(checked.call_args.args[1], context)
                self.assertIs(checked.call_args.kwargs['stop_on_divergence'], False)
                self.assertEqual(len(result['evals']), 1)
                actual = result['evals'][0]
                self.assertEqual(actual['passed'], case['passed'])
                self.assertEqual(actual['n_diverged'], 1)
                self.assertEqual(actual['max_u'], .95)
                self.assertTrue(actual['diverged'])
                self.assertEqual(actual['acc_drop_pp'], case['drop'])
                if case['passed']:
                    self.assertIs(result['selected'], candidate)
                else:
                    self.assertIsNone(result['selected'])


if __name__ == '__main__':
    unittest.main()
