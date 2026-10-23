"""R8f2 final Mock statistics and the engines used by the Agent loop."""

import copy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import agent, agent_tools, tuner
from pt_agent.tensor_sim import clear_caches


SOURCE = ('@fhe.program(slots=16384)\ndef feedback_toy(v: Cipher[1, 2]):\n'
          '    x = pack(v, layout="flat")\n'
          '    h = linear(x, w=W["identity"], algo="bsgs")\n'
          '    a = act(h, fn="relu", method="cheb", deg=15, domain=(-8., 8.))\n'
          '    y = linear(a, w=W["identity"], algo="bsgs")\n'
          '    return y\n')


def context():
    images = np.array([[-2., 1.], [1., -2.], [-1., 2.], [2., -1.]], dtype=np.float64)
    reference = np.maximum(images, 0)
    return SimpleNamespace(weights={'identity': np.eye(2)}, profile='qwen48',
                           weight_shapes={'identity': (2, 2)},
                           val_images=images, val_exact_logits=reference,
                           val_labels=reference.argmax(axis=1), val_exact_top1=100.,
                           probe_images=images, probe_act_inputs=[images.copy()],
                           act_tensors=['hidden'], val_split='synthetic')


class FinalMockFeedbackTests(unittest.TestCase):
    def setUp(self):
        clear_caches()
        self.ctx = context()
        self.program = parse(SOURCE)

    def tearDown(self):
        clear_caches()

    def test_tensor_final_statistics_match_mock_and_recheck_once(self):
        expected = tuner.tune(self.program, self.ctx, engine='mock')
        clear_caches()
        with patch.object(tuner, 'validate', wraps=agent_tools.validate) as validations:
            actual = tuner.tune(self.program, self.ctx, engine='tensor')

        self.assertEqual(actual.source, expected.source)
        self.assertEqual(actual.changes, expected.changes)
        self.assertEqual(actual.feasible, expected.feasible)
        self.assertEqual(actual.history[-1]['phase'], 'result')
        self.assertEqual(actual.history[-1], expected.history[-1])
        self.assertIsNotNone(actual.history[-1]['boot_input_max_abs'])
        self.assertTrue(actual.history[-1]['worst_acts'])

        calls = validations.call_args_list
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(call.kwargs['engine'] == 'tensor' for call in calls[:-1]))
        last = calls[-1]
        self.assertEqual(last.kwargs['engine'], 'mock')
        bound = inspect.signature(agent_tools.validate).bind(*last.args, **last.kwargs)
        bound.apply_defaults()
        self.assertIs(bound.arguments['ctx'], self.ctx)
        self.assertIsNone(bound.arguments['n'])
        self.assertTrue(bound.arguments['stop_on_divergence'])
        self.assertEqual(bound.arguments['boot_nodes'], actual.cost_after['boot_nodes'])
        self.assertEqual(print_program(bound.arguments['program']), actual.source)

    def test_mock_recheck_mismatch_raises(self):
        validate = agent_tools.validate
        for field in ('feasible', 'agree'):
            with self.subTest(field=field):
                clear_caches()
                mock_calls = []

                def inconsistent(program, ctx, n=None, **kwargs):
                    result = validate(program, ctx, n, **kwargs)
                    if kwargs.get('engine') == 'mock':
                        mock_calls.append(print_program(program))
                        result = copy.deepcopy(result)
                        result[field] = not result[field] if field == 'feasible' else result[field]-1
                    return result

                with patch.object(tuner, 'validate', side_effect=inconsistent):
                    with self.assertRaises(RuntimeError) as caught:
                        tuner.tune(self.program, self.ctx, engine='tensor')
                self.assertEqual(len(mock_calls), 1)
                self.assertIn(field, str(caught.exception))

    def test_agent_evaluate_uses_mock_and_tune_uses_tensor(self):
        evaluate_impl, tune_impl = agent_tools.evaluate, tuner.tune
        engines = dict(evaluate=[], tune=[])

        def record_evaluate(*args, **kwargs):
            bound = inspect.signature(evaluate_impl).bind(*args, **kwargs)
            bound.apply_defaults()
            engines['evaluate'].append(bound.arguments['engine'])
            return evaluate_impl(*args, **kwargs)

        def record_tune(*args, **kwargs):
            bound = inspect.signature(tune_impl).bind(*args, **kwargs)
            bound.apply_defaults()
            engines['tune'].append(bound.arguments['engine'])
            return tune_impl(*args, **kwargs)

        class FixedReply:
            def chat(self, messages, *, purpose):
                return dict(content=json.dumps(dict(program=SOURCE)), seconds=0,
                            usage=dict(prompt_tokens=0, completion_tokens=0))

        def select(candidates, ctx):
            return dict(selected=min(candidates, key=lambda item: item.latency_pred_us)
                        if candidates else None, evals=[])

        with TemporaryDirectory() as directory, \
                patch.object(agent, 'build_system_prompt', return_value='offline-system'), \
                patch.object(agent, 'initial_message', return_value='offline-model'), \
                patch.object(agent, 'evaluate', side_effect=record_evaluate) as evaluate_calls, \
                patch.object(agent, 'tune', side_effect=record_tune), \
                patch.object(agent, 'confirm_select', side_effect=select):
            result = agent.run_agent(self.ctx, FixedReply(), max_iters=1,
                                     trace_path=Path(directory)/'trace.jsonl')
        self.assertEqual(result['iterations'], 1)
        self.assertEqual(engines, dict(evaluate=['mock'], tune=['tensor']))
        self.assertNotIn('engine', evaluate_calls.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
