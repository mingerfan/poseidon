import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.data import cifar
from pt_agent.dsl.parser import parse
from pt_agent.experiments import r8e6_reconfirm
from pt_agent.paths import pt_result
from pt_agent.planners.agent import run_agent
from pt_agent.planners.agent_tools import AgentContext, build_context
from pt_agent.planners.confirm import (
    MAX_CONFIRM_EVALS, Candidate, build_confirm_context, confirm_select,
)
from pt_agent.planners.tuner import TuneResult


TRACE = pt_result('r8e5-dev-20261022T041646/resnet20/agent_trace.jsonl')


def source(name):
    return (f'@fhe.program(slots=16384)\ndef {name}(v: Cipher[1, 2]):\n'
            '    x = pack(v, layout="flat")\n    return x\n')


def validation(passed):
    return dict(feasible=passed, top1=90.0 if passed else 88.0,
                exact_top1=90.0, acc_drop_pp=0.0 if passed else 2.0,
                max_u=.8, diverged=False, n_diverged=0, boot_input_bound=None,
                boot_input_max_abs=0.0, agree=2, total=2)


def feedback(latency):
    return dict(stage='complete', feasible=True, exact=dict(ok=True),
                cost=dict(latency_pred_us=latency, boots_pred=1),
                validation=validation(True))


class FakeClient:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.calls = []

    def chat(self, messages, *, purpose):
        self.calls.append(copy.deepcopy(messages))
        return dict(content=json.dumps(dict(program=next(self.sources))),
                    seconds=0, usage=dict(prompt_tokens=1, completion_tokens=2))


class ConfirmTests(unittest.TestCase):
    def test_01_three_splits_and_encrypted_indices(self):
        groups = [set(cifar.HELDOUT), set(cifar.CONFIRM), set(cifar.REPORT)]
        for index, group in enumerate(groups):
            for other in groups[index+1:]:
                self.assertFalse(group & other)
        self.assertEqual(set.union(*groups), set(range(10000)))
        self.assertEqual(groups[0], set(range(9000, 10000)))
        self.assertEqual(groups[1], set(range(8000, 9000)))
        self.assertEqual(groups[2], set(range(8000)))
        encrypted = json.loads(cifar.SELECTIONS.read_text())['encrypted_test']
        self.assertEqual(len(encrypted), 40)
        self.assertTrue(all(entry['index'] in cifar.REPORT for entry in encrypted))
        for reader, expected in ((cifar.confirm_set, range(8000, 9000)),
                                 (cifar.heldout_set, range(9000, 10000)),
                                 (cifar.report_test_set, range(8000))):
            with self.subTest(reader=reader.__name__):
                images, labels = reader()
                self.assertEqual(len(images), len(expected))
                pixels, expected_labels = cifar.read_rows('test', [expected[0], expected[-1]])
                np.testing.assert_array_equal(images[[0, -1]], cifar.normalize(pixels))
                np.testing.assert_array_equal(labels[[0, -1]], expected_labels)

    def test_02_context_normalization_reshape_and_shared_probes(self):
        normalization = ((.1, .2, .3), (.4, .5, .6))
        graph = SimpleNamespace(inputs=['input'], outputs=['output'], nodes=[],
                                tensors={'input': SimpleNamespace(shape=(1, 3072))},
                                weights={}, name='flat')
        images = np.arange(3*3072, dtype=np.float64).reshape(3, 3, 32, 32)
        labels = np.array([0, 1, 0])
        logits = np.array([[2., 1.], [1., 2.], [2., 1.]])

        def reference(graph_arg, inputs, **kwargs):
            self.assertIs(graph_arg, graph)
            self.assertEqual(inputs['input'].shape, (3, 3072))
            return {'output': logits.copy()}

        with patch('pt_agent.planners.agent_tools.heldout_set', return_value=(images, labels)) as heldout, \
                patch('pt_agent.planners.agent_tools.reference_forward', side_effect=reference):
            context = build_context(graph, {}, n_val=3, normalization=normalization)
        self.assertIs(context.normalization, normalization)
        heldout.assert_called_once_with(3, normalization=normalization)
        with patch('pt_agent.planners.confirm.confirm_set', return_value=(images, labels)) as confirmation, \
                patch('pt_agent.planners.confirm.reference_forward', side_effect=reference):
            actual = build_confirm_context(context)
        confirmation.assert_called_once_with(normalization=normalization)
        self.assertIsInstance(actual, AgentContext)
        self.assertIsNot(actual, context)
        self.assertEqual(actual.val_split, 'test[8000:9000]')
        self.assertEqual(context.val_split, 'test[9000:10000]')
        self.assertEqual(actual.val_images.shape, (3, 3072))
        np.testing.assert_array_equal(actual.val_images, images.reshape(3, 3072))
        np.testing.assert_array_equal(actual.val_labels, labels)
        np.testing.assert_array_equal(actual.val_exact_logits, logits)
        self.assertEqual(actual.val_exact_top1, 100.0)
        for field in ('graph', 'weights', 'calibration', 'weight_shapes', 'probe_images',
                      'probe_act_inputs', 'needed_margin', 'act_tensors', 'normalization'):
            self.assertIs(getattr(actual, field), getattr(context, field))

    def test_03_order_dedup_and_first_passing_candidate(self):
        first = Candidate(source('first'), 'llm', 1, 10., 1)
        second = Candidate(source('second'), 'tuned', 1, 10., 1)
        third = Candidate(source('third'), 'llm', 2, 10., 2)
        duplicate = Candidate(first.source, 'llm', 4, 11., 1)
        fourth = Candidate(source('fourth'), 'llm', 4, 30., 3)
        pool = [fourth, second, duplicate, third, first]
        context = SimpleNamespace(val_split='test[9000:10000]')
        confirmed_context = object()
        with patch('pt_agent.planners.confirm.build_confirm_context', return_value=confirmed_context) as build, \
                patch('pt_agent.planners.confirm.validate',
                      side_effect=[validation(False), validation(False), validation(True)]) as validate:
            result = confirm_select(pool, context)
        build.assert_called_once_with(context)
        self.assertIs(result['selected'], third)
        self.assertEqual(len(result['evals']), 3)
        self.assertEqual([call.args[0].name for call in validate.call_args_list],
                         ['first', 'second', 'third'])
        self.assertTrue(all(call.args[1] is confirmed_context for call in validate.call_args_list))
        self.assertEqual([row['passed'] for row in result['evals']], [False, False, True])
        self.assertEqual([(row['iteration'], row['origin']) for row in result['evals']],
                         [(1, 'llm'), (1, 'tuned'), (2, 'llm')])
        for row, candidate in zip(result['evals'], (first, second, third)):
            self.assertEqual((row['latency_pred_us'], row['boots_pred']),
                             (candidate.latency_pred_us, candidate.boots_pred))
            self.assertTrue({'top1', 'exact_top1', 'acc_drop_pp', 'max_u', 'diverged'} <= row.keys())

    def test_04_all_fail_is_bounded_and_duplicate_sources_are_once(self):
        pool = [Candidate(source(f'candidate{i}'), 'llm', i, float(i), i)
                for i in range(MAX_CONFIRM_EVALS+3)]
        pool.extend([Candidate(pool[0].source, 'tuned', 20, .5, 1),
                     Candidate(pool[1].source, 'tuned', 21, 1.5, 1)])
        with patch('pt_agent.planners.confirm.build_confirm_context', return_value=object()), \
                patch('pt_agent.planners.confirm.validate', return_value=validation(False)) as validate:
            result = confirm_select(list(reversed(pool)),
                                    SimpleNamespace(val_split='test[9000:10000]'))
        self.assertIsNone(result['selected'])
        self.assertEqual(len(result['evals']), MAX_CONFIRM_EVALS)
        self.assertEqual(validate.call_count, MAX_CONFIRM_EVALS)
        names = [call.args[0].name for call in validate.call_args_list]
        self.assertEqual(names, [f'candidate{i}' for i in range(MAX_CONFIRM_EVALS)])
        self.assertEqual(len(set(names)), len(names))

    def test_05_agent_keeps_every_feasible_llm_and_tuned_candidate(self):
        sources = [source('raw_first'), source('raw_second')]
        tuned_sources = [source('tuned_first'), source('tuned_second')]
        client = FakeClient(sources)
        selected = None
        confirmation = dict(selected=None, evals=[dict(confirmation_only='private-confirm-result')])
        seen = []

        def tune_result(index):
            return TuneResult(parse(tuned_sources[index]), tuned_sources[index], [],
                              dict(latency_pred_us=40.-10*index, boots_pred=1),
                              dict(latency_pred_us=35.-10*index, boots_pred=1),
                              0, 1, .01, [validation(True)], True)

        def choose(candidates, context, **kwargs):
            nonlocal selected
            self.assertEqual(len(client.calls), 2)
            seen.extend(candidates)
            selected = next(candidate for candidate in candidates
                            if candidate.origin == 'llm' and candidate.iteration == 1)
            return dict(confirmation, selected=selected)

        with TemporaryDirectory() as temporary, \
                patch('pt_agent.planners.agent.build_system_prompt', return_value='offline-system'), \
                patch('pt_agent.planners.agent.initial_message', return_value='{}'), \
                patch('pt_agent.planners.agent.evaluate', side_effect=[feedback(40.), feedback(30.)]), \
                patch('pt_agent.planners.agent.tune', side_effect=[tune_result(0), tune_result(1)]) as tune, \
                patch('pt_agent.planners.agent.confirm_select', side_effect=choose) as confirm:
            result = run_agent(object(), client, max_iters=2,
                               trace_path=Path(temporary)/'trace.jsonl')
        self.assertEqual(tune.call_count, 2)
        self.assertEqual(confirm.call_count, 1)
        self.assertEqual(result['pool_size'], 4)
        self.assertEqual([(c.iteration, c.origin, c.source) for c in seen],
                         [(1, 'llm', sources[0]), (1, 'tuned', tuned_sources[0]),
                          (2, 'llm', sources[1]), (2, 'tuned', tuned_sources[1])])
        self.assertEqual(result['status'], 'feasible')
        self.assertEqual(result['best_program'], selected.source)
        self.assertEqual(result['best_iteration'], selected.iteration)
        self.assertIs(result['confirm']['selected'], selected)
        self.assertEqual(result['confirm']['evals'], confirmation['evals'])
        self.assertNotIn('private-confirm-result', json.dumps(client.calls))
        self.assertNotIn('test[8000:9000]', json.dumps(client.calls))

    def test_06_agent_confirmation_failure_has_no_frozen_program(self):
        client = FakeClient([source('first'), source('second')])
        with TemporaryDirectory() as temporary, \
                patch('pt_agent.planners.agent.build_system_prompt', return_value='offline-system'), \
                patch('pt_agent.planners.agent.initial_message', return_value='{}'), \
                patch('pt_agent.planners.agent.evaluate', side_effect=[feedback(40.), feedback(30.)]), \
                patch('pt_agent.planners.agent.confirm_select', return_value=dict(selected=None, evals=[])):
            result = run_agent(object(), client, max_iters=2,
                               trace_path=Path(temporary)/'trace.jsonl', tuner=False)
        self.assertEqual(result['status'], 'confirm_failed')
        self.assertIsNone(result['best_program'])
        self.assertEqual(result['pool_size'], 2)
        self.assertEqual(len(client.calls), 2)

    def test_07_reconstructs_feasible_resnet20_trace_sources(self):
        expected = []
        for line in TRACE.read_text().splitlines():
            row = json.loads(line)
            if row['fb'].get('feasible'):
                assistant = next(message['content'] for message in row['messages']
                                 if message['role'] == 'assistant')
                expected.append((row['iteration'], 'llm', json.loads(assistant)['program']))
            if row.get('tuned') and row['tuned'].get('feasible'):
                expected.append((row['iteration'], 'tuned', row['tuned']['source']))
        with patch('pt_agent.experiments.r8e6_reconfirm.estimate',
                   return_value=dict(latency_pred_us=1., boots_pred=0)) as estimate:
            candidates, comparisons = r8e6_reconfirm.reconstruct_candidates(TRACE, {}, profile='qwen48')
        self.assertEqual(len(candidates), len(expected))
        self.assertEqual(estimate.call_count, len(expected))
        self.assertEqual(len(comparisons), len(expected))
        self.assertCountEqual([(candidate.iteration, candidate.origin, candidate.source)
                               for candidate in candidates], expected)
        for candidate in candidates:
            with self.subTest(iteration=candidate.iteration, origin=candidate.origin):
                parse(candidate.source)
                self.assertEqual(candidate.latency_pred_us, 1.)
                self.assertEqual(candidate.boots_pred, 0)


if __name__ == '__main__':
    unittest.main()
