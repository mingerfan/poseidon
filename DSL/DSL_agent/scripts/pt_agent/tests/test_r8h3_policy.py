"""Offline tuning policy and frozen-trace trigger equivalence."""

import copy
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.paths import pt_result
from pt_agent.planners import agent
from pt_agent.planners.tuner import TuneResult
from pt_agent.tests._artifacts import require
from pt_agent.tests.test_r8h_tune_all import FakeClient, feedback, source, tune_result


SOURCE_RUN = pt_result('r8e5-dev-20261022T041646')
NAMES = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
TRACES = [SOURCE_RUN / name / 'agent_trace.jsonl' for name in NAMES]


class TunePolicyTests(unittest.TestCase):
    def test_01_default_only_tunes_new_best(self):
        programs = [source(index) for index in range(3)]
        with TemporaryDirectory() as directory, \
                patch.object(agent, 'build_system_prompt', return_value='offline-system'), \
                patch.object(agent, 'initial_message', return_value='offline-model'), \
                patch.object(agent, 'evaluate', side_effect=[feedback(cost) for cost in (100, 120, 90)]), \
                patch.object(agent, 'tune', side_effect=[tune_result(programs[0], 100, 100),
                                                       tune_result(programs[2], 90, 90)]) as tuning, \
                patch.object(agent, 'confirm_select', return_value=dict(selected=None, evals=[])):
            path = Path(directory) / 'trace.jsonl'
            result = agent.run_agent(object(), FakeClient(programs), max_iters=3, trace_path=path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(tuning.call_count, 2)
        self.assertEqual([row['iteration'] for row in rows if row['tuned'] is not None], [1, 3])
        self.assertEqual(result['iterations'], 3)
        self.assertEqual(result['tune_policy'], 'new_best')
        self.assertTrue(all(row['tune_policy'] == 'new_best' for row in rows))
        self.assertTrue(all(row['tuned_reused_from'] is None for row in rows))

    def test_02_invalid_policy_fails_before_any_work(self):
        with patch.object(agent, 'build_system_prompt') as prompt, \
                patch.object(agent, 'evaluate') as evaluate, patch.object(agent, 'tune') as tune:
            with self.assertRaisesRegex(ValueError, 'tune_policy'):
                agent.run_agent(object(), object(), trace_path=Path('unused.jsonl'), tune_policy='invalid')
        prompt.assert_not_called()
        evaluate.assert_not_called()
        tune.assert_not_called()

    def test_03_entrypoints_accept_default_and_all(self):
        entries = (
            ('pt_agent.run', '_inside', ['--model', 'mlp', '--planner', 'agent', '--stages', 'plan']),
            ('pt_agent.experiments.r8b_dev', '_inside', ['--model', 'mlp', '--baseline-rule-report', 'unused.json']),
            ('pt_agent.experiments.r8g_mlp_rerun', '_inside', []),
            ('pt_agent.experiments.r8h_rerun', 'inside', ['--model', 'mlp']),
            ('pt_agent.experiments.r8h2_reconfirm', 'inside', ['--model', 'resnet20', '--agent']),
        )
        with TemporaryDirectory() as directory:
            common = ['--inside', '--out', directory, '--models-dir', directory,
                      '--budget-path', str(Path(directory) / 'unused-calls.jsonl')]
            for name, target, extra in entries:
                module = importlib.import_module(name)
                for selected in ('new_best', 'all'):
                    flags = [] if selected == 'new_best' else ['--tune-policy', 'all']
                    if name != 'pt_agent.run':
                        flags += ['--call-limit', '71']
                    with self.subTest(module=name, policy=selected), \
                            patch.object(sys, 'argv', [name, *common, *extra, *flags]), \
                            patch.object(module, target, return_value=0) as inside:
                        self.assertEqual(module.main(), 0)
                        self.assertEqual(inside.call_args.args[0].tune_policy, selected)
            from pt_agent.experiments.r8g_mlp_rerun import _run_args

            args = SimpleNamespace(out=Path(directory), models_dir=Path(directory),
                                   budget_path=Path(directory) / 'unused-calls.jsonl', call_limit=71)
            self.assertEqual(_run_args(args).tune_policy, 'new_best')
            args.tune_policy = 'all'
            self.assertEqual(_run_args(args).tune_policy, 'all')


@require(TRACES)
class FrozenTracePolicyTests(unittest.TestCase):
    def compare(self, name):
        rows = [json.loads(line) for line in (SOURCE_RUN / name / 'agent_trace.jsonl').read_text().splitlines()]
        expected = [row['iteration'] for row in rows if row.get('tuned') is not None]
        called = []

        class ReplayClient:
            current = None

            def chat(self, messages, *, purpose):
                self.current = rows[int(purpose.rsplit('-', 1)[1]) - 1]
                content = next(message['content'] for message in self.current['messages']
                               if message['role'] == 'assistant')
                return dict(content=content, seconds=0, usage={})

        client = ReplayClient()

        def fake_tune(program, ctx, **kwargs):
            called.append(client.current['iteration'])
            stored = copy.deepcopy(client.current['tuned'])
            self.assertIsNotNone(stored, 'default policy tuned an iteration absent from the frozen trace')
            return TuneResult(program=parse(stored['source']), **stored)

        with TemporaryDirectory() as directory, \
                patch.object(agent, 'build_system_prompt', return_value='offline-system'), \
                patch.object(agent, 'initial_message', return_value='offline-model'), \
                patch.object(agent, 'feedback_message', return_value='offline-feedback'), \
                patch.object(agent, 'evaluate', side_effect=lambda *_: copy.deepcopy(client.current['fb'])), \
                patch.object(agent, 'tune', side_effect=fake_tune), \
                patch.object(agent, 'confirm_select', return_value=dict(selected=None, evals=[])):
            path = Path(directory) / 'trace.jsonl'
            result = agent.run_agent(object(), client, max_iters=len(rows), trace_path=path)
            replayed = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(called, expected)
        self.assertEqual(result['iterations'], len(rows))
        self.assertEqual([row['iteration'] for row in replayed if row['tuned'] is not None], expected)
        self.assertTrue(all(row['tune_policy'] == 'new_best' for row in replayed))
        print('TRIGGER_SEQUENCE ' + json.dumps(dict(model=name, expected=expected, actual=called)), flush=True)

    def test_resnet20(self):
        self.compare('resnet20')

    def test_resnet20_silu(self):
        self.compare('resnet20_silu')

    def test_cnn_plain(self):
        self.compare('cnn_plain')

    def test_mlp(self):
        self.compare('mlp')


if __name__ == '__main__':
    unittest.main()
