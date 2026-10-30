"""A1 §6.11 keeps the pre-revision rule available and scopes the new rule to v8."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.printer import print_program
from pt_agent.llm import LLMError
from pt_agent.planners import agent, transformer_tuner
from pt_agent.planners.tuner import TuneResult
from pt_agent.tests import test_a1_session_steps as session_tests
from pt_agent.tests.test_a1_agent import SOURCE


DEFAULT = 'v8-6.11'
PRE = 'v8-pre-6.11'
Client, feedback, reply = session_tests.Client, session_tests.feedback, session_tests.reply


class AgentRuleTests(unittest.TestCase):
    def equivalent(self, *args, **kwargs):
        return session_tests.SessionStepTests.equivalent(self, *args, **kwargs)

    def run_fake(self, feedbacks, *, version='r8-v8', rule=DEFAULT, once=False,
                 policy='new_best', sources=None, drop_rule=False):
        sources = sources or [SOURCE.replace('deg=15', 'deg='+str(15+16*i))
                              for i in range(len(feedbacks))]
        ctx = SimpleNamespace(prompt_version=version)
        client = Client([reply(source) for source in sources])
        tuning_rules, pools = [], []

        def tuning(program, *args, **kwargs):
            tuning_rules.append(transformer_tuner._tune_rule.get())
            cost = dict(latency_pred_us=5, boots_pred=1)
            return TuneResult(program, print_program(program), [], dict(cost), dict(cost),
                              0, 1, 1., [feedback()['validation']], True)

        def confirming(pool, *args, **kwargs):
            pools.extend(asdict(item) for item in pool)
            selected = min(pool, key=lambda c: (c.latency_pred_us, c.iteration, c.origin)) if pool else None
            return dict(selected=selected, evals=[],
                        deduplicated=dict(before=len(pool), after=len(pool)), duplicates=[])

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(agent, 'build_system_prompt', return_value='system'), \
                patch.object(agent, 'initial_message', return_value='initial'), \
                patch.object(agent, 'evaluate', side_effect=copy.deepcopy(feedbacks)), \
                patch.object(agent, 'tune', side_effect=tuning), \
                patch.object(agent, 'confirm_select', side_effect=confirming):
            trace = Path(tmp)/'trace.jsonl'
            kwargs = dict(max_iters=len(sources), trace_path=trace, tune_policy=policy)
            if rule is not None:
                kwargs['tune_rule'] = rule
            state = None
            if drop_rule:
                state = agent.new_agent_state(ctx, max_iters=len(sources), tune_policy=policy)
                del state['tune_rule']
                while state['phase'] != 'done':
                    agent.agent_step(ctx, state, trace_path=trace, client=client)
                result = agent.agent_result(state)
            else:
                result = (agent._run_agent_once if once else agent.run_agent)(ctx, client, **kwargs)
            raw_trace = trace.read_text()
        result.pop('seconds')
        return dict(result=result, trace=[json.loads(line) for line in raw_trace.splitlines()],
                    raw_trace=raw_trace, messages=client.messages, tuning_rules=tuning_rules,
                    pool=pools, state=state)

    def test_infeasible_cost_does_not_set_min_and_later_feasible_is_tuned(self):
        # Same replies and feedback as the explicitly preserved pre-rule test.
        result = self.equivalent([reply(), reply(SOURCE.replace('deg=15', 'deg=31'))],
                                 [feedback(10, False), feedback(20)], tune_feasible=False)
        self.assertEqual(len(result['tunes']), 2)
        self.assertEqual(result['result']['first_feasible_iteration'], 2)
        self.assertTrue(result['result']['history'][1]['new_llm_best'])
        self.assertEqual(result['result']['history'][1]['llm_min'], 20)
        self.assertIsNone(result['result']['history'][0]['llm_min'])
        self.assertFalse(result['result']['history'][0]['new_llm_best'])

    def test_diverged_program_does_not_update_min_or_mark_tuning(self):
        diverged = feedback(1, False)
        diverged['validation']['diverged'] = True
        for once in (False, True):
            with self.subTest(once=once):
                result = self.run_fake([diverged, feedback(20)], once=once)
                first, second = result['result']['history']
                self.assertIsNone(first['llm_min'])
                self.assertFalse(first['new_llm_best'])
                self.assertFalse(first['has_tuned'])
                self.assertEqual(second['llm_min'], 20)
                self.assertTrue(second['new_llm_best'])
                self.assertEqual(result['tuning_rules'], [DEFAULT])

    def test_tuned_cost_does_not_update_raw_min(self):
        result = self.run_fake([feedback(20), feedback(10)])
        self.assertEqual([row['llm_min'] for row in result['result']['history']], [20, 10])
        self.assertEqual([row['new_llm_best'] for row in result['result']['history']], [True, True])
        self.assertEqual(result['tuning_rules'], [DEFAULT, DEFAULT])
        self.assertEqual(result['result']['best']['latency_pred_us'], 5)

    def test_v4_through_v7_reject_nondefault_rule_before_any_work(self):
        for version in ('r8-v4', 'r8-v5', 'r8-v6', 'r8-v7'):
            for entry in (agent.run_agent, agent._run_agent_once, agent.new_agent_state):
                with self.subTest(version=version, entry=entry.__name__):
                    ctx = SimpleNamespace(prompt_version=version)
                    with patch.object(agent, 'build_system_prompt') as prompt:
                        with self.assertRaises(ValueError):
                            if entry is agent.new_agent_state:
                                entry(ctx, tune_rule=PRE)
                            else:
                                entry(ctx, None, trace_path=Path('unused'), tune_rule=PRE)
                        prompt.assert_not_called()

    def test_legacy_versions_keep_old_trigger_trace_and_result_fields(self):
        for version in ('r8-v4', 'r8-v5', 'r8-v6', 'r8-v7'):
            with self.subTest(version=version):
                default = self.run_fake([feedback(10, False), feedback(20)], version=version)
                explicit = self.run_fake([feedback(10, False), feedback(20)], version=version, rule=DEFAULT)
                self.assertEqual(default, explicit)
                self.assertNotIn('tune_rule', default['result'])
                self.assertNotIn('tune_rule', default['raw_trace'])
                self.assertEqual([r['new_llm_best'] for r in default['result']['history']], [True, False])
                self.assertEqual([r['llm_min'] for r in default['result']['history']], [10, 10])
                self.assertEqual(len(default['tuning_rules']), 1)
                self.assertEqual(set(default['trace'][0]), {
                    'iteration', 'messages', 'fb', 'usage', 'seconds', 'tuned',
                    'tuned_reused_from', 'tune_policy'})

    def test_feasible_inputs_keep_feedback_and_results_apart_from_rule_metadata(self):
        for once in (False, True):
            old = self.run_fake([feedback(30), feedback(20)], rule=PRE, once=once)
            new = self.run_fake([feedback(30), feedback(20)], rule=DEFAULT, once=once)
            self.assertEqual(old['messages'], new['messages'])
            self.assertEqual(old['pool'], new['pool'])
            self.assertEqual(old['tuning_rules'], [PRE, PRE])
            self.assertEqual(new['tuning_rules'], [DEFAULT, DEFAULT])
            for result, rule in ((old, PRE), (new, DEFAULT)):
                self.assertEqual(result['result'].pop('tune_rule'), rule)
                for row in result['trace']:
                    self.assertEqual(row.pop('tune_rule'), rule)
                    self.assertNotIn('tune_rule', row['fb'])
                    self.assertNotIn('tune_rule', row['tuned'])
            self.assertEqual(old['result'], new['result'])
            self.assertEqual(old['trace'], new['trace'])
        self.assertEqual(transformer_tuner._tune_rule.get(), DEFAULT)

    def test_cache_pool_and_rule_survive_process_boundaries(self):
        for rule in (PRE, DEFAULT):
            with self.subTest(rule=rule):
                result = self.equivalent([reply()]*3, [feedback()]*3,
                                         tune_policy='all', tune_rule=rule)
                self.assertEqual(len(result['tunes']), 1)
                self.assertEqual([r['tuned_reused_from'] for r in result['trace']], [None, 1, 1])
                self.assertEqual(result['result']['pool_size'], 6)
                self.assertEqual(result['result']['tune_rule'], rule)
                self.assertEqual([row['tune_rule'] for row in result['trace']], [rule]*3)

    def test_legacy_session_without_rule_uses_pre_revision(self):
        result = self.run_fake([feedback(10, False), feedback(20)], drop_rule=True)
        self.assertEqual(result['tuning_rules'], [PRE])
        self.assertEqual(result['result']['history'][1]['llm_min'], 10)
        self.assertFalse(result['result']['history'][1]['new_llm_best'])
        self.assertEqual(result['result']['tune_rule'], PRE)

    def test_length_interface_and_exhaustion_are_identical_in_both_rules(self):
        cases = [([LLMError('budget_exhausted')], []),
                 ([dict(content='', seconds=1., usage={}, finish_reason='length'), reply()], [feedback()]),
                 ([reply(), reply()], [feedback(stage='interface'), feedback()])]
        for rule in (PRE, DEFAULT):
            for replies, feedbacks in cases:
                with self.subTest(rule=rule, first=replies[0]):
                    result = self.equivalent(replies, feedbacks, tuner=False, tune_rule=rule)
                    self.assertEqual(result['result']['tune_rule'], rule)
                    self.assertTrue(all(row['tune_rule'] == rule for row in result['trace']))

    def test_invalid_rule_is_rejected(self):
        ctx = SimpleNamespace(prompt_version='r8-v8')
        for entry in (agent.run_agent, agent._run_agent_once, agent.new_agent_state):
            with self.subTest(entry=entry.__name__), self.assertRaises(ValueError):
                if entry is agent.new_agent_state:
                    entry(ctx, tune_rule='unknown')
                else:
                    entry(ctx, None, trace_path=Path('unused'), tune_rule='unknown')


if __name__ == '__main__':
    unittest.main()
