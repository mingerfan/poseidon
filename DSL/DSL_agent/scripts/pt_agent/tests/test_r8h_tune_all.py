"""R8h orchestration tests: fake replies, evaluations, tuning and confirmation."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import agent
from pt_agent.planners.tuner import TuneResult


def source(index=0):
    return ('@fhe.program(slots=16384)\ndef net(v: Cipher[1, 2]):\n'
            '    x = pack(v, layout="flat")\n'
            f'    y = act(x, fn="relu", method="cheb", deg=15, domain=({-8.-index}, {8.+index}))\n'
            '    return y\n')


def feedback(latency):
    return dict(stage='complete', feasible=True, exact=dict(ok=True),
                validation=dict(agree=2, total=2, max_u=.5, diverged=False,
                                boot_input_bound=None, boot_input_max_abs=1., worst_acts=[]),
                cost=dict(latency_pred_us=latency, boots_pred=1))


def tune_result(text, before, after, *, changes=None):
    program = parse(text)
    return TuneResult(program, print_program(program), changes or [],
                      dict(latency_pred_us=before, boots_pred=1),
                      dict(latency_pred_us=after, boots_pred=1), 0, 1, .25,
                      [dict(agree=2, total=2, max_u=.5, feasible=True,
                            boot_input_max_abs=1., worst_acts=[])], True)


class FakeClient:
    def __init__(self, programs):
        self.programs = iter(programs)
        self.messages = []

    def chat(self, messages, *, purpose):
        self.messages.append(copy.deepcopy(messages))
        return dict(content=json.dumps(dict(program=next(self.programs))), seconds=0,
                    usage=dict(prompt_tokens=0, completion_tokens=0))


class TuneAllTests(unittest.TestCase):
    def run_loop(self, programs, evaluations, tunes, *, max_iters=None):
        client = FakeClient(programs)
        captured = []

        def select(candidates, ctx):
            captured.extend(candidates)
            selected = min(candidates, key=lambda item: (item.latency_pred_us, item.iteration, item.origin)) \
                if candidates else None
            return dict(selected=selected, evals=[])

        with TemporaryDirectory() as directory, \
                patch.object(agent, 'build_system_prompt', return_value='offline-system'), \
                patch.object(agent, 'initial_message', return_value='offline-model'), \
                patch.object(agent, 'evaluate', side_effect=evaluations), \
                patch.object(agent, 'tune', side_effect=tunes) as tuning, \
                patch.object(agent, 'confirm_select', side_effect=select) as confirm:
            trace = Path(directory) / 'trace.jsonl'
            result = agent.run_agent(object(), client, max_iters=max_iters or len(programs), trace_path=trace)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            calls = tuning.call_args_list
        self.assertEqual(confirm.call_count, 1)
        return result, rows, calls, captured, client

    def test_01_every_eligible_candidate_is_tuned(self):
        programs = [source(index) for index in range(3)]
        costs = [100, 120, 90]
        result, rows, calls, pool, client = self.run_loop(
            programs, [feedback(cost) for cost in costs],
            [tune_result(text, cost, cost) for text, cost in zip(programs, costs)])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['iterations'], 3)
        self.assertTrue(all(row['tuned'] is not None for row in rows))
        self.assertTrue(all(row['tuned_reused_from'] is None for row in rows))
        self.assertFalse(result['history'][1]['new_llm_best'])
        self.assertEqual([(c.iteration, c.origin) for c in pool],
                         [(1, 'llm'), (1, 'tuned'), (2, 'llm'), (2, 'tuned'), (3, 'llm'), (3, 'tuned')])
        for messages in client.messages[1:]:
            self.assertIn('tuned', json.loads(messages[-1]['content']))
        for call in calls:
            self.assertEqual(call.kwargs, dict(budget_s=2400, max_sweeps=1))

    def test_02_normalized_duplicates_reuse_the_first_result(self):
        original = source()
        duplicate = '\n\n' + original.replace('    ', '        ')
        self.assertEqual(print_program(parse(original)), print_program(parse(duplicate)))
        act_line = next(op.lineno for op in parse(original).ops if op.kind == 'act')
        change = dict(line=act_line, call_lines=[], kind='domain', old=(-8., 8.), new=(-7., 7.))
        tuned = tune_result(original, 100, 90, changes=[change])
        result, rows, calls, pool, _ = self.run_loop(
            [original, duplicate], [feedback(100), feedback(100)], [tuned])
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows[1]['tuned_reused_from'], 1)
        self.assertEqual(result['history'][1]['tuned_reused_from'], 1)
        self.assertEqual(rows[0]['tuned'], rows[1]['tuned'])
        self.assertEqual(rows[1]['tuned']['changes'][0]['line_text'], original.splitlines()[act_line-1].strip())
        self.assertEqual(len([candidate for candidate in pool if candidate.origin == 'tuned']), 2)
        self.assertNotIn('line_text', tuned.changes[0])

    def test_03_ineligible_candidates_do_not_trigger_tuning(self):
        cases = []
        bad = feedback(100)
        bad.update(feasible=False, exact=dict(ok=False))
        cases.append(bad)
        bad = feedback(100)
        bad.update(feasible=False)
        bad['validation']['diverged'] = True
        cases.append(bad)
        bad = feedback(100)
        bad.update(stage='parse', feasible=False, exact=None, validation={})
        cases.append(bad)
        bad = feedback(100)
        bad.update(feasible=False)
        bad['validation'].update(boot_input_bound=.5, boot_input_max_abs=1.)
        cases.append(bad)
        for index, fb in enumerate(cases):
            with self.subTest(index=index):
                _, rows, calls, pool, _ = self.run_loop([source()], [fb], [])
                self.assertEqual(len(calls), 0)
                self.assertIsNone(rows[0]['tuned'])
                self.assertEqual(pool, [])

    def test_04_slower_raw_candidate_can_win_after_tuning_and_reset_stale(self):
        programs = [source(index) for index in range(4)]
        before, after = [100, 120, 130, 140], [100, 70, 90, 95]
        tunes = [tune_result(text.replace('def net(', f'def tuned{index}('), old, new)
                 for index, (text, old, new) in enumerate(zip(programs, before, after))]
        result, rows, calls, pool, client = self.run_loop(
            programs, [feedback(cost) for cost in before], tunes, max_iters=6)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(client.messages), 4)
        self.assertEqual(result['iterations'], 4)
        self.assertEqual(result['best_iteration'], 2)
        self.assertEqual(result['best_source'], 'tuned')
        self.assertEqual(result['best']['latency_pred_us'], 70)
        self.assertEqual(result['confirm']['selected'].source, tunes[1].source)
        self.assertIn((2, 'tuned', 70), [(c.iteration, c.origin, c.latency_pred_us) for c in pool])
        self.assertEqual([row['tuned']['cost_after']['latency_pred_us'] for row in rows], after)


if __name__ == '__main__':
    unittest.main()
