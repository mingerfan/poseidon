"""E9a: the tuner-only session uses the saved-reply path and never asks for an LLM client (API 0)."""

import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import e9a_tuner_only as e9a


class CommandTest(unittest.TestCase):
    def test_saved_reply_has_the_session_format(self):
        reply = e9a.saved_reply('@fhe.program(slots=16384)\ndef f(x): return x\n')
        self.assertEqual(sorted(reply), ['content', 'finish_reason', 'id', 'seconds', 'usage'])
        self.assertEqual(json.loads(reply['content'])['program'].splitlines()[0], '@fhe.program(slots=16384)')

    def test_step_command_uses_the_current_pipeline(self):
        command = e9a.step_command('session-step', Path('/o'), 'bert', Path('/m'))
        self.assertEqual(command[command.index('--tune-rule')+1], 'v8-6.25')
        self.assertEqual(command[command.index('--placement')+1], 'program-plan')
        self.assertEqual(command[command.index('--prompt-version')+1], 'r8-v11')

    def test_evaluate_reads_the_reply_and_search_is_sweep(self):
        first = e9a.session_arguments(Path('/s'), 'evaluate', 'tuner-bert-R0', Path('/c'), Path('/l'), Path('/r.json'))
        self.assertEqual(first[first.index('--response-file')+1], '/r.json')
        self.assertEqual(first[first.index('--session-iters')+1], '1')
        for phase in ('tune', 'confirm'):
            later = e9a.session_arguments(Path('/s'), phase, 'tuner-bert-R0', Path('/c'), Path('/l'), Path('/r.json'))
            self.assertNotIn('--response-file', later)
            self.assertEqual(later[later.index('--tune-search')+1], 'sweep')
            self.assertEqual(later[later.index('--confirm-rule')+1], 'c2')

    def test_cases_and_rule_programs(self):
        self.assertEqual(e9a.CASES, (('bert', 'R0'), ('bert', 'R1'), ('qwen', 'R0'), ('qwen', 'R1')))
        self.assertEqual(set(e9a.RULE_ROOTS), {'bert', 'qwen'})


class SummaryTest(unittest.TestCase):
    def test_summary_row(self):
        entry = dict(model='bert', base='R0', rule=dict(boots=dict(predicted=5, exported=5, compiled=5)),
                     session=dict(status='feasible', history=[dict(feasible=True, boots_pred=5, max_u=0.5,
                                  tuned=dict(feasible=True, boots_pred=4, max_u=0.6, seconds=1.0))]),
                     acceptance=dict(boots=dict(predicted=4, exported=4, compiled=4),
                                     report100=dict(tensor=dict(agree=100, total=100, logit_rel_err=0.001))))
        with tempfile.TemporaryDirectory() as tmp:
            lines = e9a.summarize(Path(tmp), dict(cases={'bert-R0': entry}))
            self.assertIn('| bert | R0 | 5/5/5 | True / 5 / 0.5 | True / 4 / 0.6 / 1 | feasible | 4/4/4 | 100/100，logit 0.001 / — |', lines)
            self.assertTrue((Path(tmp)/'E9a.md').exists())


if __name__ == '__main__':
    unittest.main()
