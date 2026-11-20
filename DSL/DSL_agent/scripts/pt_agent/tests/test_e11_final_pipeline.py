"""Offline checks of the E11 driver: the first reply, saved replies, the step arguments and the summary (outer python3)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pt_agent.experiments import e11_final_pipeline as e11


class FirstReply(unittest.TestCase):
    def test_rule_program_first_then_the_inner_client(self):
        inner = mock.Mock()
        inner.chat.return_value = dict(content='{"program": "llm"}')
        client = e11.FirstReplyClient('rule source', inner)
        first = client.chat([], purpose='agent-iter-1')
        self.assertEqual(json.loads(first['content']), dict(program='rule source'))
        inner.chat.assert_not_called()
        self.assertEqual(client.chat(['m'], purpose='agent-iter-2')['content'], '{"program": "llm"}')
        inner.chat.assert_called_once_with(['m'], purpose='agent-iter-2')
        self.assertEqual(client.served, 2)


class SavedReplies(unittest.TestCase):
    def test_takes_the_first_assistant_messages(self):
        rows = [dict(iteration=1, messages=[dict(role='assistant', content='a'), dict(role='user', content='f')]),
                dict(iteration=2, messages=[]),
                dict(iteration=3, messages=[dict(role='assistant', content='b'), dict(role='user', content='g')]),
                dict(iteration=4, messages=[dict(role='assistant', content='c'), dict(role='user', content='h')])]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)/'agent_trace.jsonl'
            trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with mock.patch.dict(e11.REPLAY_TRACES, mlp=trace):
                self.assertEqual(e11.saved_replies('mlp'), ['a', 'b', 'c'])
                self.assertEqual(e11.saved_replies('mlp', count=2), ['a', 'b'])
            (Path(tmp)/'mlp/session').mkdir(parents=True)
            (Path(tmp)/'mlp/session/agent_trace.jsonl').write_text(trace.read_text())
            self.assertEqual(e11.saved_replies('mlp', source=Path(tmp)), ['b', 'c'])


class TransformerArguments(unittest.TestCase):
    def arguments(self, phase, iteration, mode, files=('reply-1.json', 'reply-2.json'), variant='e11'):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'bert'
            folder.mkdir()
            for name in files:
                (folder/name).write_text('{}')
            args = mock.Mock(mode=mode, ledger_cap=450, variant=variant)
            return folder, e11.transformer_arguments(folder/'session', phase, iteration, folder, args, Path(tmp)/'ledger')

    def test_settings(self):
        folder, extra = self.arguments('tune', 1, 'live')
        for flag, value in (('--tune-policy', 'all'), ('--stale-limit', '1'), ('--focus-retune', 'on'),
                            ('--session-iters', '4'), ('--confirm-rule', 'c2'), ('--tune-search', 'sweep'),
                            ('--ledger-cap', '450')):
            self.assertEqual(extra[extra.index(flag)+1], value)
        self.assertNotIn('--response-file', extra)
        self.assertEqual(extra[extra.index('--numeric-lock')+1], 'off')
        self.assertEqual(extra[extra.index('--session-label')+1], 'E11')

    def test_e12_locks_settings_and_keeps_all_rewrites(self):
        folder, extra = self.arguments('tune', 1, 'live', variant='e12')
        self.assertEqual(extra[extra.index('--numeric-lock')+1], 'on')
        self.assertEqual(extra[extra.index('--stale-limit')+1], '3')
        self.assertEqual(extra[extra.index('--session-label')+1], 'E12')

    def test_e13_keeps_all_rewrites_without_the_lock(self):
        folder, extra = self.arguments('tune', 1, 'live', variant='e13')
        self.assertEqual(extra[extra.index('--numeric-lock')+1], 'off')
        self.assertEqual(extra[extra.index('--stale-limit')+1], '3')
        self.assertEqual(extra[extra.index('--session-label')+1], 'E13')

    def test_first_evaluate_always_uses_the_rule_program(self):
        folder, extra = self.arguments('evaluate', 1, 'live')
        self.assertEqual(extra[extra.index('--response-file')+1], str(folder/'reply-1.json'))

    def test_later_evaluates_call_the_provider_in_live_mode(self):
        folder, extra = self.arguments('evaluate', 2, 'live')
        self.assertNotIn('--response-file', extra)

    def test_later_evaluates_use_saved_replies_in_replay_mode(self):
        folder, extra = self.arguments('evaluate', 2, 'replay')
        self.assertEqual(extra[extra.index('--response-file')+1], str(folder/'reply-2.json'))
        with self.assertRaises(RuntimeError):
            self.arguments('evaluate', 3, 'replay')


class IterationSteps(unittest.TestCase):
    def test_reads_reply_tuned_and_focus(self):
        rows = [dict(iteration=1, fb=dict(feasible=True, cost=dict(boots_pred=210)), seconds=0.0,
                     tuned=dict(cost_after=dict(boots_pred=143), seconds=400.0)),
                dict(iteration=2, fb=dict(feasible=True, cost=dict(boots_pred=150)), seconds=90.0,
                     tuned=dict(cost_after=dict(boots_pred=120), seconds=60.0, focus_lines=[5, 9]))]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'agent_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            steps = e11.iteration_steps(Path(tmp))
        self.assertEqual([(s['reply'], s['tuned'], s['focus']) for s in steps], [(210, 143, None), (150, 120, 2)])


class RulePrograms(unittest.TestCase):
    def test_transformers_start_from_r1_and_cnns_from_e1(self):
        self.assertEqual(e11.rule_program('bert').parent.name, 'R1')
        self.assertTrue(str(e11.rule_program('resnet20')).endswith('program.ptdsl'))


if __name__ == '__main__':
    unittest.main()
