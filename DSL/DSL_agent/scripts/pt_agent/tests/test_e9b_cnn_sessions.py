"""E9b: call limits, replay client, session outcome and the outer control flow (API 0, no Nix)."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import e9b_cnn_sessions as e9b

AGENT_HEADER = "@fhe.program(slots=16384, refresh='agent', placement='program-plan', l_max=5, boot_out_level=5)"
RULE_HEADER = '@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")'


def trace_row(iteration, *, feasible=True, tuned=None, placement=e9b.PLACEMENT, model=e9b.PLACEMENT):
    return dict(iteration=iteration, placement=placement, tuned=tuned,
                fb=dict(stage='complete', feasible=feasible, cost=dict(boots_pred=11, placement_model=model)))


def good_report(calls=3):
    return dict(complete=True, placement=e9b.PLACEMENT, llm_calls=calls,
                agent=dict(status='feasible', placement=e9b.PLACEMENT, llm_calls=calls))


class LimitTest(unittest.TestCase):
    def test_each_session_gets_six_calls_under_the_ceiling(self):
        self.assertEqual(e9b.call_limit(415), 421)
        self.assertEqual(e9b.call_limit(433), 439)
        self.assertEqual(e9b.call_limit(436), 439)
        self.assertEqual(e9b.call_limit(439), 439)

    def test_four_full_sessions_end_on_the_ceiling(self):
        count = 415
        for _ in e9b.ORDER:
            count = e9b.call_limit(count)
        self.assertEqual(count, e9b.API_CEILING)
        self.assertLess(e9b.API_CEILING, 450)


class CommandTest(unittest.TestCase):
    def test_session_command(self):
        command = e9b.session_command('mlp', Path('/o'), Path('/l.jsonl'), 421)
        self.assertEqual(command[1:4], ['-B', '-m', 'pt_agent.tools.nixrun'])
        self.assertEqual(command[command.index('--seconds')+1], str(e9b.SESSION_SECONDS))
        self.assertEqual(command[command.index('--worker')+1], 'session')
        self.assertEqual(command[command.index('--call-limit')+1], '421')
        self.assertEqual(command[command.index('--budget-path')+1], '/l.jsonl')
        self.assertNotIn('--replay', command)
        replay = e9b.session_command('mlp', Path('/o'), Path('/l.jsonl'), 415, replay=Path('/t.jsonl'))
        self.assertEqual(replay[replay.index('--replay')+1], '/t.jsonl')

    def test_other_commands(self):
        compile_ = e9b.compile_command('resnet20', Path('/p.ptdsl'), Path('/c'))
        self.assertEqual([compile_[compile_.index(name)+1] for name in ('--worker', '--model', '--program', '--out')],
                         ['compile', 'resnet20', '/p.ptdsl', '/c'])
        test = e9b.test_command('mlp', Path('/s'), Path('/t'))
        self.assertEqual([test[test.index(name)+1] for name in ('--worker', '--session', '--out')], ['test', '/s', '/t'])
        header = e9b.header_command(Path('/rule.ptdsl'), Path('/h'))
        self.assertIn('pt_agent.experiments.e9a_tuner_only', header)
        self.assertEqual(header[header.index('--worker')+1], 'header')

    def test_rule_programs_and_smoke_trace_exist(self):
        self.assertEqual(set(e9b.RULE_RUNS), set(e9b.ORDER))
        for run in e9b.RULE_RUNS.values():
            self.assertTrue((e9b.RESULTS_DIR/run/'program.ptdsl').is_file(), run)
        self.assertTrue(e9b.SMOKE_TRACE.is_file())


class ReplayTest(unittest.TestCase):
    def test_replies_come_back_in_order_then_the_budget_error(self):
        from pt_agent.llm import LLMError
        client = e9b.ReplayClient(['a', 'b'])
        first = client.chat([], purpose='agent-iter-1')
        self.assertEqual((first['content'], first['finish_reason'], first['usage']), ('a', 'stop', {}))
        self.assertEqual(client.chat([], purpose='agent-iter-2')['content'], 'b')
        with self.assertRaises(LLMError) as caught:
            client.chat([], purpose='agent-iter-3')
        self.assertEqual(caught.exception.code, 'budget_exhausted')
        self.assertEqual(client.used, 2)

    def test_trace_rows_without_a_reply_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl'
            rows = [dict(iteration=1, messages=[dict(role='assistant', content='one'), dict(role='user', content='fb')]),
                    dict(iteration=2, messages=[]),
                    dict(iteration=3, messages=[dict(role='assistant', content='three'), dict(role='user', content='fb')])]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(e9b.ReplayClient.from_trace(path).replies, ['one', 'three'])


class NamespaceTest(unittest.TestCase):
    def test_session_arguments(self):
        args = e9b.run_namespace(Path('/o'), 'cnn_plain', Path('/m'), e9b.SESSION_STAGES, budget_path=Path('/l'), limit=421)
        self.assertEqual((args.planner, args.prompt_version, args.placement, args.tune_policy, args.tuner),
                         ('agent', 'r8-v4', 'program-plan', 'new_best', 'on'))
        self.assertEqual((args.agent_iters, args.llm_call_limit, args.stages), (6, 421, ['plan', 'check', 'mock']))
        self.assertEqual((args.llm_model, args.llm_reasoning, args.llm_max_tokens, args.tuner_budget_s),
                         ('deepseek-flash', 'high', 32768, 2400))
        self.assertNotIn('test', args.stages)

    def test_test_arguments_make_no_call(self):
        args = e9b.run_namespace(Path('/o'), 'mlp', Path('/m'), ('test',))
        self.assertEqual((args.stages, args.final, args.llm_call_limit, args.budget_path), (['test'], True, 0, None))


class OutcomeTest(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(e9b.session_outcome(good_report(), True), 'frozen')
        self.assertEqual(e9b.session_outcome(dict(complete=False, agent=dict(status='confirm_failed')), False), 'confirm_failed')
        self.assertEqual(e9b.session_outcome(dict(complete=False, agent=dict(status='error', error='http_error')), False),
                         'provider_error')
        self.assertEqual(e9b.session_outcome(dict(complete=False, error='KeyError'), False), 'failed')
        self.assertEqual(e9b.session_outcome(dict(complete=False, agent=dict(status='feasible')), True), 'failed')

    def test_header_check_accepts_both_quote_styles(self):
        self.assertTrue(e9b.placed(AGENT_HEADER))
        self.assertTrue(e9b.placed(AGENT_HEADER.replace("'", '"')))
        self.assertFalse(e9b.placed(RULE_HEADER))
        self.assertFalse(e9b.placed(''))


class SmokeTest(unittest.TestCase):
    def rows(self):
        return [trace_row(1, tuned=dict(feasible=True)), trace_row(2, feasible=False), trace_row(3)]

    def test_clean_replay_has_no_problem(self):
        self.assertEqual(e9b.smoke_problems(good_report(), self.rows(), AGENT_HEADER, 3, 415, 415), [])

    def test_each_deviation_is_reported(self):
        cases = dict(
            incomplete=(dict(good_report(), complete=False, error='x'), self.rows(), AGENT_HEADER, 3, 415, 415),
            no_placement=(dict(good_report(), placement=None), self.rows(), AGENT_HEADER, 3, 415, 415),
            trace_placement=(good_report(), [trace_row(1, tuned=dict(a=1)), trace_row(2, placement=None), trace_row(3)],
                             AGENT_HEADER, 3, 415, 415),
            depth_feedback=(good_report(), [trace_row(1, tuned=dict(a=1)), trace_row(2, model='depth-dp'), trace_row(3)],
                            AGENT_HEADER, 3, 415, 415),
            no_tuner=(good_report(), [trace_row(1), trace_row(2), trace_row(3)], AGENT_HEADER, 3, 415, 415),
            rule_header=(good_report(), self.rows(), RULE_HEADER, 3, 415, 415),
            fewer_replies=(good_report(2), self.rows()[:2], AGENT_HEADER, 3, 415, 415),
            ledger=(good_report(), self.rows(), AGENT_HEADER, 3, 415, 416))
        for name, arguments in cases.items():
            self.assertTrue(e9b.smoke_problems(*arguments), name)


class AccountTest(unittest.TestCase):
    def test_one_registry_row_per_new_call(self):
        calls = [dict(ts='t%d' % index, purpose='agent-iter-1') for index in range(1, 419)]
        appended, recorded = [], {417}
        with patch('pt_agent.registry.append', side_effect=appended.append), \
             patch('pt_agent.registry.git_rev', return_value='rev'), \
             patch.object(e9b.r8g, '_calls', return_value=calls):
            total = e9b.account(Path('/l'), 415, 'mlp', Path('/o'), recorded, wait=True)
        self.assertEqual(total, 418)
        self.assertEqual([row['metrics']['global_call'] for row in appended], [416, 418])
        self.assertTrue(all(row['round'] == 'E9' and row['mode'] == 'llm-call' and row['model'] == 'mlp' for row in appended))
        self.assertEqual(recorded, {416, 417, 418})

    def test_a_locked_ledger_is_left_for_the_next_poll(self):
        with patch('pt_agent.registry.append') as append, patch.object(e9b.r8g, '_calls', return_value=None):
            self.assertIsNone(e9b.account(Path('/l'), 415, 'mlp', Path('/o'), set(), wait=False))
        append.assert_not_called()


class SessionChildTest(unittest.TestCase):
    SCRIPT = ('import json, sys, pathlib; out = pathlib.Path(sys.argv[1]); '
              'out.joinpath("program.ptdsl").write_text(sys.argv[2]+"\\ndef f(v):\\n    return v\\n"); '
              'out.joinpath("REPORT.json").write_text(json.dumps(dict(complete=True, placement="program-plan", '
              'llm_calls=0, frozen_source_lines=3, agent=dict(status="feasible", iterations=2, seconds=1.0))))')

    def test_a_frozen_session_is_recorded_without_a_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root/'calls.jsonl'
            ledger.write_text(''.join(json.dumps(dict(ts='t', purpose='p'))+'\n' for _ in range(5)))
            out, appended = root/'session', []
            with patch('pt_agent.registry.append', side_effect=appended.append), \
                 patch('pt_agent.registry.git_rev', return_value='rev'):
                result = e9b.session_child([sys.executable, '-c', self.SCRIPT, str(out), AGENT_HEADER], out, model='mlp',
                                           mode='unit', ledger=ledger, before=5, recorded=set())
            self.assertEqual((result['outcome'], result['frozen'], result['api_calls'], result['exit']), ('frozen', True, 0, 0))
            self.assertEqual((result['status'], result['iterations'], result['driver_error']), ('feasible', 2, None))
            self.assertTrue(e9b.placed(result['header']))
            self.assertEqual(json.loads((out/'SESSION.json').read_text())['api_after'], 5)
            self.assertEqual([row['mode'] for row in appended], ['unit'])

    def test_a_child_without_a_report_counts_as_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root/'calls.jsonl'
            ledger.write_text('')
            with patch('pt_agent.registry.append'), patch('pt_agent.registry.git_rev', return_value='rev'):
                result = e9b.session_child([sys.executable, '-c', 'raise SystemExit(3)'], root/'s', model='mlp', mode='unit',
                                           ledger=ledger, before=0, recorded=set())
            self.assertEqual((result['outcome'], result['frozen'], result['exit'], result['header']), ('failed', False, 3, None))


class ModelFlowTest(unittest.TestCase):
    def flow(self, session, *, ledger=415, compile_=None):
        """run_model with every child replaced; returns (entry or error text, names of the steps that ran)."""
        steps = []
        session = dict(dict(api_before=ledger, api_after=ledger+session.get('api_calls', 0), driver_error=None,
                            agent_error=None, error=None), **session)

        def compiled(model, program, out, mode):
            steps.append('compile')
            return compile_ or dict(complete=True, boots=dict(predicted=9, exported=9, compiled=9), three_way_equal=True)

        def child(command, out, *, model, mode):
            steps.append('test')
            return dict(complete=True, exit=0, seconds=1.0, top1_test=55.9, torch_top1_report=56.0, n_diverged=0)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(e9b, 'rule_baseline', side_effect=lambda folder, model: steps.append('rule') or dict(complete=True)), \
             patch.object(e9b, 'session_child', side_effect=lambda *a, **k: steps.append('session') or session), \
             patch.object(e9b, 'compiled', side_effect=compiled), patch.object(e9b.e9a, 'child', side_effect=child), \
             patch.object(e9b, 'recorded_calls', return_value=set()), \
             patch.object(e9b, 'ledger_rows', return_value=[{}]*ledger), patch.object(e9b.shutil, 'copyfile'):
            try:
                return e9b.run_model(Path(tmp), 'mlp', Path(tmp)/'calls.jsonl'), steps
            except RuntimeError as exc:
                return str(exc), steps

    def test_frozen_session_is_compiled_and_tested(self):
        entry, steps = self.flow(dict(outcome='frozen', frozen=True, api_calls=3))
        self.assertEqual(steps, ['rule', 'session', 'compile', 'test'])
        self.assertEqual(entry['test']['top1_test'], 55.9)

    def test_failed_compile_skips_the_report_set(self):
        entry, steps = self.flow(dict(outcome='frozen', frozen=True, api_calls=3), compile_=dict(complete=False, error='x'))
        self.assertEqual(steps, ['rule', 'session', 'compile'])
        self.assertNotIn('test', entry)

    def test_confirm_failure_is_recorded_and_the_round_continues(self):
        entry, steps = self.flow(dict(outcome='confirm_failed', frozen=False, api_calls=6))
        self.assertEqual(steps, ['rule', 'session'])
        self.assertEqual(entry['session']['outcome'], 'confirm_failed')

    def test_provider_error_stops_the_round(self):
        error, steps = self.flow(dict(outcome='provider_error', frozen=False, api_calls=1, agent_error='http_error'))
        self.assertIn('provider_error: http_error', error)
        self.assertEqual(steps, ['rule', 'session'])

    def test_no_session_starts_at_the_ceiling(self):
        error, steps = self.flow(dict(outcome='frozen', frozen=True), ledger=e9b.API_CEILING)
        self.assertIn('approved ceiling', error)
        self.assertEqual(steps, ['rule'])

    def test_a_session_past_its_limit_stops_the_round(self):
        error, steps = self.flow(dict(outcome='frozen', frozen=True, api_calls=7))
        self.assertIn('call limit', error)
        self.assertEqual(steps, ['rule', 'session'])


class RunTest(unittest.TestCase):
    def run_round(self, smoke, *, smoke_only=False, models=None, fail=None):
        from types import SimpleNamespace
        ran = []

        def run_model(root, model, ledger, entry):
            ran.append(model)
            entry['session'] = dict(outcome='frozen')
            if model == fail:
                raise RuntimeError(model+' session provider_error: http_error')

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.object(e9b, 'smoke', return_value=smoke), patch.object(e9b, 'ledger_rows', return_value=[{}]*415), \
             patch.object(e9b, 'run_model', side_effect=run_model), \
             patch('pt_agent.registry.append'), patch('pt_agent.registry.git_rev', return_value='rev'), \
             patch('traceback.print_exc'), patch('builtins.print'):
            root = Path(tmp)
            code = e9b.run(SimpleNamespace(root=root, models=models, smoke_only=smoke_only))
            return code, ran, json.loads((root/'E9b.json').read_text()), (root/'STOP-b.json').exists(), (root/'E9b.md').read_text()

    def test_a_failed_replay_starts_no_session(self):
        code, ran, data, stopped, _ = self.run_round(dict(complete=False, problems=['the tuner did not run']))
        self.assertEqual((code, ran, data['complete'], stopped), (1, [], False, True))
        self.assertIn('the tuner did not run', data['error'])

    def test_smoke_only(self):
        code, ran, data, stopped, text = self.run_round(dict(complete=True, problems=[]), smoke_only=True)
        self.assertEqual((code, ran, data['complete'], stopped), (0, [], True, False))
        self.assertIn('通过', text)

    def test_models_run_in_order(self):
        code, ran, data, _, _ = self.run_round(dict(complete=True, problems=[]))
        self.assertEqual((code, ran, data['complete']), (0, list(e9b.ORDER), True))
        code, ran, data, _, _ = self.run_round(dict(complete=True, problems=[]), models=['resnet20', 'mlp'])
        self.assertEqual((ran, data['complete']), (['mlp', 'resnet20'], True))


    def test_a_stopped_model_keeps_its_partial_entry_and_stays_open(self):
        code, ran, data, stopped, _ = self.run_round(dict(complete=True, problems=[]), fail='cnn_plain')
        self.assertEqual((code, ran, stopped, data['complete']), (1, ['mlp', 'cnn_plain'], True, False))
        self.assertEqual((sorted(data['models']), sorted(data['stopped'])), (['mlp'], ['cnn_plain']))
        self.assertEqual(data['stopped']['cnn_plain']['session'], dict(outcome='frozen'))


class SummaryTest(unittest.TestCase):
    def test_table_row(self):
        boots = lambda value: dict(predicted=value, exported=value, compiled=value)
        data = dict(api_before=415, api_after=418, smoke=dict(complete=True, compile=dict(boots=boots(11)), problems=[]),
                    models=dict(mlp=dict(
                        rule=dict(boots=boots(16), candidates={'depth-dp': dict(boots=18)}),
                        session=dict(outcome='frozen', api_calls=3, iterations=3, seconds=90.5, frozen_source_lines=7),
                        compile=dict(boots=boots(12), candidates={'depth-dp': dict(boots=13)}),
                        test=dict(top1_test=55.925, torch_top1_report=56.0375, n_diverged=0))))
        with tempfile.TemporaryDirectory() as tmp:
            lines = e9b.summarize(Path(tmp), data)
            self.assertEqual((Path(tmp)/'E9b.md').read_text(), '\n'.join(lines)+'\n')
        row = lines[-1]
        self.assertIn('| mlp | 18 / 16/16/16 | frozen / 3 / 3 / 90.5 | 7 | 13 / 12/12/12 | -25.0% |', row)
        self.assertIn('55.92 / 56.04 / 0 |', row)
        self.assertIn('415 → 418', lines[2])

    def test_missing_parts_show_a_dash(self):
        data = dict(api_before=415, api_after=421, smoke=dict(complete=True), models=dict(
            mlp=dict(rule=dict(complete=False, error='x'), session=dict(outcome='confirm_failed', api_calls=6, iterations=6))))
        with tempfile.TemporaryDirectory() as tmp:
            row = e9b.summarize(Path(tmp), data)[-1]
        self.assertIn('| mlp | — / —/—/— | confirm_failed / 6 / 6 / — | — | — / —/—/— | — | — / — / — |', row)


if __name__ == '__main__':
    unittest.main()
