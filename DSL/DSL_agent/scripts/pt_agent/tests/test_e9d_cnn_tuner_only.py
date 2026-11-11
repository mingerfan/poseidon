"""E9d: echo client, command, per-iteration summary and the outer control flow (API 0, no Nix)."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import e9d_cnn_tuner_only as e9d

AGENT_HEADER = "@fhe.program(slots=16384, refresh='agent', placement='program-plan', l_max=5, boot_out_level=5)"


def program_of(reply):
    return json.loads(reply['content'])['program']


class NextSourceTest(unittest.TestCase):
    def test_a_feasible_tuner_result_becomes_the_next_reply(self):
        self.assertEqual(e9d.next_source('old', dict(tuned=dict(feasible=True, source='new'))), 'new')

    def test_the_previous_reply_is_kept_otherwise(self):
        for row in (None, {}, dict(tuned=None), dict(tuned=dict(feasible=False, source='new')),
                    dict(tuned=dict(feasible=True)), dict(tuned=dict(feasible=True, source=None))):
            self.assertEqual(e9d.next_source('old', row), 'old', row)


class EchoClientTest(unittest.TestCase):
    def test_rule_program_first_then_the_tuner_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)/'agent_trace.jsonl'
            client = e9d.EchoClient('rule', trace)
            first = client.chat([], purpose='agent-iter-1')
            self.assertEqual((program_of(first), first['finish_reason'], first['usage']), ('rule', 'stop', {}))
            self.assertFalse(trace.exists())
            trace.write_text(json.dumps(dict(iteration=1, tuned=dict(feasible=True, source='sweep-1')))+'\n')
            self.assertEqual(program_of(client.chat([], purpose='agent-iter-2')), 'sweep-1')
            with trace.open('a') as stream:
                stream.write(json.dumps(dict(iteration=2, tuned=None))+'\n')
            self.assertEqual(program_of(client.chat([], purpose='agent-iter-3')), 'sweep-1')
            with trace.open('a') as stream:
                stream.write(json.dumps(dict(iteration=3, tuned=dict(feasible=False, source='bad')))+'\n')
            self.assertEqual(program_of(client.chat([], purpose='agent-iter-4')), 'sweep-1')
            self.assertEqual(client.sent, ['rule', 'sweep-1', 'sweep-1', 'sweep-1'])

    def test_a_trace_out_of_step_with_the_replies_stops_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)/'agent_trace.jsonl'
            client = e9d.EchoClient('rule', trace)
            client.chat([], purpose='agent-iter-1')
            trace.write_text('')
            with self.assertRaises(RuntimeError):
                client.chat([], purpose='agent-iter-2')
            self.assertEqual(client.sent, ['rule'])


class CommandTest(unittest.TestCase):
    def test_session_command(self):
        command = e9d.session_command('resnet20', Path('/rule.ptdsl'), Path('/o'), Path('/m'))
        self.assertEqual(command[1:4], ['-B', '-m', 'pt_agent.tools.nixrun'])
        self.assertEqual(command[command.index('--seconds')+1], str(e9d.e9b.SESSION_SECONDS))
        self.assertIn('pt_agent.experiments.e9d_cnn_tuner_only', command)
        self.assertEqual([command[command.index(name)+1] for name in ('--worker', '--model', '--program', '--out', '--models-dir')],
                         ['session', 'resnet20', '/rule.ptdsl', '/o', '/m'])
        self.assertNotIn('--budget-path', command)

    def test_the_loop_matches_the_agent_sessions(self):
        self.assertEqual((e9d.ORDER, e9d.ITERATIONS, e9d.PLACEMENT), (e9d.e9b.ORDER, 6, 'program-plan'))
        args = e9d.e9b.run_namespace(Path('/o'), 'mlp', Path('/m'), e9d.e9b.SESSION_STAGES, iters=e9d.ITERATIONS)
        self.assertEqual((args.agent_iters, args.budget_path, args.llm_call_limit, args.tuner, args.tune_policy),
                         (6, None, 0, 'on', 'new_best'))


class SweepsTest(unittest.TestCase):
    def test_reply_and_tuner_counts_per_iteration(self):
        session = dict(trace=[
            dict(iteration=1, feasible=True, feedback=dict(cost=dict(boots_pred=210)),
                 tuned=dict(feasible=True, cost_after=dict(boots_pred=180))),
            dict(iteration=2, feasible=True, feedback=dict(cost=dict(boots_pred=180)), tuned=None),
            dict(iteration=3, feasible=False, feedback={}, tuned=None)])
        self.assertEqual(e9d.sweeps(session), [
            dict(iteration=1, feasible=True, reply=210, tuned=180, tuned_feasible=True),
            dict(iteration=2, feasible=True, reply=180, tuned=None, tuned_feasible=None),
            dict(iteration=3, feasible=False, reply=None, tuned=None, tuned_feasible=None)])
        self.assertEqual(e9d.sweeps({}), [])


class ReferenceTest(unittest.TestCase):
    def test_counts_come_from_the_e9b_summary(self):
        boots = lambda value: dict(boots=dict(predicted=value, exported=value, compiled=value))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(e9d.reference(root), {})
            (root/'E9b.json').write_text(json.dumps(dict(models=dict(
                resnet20_silu=dict(rule=boots(210), compile=boots(91), test=dict(top1_test=92.1625)),
                mlp=dict(rule=dict(complete=False))))))
            self.assertEqual(e9d.reference(root), dict(resnet20_silu=dict(rule=210, agent=91, agent_top1=92.1625),
                                                       mlp=dict(rule=None, agent=None, agent_top1=None)))


class ModelFlowTest(unittest.TestCase):
    def flow(self, session, *, compile_=None):
        """run_model with every child replaced; returns (entry or error text, names of the steps that ran)."""
        steps = []
        session = dict(dict(api_calls=0, driver_error=None, agent_error=None, error=None, header=AGENT_HEADER, trace=[]),
                       **session)

        def compiled(model, program, out, mode):
            steps.append('compile')
            return compile_ or dict(complete=True, boots=dict(predicted=9, exported=9, compiled=9), three_way_equal=True)

        def child(command, out, *, model, mode):
            steps.append('test')
            return dict(complete=True, exit=0, seconds=1.0, top1_test=55.9, torch_top1_report=56.0, n_diverged=0)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(e9d.e9b, 'session_child', side_effect=lambda *a, **k: steps.append('session') or session), \
             patch.object(e9d.e9b, 'compiled', side_effect=compiled), patch.object(e9d.e9a, 'child', side_effect=child), \
             patch.object(e9d.e9b, 'recorded_calls', return_value=set()), \
             patch.object(e9d, 'ledger_rows', return_value=[{}]*430), patch.object(e9d.shutil, 'copyfile'):
            try:
                return e9d.run_model(Path(tmp), 'mlp', Path(tmp)/'calls.jsonl'), steps
            except RuntimeError as exc:
                return str(exc), steps

    def test_frozen_session_is_compiled_and_tested(self):
        entry, steps = self.flow(dict(outcome='frozen', frozen=True))
        self.assertEqual(steps, ['session', 'compile', 'test'])
        self.assertEqual((entry['test']['top1_test'], entry['sweeps']), (55.9, []))
        self.assertTrue(entry['program'].endswith('program.ptdsl'))

    def test_failed_compile_skips_the_report_set(self):
        entry, steps = self.flow(dict(outcome='frozen', frozen=True), compile_=dict(complete=False, error='x'))
        self.assertEqual(steps, ['session', 'compile'])
        self.assertNotIn('test', entry)

    def test_confirm_failure_is_recorded_and_the_round_continues(self):
        entry, steps = self.flow(dict(outcome='confirm_failed', frozen=False))
        self.assertEqual(steps, ['session'])
        self.assertEqual(entry['session']['outcome'], 'confirm_failed')

    def test_a_new_ledger_row_stops_the_round(self):
        error, steps = self.flow(dict(outcome='frozen', frozen=True, api_calls=1))
        self.assertIn('ledger changed', error)
        self.assertEqual(steps, ['session'])

    def test_a_failed_session_stops_the_round(self):
        error, steps = self.flow(dict(outcome='failed', frozen=False, error='KeyError'))
        self.assertIn('failed: KeyError', error)
        self.assertEqual(steps, ['session'])

    def test_a_frozen_program_without_the_placement_header_stops_the_round(self):
        error, steps = self.flow(dict(outcome='frozen', frozen=True, header='@fhe.program(slots=16384)'))
        self.assertIn('header', error)
        self.assertEqual(steps, ['session'])


class RunTest(unittest.TestCase):
    def run_round(self, *, models=None, fail=None, ledger=(430, 430)):
        from types import SimpleNamespace
        ran, counts = [], iter([ledger[0]]+[ledger[1]]*20)

        def run_model(root, model, ledger_path, entry):
            ran.append(model)
            entry['session'] = dict(outcome='frozen')
            if model == fail:
                raise RuntimeError(model+' session failed: KeyError')

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.object(e9d, 'ledger_rows', side_effect=lambda path: [{}]*next(counts)), \
             patch.object(e9d, 'run_model', side_effect=run_model), \
             patch('pt_agent.registry.append'), patch('pt_agent.registry.git_rev', return_value='rev'), \
             patch('traceback.print_exc'), patch('builtins.print'):
            root = Path(tmp)
            code = e9d.run(SimpleNamespace(root=root, models=models))
            return code, ran, json.loads((root/'E9d.json').read_text()), (root/'STOP-d.json').exists()

    def test_models_run_in_order(self):
        code, ran, data, stopped = self.run_round()
        self.assertEqual((code, ran, data['complete'], stopped), (0, list(e9d.ORDER), True, False))
        code, ran, data, _ = self.run_round(models=['resnet20', 'mlp'])
        self.assertEqual((ran, data['complete']), (['mlp', 'resnet20'], True))

    def test_a_stopped_model_keeps_its_partial_entry(self):
        code, ran, data, stopped = self.run_round(fail='cnn_plain')
        self.assertEqual((code, ran, stopped, data['complete']), (1, ['mlp', 'cnn_plain'], True, False))
        self.assertEqual((sorted(data['models']), sorted(data['stopped'])), (['mlp'], ['cnn_plain']))

    def test_a_changed_ledger_stops_after_the_first_model(self):
        code, ran, data, stopped = self.run_round(ledger=(430, 431))
        self.assertEqual((code, ran, stopped, data['complete']), (1, ['mlp'], True, False))
        self.assertIn('changed the API ledger', data['error'])


class SummaryTest(unittest.TestCase):
    def test_table_row(self):
        boots = lambda value: dict(predicted=value, exported=value, compiled=value)
        data = dict(api_before=430, api_after=430, reference=dict(resnet20_silu=dict(rule=210, agent=91, agent_top1=92.16)),
                    models=dict(resnet20_silu=dict(
                        session=dict(outcome='frozen', iterations=3, seconds=1200.5, best_source='tuned', best_iteration=2),
                        sweeps=[dict(reply=210, tuned=180), dict(reply=180, tuned=140), dict(reply=140, tuned=None)],
                        compile=dict(boots=boots(140)),
                        test=dict(top1_test=92.3, torch_top1_report=92.5, n_diverged=2))))
        with tempfile.TemporaryDirectory() as tmp:
            lines = e9d.summarize(Path(tmp), data)
            self.assertEqual((Path(tmp)/'E9d.md').read_text(), '\n'.join(lines)+'\n')
        row = lines[-1]
        self.assertIn('| resnet20_silu | 210 | 210 → 180；180 → 140；140 → — | frozen / 3 / 1200 | 精调器输出，第 2 轮 |', row)
        self.assertIn('| 140/140/140 | -33.3% | 91 | -35.0% | 92.3 / 92.5 / 2 |', row)
        self.assertIn('430 → 430', lines[2])

    def test_missing_parts_show_a_dash(self):
        data = dict(api_before=430, api_after=430, reference={}, models=dict(
            mlp=dict(session=dict(outcome='confirm_failed', iterations=6), sweeps=[])))
        with tempfile.TemporaryDirectory() as tmp:
            row = e9d.summarize(Path(tmp), data)[-1]
        self.assertIn('| mlp | — | — | confirm_failed / 6 / — | — | —/—/— | — | — | — | — / — / — |', row)


if __name__ == '__main__':
    unittest.main()
