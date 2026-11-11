"""E9c: the BERT replay feeds saved replies in order, creates no LLM client and leaves the ledger unchanged."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pt_agent.experiments import e9c_bert_replay as e9c
from pt_agent.tests._artifacts import require

SPEC = dict(id='T1', root='/unused', calls='calls', session='session', acceptance='acceptance', replies=3)


def saved(count):
    return [Path(f'/replies/call-{index:03d}.response.json') for index in range(1, count+1)]


class ArgumentTest(unittest.TestCase):
    def test_evaluate_reads_the_reply_of_its_iteration(self):
        for iteration in (1, 2, 3):
            extra = e9c.step_arguments(Path('/s'), 'evaluate', 'replay-T1', Path('/c'), Path('/l'), saved(3), iteration)
            self.assertEqual(extra[extra.index('--response-file')+1], f'/replies/call-{iteration:03d}.response.json')
            self.assertEqual(extra[extra.index('--session-iters')+1], '3')

    def test_later_steps_use_the_final_rules_without_a_reply(self):
        for phase in ('tune', 'confirm'):
            extra = e9c.step_arguments(Path('/s'), phase, 'replay-T1', Path('/c'), Path('/l'), saved(3), 2)
            self.assertNotIn('--response-file', extra)
            self.assertEqual(extra[extra.index('--tune-search')+1], 'sweep')
            self.assertEqual(extra[extra.index('--confirm-rule')+1], 'c2')

    def test_an_iteration_without_a_saved_reply_is_refused(self):
        for iteration in (0, 4):
            with self.assertRaises(RuntimeError):
                e9c.step_arguments(Path('/s'), 'evaluate', 'replay-T1', Path('/c'), Path('/l'), saved(3), iteration)

    def test_step_limit(self):
        self.assertEqual(e9c.step_limit(5), 11)

    def test_specs(self):
        self.assertEqual([spec['id'] for spec in e9c.SPECS], ['E5d', 'E6-s1', 'E6r2-p1', 'E6r2-p2'])
        self.assertEqual([spec['replies'] for spec in e9c.SPECS], [7, 5, 5, 7])
        self.assertEqual((e9c.RULE, e9c.PLACEMENT, e9c.MODEL), ('v8-6.25', 'program-plan', 'bert'))


@require(*(Path(spec['root'])/spec['session']/'AGENT.json' for spec in e9c.SPECS))
class OriginalTest(unittest.TestCase):
    def test_saved_replies_and_accepted_counts(self):
        compiled = []
        for spec in e9c.SPECS:
            self.assertEqual(len(e9c.saved_replies(spec)), spec['replies'])
            old = e9c.original(spec)
            self.assertEqual((old['status'], old['tune_rule']), ('feasible', 'v8-6.24'))
            self.assertEqual(len(old['history']), spec['replies'])
            compiled.append(old['boots']['compiled'])
        self.assertEqual(compiled, [7175, 6808, 7141, 6977])

    def test_a_wrong_reply_count_stops(self):
        with self.assertRaises(RuntimeError):
            e9c.saved_replies(dict(e9c.SPECS[0], replies=6))


class ReplayTest(unittest.TestCase):
    """The step loop with a scripted child: each child writes the next session state."""

    def run_replay(self, script, status='feasible'):
        commands = []

        def child(command, out, *, model, mode):
            commands.append((mode, list(command)))
            session = Path(command[command.index('--root')+1])
            state, result = script[len(commands)-1] if len(commands) <= len(script) else script[-1]
            session.mkdir(parents=True, exist_ok=True)
            (session/'SESSION.json').write_text(json.dumps(state))
            if state['phase'] == 'done':
                (session/'AGENT.json').write_text(json.dumps(dict(
                    status=status, iterations=2, best_iteration=2, best=dict(boots_pred=90), tune_rule='v8-6.25',
                    tune_search='sweep', confirm_rule='c2',
                    history=[dict(iteration=1, feasible=False), dict(iteration=2, feasible=True, boots_pred=100, max_u=0.9,
                                                                   tuned=dict(feasible=True, boots_pred=90, max_u=0.95, seconds=3.0))],
                    confirm=dict(selected=dict(origin='tuned', iteration=2, boots_pred=90, source='x'),
                                 evals=[dict(origin='tuned', iteration=2, boots_pred=90, passed=True, agree=20, total=20)]))))
            return dict(result, seconds=1.0)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(e9c, 'saved_replies', return_value=saved(3)), \
                mock.patch.object(e9c.e9a, 'child', side_effect=child), \
                mock.patch.object(e9c.e9a, 'register') as register, \
                mock.patch('pt_agent.planners.transformer_session.load_state', side_effect=lambda path: json.loads(Path(path).read_text())):
            folder = Path(tmp)/'c'/'T1'
            folder.mkdir(parents=True)
            try:
                result = e9c.replay(Path(tmp), folder, SPEC, Path('/model'), Path('/ledger'))
            except RuntimeError as exc:
                result = exc
            report = json.loads((folder/'session/REPORT.json').read_text()) if (folder/'session/REPORT.json').exists() else None
        return result, commands, register, report

    def test_replies_are_fed_in_order_and_the_result_is_frozen(self):
        ok = dict(complete=True, exit=0)
        script = [(dict(phase='evaluate', iteration=2), ok), (dict(phase='tune', iteration=2), ok),
                  (dict(phase='evaluate', iteration=3), ok), (dict(phase='confirm', iteration=3), ok),
                  (dict(phase='done', iteration=3), ok)]
        result, commands, register, report = self.run_replay(script)
        self.assertEqual([mode for mode, _ in commands], ['e9c-T1-evaluate', 'e9c-T1-evaluate', 'e9c-T1-tune',
                                                          'e9c-T1-evaluate', 'e9c-T1-confirm'])
        replies = [command[command.index('--response-file')+1] for _, command in commands if '--response-file' in command]
        self.assertEqual(replies, [f'/replies/call-00{index}.response.json' for index in (1, 2, 3)])
        for _, command in commands:
            self.assertEqual(command[command.index('--tune-rule')+1], 'v8-6.25')
            self.assertEqual(command[command.index('--placement')+1], 'program-plan')
            self.assertEqual(command[command.index('--session-label')+1], 'replay-T1')
        self.assertTrue(result['complete'] and result['frozen'])
        self.assertEqual((result['main_api_calls'], result['saved_replies'], len(result['steps'])), (0, 3, 5))
        self.assertEqual(result['selected'], dict(origin='tuned', iteration=2, boots_pred=90))
        self.assertEqual(result['history'][1]['tuned'], dict(feasible=True, boots_pred=90, max_u=0.95, seconds=3.0))
        self.assertEqual(report['label'], e9c.NOTE)
        register.assert_called_once()

    def test_a_session_without_a_passing_candidate_is_complete_and_unfrozen(self):
        ok = dict(complete=True, exit=0)
        result, _, _, _ = self.run_replay([(dict(phase='done', iteration=1), ok)], status='confirm_failed')
        self.assertTrue(result['complete'])
        self.assertFalse(result['frozen'])

    def test_a_failed_step_ends_the_replay(self):
        result, commands, register, report = self.run_replay(
            [(dict(phase='evaluate', iteration=2), dict(complete=False, exit=1, error='boom'))])
        self.assertEqual(len(commands), 1)
        self.assertEqual((result['complete'], result['error'], result['main_api_calls']), (False, 'boom', 0))
        self.assertFalse(report['complete'])
        register.assert_not_called()

    def test_a_session_that_never_ends_is_stopped(self):
        result, commands, _, _ = self.run_replay([(dict(phase='tune', iteration=1), dict(complete=True, exit=0))])
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(len(commands), e9c.step_limit(3))


class RunTest(unittest.TestCase):
    def run_driver(self, sessions=None, grow=False, existing=None):
        started = []

        def run_session(root, spec, model_dir, ledger):
            started.append(spec['id'])
            if grow:
                with ledger.open('a') as handle:
                    handle.write('{}\n')
            return dict(id=spec['id'], session=dict(status='feasible', frozen=True))

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)/'results'
            results.mkdir()
            (results/'llm_calls.jsonl').write_text('{}\n{}\n')
            root = Path(tmp)/'root'
            if existing:
                root.mkdir()
                (root/'E9c.json').write_text(json.dumps(dict(api_before=2, sessions={name: dict(id=name) for name in existing})))
            with mock.patch.object(e9c, 'RESULTS_DIR', results), mock.patch.object(e9c, 'run_session', side_effect=run_session), \
                    mock.patch.object(e9c.e9a, 'register') as register, mock.patch.dict('os.environ'), \
                    mock.patch.object(e9c.e7, 'inventory', return_value=dict(models=dict(bert='/model'))), \
                    mock.patch('traceback.print_exc'), mock.patch('builtins.print'):
                code = e9c.run(mock.Mock(root=root, sessions=sessions))
            data = json.loads((root/'E9c.json').read_text())
            stopped = (root/'STOP-c.json').exists()
            summary = (root/'E9c.md').exists()
        return code, started, data, stopped, summary, register

    def test_all_sessions_run_in_order_with_an_unchanged_ledger(self):
        code, started, data, stopped, summary, register = self.run_driver()
        self.assertEqual((code, stopped, summary), (0, False, True))
        self.assertEqual(started, ['E5d', 'E6-s1', 'E6r2-p1', 'E6r2-p2'])
        self.assertEqual((data['api_before'], data['api_after'], data['complete'], data['error']), (2, 2, True, None))
        self.assertEqual(register.call_args.args[1], 'e9c-summary')

    def test_a_ledger_change_stops_the_run(self):
        code, started, data, stopped, _, _ = self.run_driver(grow=True)
        self.assertEqual((code, started, stopped), (1, ['E5d'], True))
        self.assertFalse(data['complete'])
        self.assertIn('ledger', data['error'])

    def test_selected_sessions_only(self):
        code, started, data, _, _, _ = self.run_driver(sessions=['E6-s1'])
        self.assertEqual((code, started, data['complete']), (0, ['E6-s1'], True))

    def test_finished_sessions_are_kept(self):
        code, started, data, _, _, _ = self.run_driver(existing=['E5d', 'E6-s1'])
        self.assertEqual((code, started), (0, ['E6r2-p1', 'E6r2-p2']))
        self.assertEqual(sorted(data['sessions']), ['E5d', 'E6-s1', 'E6r2-p1', 'E6r2-p2'])


class SummaryTest(unittest.TestCase):
    def test_change(self):
        self.assertAlmostEqual(e9c.change(dict(compiled=200), dict(compiled=190)), -5.0)
        self.assertIsNone(e9c.change(dict(compiled=200), None))
        self.assertIsNone(e9c.change(None, dict(compiled=190)))

    def test_summary_row(self):
        history = [dict(iteration=1, feasible=False), dict(iteration=2, feasible=True, boots_pred=200,
                                                           tuned=dict(feasible=True, boots_pred=190))]
        entry = dict(id='E5d',
                     original=dict(tune_rule='v8-6.24', tune_search=None, confirm_rule=None, history=history,
                                   boots=dict(predicted=200, exported=200, compiled=200)),
                     session=dict(status='feasible', saved_replies=2, history=history,
                                  selected=dict(origin='tuned', iteration=2, boots_pred=190)),
                     acceptance=dict(boots=dict(predicted=190, exported=190, compiled=190),
                                     report100=dict(tensor=dict(agree=100, total=100, logit_rel_err=0.002))))
        with tempfile.TemporaryDirectory() as tmp:
            lines = e9c.summarize(Path(tmp), dict(sessions={'E5d': entry}))
            self.assertIn('| E5d | 2 | v8-6.24/sweep/c1 | F 200→190 | 200/200/200 | F 200→190 | feasible | tuned 第 2 轮 | '
                          '190/190/190 | -5.0% | 100/100，logit 0.002 / — |', lines)
            self.assertTrue((Path(tmp)/'E9c.md').exists())

    def test_an_error_replaces_the_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = e9c.summarize(Path(tmp), dict(sessions={'E5d': dict(id='E5d', original={}, error='RuntimeError: x')}))
        self.assertTrue(any('RuntimeError: x' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
