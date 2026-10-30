"""A1 §6.11 persistent migration and retained decision-only replay."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pt_agent.planners.transformer_session import load_state,save_state,migrate_tune_rule
from pt_agent.experiments.a1_transformer_agent import decision_replay
from pt_agent.paths import RESULTS_DIR
from pt_agent.tests._artifacts import require


class RuleMigrationTests(unittest.TestCase):
    def state(self):
        return dict(prompt_version='r8-v8',tune_rule='v8-pre-6.11',phase='evaluate',
            iteration=7,calls=6,pending=None,llm_min=2592782653,has_tuned=True,
            history=[dict(feasible=False,latency_pred_us=1),
                     dict(feasible=True,latency_pred_us=4564777102),
                     dict(feasible=True,latency_pred_us=3233700613),
                     dict(feasible=False,latency_pred_us=2592782653)],
            messages=[dict(role='user',content='retained feedback')],pool=['retained'],
            tune_cache={'retained':('program',1,'source')},records={(1,'llm','source'):3},
            recovery=dict(section='6.9',llm_min_after_call13=3233700613))

    def test_migration_preserves_records_and_is_idempotent(self):
        state=self.state();old=copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'SESSION.json';save_state(path,state)
            record=migrate_tune_rule(root,next_model_call=15);actual=load_state(path)
            self.assertEqual(record['old_llm_min'],2592782653)
            self.assertEqual(record['new_llm_min'],3233700613)
            self.assertEqual(record['llm_min_after_call13'],3233700613)
            self.assertEqual(record['next_iteration'],7)
            self.assertEqual(actual['tune_rule'],'v8-6.11')
            for key in old.keys()-{'tune_rule','llm_min'}:self.assertEqual(actual[key],old[key])
            retained=path.read_text()
            self.assertEqual(migrate_tune_rule(root,next_model_call=15),record)
            self.assertEqual(path.read_text(),retained)
            with self.assertRaises(ValueError):migrate_tune_rule(root,next_model_call=16)

    def test_legacy_state_is_pre_rule_without_rewriting_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SESSION.json';state=self.state();state.pop('tune_rule');save_state(path,state)
            retained=path.read_text()
            self.assertEqual(load_state(path)['tune_rule'],'v8-pre-6.11')
            self.assertEqual(path.read_text(),retained)

    def test_migration_rejects_pending_and_wrong_boundary(self):
        for updates in (dict(pending={}),dict(phase='tune'),dict(calls=5),dict(prompt_version='r8-v7')):
            with self.subTest(updates=updates),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);state=self.state();state.update(updates);save_state(root/'SESSION.json',state)
                retained=(root/'SESSION.json').read_text()
                with self.assertRaises(ValueError):migrate_tune_rule(root,next_model_call=15)
                self.assertEqual((root/'SESSION.json').read_text(),retained)

    def test_no_feasible_raw_program_resets_minimum_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=self.state();state['history']=[dict(feasible=False,latency_pred_us=1)]
            save_state(root/'SESSION.json',state)
            self.assertIsNone(migrate_tune_rule(root,next_model_call=15)['new_llm_min'])


@require(RESULTS_DIR/'A1-bert-session-20261028T025117-861c/agent_trace.jsonl',
         RESULTS_DIR/'A1-bert-session-20261028T033706-5ca7/agent_trace.jsonl')
class RetainedDecisionReplayTests(unittest.TestCase):
    def test_fourteen_calls_keep_session_boundaries_and_do_not_estimate_saved_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=decision_replay(SimpleNamespace(out=Path(tmp),legacy_sessions=[
                RESULTS_DIR/'A1-bert-session-20261028T025117-861c',
                RESULTS_DIR/'A1-bert-session-20261028T033706-5ca7']))
        self.assertEqual([r['model_call'] for r in result['rows'] if r['old_triggered']],[1,8,9,13])
        self.assertEqual([r['model_call'] for r in result['rows'] if r['new_triggered']],[1,7,8,9,12,13])
        self.assertEqual(result['rows'][-1]['new_llm_min'],3233700613)
        self.assertEqual(result['infeasible_tunes'][1]['coarse_evals'],37)
        self.assertEqual(result['historical_infeasible_tune_seconds'],5240.848565660941)
        self.assertIsNone(result['net_saved_seconds'])
        self.assertEqual((result['llm_calls'],result['tune_calls'],result['report_images']),(0,0,0))


if __name__=='__main__':unittest.main()
