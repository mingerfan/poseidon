"""Saved candidate scope and payload projection; no model or Agent calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import saved_candidate_estimates as entry


class SavedCandidateEstimateTests(unittest.TestCase):
    def test_pending_complete_evaluation_is_infeasible_uncommitted_and_uncompiled(self):
        state=dict(history=[],pending=dict(row=dict(iteration=1,stage='complete',feasible=False,
             boots_pred=9841,latency_pred_us=2599150148.,agree=999,max_u=1.2334964708186003)),
             messages=[dict(content='PRIVATE_REQUEST_TEXT')],best_program='PRIVATE_PROGRAM_TEXT')
        rows=entry.project_session(state);self.assertEqual(len(rows),1);row=rows[0]
        self.assertEqual(row['metrics']['boots_pred'],9841)
        self.assertFalse(row['metrics']['feasible'])
        self.assertEqual(row['evaluator_stage'],'complete')
        self.assertFalse(row['evaluator_stage_is_cold_pipeline_status'])
        self.assertFalse(row['pending_transaction_is_committed'])
        self.assertIsNone(row['compiled_bootstraps'])
        self.assertNotIn('PRIVATE_',json.dumps(rows))

    def test_committed_and_pending_origins_and_nonfinite_fields_remain_distinct(self):
        state=dict(history=[dict(iteration=1,boots_pred=100,tuned=dict(boots_pred=90))],
                   pending=dict(row=dict(iteration=2,feasible=False,latency_pred_us=float('inf'),
                                        boots_pred=dict(private='PROGRAM'))))
        rows=entry.project_session(state)
        self.assertEqual([r['origin']for r in rows],['llm','tuned','llm'])
        self.assertEqual([r['transaction']for r in rows],['committed_history','committed_history','pending_uncommitted'])
        self.assertIsNone(rows[-1]['metrics']['latency_pred_us'])
        self.assertIsNone(rows[-1]['metrics']['boots_pred'])
        self.assertEqual(rows[-1]['nonfinite_or_nonscalar_original_fields'],['latency_pred_us','boots_pred'])
        json.dumps(rows,allow_nan=False)

    def test_running_case_is_rejected_before_session_or_registry_read(self):
        with TemporaryDirectory()as folder:
            case=Path(folder)/'live';case.mkdir()
            with patch.object(entry.terminal.closure,'registry_rows')as registry, \
                    patch.object(entry.terminal.closure,'read')as read:
                with self.assertRaises(entry.terminal.Pending):entry.export_case(case)
                registry.assert_not_called();read.assert_not_called()

    def test_all_original_births_must_close_before_session_payload_is_read(self):
        with TemporaryDirectory()as folder:
            case=Path(folder)/'closed';case.mkdir();(case/'completion.json').write_text('{}')
            result=dict(round='BERT-FRESH-COLD-3H',model='bert-tiny-sst2',mode=entry.IDENTITIES['BERT-FRESH-COLD-3H'])
            with patch.object(entry.terminal.closure,'registry_rows',return_value={}), \
                    patch.object(entry.terminal,'closed',return_value=(result,{})), \
                    patch.object(entry.terminal,'births_closed',side_effect=entry.terminal.Pending('actual birth live')), \
                    patch.object(entry.terminal.closure,'read')as read:
                with self.assertRaises(entry.terminal.Pending):entry.export_case(case)
                read.assert_not_called()


if __name__=='__main__':unittest.main()
