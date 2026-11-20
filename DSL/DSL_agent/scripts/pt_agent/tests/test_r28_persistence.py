"""Two small checks for recovery persistence and the final endpoint only."""
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r28_common_agent as entry
from pt_agent.experiments.completion_metrics import artifact_bytes, persist_case_completion


class PersistenceChecks(unittest.TestCase):
    def test_recovery_provenance_is_saved_before_reply_is_consumed(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'program.ptdsl';source.write_text('def model(x):\n    return x\n')
            previous=root/'previous.json';info=root/'info.json'
            previous.write_text(json.dumps(dict(api_delta=1,api_after=1,worker=dict(frozen=False,source_model_info=str(info)))))
            args=Namespace(resume_result=previous,resume_source=source,model_info=info,ledger=root/'ledger.jsonl')
            result={};order=[]
            def save():
                self.assertEqual(result['recovered_api_calls'],1)
                self.assertFalse(result['recovery']['raw_reply_recovered'])
                self.assertFalse(result['recovery']['provider_request_sent_during_replay'])
                order.append('saved')
            record=dict(status=200,purpose='agent-iter-1',error=None,prompt_tokens=7,completion_tokens=9,seconds=1.5,finish_reason='stop')
            with patch.object(entry,'call_records',return_value=[record]):
                reply=entry._recover_first_reply(args,result,save=save)
                order.append('consumed')
            self.assertEqual(order,['saved','consumed'])
            self.assertEqual(json.loads(reply['content']),dict(program=source.read_text()))
            self.assertEqual(reply['usage'],dict(prompt_tokens=7,completion_tokens=9))

    def test_complete_endpoint_follows_result_and_unique_registry(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'registry.jsonl').touch();out=root/'case'
            args=Namespace(out=out,model='resnet20_silu',prepare_only=True,calls=1,memory_gib=6,
                           preflight_evidence=None,seconds=30,tuner_seconds=30,resume_source=None)
            selected=dict(model_info=root/'info.json',calibration=root/'calibration.json',
                          r13_result=root/'r13.json',r14_result=root/'r14.json')
            def process(command,**kwargs):
                kwargs['log'].write_text('fixture-revision\n')
                return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
            observed=[]
            def completion(directory,**kwargs):
                def clock():
                    result=json.loads((out/'result.json').read_text())
                    rows=[json.loads(line)for line in (root/'registry.jsonl').read_text().splitlines()]
                    self.assertEqual(result['completion_path'],str(out/'completion.json'))
                    self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]),8)
                    self.assertFalse((out/'completion.json').exists())
                    observed.append(True)
                    return kwargs['started_ns']+1000000000
                return persist_case_completion(directory,clock=clock,**kwargs)
            with patch.object(paths,'RESULTS_DIR',root),patch.object(entry,'sources',return_value=selected), \
                 patch.object(entry,'call_records',return_value=[]),patch.object(entry,'run_process',side_effect=process), \
                 patch.object(entry,'persist_case_completion',side_effect=completion):
                self.assertEqual(entry.outer(args),0)
            marker=json.loads((out/'completion.json').read_text())
            self.assertEqual(observed,[True]);self.assertEqual(marker['round'],'R28')
            self.assertEqual(marker['complete_case_elapsed_seconds'],1)
            self.assertEqual(marker['case_artifact_bytes'],artifact_bytes(out))
            self.assertIn('completion marker write excluded',marker['elapsed_scope'])
            self.assertTrue(marker['exclusive_attribution_ready'])


if __name__=='__main__':unittest.main()
