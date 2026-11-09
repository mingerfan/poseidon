"""E7 serial budget and outer registry routing, without provider requests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import e7_confirm_tail_check as entry


class DriverTests(unittest.TestCase):
    def test_qwen_order_and_confirmation_budget(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);seen=[]
            def execute(command,**kwargs):
                label=command[command.index('--session-label')+1];seen.append(label)
                self.assertEqual(command[command.index('--confirm-rule')+1],'c2')
                self.assertEqual(command[command.index('--tune-call-limit')+1],'9')
                self.assertEqual(command[command.index('--provider-state')+1],str(root/'provider-state.json'))
                out=Path(command[command.index('--out')+1])
                entry.write(out/'REPORT.json',dict(complete=True,frozen=False))
                return SimpleNamespace(returncode=0)
            with patch.object(entry.subprocess,'run',side_effect=execute),patch.object(entry.e6r2,'boundary') as boundary:
                entry.run_b(root,dict(models=dict(qwen=str(root/'model'))))
            self.assertEqual(seen,['q1','q2']);self.assertEqual(boundary.call_count,2)

    def test_new_labels_register_main_and_tune_as_e7(self):
        from pt_agent.experiments.a1_transformer_agent import account_calls
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            entry.write(out/'api-accounting.json',dict(api_before=365))
            ledger=root/'calls.jsonl'
            values=[{}]*365+[dict(ts='now',purpose='agent-iter-1'),dict(ts='now',purpose='tune-select-iter1-round1')]
            ledger.write_text(''.join(json.dumps(item)+'\n' for item in values))
            appended=[]
            with patch('pt_agent.registry.load',return_value=[]),patch('pt_agent.registry.append',side_effect=appended.append):
                account_calls(ledger,out,model='qwen',prompt_version='r8-v11',label='q1',split_version='split-v2')
            self.assertEqual([r['round'] for r in appended],['E7','E7'])
            self.assertEqual([r['metrics']['budget_namespace'] for r in appended],
                             [['qwen','r8-v11','q1'],['qwen','t1','q1']])

    def test_e7a_code_error_does_not_start_e7b(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);entry.write(root/'OFFLINE_TESTS.json',dict(complete=True));entry.write(root/'manifest.json',{})
            summary=dict(complete=False,error='code',api_before=365,api_after=365,api_delta=0,namespaces=[],accounting_equal=True,within_limits=True)
            with patch.object(entry,'ledger_rows',return_value=[{}]*365), \
                 patch.object(entry,'run_a',side_effect=RuntimeError('test code error')), \
                 patch.object(entry,'run_b') as after,patch.object(entry,'summarize',return_value=summary),patch.object(entry,'register'):
                self.assertEqual(entry.run(SimpleNamespace(root=root)),1)
                after.assert_not_called()
            self.assertIn('test code error',entry.read(root/'STOP.json')['error'])


if __name__=='__main__':unittest.main()
