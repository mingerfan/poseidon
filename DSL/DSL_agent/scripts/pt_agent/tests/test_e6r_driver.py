"""Offline E6r ordering, independent allowances, continuation and bookkeeping."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.experiments import e6r_llm_select_fix as entry
from pt_agent.planners.transformer_session import ModelBudgetClient,model_calls


class DriverTests(unittest.TestCase):
    def test_replay_budgets_and_all_saved_main_replies(self):
        for model,total,limit in (('bert',7,8),('qwen',8,16)):
            with TemporaryDirectory() as directory:
                root=Path(directory);old=root/'old';seen=[]
                for i in range(1,total+1):entry.write(old/'calls'/model/'r8-v11'/f'call-{i:03d}.response.json',{})
                def child(command,out,**kwargs):
                    phase=command[command.index('--step')+1];session=Path(command[command.index('--root')+1])
                    state=entry.read(session/'SESSION.json') if (session/'SESSION.json').exists() else dict(iteration=1)
                    i=state['iteration'];seen.append(command)
                    if phase=='confirm':state['phase']='done';entry.write(session/'AGENT.json',dict(status='confirm_failed'))
                    else:state.update(phase='confirm' if i==total else 'evaluate',iteration=i+1)
                    entry.write(session/'SESSION.json',state);return dict(complete=True,exit=0)
                with patch.object(entry,'child',side_effect=child),patch.object(entry,'register'):
                    result=entry.replay(root/'run',old,model,root/'model',root/'ledger')
                self.assertTrue(result['complete']);self.assertEqual(result['main_api_calls'],0)
                self.assertEqual(sum('--response-file' in c for c in seen),total)
                self.assertTrue(all(c[c.index('--tune-call-limit')+1]==str(limit) for c in seen))
                self.assertTrue(all(c[c.index('--session-label')+1]=='E6r-a' for c in seen))

    def test_supplement_then_replay_then_four_ordered_sessions(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';seen=[]
            entry.write(run/'OFFLINE_TESTS.json',dict(complete=True))
            entry.write(run/'network-review/manifest.json',dict(order=['approved']))
            def replay(root,old,model,*_):seen.append(('replay',model));return dict(complete=True,frozen=False)
            def process(command,**kwargs):
                model=command[command.index('--model')+1];label=command[command.index('--session-label')+1]
                seen.append(('session',model,label));self.assertEqual(command[command.index('--tune-call-limit')+1],'14')
                entry.write(Path(command[command.index('--out')+1])/'REPORT.json',dict(complete=True,frozen=False))
                return SimpleNamespace(returncode=0)
            summary=dict(api_before=283,api_after=283,api_delta=0,namespaces=[],accounting_equal=True,within_limits=True)
            def ledger(p):return [] if p.name=='registry.jsonl' else [{}]*283
            with patch.object(entry,'RESULTS_DIR',root),patch.object(entry,'ledger_rows',side_effect=ledger), \
                 patch.object(entry.e6,'configurations',return_value={m:dict(model=root/m) for m in ('bert','qwen')}), \
                 patch.object(entry,'supplement',side_effect=lambda *a:seen.append(('supplement',))), \
                 patch.object(entry,'replay',side_effect=replay),patch.object(entry,'summarize',return_value=summary), \
                 patch.object(entry.subprocess,'run',side_effect=process),patch.object(entry.subprocess,'check_output',return_value=''), \
                 patch.object(entry,'register'):
                self.assertEqual(entry.run(SimpleNamespace(root=run,old=root/'old',previous=root/'previous')),0)
            self.assertEqual(seen,[('supplement',),('replay','bert'),('replay','qwen'),
                ('session','bert','r1'),('session','qwen','r1'),('session','bert','r2'),('session','qwen','r2')])

    def test_successful_compile_supplement_is_reused_and_failure_is_preserved(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);previous=root/'old';run=root/'run'
            entry.write(previous/'E6.json',dict(e6b={'qwen s1':dict(timings={})}))
            result=dict(complete=True,exit=0,boots=dict(predicted=22,exported=22,compiled=22),
                        compile=dict(seconds=2),seconds=3)
            def compiled(command,out,**kwargs):
                entry.write(out/'REPORT.json',result);return result
            with patch.object(entry,'child',side_effect=compiled) as child:
                entry.supplement(run,previous,root/'model')
                entry.supplement(run,previous,root/'model')
                child.assert_called_once()
                self.assertEqual(entry.read(run/'E6-updated.json')['e6b']['qwen s1']['boots'],result['boots'])
                entry.write(run/'e6-qwen-s1-compile/REPORT.json',dict(complete=False,error='retained stop'))
                with self.assertRaisesRegex(RuntimeError,'retained stop'):
                    entry.supplement(run,previous,root/'model')
                child.assert_called_once()

    def test_new_namespaces_keep_old_s1_and_enforce_each_limit(self):
        from pt_agent.llm import LLMError
        stub=SimpleNamespace(chat=lambda *a,**k:dict(content='{}'))
        with TemporaryDirectory() as folder:
            for model in ('bert','qwen'):
                old=ModelBudgetClient(stub,folder,model=model,prompt_version='r8-v11',label='s1')
                old.chat([],purpose='saved')
                for version,label,limit in (('t1','E6r-a',entry.REPLAY_LIMITS[model]),
                                           ('r8-v11','r1',10),('t1','r1',14),
                                           ('r8-v11','r2',10),('t1','r2',14)):
                    client=ModelBudgetClient(stub,folder,model=model,prompt_version=version,label=label,limit=limit)
                    for _ in range(limit):client.chat([],purpose='offline')
                    with self.assertRaises(LLMError):client.chat([],purpose='offline')
                    self.assertEqual(model_calls(folder,model,version,label),limit)
                self.assertEqual(old.calls,1)

    def test_accounting_routes_new_labels_to_e6r_only(self):
        from pt_agent.experiments.a1_transformer_agent import account_calls
        with TemporaryDirectory() as directory:
            folder=Path(directory);ledger=folder/'ledger';out=folder/'step'
            entry.write(out/'api-accounting.json',dict(api_before=283))
            rows=[{}]*283+[dict(purpose='tune-select-iter4-round1',ts='now')]
            with patch('pt_agent.experiments.r8g_mlp_rerun._calls',return_value=rows), \
                 patch('pt_agent.registry.load',return_value=[]),patch('pt_agent.registry.git_rev',return_value='rev'), \
                 patch('pt_agent.registry.append') as append:
                account_calls(ledger,out,model='bert',prompt_version='r8-v11',label='E6r-a')
            record=append.call_args[0][0]
            self.assertEqual(record['round'],'E6r');self.assertEqual(record['metrics']['global_call'],284)
            self.assertEqual(record['metrics']['budget_namespace'],['bert','t1','E6r-a'])

if __name__=='__main__':unittest.main()
