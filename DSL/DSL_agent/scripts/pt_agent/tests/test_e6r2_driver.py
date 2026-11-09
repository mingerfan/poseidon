"""E6r2 order, shared breaker, cooldown and real-attempt report accounting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.experiments import e6r2_resume_after_outage as entry
from pt_agent.planners.provider_retry import ProviderState,ProviderOutage


class DriverTests(unittest.TestCase):
    def test_report_rejection_is_distinct_from_completed_execution(self):
        row=dict(acceptance_status='complete',report100={
            'tensor':dict(feasible=False,agree=100,total=100),
            'mock':dict(feasible=False,agree=100,total=100)})
        self.assertEqual(entry.report_status(row),'report_failed')
        self.assertEqual(row['acceptance_status'],'complete')
        self.assertEqual(entry.report_status(dict(acceptance_status='complete',
            report100={'tensor':dict(feasible=True),'mock':dict(feasible=True)})),'complete')
        self.assertEqual(entry.report_status(dict(acceptance_status='not_frozen',report100={})),'not_frozen')

    def test_boundary_waits_600_after_failed_session_and_circuit_bypasses_wait(self):
        with TemporaryDirectory() as temporary:
            waits=[];policy=ProviderState(Path(temporary)/'state.json',sleep=waits.append)
            entry.boundary(policy,failed_session='bert replay')
            self.assertEqual(waits,[])
            entry.boundary(policy);self.assertEqual(waits,[600])
            self.assertIsNone(policy.read()['cooldown_pending'])
            policy.update(lambda s:s.update(provider_outage=True))
            with self.assertRaises(ProviderOutage):entry.boundary(policy)
            self.assertEqual(waits,[600])

    def test_replays_and_four_new_sessions_keep_order_and_new_quota(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';seen=[]
            entry.write(run/'OFFLINE_TESTS.json',dict(complete=True))
            entry.write(run/'network-review/manifest.json',dict(order=['approved']))
            def replay(root,old,model,*_):seen.append(('replay',model));return dict(complete=True,frozen=False)
            def execute(command,**kwargs):
                model=command[command.index('--model')+1];label=command[command.index('--session-label')+1]
                self.assertEqual(command[command.index('--tune-call-limit')+1],'12')
                self.assertIn('--provider-state',command);seen.append(('session',model,label))
                entry.write(Path(command[command.index('--out')+1])/'REPORT.json',dict(complete=True,frozen=False))
                return SimpleNamespace(returncode=0)
            summary=dict(api_before=291,api_after=291,api_delta=0,namespaces=[],accounting_equal=True,within_limits=True)
            with patch.object(entry,'RESULTS_DIR',root), \
                 patch.object(entry,'ledger_rows',side_effect=lambda p:[] if p.name=='registry.jsonl' else [{}]*291), \
                 patch.object(entry.e6,'configurations',return_value={m:dict(model=root/m) for m in ('bert','qwen')}), \
                 patch.object(entry,'replay',side_effect=replay),patch.object(entry,'summarize',return_value=summary), \
                 patch.object(entry.subprocess,'run',side_effect=execute),patch.object(entry.subprocess,'check_output',return_value=''), \
                 patch.object(entry,'register'):
                self.assertEqual(entry.run(SimpleNamespace(root=run,old=root/'old',previous=root/'previous')),0)
            self.assertEqual(seen,[('replay','bert'),('replay','qwen'),('session','bert','p1'),
                ('session','qwen','p1'),('session','bert','p2'),('session','qwen','p2')])

    def test_breaker_ends_round_before_qwen_or_any_new_session(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';seen=[]
            entry.write(run/'OFFLINE_TESTS.json',dict(complete=True))
            entry.write(run/'network-review/manifest.json',dict(order=[]))
            def replay(*args):
                seen.append(args[2]);ProviderState(run/'provider-state.json').update(lambda s:s.update(
                    provider_outage=True,outage_window=dict(start='first',end='third')))
                return dict(complete=False,failure_class='provider',error='provider_outage')
            summary=dict(api_before=291,api_after=291,api_delta=0,namespaces=[],accounting_equal=True,within_limits=True)
            with patch.object(entry,'RESULTS_DIR',root), \
                 patch.object(entry,'ledger_rows',side_effect=lambda p:[] if p.name=='registry.jsonl' else [{}]*291), \
                 patch.object(entry.e6,'configurations',return_value={m:dict(model=root/m) for m in ('bert','qwen')}), \
                 patch.object(entry,'replay',side_effect=replay),patch.object(entry,'summarize',return_value=summary), \
                 patch.object(entry.subprocess,'run') as process,patch.object(entry.subprocess,'check_output',return_value=''), \
                 patch.object(entry,'register'):
                self.assertEqual(entry.run(SimpleNamespace(root=run,old=root/'old',previous=root/'previous')),1)
                process.assert_not_called()
            self.assertEqual(seen,['bert'])
            stop=entry.read(run/'STOP.json');self.assertEqual(stop['failure_class'],'provider_outage')
            self.assertEqual(stop['provider_state']['outage_window'],dict(start='first',end='third'))

    def test_attempt_report_includes_retries_waits_and_counts_tokens_once(self):
        from pt_agent.planners.provider_retry import RetryingBudgetClient
        from pt_agent.tests.test_e6r2_provider import Stub,error
        with TemporaryDirectory() as temporary:
            root=Path(temporary);session=entry.session_path(root,'e6r2-a','bert','E6r2-a')
            entry.write(session/'REPORT.json',dict(complete=True,seconds=20))
            reply=dict(content='ok',seconds=2,usage=dict(prompt_tokens=5,completion_tokens=1))
            client=RetryingBudgetClient(Stub([error(),reply]),root/'calls',model='bert',prompt_version='t1',
                label='E6r2-a',limit=8,provider_state=root/'provider-state.json',sleep=lambda _:None)
            client.chat([],purpose='tune-select-iter4-round1')
            row=entry.session_summary(session)
            self.assertEqual(row['tuning_calls'],2);self.assertEqual(row['logical_requests']['tuning'],1)
            self.assertEqual(row['retry_count'],1);self.assertEqual(row['retry_wait_s'],300)
            self.assertEqual(row['tokens']['tuning'],6)
            self.assertEqual({k:row['tune_retry_stats']['4'][k] for k in ('calls','retries','wait_s')},dict(calls=2,retries=1,wait_s=300))
            self.assertGreaterEqual(row['tune_retry_stats']['4']['llm_s'],2)
            self.assertFalse(row['provider_outage_triggered'])

    def test_main_and_tune_clients_use_the_shared_retry_state(self):
        from pt_agent.experiments import a1_transformer_agent as entrypoint
        from pt_agent.planners import agent_tools,transformer_session
        from pt_agent.planners.provider_retry import RetryingBudgetClient
        from pt_agent.tests.test_e6r2_provider import Stub
        with TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir();(model/'calibration.json').write_text('{}')
            session=root/'session';session.mkdir();(session/'SESSION.json').write_text('{}')
            for phase in ('evaluate','tune'):
                ctx=SimpleNamespace();state=dict(phase=phase,max_iters=10,history=[],pending=dict(iteration=4))
                args=SimpleNamespace(model='bert',model_dir=model,prompt_version='r8-v11',placement='program-plan',
                    tune_search='llm-select',session_label='E6r2-a',tune_call_limit=8,session_iters=7,
                    out=root/'out',root=session,step=phase,call_dir=root/'calls',budget_path=root/'ledger',
                    response_file=None,tune_rule='v8-6.24',provider_state=root/'provider-state.json')
                def advance(ctx,directory,**kwargs):
                    client=kwargs['client'] if phase=='evaluate' else kwargs['tune_client']
                    self.assertIsInstance(client,RetryingBudgetClient);self.assertEqual(client.policy.path,args.provider_state)
                    return dict(complete=True)
                with patch('pt_agent.models.info.ModelInfo.load',return_value=SimpleNamespace()), \
                     patch.object(agent_tools,'build_context',return_value=ctx),patch.object(entrypoint,'samples_for',return_value=None), \
                     patch.object(transformer_session,'load_state',return_value=state),patch.object(transformer_session,'session_step',side_effect=advance), \
                     patch('pt_agent.llm.LLMClient',return_value=Stub([])):
                    entrypoint.session_step(args)

if __name__=='__main__':unittest.main()
