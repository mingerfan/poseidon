"""Only synthetic saved metadata; no model, Provider or candidate evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import r30_continue_common_session as continuation
from pt_agent.experiments import vgg_continuation_terminal_audit as audit
from pt_agent.planners.transformer_session import save_state


class ReadbackChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve();self.case=self.root/'failed';self.agent=self.case/'work/agent'
        self.agent.mkdir(parents=True);self.registry=self.root/'registry.jsonl';self.ledger=self.root/'ledger.jsonl'
        self.out=self.root/'derived-audit';self.program=self.agent/'program.ptdsl'
        self.text='def model(x):\n    return x\n';self.program.write_text(self.text)
        source_case=self.root/'historical'
        self.source=dict(original_case=str(source_case),model_info=str(source_case/'model_info.json'),
            calibration=str(source_case/'calibration.json'),original_complete_seconds=3575.)
        self.replies=[dict(content=json.dumps(dict(program=self.text)),usage=dict(prompt_tokens=1,completion_tokens=2),
            seconds=.1,finish_reason='stop')for _ in range(6)]
        messages=[dict(role='system',content='synthetic grammar'),dict(role='user',content='synthetic task')]
        for i,reply in enumerate(self.replies,1):
            messages.extend([dict(role='assistant',content=reply['content']),dict(role='user',content='feedback'+str(i))])
        self.initial=dict(phase='tune',iteration=4,calls=4,history=[dict(iteration=i)for i in range(1,4)],
            messages=messages[:8],pending=dict(reply=self.replies[3]),prompt_tokens=4,completion_tokens=8)
        selected=dict(iteration=3,origin='llm',source=self.text)
        confirmation=dict(selected=selected,evals=[dict(iteration=3,origin='llm',passed=True,diverged=False,
            n_diverged=0,total=1000,acc_drop_pp=0.,max_u_calibrated=.8,max_u_analytic=0.,
            worst_acts=[dict(domain=(-2.,2.))])])
        self.state=dict(continuation.OPTIONS,phase='done',iteration=7,calls=6,failed=None,
            history=[dict(iteration=i)for i in range(1,7)],messages=messages,confirmation=confirmation,
            best_program=self.text,prompt_tokens=6,completion_tokens=12)
        check=dict(passed=True,finite=True,dtype_ok=True,shape_ok=True,
            criterion='1e-12+1e-12*abs(reference)',max_error_ratio=0.)
        self.worker=dict(model=continuation.MODEL,status='ok',error=None,tuner_enabled=True,
            recovered_api_calls=4,actual_api_calls=2,source_model_info=self.source['model_info'],
            source_calibration=self.source['calibration'],report_samples=0,compiler_runs=0,real_ckks_runs=0,
            agent=dict(llm_calls=6,confirm=deepcopy(confirmation),best_program=self.text),
            frozen=True,quality_acceptance=dict(passed=True),selected_program=str(self.program),
            strict_math_probes=[dict(sample_id='test:'+str(9000+i),exact=dict(check),approximate=dict(check))for i in range(3)],
            boots_pred=218,latency_pred_us=123.)
        # This is precisely the representation changed by ordinary worker JSON.
        self.worker['agent']['confirm']['evals'][0]['worst_acts'][0]['domain']=[-2.,2.]
        save_state(self.case/'inherited-agent/SESSION.json',self.initial)
        save_state(self.agent/'SESSION.json',self.state)
        self.write(self.agent/'worker-result.json',self.worker)
        for number in (5,6):
            self.write(self.agent/'api'/(str(number)+'.request.json'),
                dict(purpose='agent-iter-'+str(number),messages=messages[:2*number]))
            self.write(self.agent/'api'/(str(number)+'.response.json'),self.replies[number-1])
            path=self.agent/'sources'/('iteration-'+str(number)+'.ptdsl');path.parent.mkdir(exist_ok=True);path.write_text(self.text)
        (self.agent/'agent_trace.jsonl').write_text(''.join(json.dumps(dict(iteration=i))+'\n'for i in range(1,7)))
        self.events=[dict(type='begin',stage_id='agent_confirm_select',event_id='one',start_ns=1),
            dict(type='end',stage_id='agent_confirm_select',event_id='one',start_ns=1,end_ns=2,status='ok',error=None)]
        (self.agent/'events.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in self.events))
        self.ledger.write_text(''.join(json.dumps(dict(status=200,error=None,purpose='agent-iter-'+str(i),
            prompt_tokens=1,completion_tokens=2,seconds=.1,finish_reason='stop'))+'\n'for i in range(1,7)))
        self.record=dict(round=continuation.ROUND,mode=continuation.MODE,model=continuation.MODEL,
            status='error',error=audit.ERROR,worker_attempts=1,worker_completed=0,original_api_calls=4,
            actual_new_api_calls=2,cold_start=False,resource_monitor_error=None,source=self.source,
            original_time_seconds=3575.,variant=continuation.VARIANT,api_before=4,api_after=6,worker=self.worker,
            process=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[]))
        self.write(self.case/'result.json',self.record)
        self.write(self.case/'completion.json',dict(status='error',required_artifacts_persisted=True,
            registry_persisted=True,complete_case_elapsed_seconds=4727.))
        row=dict(run_id=self.case.name,round=continuation.ROUND,model=continuation.MODEL,mode=continuation.MODE,
            git_rev=None,metrics={k:self.record.get(k)for k in audit.OLD_METRICS},artifacts_dir=str(self.case),ts='synthetic')
        self.registry.write_text(json.dumps(row)+'\n')
        self.gate=patch.object(continuation,'check_source',return_value=(self.initial,self.source))
        self.gate.start();self.addCleanup(self.gate.stop)
        self.transport=patch('pt_agent.llm.subprocess.run',side_effect=AssertionError('No Provider process is allowed'))
        self.transport.start();self.addCleanup(self.transport.stop)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_confirmation_tuple_list_boundary_keeps_full_frozen_checks(self):
        self.assertEqual(continuation.check_worker(self.worker,self.state,self.source,2,agent_dir=self.agent),'ok')

    def test_real_confirmation_scalar_and_threshold_changes_rejected(self):
        for field,value in (('acc_drop_pp',.1),('max_u_calibrated',.91),('passed',False)):
            changed=deepcopy(self.worker);changed['agent']['confirm']['evals'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                continuation.check_worker(changed,self.state,self.source,2,agent_dir=self.agent)

    def test_changed_selected_AST_or_math_evidence_rejected(self):
        for change in ('program','math'):
            changed=deepcopy(self.worker)
            if change=='program':self.program.write_text('def model(x):\n    return y\n')
            else:changed['strict_math_probes'][0]['exact']['passed']=False
            with self.subTest(change=change),self.assertRaises(ValueError):
                continuation.check_worker(changed,self.state,self.source,2,agent_dir=self.agent)
            self.program.write_text(self.text)

    def test_same_nonfinite_diagnostic_keeps_representation_without_changing_quality(self):
        state=deepcopy(self.state);worker=deepcopy(self.worker)
        state['confirmation']['evals'][0]['diagnostic']=float('nan')
        worker['agent']['confirm']['evals'][0]['diagnostic']=float('nan')
        self.assertEqual(continuation.check_worker(worker,state,self.source,2,agent_dir=self.agent),'ok')
        worker['agent']['confirm']['evals'][0]['diagnostic']=0.
        with self.assertRaises(ValueError):continuation.check_worker(worker,state,self.source,2,agent_dir=self.agent)

    def test_all_saved_gates_and_provider_usage_verify_without_new_execution(self):
        proof=audit.audit_source(self.case,ledger=self.ledger,registry=self.registry)
        self.assertTrue(proof['formal_frozen']);self.assertEqual(proof['continued_api_calls'],2)
        self.assertEqual(proof['strict_math_checks'],6);self.assertEqual(proof['confirmation_runs_verified'],1)
        self.assertEqual(proof['original_continuation_complete_seconds'],4727.)

    def test_provider_ledger_change_is_refused(self):
        rows=[json.loads(x)for x in self.ledger.read_text().splitlines()];rows[-1]['prompt_tokens']=2
        self.ledger.write_text(''.join(json.dumps(x)+'\n'for x in rows))
        with self.assertRaises(ValueError):audit.audit_source(self.case,ledger=self.ledger,registry=self.registry)

    def test_second_confirmation_or_unclosed_process_refused(self):
        for change in ('confirm','remaining'):
            if change=='confirm':
                (self.agent/'events.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in self.events*2))
            else:
                self.write(self.case/'result.json',dict(self.record,process=dict(self.record['process'],remaining_pids=[7])))
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.audit_source(self.case,ledger=self.ledger,registry=self.registry)
            (self.agent/'events.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in self.events))

    def test_original_error_or_registry_mismatch_is_refused(self):
        for change in ('error','registry'):
            if change=='error':self.write(self.case/'result.json',dict(self.record,error='Different failure'))
            else:self.registry.write_text(self.registry.read_text()*2)
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.audit_source(self.case,ledger=self.ledger,registry=self.registry)
            self.write(self.case/'result.json',self.record)

    def test_execute_writes_only_new_case_and_one_eight_field_row(self):
        source_text=(self.case/'result.json').read_text()
        args=type('Args',(),dict(out=self.out,source_case=self.case,ledger=self.ledger,registry=self.registry))()
        with patch.object(audit.paths,'RESULTS_DIR',self.root):result=audit.execute(args)
        self.assertEqual(result['status'],'ok');self.assertEqual(result['api_calls'],0)
        self.assertEqual(result['new_confirmations'],0);self.assertEqual(result['model_runs'],0)
        self.assertEqual(result['selected_program'],str(self.program))
        self.assertEqual((self.case/'result.json').read_text(),source_text)
        rows=[json.loads(x)for x in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertEqual(set(rows[-1]),audit.FIELDS)
        self.assertTrue(json.loads((self.out/'completion.json').read_text())['registry_persisted'])


if __name__=='__main__':unittest.main()
