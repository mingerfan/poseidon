"""Synthetic whole-session restoration; no model, network or backend process."""
from argparse import Namespace
import ast
from contextlib import ExitStack, nullcontext, redirect_stderr
from copy import deepcopy
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from pt_agent.experiments import r28_common_agent as common
from pt_agent.experiments import r30_continue_common_session as entry
from pt_agent.planners.transformer_session import load_state, save_state
from pt_agent.tests.test_r30_no_tuner import _isolated_agent


def isolated_agent():
    module=_isolated_agent()
    source=Path(common.__file__).parents[1]/'planners/agent.py'
    names={'_finish_tune', 'saved_tune_result', 'agent_result'}
    definitions=[node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:raise AssertionError('Trusted state functions changed')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), module.__dict__)
    module.convergent_rule=lambda rule:True
    module.tune_rule_scope=lambda rule:nullcontext()
    module.changed_lines=lambda source, previous:set()
    return module


@dataclass
class SyntheticTune:
    program:str
    source:str
    feasible:bool
    cost_after:dict
    history:list
    changes:list
    coarse_evals:int=1
    full_evals:int=1
    seconds:float=.01
    sweeps:int=1
    converged:bool=True


class ContinuationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        guard=patch('pt_agent.llm.subprocess.run',side_effect=AssertionError('Provider transport is blocked in synthetic tests'))
        guard.start();self.addCleanup(guard.stop)
        self.root=Path(self.tmp.name)
        self.case=self.root/'original-vgg'
        self.old=self.case/'work/agent';self.old.mkdir(parents=True)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.agent=isolated_agent()
        lookup=ModuleType('pt_agent.planners.transformer_tuner')
        lookup.validate_tune_search=lambda ctx:getattr(ctx, 'tune_search', 'sweep')
        self.stack.enter_context(patch.dict(sys.modules, {lookup.__name__:lookup}))
        replacements=dict(build_system_prompt=lambda ctx:'synthetic grammar',
            initial_message=lambda ctx:'synthetic original model task',
            confirmation_rule=lambda ctx, rule:'c1', apply_placement=lambda source, ctx:source,
            parse=lambda source:source, print_program=lambda program:program,
            evaluate=self.evaluate, tune=self.tune,
            feedback_message=lambda feedback, iteration, best, **kw:'feedback '+str(iteration),
            confirm_select=self.confirm)
        for key, value in replacements.items():self.stack.enter_context(patch.object(self.agent, key, value))
        self.ctx=SimpleNamespace(prompt_version='r8-v11', placement='program-plan',
            tune_search='sweep', model_info=SimpleNamespace(split={}))
        state=self.agent.new_agent_state(self.ctx, **common.agent_options(6,600))
        for iteration in range(1,5):
            reply=self.reply(iteration)
            self.write(self.old/'api'/(str(iteration)+'.request.json'),
                dict(messages=deepcopy(state['messages']), purpose='agent-iter-'+str(iteration)))
            self.write(self.old/'api'/(str(iteration)+'.response.json'), reply)
            self.agent.agent_step(self.ctx,state,trace_path=self.old/'agent_trace.jsonl',reply=reply)
            if iteration<4:self.agent.agent_step(self.ctx,state,trace_path=self.old/'agent_trace.jsonl')
        self.initial=state
        save_state(self.old/'SESSION.json', state)
        self.info=self.case/'work/intake/model_info/model_info.json'
        self.cal=self.case/'work/calibration/calibration.json'
        self.write(self.info,dict(source=dict(factory=entry.FACTORY),threshold=dict(acc_drop_pp=.5),split=dict(kind='legacy_cifar')))
        self.write(self.cal,{})
        self.write(self.case/'result.json',dict(round='R30',model=entry.MODEL,status='error',api_delta=4,
            api_before=0,api_after=4,variant='llm-first-generation+generic-deterministic-tuning',
            attempted=dict(intake=1,calibration=1,agent=1),completed=dict(intake=1,calibration=1)))
        self.write(self.case/'completion.json',dict(status='error',required_artifacts_persisted=True,
            registry_persisted=True,complete_case_elapsed_seconds=3575.021826592))
        self.write(self.old/'worker-result.json',dict(model=entry.MODEL,status='running',frozen=False,
            strict_math_probes=[],report_samples=0,compiler_runs=0,real_ckks_runs=0,
            source_model_info=str(self.info),source_calibration=str(self.cal)))
        self.write(self.old/'phase_result.json',dict(phase='agent',status='running'))
        (self.old/'events.jsonl').write_text('')
        startup=self.case/'source/pt_agent/planners/agent.py';startup.parent.mkdir(parents=True)
        startup.write_text((Path(common.__file__).parents[1]/'planners/agent.py').read_text())
        self.ledger=self.root/'calls.jsonl'
        self.ledger.write_text(''.join(json.dumps(dict(status=200,error=None,purpose='agent-iter-'+str(i)))+'\n'
            for i in range(1,5)))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    @staticmethod
    def reply(iteration):
        return dict(content=json.dumps(dict(program='def candidate(x):\n    return x + '+str(iteration)+'\n')),
            seconds=.1,finish_reason='stop',usage=dict(prompt_tokens=iteration,completion_tokens=2*iteration))

    @staticmethod
    def evaluate(source, ctx, **kwargs):
        iteration=int(source.split('+')[-1].strip())
        return dict(stage='complete',feasible=True,exact=dict(ok=True),
            cost=dict(latency_pred_us=1000/2**iteration,boots_pred=iteration),
            validation=dict(diverged=False,agree=1000,max_u=.5,max_u_calibrated=.5,max_u_analytic=0.))

    def tune(self, program, ctx, **kwargs):
        iteration=int(program.split('+')[-1].strip())
        source='def candidate(x):\n    return x + '+str(iteration)+'.5\n'
        return SyntheticTune(program=source,source=source,feasible=True,
            cost_after=dict(latency_pred_us=900/2**iteration,boots_pred=iteration),
            history=[dict(agree=1000,max_u=.5,max_u_calibrated=.5,max_u_analytic=0.)],changes=[])

    @staticmethod
    def confirm(pool, ctx, **kwargs):
        selected=min(pool,key=lambda candidate:candidate.latency_pred_us)
        row=dict(iteration=selected.iteration,origin=selected.origin,passed=True,diverged=False,
            n_diverged=0,total=1000,acc_drop_pp=.5,max_u=.5,max_u_calibrated=.5,max_u_analytic=0.)
        return dict(selected=selected,evals=[row],duplicates=[],deduplicated=dict(before=len(pool),after=len(pool)))

    def worker_args(self, inherited):
        return Namespace(model=entry.MODEL,model_info=self.info,calibration=self.cal,calls=6,tuner_seconds=600,
            tuner=True,prepare_only=False,resume_source=None,resume_session=inherited/'SESSION.json',
            resume_case=self.case,ledger=self.ledger)

    def copied(self):
        inherited=self.root/'inherited'
        entry.preserve_session(self.case,inherited,self.initial)
        return inherited

    def test_source_gate_retains_four_actual_replies_options_and_original_clock(self):
        state,summary=entry.check_source(self.case,ledger=self.ledger)
        self.assertEqual(state,self.initial)
        self.assertEqual(summary['remaining_calls'],2)
        self.assertEqual(summary['next_step'],'tune')
        self.assertIs(summary['next_step_requires_API'],False)
        self.assertEqual(summary['original_complete_seconds'],3575.021826592)
        self.assertTrue(summary['provider_history_verified'])
        self.assertTrue(summary['planner_state_function_AST_equal'])
        self.assertIsInstance(next(iter(state['records'])),tuple)
        for key, invalid in (('calls',3),('tuner',False),('max_iters',5),('tuner_budget_s',601),('phase','evaluate')):
            with self.subTest(key=key):
                changed=deepcopy(state);changed[key]=invalid
                with self.assertRaises(ValueError):entry.validate_state(changed)

    def test_source_gate_rejects_changed_response_context_and_planner_semantics(self):
        request=self.old/'api/4.request.json';original=json.loads(request.read_text())
        changed=deepcopy(original);changed['messages'][-1]['content']='different feedback';self.write(request,changed)
        with self.assertRaisesRegex(ValueError,'request history'):entry.check_source(self.case,ledger=self.ledger)
        self.write(request,original)
        response=self.old/'api/4.response.json';original=json.loads(response.read_text())
        changed=dict(original,seconds=123);self.write(response,changed)
        with self.assertRaisesRegex(ValueError,'Fourth original response'):entry.check_source(self.case,ledger=self.ledger)
        self.write(response,original)
        info=json.loads(self.info.read_text());self.write(self.info,dict(info,threshold=dict(acc_drop_pp=.6)))
        with self.assertRaisesRegex(ValueError,'quality policy'):entry.check_source(self.case,ledger=self.ledger)
        self.write(self.info,info)
        startup=self.case/'source/pt_agent/planners/agent.py'
        startup.write_text(startup.read_text().replace('state[\'calls\'] += 1','state[\'calls\'] += 2',1))
        with self.assertRaisesRegex(ValueError,'state semantics'):entry.check_source(self.case,ledger=self.ledger)

    def test_public_serializer_and_worker_restore_preserve_all_history_without_ledger_writes(self):
        original=load_state(self.old/'SESSION.json');ledger=self.ledger.read_text().splitlines()
        inherited=self.copied();work=self.root/'worker';work.mkdir();result={};saved=[]
        restored=entry.initialize_worker(self.worker_args(inherited),work,result,save=lambda:saved.append(deepcopy(result)))
        self.assertEqual(restored,original);self.assertEqual(load_state(work/'SESSION.json'),original)
        self.assertEqual(load_state(self.old/'SESSION.json'),original)
        self.assertEqual(self.ledger.read_text().splitlines(),ledger)
        self.assertEqual(result['recovered_api_calls'],4);self.assertEqual(result['actual_api_calls'],0)
        self.assertEqual(result['recovery']['initial_phase'],'tune');self.assertEqual(len(saved),1)
        self.assertEqual(len(list((work/'sources').glob('*.ptdsl'))),4)
        self.assertFalse((work/'api').exists())
        self.assertEqual(len(list((inherited/'api').glob('*.response.json'))),4)
        with self.assertRaisesRegex(ValueError,'fresh evidence'):entry.initialize_worker(self.worker_args(inherited),work,{},save=lambda:None)

    def test_actual_agent_redoes_pending_tune_before_any_new_provider_request(self):
        inherited=self.copied();state=load_state(inherited/'SESSION.json');old_messages=deepcopy(state['messages'])
        client=SimpleNamespace(chat=Mock(side_effect=AssertionError('Pending tune must use no API')))
        before=deepcopy(state)
        self.agent.agent_step(self.ctx,state,trace_path=inherited/'agent_trace.jsonl',client=client)
        client.chat.assert_not_called();self.assertEqual(state['calls'],4)
        self.assertEqual(state['history'][:3],before['history']);self.assertEqual(len(state['history']),4)
        self.assertEqual(state['messages'][:8],old_messages)
        self.assertEqual(state['messages'][8]['content'],before['pending']['reply']['content'])
        self.assertEqual(state['phase'],'evaluate');self.assertEqual(state['iteration'],5)
        self.assertIsNone(state['pending']);self.assertEqual(state['tuner_budget_s'],600)
        self.assertEqual(state['prompt_tokens'],before['prompt_tokens'])

    def finish(self, *, malformed=False):
        inherited=self.copied();work=self.root/'worker';work.mkdir();result={}
        state=entry.initialize_worker(self.worker_args(inherited),work,result,save=lambda:None)
        count=0
        def chat(messages,*,purpose):
            nonlocal count
            count+=1;self.assertLessEqual(count,2)
            reply=self.reply(4+count)
            if malformed and count==1:reply['content']='synthetic malformed JSON'
            self.write(work/'api'/(str(count)+'.request.json'),dict(messages=deepcopy(messages),purpose=purpose))
            self.write(work/'api'/(str(count)+'.response.json'),reply)
            return reply
        client=SimpleNamespace(chat=chat)
        while state['phase']!='done':self.agent.agent_step(self.ctx,state,trace_path=work/'agent_trace.jsonl',client=client)
        save_state(work/'SESSION.json',state)
        agent=self.agent.agent_result(state)
        agent['confirm']['selected']=asdict(agent['confirm']['selected'])
        program=work/'program.ptdsl';program.write_text(agent['best_program'])
        check=dict(passed=True,finite=True,dtype_ok=True,shape_ok=True,
            criterion='1e-12+1e-12*abs(reference)',max_error_ratio=.25)
        result.update(model=entry.MODEL,status='ok',frozen=True,tuner_enabled=True,actual_api_calls=count,
            source_model_info=str(self.info),source_calibration=str(self.cal),agent=agent,
            quality_acceptance=dict(passed=True),selected_program=str(program),report_samples=0,
            compiler_runs=0,real_ckks_runs=0,strict_math_probes=[dict(sample_id='test:'+str(9000+i),
                exact=deepcopy(check),approximate=deepcopy(check)) for i in range(3)])
        return state,work,result

    def test_reconstructed_prompt_context_must_match_the_retained_initial_messages(self):
        prompts=ModuleType('pt_agent.planners.prompts')
        prompts.build_system_prompt=lambda ctx:'synthetic grammar'
        prompts.initial_message=lambda ctx:'synthetic original model task'
        with patch.dict(sys.modules,{prompts.__name__:prompts}):
            self.assertEqual(entry.verify_context(self.initial,self.ctx)['next_step'],'tune')
            prompts.initial_message=lambda ctx:'changed calibration context'
            with self.assertRaisesRegex(ValueError,'original context'):entry.verify_context(self.initial,self.ctx)

    def test_malformed_new_reply_retains_the_original_agent_rejection_semantics(self):
        state,work,result=self.finish(malformed=True)
        _,source=entry.check_source(self.case,ledger=self.ledger)
        self.assertEqual(state['history'][4]['stage'],'response')
        self.assertFalse(state['history'][4]['feasible'])
        self.assertFalse((work/'sources/iteration-5.ptdsl').exists())
        self.assertTrue(entry.check_provider_history(self.initial,state,work,2)['complete_provider_conversation'])
        self.assertEqual(entry.check_worker(result,state,source,2,agent_dir=work),'ok')

    def test_full_continuation_retains_six_call_protocol_and_strict_terminal_policy(self):
        state,work,result=self.finish()
        _,source=entry.check_source(self.case,ledger=self.ledger)
        audit=entry.check_provider_history(self.initial,state,work,2)
        self.assertFalse(audit['old_ledger_records_recreated']);self.assertEqual(state['calls'],6)
        self.assertEqual(entry.check_worker(result,state,source,2,agent_dir=work),'ok')
        for role,field,value in (('quality','acc_drop_pp',.500001),('math','criterion','relaxed'),('math','finite',False)):
            with self.subTest(role=role,field=field):
                changed=deepcopy(result)
                if role=='quality':
                    changed['agent']['confirm']['evals'][0][field]=value
                    changed_state=deepcopy(state);changed_state['confirmation']=changed['agent']['confirm']
                else:
                    changed['strict_math_probes'][0]['approximate'][field]=value;changed_state=state
                with self.assertRaises(ValueError):entry.check_worker(changed,changed_state,source,2,agent_dir=work)
        bad=deepcopy(state);bad['messages'][0]['content']='new system prompt'
        with self.assertRaises(ValueError):entry.check_provider_history(self.initial,bad,work,2)

    def test_actual_worker_step_finally_preserves_pending_state_when_tuning_raises(self):
        # Execute the unchanged worker loop body with the synthetic trusted state
        # functions. This covers its durable finally without numerical imports.
        tree=ast.parse(Path(common.__file__).read_text())
        worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='worker')
        loop=next(node for node in ast.walk(worker) if isinstance(node,ast.While))
        args=['ctx','state','work','client','resumed_state','result','save']
        function=ast.FunctionDef(name='step_loop',args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg=name) for name in args],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=[loop],decorator_list=[])
        namespace=dict(agent=self.agent,save_state=save_state)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
            str(Path(common.__file__)),'exec'),namespace)
        work=self.root/'partial';work.mkdir();state=deepcopy(self.initial);result={};saved=[]
        client=SimpleNamespace(calls=0,chat=Mock())
        with patch.object(self.agent,'tune',side_effect=RuntimeError('synthetic interrupted tool')):
            with self.assertRaisesRegex(RuntimeError,'interrupted tool'):
                namespace['step_loop'](self.ctx,state,work,client,state,result,lambda:saved.append(deepcopy(result)))
        restored=load_state(work/'SESSION.json')
        self.assertEqual(restored,self.initial);self.assertEqual(restored['phase'],'tune')
        self.assertEqual(restored['calls'],4);client.chat.assert_not_called()
        self.assertEqual(saved,[dict(actual_api_calls=0)])

    def outer_args(self,out,**kwargs):
        values=dict(out=out,source_case=self.case,ledger=self.ledger,allowance=self.root/'allowance.json',
            registry=self.root/'registry.jsonl',python=Path('/trusted/python'),prepare_only=True,
            seconds=7200,memory_gib=12,case_gib=4)
        values.update(kwargs);return Namespace(**values)

    def test_new_case_preparation_has_separate_clock_unique_registry_and_no_calls(self):
        out=self.root/'new-continuation';old=load_state(self.old/'SESSION.json')
        with patch.object(entry.paths,'RESULTS_DIR',self.root),patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)), \
                patch.object(entry,'run_process',side_effect=AssertionError('Preparation must be source-only')):
            result=entry.execute(self.outer_args(out))
        self.assertEqual(result['status'],'prepared');self.assertEqual(result['actual_new_api_calls'],0)
        self.assertFalse(result['cold_start']);self.assertFalse(result['historical_elapsed_reused'])
        self.assertEqual(result['original_time_seconds'],3575.021826592)
        completion=json.loads((out/'completion.json').read_text())
        self.assertLess(completion['complete_case_elapsed_seconds'],10)
        self.assertTrue(completion['registry_persisted']);self.assertTrue(completion['required_artifacts_persisted'])
        rows=[json.loads(line) for line in self.root.joinpath('registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(load_state(self.old/'SESSION.json'),old)
        with patch.object(entry.paths,'RESULTS_DIR',self.root),self.assertRaises(ValueError):entry.execute(self.outer_args(out))

    def test_interrupted_process_retains_new_partial_and_never_becomes_completed(self):
        out=self.root/'interrupted-continuation'
        def interrupt(command,**kwargs):
            work=out/'work/agent';work.mkdir(parents=True)
            self.write(work/'worker-result.json',dict(status='running',actual_api_calls=0))
            save_state(work/'SESSION.json',self.initial)
            return dict(exit_code=-15,timed_out=False,signal=15,signals_sent=[],remaining_pids=[])
        with patch.object(entry.paths,'RESULTS_DIR',self.root),patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)), \
                patch.object(entry,'ensure_allowance'),patch.object(entry,'run_process',side_effect=interrupt):
            result=entry.execute(self.outer_args(out,prepare_only=False))
        self.assertEqual(result['status'],'error');self.assertEqual(result['worker_attempts'],1)
        self.assertEqual(result['worker_completed'],0);self.assertEqual(result['actual_new_api_calls'],0)
        self.assertEqual(load_state(out/'work/agent/SESSION.json'),self.initial)
        self.assertEqual(result['worker']['status'],'running')
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],'error')

    def test_worker_cli_limits_resumption_to_the_original_hybrid_protocol(self):
        self.assertFalse(entry._clean(dict(exit_code=0)))
        self.assertTrue(entry._clean(dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])))
        base=['--worker','--resume-session','/copy/SESSION.json','--resume-case','/old/case']
        with patch.object(common,'worker',return_value=0) as worker:
            self.assertEqual(common.main(base),0)
        self.assertTrue(worker.call_args.args[0].tuner)
        for extra in (['--no-tuner'],['--prepare-only'],['--resume-source','/source','--resume-result','/result']):
            with self.subTest(extra=extra),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):common.main(base+extra)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):common.main(base[1:])
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):common.main(['--worker','--resume-session','/copy/SESSION.json'])


if __name__=='__main__':unittest.main()
