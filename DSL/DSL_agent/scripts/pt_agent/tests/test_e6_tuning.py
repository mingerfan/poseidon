"""E6 selection, unchanged sweep and independent request allowances (offline)."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import transformer_tuner as tuner
from pt_agent.planners.agent import saved_tune_result,new_agent_state
from pt_agent.planners.prompts import build_tune_prompt
from pt_agent.planners.transformer_session import ModelBudgetClient,model_calls,validate_tune_search
from pt_agent.tests.test_a1_tuner import context
from pt_agent.tests.test_r624_floor import SOURCE,measured,run
from pt_agent.tests.test_u4_squared_softmax import fixture


class Client:
    def __init__(self,answers,limit=20):
        self.answers=iter(answers);self.calls=0;self.limit=limit;self.requests=[]
    def chat(self,messages,*,purpose):
        self.calls+=1;self.requests.append((messages,purpose))
        answer=next(self.answers)
        if callable(answer):answer=answer(messages)
        return answer if isinstance(answer,dict) and 'content' in answer else dict(
            content=json.dumps(answer),seconds=.25,usage=dict(prompt_tokens=10,completion_tokens=5),finish_reason='stop')


def simulation(client,*,degree=15,budget=2400,clock=None,cost_override=None,menu_override=None,validation_override=None):
    p=parse(SOURCE.replace('deg=15','deg='+str(degree)))
    ctx=SimpleNamespace(prompt_version='r8-v11',placement='program-plan',tune_search='llm-select',
        weights={},profile='unused',calibration={k:dict(min=-.9,max=.9) for k in ('first','second')},
        model_info=SimpleNamespace(task='causal_lm',threshold=dict(logit_rel_err=.01)))
    estimates=[];validations=[]
    def cost(candidate,*args,**kwargs):
        estimates.append(print_program(candidate))
        return dict(latency_pred_us=(cost_override(candidate) if callable(cost_override) else cost_override) or sum(p.attrs['deg'] for p in tuner.points(candidate)),boots_pred=1)
    def validate(candidate,*args,**kwargs):
        validations.append((args,kwargs));return validation_override(candidate,*args,**kwargs) if validation_override else measured(True)
    with ExitStack() as stack:
        stack.enter_context(tuner.tune_rule_scope('v8-6.24'))
        stack.enter_context(patch('pt_agent.dsl.interface.require_interface'))
        stack.enter_context(patch('pt_agent.margin.require_margin'))
        stack.enter_context(patch('pt_agent.planners.margin_refit.refit',return_value=({},dict(status='kept'))))
        stack.enter_context(patch('pt_agent.planners.approximation_failures.program_failures',return_value=[]))
        stack.enter_context(patch.object(tuner,'point_names',return_value=['first','second']))
        stack.enter_context(patch.object(tuner,'cached_estimate',side_effect=cost))
        stack.enter_context(patch.object(tuner,'validate',side_effect=validate))
        stack.enter_context(patch.object(tuner,'exact_check',return_value=dict(ok=True)))
        if menu_override is not None:stack.enter_context(patch.object(tuner,'build_tune_menu',side_effect=menu_override))
        if clock is not None:stack.enter_context(patch.object(tuner.time,'monotonic',side_effect=clock))
        result=tuner.tune(p,ctx,client=client,iteration=4,budget_s=budget,engine_profile='tensor')
    return result,estimates,validations


class MenuTests(unittest.TestCase):
    def test_real_token_menu_matches_sweep_estimated_steps(self):
        p=parse(fixture(2));ctx=context();ctx.prompt_version='r8-v11';ctx.placement='program-plan'
        ctx.model_info.task='causal_lm';ctx.model_info.threshold['logit_rel_err']=.01
        seen=[]
        def cost(p,*a,**kw):
            seen.append(print_program(p));return dict(latency_pred_us=100,boots_pred=1)
        with tuner.tune_rule_scope('v8-6.24'),patch('pt_agent.dsl.interface.require_interface'), \
             patch('pt_agent.margin.require_margin'), \
             patch('pt_agent.planners.approximation_failures.program_failures',return_value=[]), \
             patch.object(tuner,'validate',return_value=measured(True)), \
             patch.object(tuner,'exact_check',return_value=dict(ok=True)), \
             patch.object(tuner,'cached_estimate',side_effect=cost):
            menu=tuner.build_tune_menu(p,ctx,{})
            tuner.tune(p,ctx,max_sweeps=1,engine_profile='tensor')
        sources=[]
        for item in menu:
            candidate=copy.deepcopy(p)
            candidate.ops[item['point']['op_id']].attrs.update({k:v['new'] for k,v in item['changes'].items()})
            sources.append(print_program(candidate))
            self.assertIn('depth',item['local_delta']);self.assertIn('counts',item['local_delta'])
            self.assertEqual(item['line'],5)
        self.assertTrue(menu)
        self.assertEqual(set(sources),set(seen[2:]))

    def test_default_sweep_matches_prechange_snapshot(self):
        saved=json.loads(Path(__file__).with_name('e6_sweep_snapshot.json').read_text())
        from pt_agent.tests.test_a1_convergent import DominanceTests
        r,c,_=DominanceTests().simulate(dominated=False)
        results={'softmax':dict(result=saved_tune_result(r),costs=c)}
        r,c=run('v8-6.24');results['floor']=dict(result=saved_tune_result(r),costs=c)
        for value in results.values():
            for key in ('search','llm_calls','llm_s','estimates','accepted_states'):value['result'].pop(key)
            for row in value['result']['history']:  # E8 §2.2 adds decision-neutral domain-row fields
                if row.get('phase')=='domain':row.pop('reasons',None);row.pop('metrics',None)
        self.assertEqual(json.loads(json.dumps(results)),saved)

    def test_prompt_snapshot(self):
        messages=build_tune_prompt({'program':'1: x'},[{'id':'m1'}],[])
        self.assertEqual(messages[0]['content'],Path(__file__).with_name('e6_t1_prompt.txt').read_text().rstrip('\n'))
        self.assertEqual(json.loads(messages[1]['content']),dict(context={'program':'1: x'},menu=[{'id':'m1'}],previous_rounds=[]))


class SelectionTests(unittest.TestCase):
    def test_order_duplicate_unknown_stale_atomic_and_fields(self):
        client=Client([dict(candidates=[dict(moves=['m1','m1']),dict(moves=['absent']),
            dict(moves=['m1']),dict(moves=['m1']),dict(moves=['m2'])])])
        result,estimates,validations=simulation(client)
        row=next(r for r in result.history if r['phase']=='llm_round')
        self.assertEqual([r.get('rejected') for r in row['candidates']],
                         ['conflict','unknown_id',None,'not_reached',None])
        self.assertEqual((row['accepted_candidates'],len(result.changes)),(2,2))
        self.assertEqual((client.calls,result.llm_calls,result.llm_s),(1,1,.25))
        self.assertEqual(client.requests[0][1],'tune-select-iter4-round1')
        self.assertEqual(row['menu_size'],2);self.assertEqual(row['tokens']['prompt_tokens'],10)
        self.assertEqual(row['finish_reason'],'stop')
        self.assertEqual(len(estimates),6)
        self.assertEqual([v[0][1:] for v in validations[2:-1]],[ (200,),(),(200,),() ])
        # One atomic two-point proposal is estimated once and accepted together.
        other,estimates,_=simulation(Client([dict(candidates=[dict(moves=['m1','m2'])])]))
        self.assertEqual(len(estimates),3);self.assertEqual(len(other.changes),2)
        self.assertEqual(other.history[-2]['accepted_candidates'],1)

    def test_invalid_json_and_length_zero_candidates_without_retry(self):
        for reply in (dict(content='bad',finish_reason='stop'),dict(content='{"candidates":[]}',finish_reason='length')):
            client=Client([reply,reply]);result,estimates,_=simulation(client)
            row=result.history[-2]
            self.assertEqual(row['candidates'],[]);self.assertEqual(client.calls,2)
            self.assertEqual(len(estimates),2)
            self.assertIn(row['parse_error'],('length','invalid_json'))
        rows,error=tuner.parse_tune_reply(dict(content='text {"candidates":[]} trailing {"candidates":[]}',finish_reason='stop'),[])
        self.assertEqual((rows,error),([],None))

    def test_round_no_accept_call_limit_and_time_stops(self):
        client=Client([dict(candidates=[dict(moves=['m1'])])]*4)
        result,_,_=simulation(client,degree=127)
        self.assertEqual(client.calls,4);self.assertEqual(len([r for r in result.history if r['phase']=='llm_round']),4)
        client=Client([dict(candidates=[dict(moves=['m1'])])]*2)
        result,_,_=simulation(client,cost_override=100)
        self.assertEqual(client.calls,2);self.assertEqual(result.history[-2]['candidates'][0]['rejected'],'cost')
        client=Client([],limit=0);result,_,_=simulation(client)
        self.assertEqual(client.calls,0)
        now=[0.]
        def answer(messages):now[0]=2401.;return dict(candidates=[dict(moves=['m1'])])
        client=Client([answer]);result,estimates,_=simulation(client,clock=lambda:now[0])
        self.assertEqual(len(estimates),2);self.assertEqual(client.calls,1)
        self.assertEqual(result.history[-2]['candidates'][0]['rejected'],'not_reached')
        self.assertEqual(result.history[-2]['candidates'][0]['reason'],'budget')

    def test_cache_miss_counter(self):
        from pt_agent.tensor_sim import cached_estimate,estimate_calls
        from pt_agent.cost import estimate
        p=parse('@fhe.program(slots=1024)\ndef f(x: Cipher[1,2]):\n y=pack(x,layout="flat")\n z=mul(y,y)\n return z\n')
        before=estimate_calls();weights={}
        cached_estimate(p,weights,'qwen48',estimate)
        cached_estimate(p,weights,'qwen48',estimate)
        self.assertEqual(estimate_calls()-before,1)
        cached_estimate(p,weights,'qwen48',estimate,cache=False)
        self.assertEqual(estimate_calls()-before,2)


class WiringTests(unittest.TestCase):
    def test_version_and_placement_reject_before_request(self):
        with patch('pt_agent.llm.LLMClient') as provider:
            for version,placement in (('r8-v10','program-plan'),('r8-v11',None),('r8-v4',None)):
                ctx=SimpleNamespace(tune_search='llm-select',prompt_version=version,placement=placement)
                with self.assertRaises(ValueError):validate_tune_search(ctx)
                with self.assertRaises(ValueError):new_agent_state(ctx)
            provider.assert_not_called()

    def test_tune_step_creates_only_the_t1_client_and_records_label(self):
        from pt_agent.experiments import a1_transformer_agent as entry
        from pt_agent.planners import agent_tools,transformer_session
        with TemporaryDirectory() as temporary:
            root=Path(temporary);model_dir=root/'model';model_dir.mkdir()
            (model_dir/'calibration.json').write_text('{}')
            session=root/'session';session.mkdir();(session/'SESSION.json').write_text('{}')
            ctx=SimpleNamespace()
            state=dict(phase='tune',max_iters=10,history=[],pending=dict(iteration=4))
            args=SimpleNamespace(model='bert',model_dir=model_dir,prompt_version='r8-v11',placement='program-plan',
                tune_search='llm-select',session_label='E6a',tune_call_limit=12,session_iters=7,
                out=root/'out',root=session,step='tune',call_dir=root/'calls',budget_path=root/'ledger',
                response_file=None,tune_rule='v8-6.24')
            stub=SimpleNamespace(chat=lambda *a,**k:dict(content='{}'))
            seen=[]
            def advance(ctx,directory,**kwargs):
                seen.append(kwargs)
                kwargs['tune_client'].chat([],purpose='tune-select-iter4-round1')
                return dict(complete=True)
            with patch('pt_agent.models.info.ModelInfo.load',return_value=SimpleNamespace()), \
                 patch.object(agent_tools,'build_context',return_value=ctx), \
                 patch.object(entry,'samples_for',return_value=None), \
                 patch.object(transformer_session,'load_state',return_value=state), \
                 patch.object(transformer_session,'session_step',side_effect=advance), \
                 patch('pt_agent.llm.LLMClient',return_value=stub) as provider:
                result=entry.session_step(args)
            provider.assert_called_once_with(args.budget_path,model='deepseek-flash',reasoning_effort='high',
                                            max_tokens=65536,max_total_calls=403)
            self.assertIsNone(seen[0]['client'])
            self.assertEqual(seen[0]['tune_client'].limit,12)
            self.assertEqual(result['tuning_calls'],1)
            self.assertEqual(result['tune_search'],'llm-select')
            self.assertTrue((root/'calls/bert/t1/E6a/call-001.request.json').exists())
            self.assertFalse((root/'calls/bert/r8-v11').exists())

    def test_session_namespaces_do_not_consume_e5d(self):
        stub=SimpleNamespace(chat=lambda *a,**k:dict(content='{}'))
        with TemporaryDirectory() as directory:
            old=ModelBudgetClient(stub,directory,model='bert',prompt_version='r8-v11')
            for _ in range(7):old.chat([],purpose='old')
            main=ModelBudgetClient(stub,directory,model='bert',prompt_version='r8-v11',label='s1')
            tune=ModelBudgetClient(stub,directory,model='bert',prompt_version='t1',label='s1',limit=20)
            replay=ModelBudgetClient(stub,directory,model='bert',prompt_version='t1',label='E6a',limit=12)
            for client,count in ((main,10),(tune,20),(replay,12)):
                for _ in range(count):client.chat([],purpose='offline')
                from pt_agent.llm import LLMError
                with self.assertRaises(LLMError):client.chat([],purpose='offline')
            self.assertEqual(old.calls,7)
            self.assertEqual(model_calls(directory,'bert','t1','s1'),20)
            self.assertEqual(model_calls(directory,'bert','r8-v11','s1'),10)
            self.assertEqual(model_calls(directory,'bert','r8-v11','s2'),0)


if __name__=='__main__':unittest.main()
