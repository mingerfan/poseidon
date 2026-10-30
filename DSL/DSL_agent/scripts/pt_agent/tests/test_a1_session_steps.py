"""A1 §6.8: process boundaries preserve the frozen r8-v8 decisions."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.llm import LLMError
from pt_agent.planners import agent
from pt_agent.planners.confirm import Candidate
from pt_agent.planners.transformer_session import load_state,save_state,write_session_result
from pt_agent.planners.tuner import TuneResult
from pt_agent.tests.test_a1_agent import SOURCE


def feedback(latency=30,feasible=True,stage='complete'):
    if stage!='complete':return dict(stage=stage,feasible=False,exact=None,cost=None,validation=None)
    return dict(stage=stage,feasible=feasible,exact=dict(ok=True),
        cost=dict(latency_pred_us=latency,boots_pred=2),
        validation=dict(agree=6,max_u=.5 if feasible else .95,diverged=False,
            max_u_calibrated=.5 if feasible else .95,max_u_analytic=0.,
            boot_input_bound=None,boot_input_max_abs=None,worst_acts=[]))


class Client:
    def __init__(self,replies):self.replies=iter(replies);self.messages=[]
    def chat(self,messages,*,purpose):
        self.messages.append(copy.deepcopy(messages))
        reply=next(self.replies)
        if isinstance(reply,Exception):raise reply
        return copy.deepcopy(reply)


def reply(source=SOURCE,**extra):
    return dict(content=json.dumps(dict(program=source)),seconds=1.,
                usage=dict(prompt_tokens=7,completion_tokens=3),finish_reason='stop',**extra)


class SessionStepTests(unittest.TestCase):
    def equivalent(self,replies,feedbacks,*,tuner=True,tune_policy='new_best',tune_feasible=True,
                   tune_rule='v8-6.11'):
        ctx=SimpleNamespace(prompt_version='r8-v8')
        results=[]
        for split in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);trace=root/'agent_trace.jsonl';client=Client(replies)
                evals=iter(copy.deepcopy(feedbacks));tunes=[];pools=[]
                def tuning(program,*args,**kwargs):
                    tunes.append(print_program(program))
                    cost=dict(latency_pred_us=5,boots_pred=1)
                    confirmed=feedback(feasible=tune_feasible)['validation']
                    return TuneResult(program,print_program(program),[],dict(cost),dict(cost),
                                      2,1,2.,[confirmed],tune_feasible)
                def confirming(pool,*args,**kwargs):
                    pools.append([asdict(item) for item in pool])
                    ordered=sorted(pool,key=lambda c:(c.latency_pred_us,c.iteration,c.origin))
                    return dict(selected=ordered[0] if ordered else None,evals=[],
                                deduplicated=dict(before=len(pool),after=len(pool)),duplicates=[])
                with patch.object(agent,'build_system_prompt',return_value='system'), \
                     patch.object(agent,'initial_message',return_value='initial'), \
                     patch.object(agent,'evaluate',side_effect=lambda *a,**k:next(evals)), \
                     patch.object(agent,'tune',side_effect=tuning), \
                     patch.object(agent,'confirm_select',side_effect=confirming):
                    kwargs=dict(max_iters=len(replies),tuner=tuner,tune_policy=tune_policy,engine_profile='tensor',
                                tune_rule=tune_rule)
                    if split:
                        state=agent.new_agent_state(ctx,**kwargs)
                        while state['phase']!='done':
                            agent.agent_step(ctx,state,trace_path=trace,client=client)
                            save_state(root/'SESSION.json',state)
                            state=load_state(root/'SESSION.json')
                        result=agent.agent_result(state)
                        for field in ('messages','history','pool','records','best','first_feasible','llm_min',
                                      'has_tuned','tune_cache','calls','prompt_tokens','completion_tokens'):
                            self.assertIn(field,state)
                    else:
                        result=agent._run_agent_once(ctx,client,trace_path=trace,**kwargs)
                    frozen=write_session_result(result,root)
                result.pop('seconds');frozen.pop('seconds')
                results.append(dict(result=result,record=frozen,pool=pools,messages=client.messages,
                    trace=[json.loads(line) for line in trace.read_text().splitlines()],tunes=tunes,
                    program=(root/'program.ptdsl').read_text() if (root/'program.ptdsl').exists() else None))
        self.assertEqual(results[0],results[1])
        return results[1]

    def test_success_trace_pool_order_tuning_and_freeze(self):
        result=self.equivalent([reply(),reply(SOURCE.replace('deg=15','deg=31'))],
                               [feedback(30),feedback(20)])
        self.assertEqual(result['result']['status'],'feasible')
        self.assertEqual(result['result']['first_feasible_iteration'],1)
        self.assertEqual(len(result['tunes']),2)

    def test_infeasible_cost_still_sets_llm_min_and_prevents_later_tune(self):
        result=self.equivalent([reply(),reply(SOURCE.replace('deg=15','deg=31'))],
                               [feedback(10,False),feedback(20)],tune_feasible=False,
                               tune_rule='v8-pre-6.11')
        self.assertEqual(len(result['tunes']),1)
        self.assertEqual(result['result']['first_feasible_iteration'],2)
        self.assertFalse(result['result']['history'][1]['new_llm_best'])
        self.assertEqual(result['result']['history'][1]['llm_min'],10)

    def test_budget_exhaustion(self):
        result=self.equivalent([LLMError('budget_exhausted')],[])
        self.assertEqual(result['result']['error'],'budget_exhausted')
        self.assertIsNone(result['program'])

    def test_length_then_success(self):
        length=dict(content='',seconds=1.,usage={},finish_reason='length')
        result=self.equivalent([length,reply()],[feedback()],tuner=False)
        self.assertEqual(result['trace'][0]['fb']['stage'],'length')

    def test_interface_then_success(self):
        renamed=SOURCE.replace('x:','v:').replace('pack(x,','pack(v,')
        result=self.equivalent([reply(renamed),reply()],[feedback(stage='interface'),feedback()],tuner=False)
        self.assertEqual(result['trace'][0]['fb']['stage'],'interface')

    def test_normalized_tune_cache_survives_round_boundaries(self):
        result=self.equivalent([reply(),reply(),reply()],[feedback(),feedback(),feedback()],tune_policy='all')
        self.assertEqual(len(result['tunes']),1)
        self.assertEqual([row['tuned_reused_from'] for row in result['trace']],[None,1,1])

    def test_stale_stop_is_identical(self):
        result=self.equivalent([reply()]*5,[feedback()]*5,tuner=False)
        self.assertEqual(result['result']['iterations'],3)

    def test_analytic_stop_skips_confirmation(self):
        result=self.equivalent([reply()],[dict(feedback(feasible=False),stop='analytic_domain_violation')],tuner=False)
        self.assertEqual(result['pool'],[])
        self.assertEqual(result['result']['error'],'analytic_domain_violation')

    def test_atomic_state_preserves_keys_tuples_and_nonfinite_values(self):
        state=dict(records={(1,'llm',SOURCE):({'max_u':float('inf')},3)},
                   pending=dict(domain=(-1.,1.)),calls=8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';save_state(path,state)
            self.assertEqual(load_state(path),state)
            self.assertFalse(path.with_name('state.json.tmp').exists())

    def test_v4_through_v7_keep_original_entry(self):
        for version in ('r8-v4','r8-v5','r8-v6','r8-v7'):
            with self.subTest(version=version),patch.object(agent,'_run_agent_once',return_value='original') as old:
                self.assertEqual(agent.run_agent(SimpleNamespace(prompt_version=version),None,trace_path=Path('unused')),'original')
                old.assert_called_once()


if __name__=='__main__':unittest.main()
