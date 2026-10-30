"""Recorded offline token Agent sessions, budget persistence and split isolation."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.analyzer import calibrate
from pt_agent.intake.graph import Graph,Node,TensorInfo
from pt_agent.models.info import ModelInfo
from pt_agent.planners.agent_tools import build_context
from pt_agent.planners import agent,confirm,prompts
from pt_agent.planners.transformer_session import ModelBudgetClient,run_session
from pt_agent.llm import LLMError
from pt_agent.tests._artifacts import COMPILER,require


SOURCE='''@fhe.program(slots=1024)
def token_demo(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 v=linear(a,w=W["matrix"],algo="bsgs")
 z=act(v,fn="tanh",method="cheb",deg=15,domain=(-1.,1.))
 y=flatten(z,start_dim=1)
 return y
'''


class Samples:
    def __init__(self):self.reads=[]
    def get(self,split,n=None):
        self.reads.append(split)
        if split=='report':raise AssertionError('report read before freeze')
        return dict(x=np.linspace(-.3,.3,12).reshape(6,1,2)[:n]),None


def fixture():
    nodes=[Node('n0','linear',['x'],['v'],dict(weight='matrix',bias=None),''),
           Node('n1','act',['v'],['z'],dict(fn='tanh'),''),
           Node('n2','flatten',['z'],['y'],dict(start_dim=1,end_dim=-1),'')]
    tensors={key:TensorInfo(key,shape,'act') for key,shape in
             [('x',(1,1,2)),('v',(1,1,2)),('z',(1,1,2)),('y',(1,2))]}
    graph=Graph('token_demo',nodes,tensors,dict(matrix=np.eye(2)),['x'],['y'])
    graph.intake_stats=dict(transformer_normalized=True)
    info=ModelInfo(dict(type='torch',intake=graph.intake_stats),graph,{},
                   [dict(name='x',shape=[1,1,2])],None,'classification','classification_top1',
                   dict(agreement_pct=99.5),{})
    samples=Samples();arrays,_=samples.get('calibration')
    return info,calibrate(graph,arrays),samples


class RecordedClient:
    def __init__(self,replies):self.replies=iter(replies);self.messages=[]
    def chat(self,messages,*,purpose):
        self.messages.append(copy.deepcopy(messages))
        value=next(self.replies)
        if isinstance(value,Exception):raise value
        return dict(content=value,seconds=0.,usage=dict(prompt_tokens=1,completion_tokens=1))


class ClientBudgetTests(unittest.TestCase):
    def test_hard_budget_persists_across_clients_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=RecordedClient(['{}',LLMError('transport_timeout')])
            client=ModelBudgetClient(raw,tmp,limit=2)
            client.chat([],purpose='first')
            with self.assertRaises(LLMError):client.chat([],purpose='second')
            restored=ModelBudgetClient(raw,tmp,limit=2)
            with self.assertRaisesRegex(LLMError,'budget_exhausted'):restored.chat([],purpose='third')
            self.assertEqual(restored.calls,2);self.assertEqual(len(raw.messages),2)
            self.assertTrue((Path(tmp)/'call-001.response.json').exists())
            self.assertTrue((Path(tmp)/'call-002.error.json').exists())

    def test_cap_is_at_most_forty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):ModelBudgetClient(object(),tmp,limit=41)


@require(COMPILER)
class TokenAgentTests(unittest.TestCase):
    def test_u1_token_pipeline_report_follows_freeze_before_native(self):
        from pt_agent import transformer_run
        from pt_agent.experiments import a1_transformer_agent as experiment
        from pt_agent.data.samples import StoredSplits
        info,calibration,samples=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model_dir=root/'model';out=root/'run';out.mkdir()
            info.save(model_dir);data=model_dir/'samples';data.mkdir()
            arrays,_=samples.get('calibration')
            for split in ('calibration','heldout','confirm'):
                np.savez(data/(split+'.npz'),**arrays)
            np.savez(data/'report.npz',x=np.resize(arrays['x'],(160,1,2)))
            events=[];original=StoredSplits.get
            def get(reader,split,n=None):
                if split=='report':
                    self.assertTrue((out/'FROZEN.json').exists());events.append('report')
                return original(reader,split,n)
            def native(*args):
                self.assertTrue((out/'report/REPORT.json').exists())
                events.append('compiler-managed');return dict(complete=True)
            options=SimpleNamespace(out=out,model_info=model_dir/'model_info.json',planner='agent',
                stages=['plan','check','mock','count','export','compile','test'],agent_iters=1,tuner='off',
                agent_client=RecordedClient([json.dumps(dict(program=SOURCE))]))
            with patch.object(StoredSplits,'get',new=get),patch.object(experiment,'native_stage',side_effect=native):
                self.assertEqual(transformer_run.inside(options),0)
            self.assertEqual(events,['report','compiler-managed'])
            report=json.loads((out/'REPORT.json').read_text())
            self.assertTrue(report['complete']);self.assertEqual(report['report']['report_images'],160)
            self.assertEqual((out/'program.ptdsl').read_text(),SOURCE)

    def test_report_guard_prevents_split_read_without_freeze(self):
        from pt_agent.experiments.a1_transformer_agent import evaluate_frozen
        info,calibration,samples=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'program.ptdsl';path.write_text(SOURCE)
            with self.assertRaises(FileNotFoundError):
                evaluate_frozen(info,calibration,samples,path,root/'report',frozen=root/'missing.json')
        self.assertNotIn('report',samples.reads)

    def test_replay_success_infeasible_feedback_parse_and_response_failure(self):
        info,calibration,samples=fixture()
        replies=['not JSON',json.dumps(dict(program='invalid python')),
                 json.dumps(dict(program=SOURCE.replace('domain=(-1.,1.)','domain=(-.1,.1)'))),
                 json.dumps(dict(program=SOURCE))]
        raw=RecordedClient(replies)
        with tempfile.TemporaryDirectory() as tmp:
            result=run_session(info,calibration,samples,raw,Path(tmp),max_iters=4,tuner=False)
            rows=[json.loads(line) for line in (Path(tmp)/'agent_trace.jsonl').read_text().splitlines()]
            self.assertEqual(result['status'],'feasible')
            self.assertEqual([r['fb']['stage'] for r in rows],['response','parse','complete','complete'])
            self.assertFalse(rows[2]['fb']['feasible']);self.assertTrue(rows[3]['fb']['feasible'])
            self.assertEqual((Path(tmp)/'sources/iteration-4.ptdsl').read_text(),SOURCE)
            self.assertTrue((Path(tmp)/'FROZEN.json').exists())
            self.assertIn('not feasible',raw.messages[3][-1]['content'])
        self.assertNotIn('report',samples.reads)
        self.assertEqual(samples.reads.count('confirm'),1)

    def test_replay_exhaustion_does_not_make_an_extra_request(self):
        info,calibration,samples=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            raw=RecordedClient(['bad JSON'])
            client=ModelBudgetClient(raw,Path(tmp)/'calls',limit=1)
            result=run_session(info,calibration,samples,client,Path(tmp)/'session',max_iters=8,tuner=False)
            self.assertEqual(result['status'],'error');self.assertEqual(result['error'],'budget_exhausted')
            self.assertEqual(client.calls,1);self.assertEqual(len(raw.messages),1)
            self.assertFalse((Path(tmp)/'session/FROZEN.json').exists())
        self.assertNotIn('report',samples.reads)

    def test_v7_tensor_tuner_and_single_confirm_dispatch(self):
        info,calibration,samples=fixture()
        raw=RecordedClient([json.dumps(dict(program=SOURCE))])
        with tempfile.TemporaryDirectory() as tmp:
            result=run_session(info,calibration,samples,raw,Path(tmp),max_iters=1,tuner=True,tuner_budget_s=10)
            self.assertEqual(result['status'],'feasible')
            self.assertTrue(result['history'][0]['tuned']['feasible'])
        self.assertNotIn('report',samples.reads)

    def test_v7_prompt_all_inputs_no_heldout_domain_hints_or_target_answer(self):
        info,calibration,samples=fixture()
        ctx=build_context(info,calibration,samples=samples);ctx.prompt_version='r8-v7'
        message=json.loads(prompts.initial_message(ctx));system=prompts.build_system_prompt(ctx)
        self.assertEqual(message['inputs'],[dict(name='x',shape=[1,1,2],kind='act')])
        self.assertNotIn('needed_margin',json.dumps(message))
        self.assertNotIn(SOURCE,system)
        self.assertIn('final_inv',system);self.assertIn('inv_budget',system)
        self.assertIn('calibration only',system)

    def test_confirm_retains_max_u_and_divergence_requirements(self):
        info,calibration,samples=fixture();ctx=build_context(info,calibration,samples=samples)
        ctx.prompt_version='r8-v7';ctx.val_split='confirm'
        candidate=confirm.Candidate(SOURCE,'llm',1,10,0)
        for max_u,diverged in ((.91,False),(.5,True)):
            measured=dict(agreement_pct=100.,max_u=max_u,diverged=diverged,n_diverged=int(diverged),
                          feasible=False,boot_input_bound=None,boot_input_max_abs=None,agree=6,total=6)
            with patch.object(confirm,'validate',return_value=measured):
                result=confirm.confirm_select([candidate],ctx,engine_profile='tensor')
            self.assertIsNone(result['selected'])


if __name__=='__main__':unittest.main()
