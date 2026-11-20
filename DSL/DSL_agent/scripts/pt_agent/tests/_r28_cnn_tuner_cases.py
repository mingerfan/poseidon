"""Actual generic tuner on a deterministic tiny CNN, with synthetic inputs."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class CNNTunerCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from pt_agent.planners import transformer_tuner
        cls.np, cls.tuner = np, transformer_tuner

    def setUp(self):
        from pt_agent.planners import agent
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        from pt_agent.experiments.r28_common_agent import observed_tools
        temporary=TemporaryDirectory(prefix='generic-cnn-root-adapter-')
        self.addCleanup(temporary.cleanup)
        self.agent=agent
        context=observed_tools(agent,StageRecorder(Path(temporary.name)/'events.jsonl'))
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def fixture(self, heldout_magnitude=.6):
        np = self.np
        from pt_agent.analyzer import calibrate, reference_forward
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.planners.agent_tools import build_context
        from pt_agent.dsl.parser import parse

        shapes = dict(x=(1,1,2,2), convolved=(1,2,2,2), activated=(1,2,2,2),
                      pooled=(1,2), logits=(1,2), kernel=(2,1,1,1), head=(2,2))
        tensors = {name: TensorInfo(name, shape, 'input' if name == 'x' else
                    'weight' if name in ('kernel','head') else 'act') for name, shape in shapes.items()}
        graph = Graph('synthetic_cnn', [
            Node('conv','conv2d',['x'],['convolved'],dict(weight='kernel',stride=(1,1),padding=(0,0),dilation=(1,1),groups=1),'synthetic'),
            Node('act','act',['convolved'],['activated'],dict(fn='silu'),'synthetic'),
            Node('pool','mean',['activated'],['pooled'],dict(axes=(2,3),keepdim=False),'synthetic'),
            Node('head','linear',['pooled'],['logits'],dict(weight='head'),'synthetic')],
            tensors, dict(kernel=np.ones((2,1,1,1),dtype=np.float64),
                          head=np.asarray([[1.,0.],[-1.,0.]],dtype=np.float64)), ['x'], ['logits'])
        training = np.asarray([-.5,-.2,.2,.5],dtype=np.float64)[:,None,None,None] * np.ones((4,1,2,2))
        heldout = np.asarray([-heldout_magnitude,-.1,.1,heldout_magnitude],dtype=np.float64)[:,None,None,None] * np.ones((4,1,2,2))
        class Samples:
            def __init__(self):
                self.accesses=[]
            def get(self, split, n=None):
                self.accesses.append(split)
                if split not in ('calibration','heldout'):
                    raise AssertionError('Tuning must preserve confirmation and report samples')
                inputs=training if split=='calibration' else heldout
                labels=reference_forward(graph,{'x':inputs})['logits'].argmax(1)
                return {'x':inputs[:n]},labels[:n]
        samples=Samples()
        info=ModelInfo(dict(type='torch',path='synthetic-only'),graph,dict(mode='original tensors'),
            [dict(name='x',kind='image',shape=[1,1,2,2],dtype='float64')],None,
            'classification','classification_top1',dict(acc_drop_pp=.5),
            dict(kind='synthetic_training',version='synthetic-v1',calibration=list(range(4)),
                 heldout=list(range(4,8)),confirm=list(range(8,12)),report=list(range(12,16))),[])
        calibration=calibrate(graph,{'x':training},batch=2)
        ctx=build_context(info,calibration,n_val=4,samples=samples)
        ctx.prompt_version='r8-v11';ctx.placement='program-plan';ctx.tune_search='sweep'
        source='''@fhe.program(slots=1024, refresh="agent", placement="program-plan", l_max=5, boot_out_level=5)
def candidate(x: Cipher[1,1,2,2]):
    p = pack(x, layout="mux", gap=1)
    c = conv2d(p, w=W["kernel"], stride=(1,1), pad=(0,0), algo="mpcb")
    a = act(c, fn="silu", method="cheb", deg=15, domain=(-2.0,2.0))
    m = mean(a, axis=(2,3))
    y = linear(m, w=W["head"], algo="bsgs")
    return y
'''
        return parse(source),ctx,samples

    def tuned(self, program, ctx):
        with self.tuner.tune_rule_scope('v8-6.25'):
            return self.agent.tune(program,ctx,budget_s=2,max_sweeps=1,
                                   engine='tensor',engine_profile='tensor',cache=False)

    def test_actual_generic_tune_cnn_retains_domain_fields_and_packed_math(self):
        from pt_agent.analyzer import reference_forward
        from pt_agent.transformer_validation import execute, validate
        from pt_agent.margin import check_margin
        program,ctx,samples=self.fixture()
        result=self.tuned(program,ctx)
        self.assertTrue(result.history)
        last=result.history[-1]
        self.assertEqual(last['phase'],'result')
        for name in ('max_u_calibrated','max_u_analytic','worst_acts'):
            self.assertIn(name,last)
        self.assertTrue(check_margin(result.program,ctx)['ok'])
        self.assertLessEqual(last['max_u_calibrated'],.90)
        self.assertGreater(last['max_u_calibrated'],0.0)
        self.assertLessEqual(last['max_u_analytic'],1.)
        self.assertTrue(result.feasible)
        arrays={name:value[:1] for name,value in ctx.val_inputs.items()}
        exact=reference_forward(ctx.graph,arrays)[ctx.graph.outputs[0]]
        tensor=execute(result.program,ctx.weights,arrays,engine='tensor',exact=True)
        mock=execute(result.program,ctx.weights,arrays,engine='mock',exact=True)
        for observed in (tensor,mock):
            self.assertTrue(self.np.all(self.np.abs(observed-exact)<=1e-12+1e-12*self.np.abs(exact)))
        measured=validate(result.program,ctx,engine='mock',stop_on_divergence=False)
        self.assertEqual(measured['total'],4)
        self.assertTrue(measured['feasible'])
        self.assertIn('max_u_calibrated',measured)
        self.assertNotIn('confirm',samples.accesses)
        self.assertNotIn('report',samples.accesses)

    def test_actual_alignment_uses_calibration_and_ignores_heldout_needed_margin(self):
        domains=[]
        for magnitude in (.6,.85):
            program,ctx,samples=self.fixture(magnitude)
            ctx.needed_margin={name:1e9 for name in ctx.needed_margin}
            with patch.object(self.tuner,'margin_domain',wraps=self.tuner.margin_domain) as margin:
                result=self.tuned(program,ctx)
            self.assertTrue(margin.call_args_list)
            for call in margin.call_args_list:
                self.assertEqual(call.args,(-.5,.5))
                self.assertEqual(call.kwargs['factor'],2.0)
            domain=next(op.attrs['domain'] for op in result.program.ops if op.kind=='act')
            domains.append(tuple(domain))
            alignment=next(row for row in result.history if row['phase']=='domain')
            self.assertTrue(alignment['accepted'],alignment)
        self.assertEqual(domains,[(-1.,1.),(-1.,1.)])


if __name__=='__main__':
    unittest.main()
