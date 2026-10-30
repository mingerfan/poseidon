"""Softmax moves rebuild calibration domains and require both validation levels."""

import copy
from itertools import islice
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.intake.graph import Node,TensorInfo
from pt_agent.planners import transformer_tuner as tuner,transformer_rule as rule
from pt_agent.tests.test_u4_squared_softmax import fixture


def context():
    node=Node('s','softmax',['x'],['p'],dict(keep='all',mask='m'),'')
    graph=SimpleNamespace(nodes=[node],tensors=dict(x=TensorInfo('x',(1,3,7),'act')))
    return SimpleNamespace(graph=graph,calibration={'p::exp':dict(min=-.3,max=.3)},
        act_tensors=['p::exp','p::inv'],prompt_version='r8-v7',profile='qwen48',weights={},
        model_info=SimpleNamespace(threshold=dict(agreement_pct=99.)))


def measured(*,feasible=True,u=.5,agreement=100):
    return dict(feasible=feasible,max_u=u,agreement_pct=agreement,diverged=False,agree=6,total=6)


class TunerTests(unittest.TestCase):
    def test_moves_have_adjacent_k_rho_and_calibration_bounds(self):
        ctx=context();p=parse(fixture(2));op=next(op for op in p.ops if op.kind=='softmax')
        candidates=list(islice(tuner.softmax_moves(p,ctx,op.id),4))
        self.assertEqual([move['kind'] for move,a in candidates],['steps','steps','inv_budget','inv_budget'])
        self.assertEqual([(a['steps'],a['inv_budget']) for _,a in candidates],[(1,.25),(3,.25),(2,.5),(2,.1)])
        for move,attrs in candidates:
            self.assertIsNotNone(attrs['final_inv_method'])
            expected=rule.budget_softmax_adjustments(ctx.calibration['p::exp'],7,attrs['steps'],attrs['inv_budget'])
            for role in ('exp','inv','step_inv','final_inv'):
                self.assertEqual(attrs[role+'_domain'],expected[role+'_domain'])

    def simulate(self,coarse,full):
        p=parse(fixture(2));ctx=context();op=next(op for op in p.ops if op.kind=='softmax')
        attrs=dict(op.attrs,steps=1)
        results=[measured(),measured(),coarse]+([full] if coarse['feasible'] else [])+[measured()]
        def cost(program,*args,**kwargs):
            k=next(op.attrs['steps'] for op in program.ops if op.kind=='softmax')
            return dict(latency_pred_us=100. if k==2 else 10.,boots_pred=k)
        with patch.object(tuner,'cached_estimate',side_effect=cost), \
             patch.object(tuner,'exact_check',return_value=dict(ok=True)), \
             patch.object(tuner,'validate',side_effect=results) as validation, \
             patch.object(tuner,'softmax_moves',return_value=iter([(dict(kind='steps'),attrs)])):
            result=tuner.tune(p,ctx,engine_profile='tensor')
        return result,validation

    def test_full_failure_rejects_move_even_after_coarse_pass(self):
        result,validation=self.simulate(measured(),measured(feasible=False,agreement=98))
        self.assertEqual(result.changes,[])
        self.assertFalse(next(row for row in result.history if row['phase']=='softmax')['accepted'])
        self.assertEqual(validation.call_args_list[2].args[2],200)

    def test_coverage_failure_rejects_without_full_validation(self):
        result,validation=self.simulate(measured(feasible=False,u=1.1),None)
        self.assertEqual(result.changes,[]);self.assertEqual(validation.call_count,4)

    def test_accepts_latency_drop_only_after_both_levels(self):
        result,validation=self.simulate(measured(),measured())
        self.assertEqual(len(result.changes),1)
        self.assertEqual(result.changes[0]['move']['kind'],'steps')
        self.assertIn('call_lines',result.changes[0])
        self.assertEqual(result.cost_after['latency_pred_us'],10.)


if __name__=='__main__':unittest.main()
