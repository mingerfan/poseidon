"""Offline fake-validation checks of the new knobs and legacy cheb sequence."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.backends import CountBackend
from pt_agent.dsl.parser import parse
from pt_agent.lower import lower
from pt_agent.planners import tuner
from pt_agent.tests.test_u5_lower import source


class TunerTests(unittest.TestCase):
    def run_tune(self,settings):
        p=parse(source(settings))
        def cost(candidate,*args,**kwargs):
            b=CountBackend(candidate.header.slots)
            lower(candidate,{},b)
            return dict(latency_pred_us=100*b.stats()['max_depth'],boots_pred=0,boot_nodes=[],hotspots=[])
        def validate(candidate,*args,**kwargs):
            return dict(feasible=True,diverged=False,acc_drop_pp=0.0,agree=1,total=1,max_u=0.1,
                        top1=100.,exact_top1=100.,rows=[dict(obs_min=None,obs_max=None)],worst_acts=[])
        with patch.object(tuner,'cached_estimate',side_effect=cost), \
             patch.object(tuner,'validate',side_effect=validate), \
             patch.object(tuner,'_exact_check',return_value=dict(ok=True)):
            result=tuner.tune(p,SimpleNamespace(weights={},profile='qwen48'),engine='mock')
        return result,[row for row in result.history if row['phase']=='coarse']

    def test_remez_uses_same_ladder(self):
        result,rows=self.run_tune('fn="exp",method="remez",deg=31,domain=(-2.,2.)')
        self.assertEqual([(r['kind'],r['old'],r['new']) for r in rows],[('deg',31,15)])
        self.assertEqual(result.program.ops[-1].attrs['method'],'remez')

    def test_newton_tries_iterations_then_initializer_degree(self):
        result,rows=self.run_tune('fn="inv",method="newton",deg0=15,iters=2,domain=(0.25,2.)')
        self.assertEqual([(r['kind'],r['old'],r['new']) for r in rows],[('iters',2,1),('deg0',15,7)])
        self.assertEqual(result.program.ops[-1].attrs['method'],'newton')

    def test_composite_search_lowers_depth(self):
        result,rows=self.run_tune('fn="relu",method="composite",degrees=(3,3,1),eps=0.2,domain=(-2.,2.)')
        self.assertEqual([r['kind'] for r in rows],['degrees'])
        self.assertLess(rows[0]['search']['depth'],7)
        self.assertEqual(result.program.ops[-1].attrs['method'],'composite')

    def test_composite_without_lower_candidate_keeps_program(self):
        result,rows=self.run_tune('fn="relu",method="composite",degrees=(3,3),eps=0.2,domain=(-2.,2.)')
        self.assertEqual(rows,[])
        self.assertEqual(result.program.ops[-1].attrs['degrees'],(3,3))
        self.assertTrue(any(r['phase']=='skip_search' for r in result.history))

    def test_cheb_sequence_remains_one_ladder_step(self):
        result,rows=self.run_tune('fn="exp",method="cheb",deg=31,domain=(-2.,2.)')
        self.assertEqual([(r['kind'],r['old'],r['new']) for r in rows],[('deg',31,15)])
        self.assertNotIn('search',rows[0])
        self.assertEqual(result.program.ops[-1].attrs['method'],'cheb')


if __name__=='__main__':
    unittest.main()
