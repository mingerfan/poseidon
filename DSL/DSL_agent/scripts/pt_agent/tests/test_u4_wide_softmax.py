"""U4 §11.7: capped squaring, classified failures and frozen BERT text."""

import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.models.info import ModelInfo
from pt_agent.planners import transformer_rule as rule
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests._artifacts import RESULTS_DIR, require
from pt_agent.tests.test_u4_kernels import mock
from pt_agent.tests.test_u4_squared_softmax import fixture


class WideSoftmaxTests(unittest.TestCase):
    def test_readonly_score_categories_keep_padding_coordinates(self):
        from pt_agent.experiments.u4_wide_softmax import score_categories
        scores=np.array([[[[1.,2.,30.],[4.,5.,60.],[70.,80.,90.]]]])
        original=scores.copy()
        rows=score_categories(scores,np.array([[1.,1.,0.]]))
        self.assertEqual(rows,dict(
            valid_query_valid_key=dict(count=4,minimum=1.,maximum=5.),
            padding_key=dict(count=3,minimum=30.,maximum=90.),
            padding_query=dict(count=3,minimum=70.,maximum=90.)))
        np.testing.assert_array_equal(scores,original)

    def test_capped_rule_and_analytical_domains(self):
        for lo, hi, k0 in ((-2., 3., 3), (-100., 100., 9), (-200., 200., 10)):
            with self.subTest(k0=k0):
                cfg=rule.squared_softmax_adjustments(dict(min=lo,max=hi),32)
                a0,b0=lo*1.5-hi,hi*1.5-hi
                self.assertEqual(max(0,math.ceil(math.log2(b0-a0))),k0)
                k=min(k0,8);a,b=a0/(2**k),b0/(2**k)
                self.assertEqual(cfg,dict(shift=hi,steps=k,inv_scale=32*math.exp(b),
                    exp_domain=(a,b),inv_domain=(math.exp(a)/(32*math.exp(b)),1.0),
                    step_inv_domain=(1/48,1.5)))
                self.assertAlmostEqual(1/cfg['inv_domain'][0],32*math.exp(b-a))
                if k0>8:self.assertGreater(b-a,1)

    def test_infeasible_points_complete_table(self):
        tensor=SimpleNamespace(shape=(1,32,32),kind='tokens')
        nodes=[SimpleNamespace(id=f'n{i}',op='softmax',inputs=['x'],outputs=[f'y{i}'],
                               attrs=dict(keep='causal')) for i in (15,42)]
        graph=SimpleNamespace(inputs=['x'],outputs=['y42'],name='wide',nodes=nodes,
                              tensors=dict(x=tensor,y15=tensor,y42=tensor))
        calib={f'y{i}::exp':dict(min=-200,max=200) for i in (15,42)}
        def unavailable(fn,domain,**kwargs):
            return dict(fn=fn,domain=domain,selected=None,status='infeasible',
                        target=kwargs['target'],candidates=[])
        with patch.object(rule,'select_approximation',side_effect=unavailable) as select:
            with self.assertRaises(rule.ApproximationLimitError) as raised:rule.plan(graph,calib)
        self.assertEqual(select.call_count,6)
        self.assertEqual([x['key'] for x in raised.exception.issues],
                         [f'y{i}::{role}' for i in (15,42) for role in ('exp','inv','step_inv')])
        self.assertTrue(all(x['report']['status']=='infeasible' for x in raised.exception.issues))
        self.assertEqual(select.call_args_list[0].kwargs['target'],1e-3/256)

    def test_exact_identity_k8_wide_scores(self):
        rng=np.random.default_rng(20261027)
        x=rng.uniform(-230,170,(2,3,7));x[0,0,:2]=(-230,170)
        self.assertGreater((x.max()-x.min())/256,1)
        mask=np.array([[1,1,0,1,0,1,0],[1,0,1,1,1,0,1]],dtype=float)
        for causal in (False,True):
            with self.subTest(causal=causal):
                expected=run_tensor(parse(fixture(0,causal=causal)),{},dict(x=x,m=mask),act_mode='exact')
                text=fixture(8,causal=causal)
                tensor=run_tensor(parse(text),{},dict(x=x,m=mask),act_mode='exact')
                actual,_,_=mock(text,{},[x,mask],exact=True)
                np.testing.assert_allclose(tensor,expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)


BERT=RESULTS_DIR/'u4-bert-prepare-20261026T115827230385/model'
PROGRAM=RESULTS_DIR/'u4-points-squared-20261026T132023740575/program.ptdsl'


@require(BERT/'model_info.json',BERT/'calibration.json',PROGRAM)
class BertRuleTextTests(unittest.TestCase):
    def test_absolute_only_bert_rule_text_unchanged(self):
        info=ModelInfo.load(BERT)
        calibration=json.loads((BERT/'calibration.json').read_text())
        original=rule.select_approximation
        def absolute_only(*args,**kwargs):
            report=dict(original(*args,**kwargs))
            candidates=[c for c in report['candidates'] if c['fit']=='absolute' and c['feasible']]
            chosen=min(candidates,key=lambda c:(c['depth'],c['relative_error'],c['method'],
                        c.get('deg',c.get('deg0',0)),c.get('iters',0)))
            report['selected']={k:chosen[k] for k in ('method','deg','deg0','iters','fit') if k in chosen}
            return report
        with patch.object(rule,'select_approximation',side_effect=absolute_only):
            self.assertEqual(rule.plan(info.graph,calibration),PROGRAM.read_text())


if __name__=='__main__':unittest.main()
