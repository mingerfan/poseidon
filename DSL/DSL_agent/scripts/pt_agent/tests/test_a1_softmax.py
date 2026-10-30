"""A1 normalization budgets, independent identity and portable coefficients."""

from pathlib import Path
import importlib
import tempfile
from types import FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.export import export_package
from pt_agent.kernels import transformer as tr
from pt_agent.lower import lower
from pt_agent.planners import transformer_rule as rule
from pt_agent.tests.test_p2_export import package_modules,emit_pair
from pt_agent.tests.test_u4_kernels import mock
from pt_agent.tests.test_u4_squared_softmax import fixture
from pt_agent.tests._artifacts import RESULTS_DIR,require
from pt_agent.transformer_tensor import run
from pt_agent.transformer_validation import points


def text(steps=2,*,causal=False,final=True):
    source=fixture(steps,causal=causal)
    if final:
        source=source.replace('keep=', 'final_inv_method="remez",final_inv_domain=(0.02,2.4),'
                              'final_inv_deg=31,final_inv_fit="relative",inv_budget=0.25,keep=')
    return source


class SoftmaxBudgetTests(unittest.TestCase):
    def test_row_scalar_cancels_mask_causal_k1_to_8(self):
        rng=np.random.default_rng(20261027)
        for causal in (False,True):
            x=rng.normal(size=(2,3,7))*2
            mask=np.array([[1,0,1,1,0,1,0],[1,1,0,1,1,0,1]],float)
            arrays=dict(x=x,m=mask)
            expected=run(parse(fixture(0,causal=causal)),{},arrays,act_mode='exact')
            for rho in rule.INV_BUDGETS:
                for k in range(1,9):
                    seen=[]
                    def evaluate(op,role,value,attrs,mask=None):
                        if role=='exp':return np.exp(value)
                        seen.append(role)
                        inverse=1/value
                        if role!='final_inv':inverse=inverse*rng.uniform(1-rho,1+rho,size=value.shape)
                        return inverse
                    source=text(k,causal=causal).replace('inv_budget=0.25','inv_budget='+str(rho))
                    actual=run(parse(source),{},arrays,point_evaluator=evaluate)
                    with self.subTest(causal=causal,rho=rho,k=k):
                        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                        self.assertEqual(seen,['inv']+[f'step_inv[{i}]' for i in range(k-1)]+['final_inv'])

    def test_parser_optional_final_and_budget_contract(self):
        p=parse(text())
        self.assertEqual(print_program(p),print_program(parse(print_program(p))))
        self.assertEqual([point.role for point in points(p)],['exp','inv','step_inv[0]','final_inv'])
        self.assertEqual(points(parse(text(1)))[-1].role,'final_inv')
        for source in (text().replace('steps=2','steps=0'),
                       text().replace('inv_budget=0.25','inv_budget=0.2'),
                       text().replace('final_inv_method="remez",',''),
                       text().replace('final_inv_domain=(0.02,2.4),',''),
                       text(final=False).replace('keep=','inv_budget=0.25,keep=')):
            with self.subTest(source=source),self.assertRaises(Exception):parse(source)
        self.assertNotIn('final_inv',print_program(parse(text(final=False))))

    def test_analytical_domains_cover_calibration_with_row_errors(self):
        scores=np.sin(np.arange(42)).reshape(2,3,7)*3
        calib=dict(min=float(scores.min()),max=float(scores.max()))
        for rho in rule.INV_BUDGETS:
            for k in range(9):
                cfg=rule.budget_softmax_adjustments(calib,7,k,rho)
                lo,hi=cfg['step_inv_domain']
                self.assertEqual((lo,hi),((1-rho)**2/(1.5*7),1.5*(1+rho)**2))
                x=(scores-cfg['shift'])/(2**k)
                self.assertGreaterEqual(x.min(),cfg['exp_domain'][0]);self.assertLessEqual(x.max(),cfg['exp_domain'][1])
                e=np.exp(x);s=e.sum(-1,keepdims=True)/cfg['inv_scale']
                self.assertGreaterEqual(s.min(),cfg['inv_domain'][0]);self.assertLessEqual(s.max(),cfg['inv_domain'][1])
                for keep in (np.ones_like(e),np.broadcast_to([1,0,1,1,0,1,0],e.shape)):
                    p=e*keep;p=p/p.sum(-1,keepdims=True)
                    for index in range(k):
                        c=np.array([1-rho,1+rho])[:,None,None]
                        p=p*c;q=p*p;total=q.sum(-1,keepdims=True)
                        self.assertGreaterEqual(total.min(),lo);self.assertLessEqual(total.max(),hi)
                        p=q/total

    def test_k_selection_uses_normalized_latency_not_depth_or_k0(self):
        def candidate(calib,n,k,rho):
            return dict(steps=k,rho=rho,attrs=dict(steps=k),reports={},feasible=k!=2)
        with patch.object(rule,'softmax_budget_candidate',side_effect=candidate), \
             patch.object(rule,'_softmax_latency',side_effect=lambda a,*args,**kwargs:(a['steps']-1)**2+100) as cost:
            selection=rule.select_softmax_budget(dict(min=-100,max=100),(1,2,3,7),rho=.25)
        self.assertEqual(selection['selected']['steps'],1)
        self.assertEqual([c['steps'] for c in selection['candidates']],list(range(9)))
        self.assertEqual(cost.call_count,8)

    def test_targets_first_intermediate_final_and_zero_k(self):
        calls=[]
        def select(fn,domain,**kwargs):
            calls.append((fn,domain,kwargs))
            return dict(selected=dict(method='remez',deg=7),candidates=[])
        with patch.object(rule,'select_approximation',side_effect=select):
            self.assertTrue(rule.softmax_budget_candidate(dict(min=-2,max=2),7,3,.25)['feasible'])
        self.assertEqual([row[2]['target'] for row in calls],[1e-3/8,.25,.25,1e-3])
        calls.clear()
        with patch.object(rule,'select_approximation',side_effect=select):
            candidate=rule.softmax_budget_candidate(dict(min=-2,max=2),7,0,.25)
        self.assertEqual([row[2]['target'] for row in calls],[1e-3,1e-3])
        self.assertNotIn('inv_budget',candidate['attrs'])

    def test_mock_tensor_p2_coefficients_and_final_diagnostic(self):
        source=text().replace('inv_domain=(0.01,100),inv_deg0=7','inv_domain=(0.1,10),inv_deg0=15')
        arrays=[np.sin(np.arange(42)).reshape(2,3,7)*.2,np.array([[1,1,0,1,1,0,1],[1,1,1,1,0,1,1]],float)]
        observed=[];program=parse(source)
        actual=run(program,{},dict(zip(('x','m'),arrays)),observer=lambda name,*args:observed.append(name))
        expected,_,_=mock(source,{},arrays)
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
        self.assertEqual(observed[-1],program.outputs[0]+'::final_inv')
        with tempfile.TemporaryDirectory() as tmp:
            package=export_package(source,program,{},Path(tmp))
            left,right=emit_pair(source,{},package)
            self.assertEqual(left[0],right[0]);self.assertEqual(left[1].keys(),right[1].keys())
            for key in left[1]:np.testing.assert_array_equal(left[1][key],right[1][key])
            with package_modules(package) as runtime,patch('numpy.linalg.solve',side_effect=AssertionError('refit')):
                portable=runtime.run_numpy({},arrays)[0]
            np.testing.assert_array_equal(portable,expected)


SNAPSHOT=RESULTS_DIR/'u4a-remaining-20261027T031028306606/baseline-tree'
OLD_PACKAGE=RESULTS_DIR/'u4-qwen-relative-managed-20261027T101309756974/managed/Qwen2_PTAgent_DSL'


@require(OLD_PACKAGE/'qwen2/core/transformer.py',SNAPSHOT/'DSL/DSL_agent/scripts/pt_agent/capabilities.py')
class ExistingEmitTests(unittest.TestCase):
    def test_absent_final_keeps_existing_emit(self):
        for k in (0,2):
            source=text(k,final=False)
            current=EmitBackend(1024);lower(parse(source),{},current)
            with tempfile.TemporaryDirectory() as tmp:
                package=export_package(source,parse(source),{},Path(tmp))
                with package_modules(package) as runtime,package_modules(OLD_PACKAGE) as retained:
                    configuration=runtime.read_config()
                    old_ops=importlib.import_module(retained.__package__+'.ops')
                    forward=FunctionType(runtime.model.forward.__code__,
                                         dict(runtime.model.forward.__globals__,ops=old_ops))
                    with patch.object(retained,'read_config',return_value=configuration), \
                         patch.object(retained,'model',SimpleNamespace(forward=forward)):
                        before=EmitBackend(1024);retained.invoke(before,{})
            left,right=before.render(),current.render()
            self.assertEqual(left[0],right[0]);self.assertEqual(left[1].keys(),right[1].keys())
            for key in left[1]:np.testing.assert_array_equal(left[1][key],right[1][key])

    def test_v4_to_v6_prompt_texts_equal_historical_snapshot(self):
        from pt_agent.experiments.u2_multi_layout import baseline_package
        from pt_agent.capabilities import render_capabilities
        from pt_agent.planners import prompts
        from pt_agent.planners.agent_tools import build_context
        from pt_agent.tests.test_a1_agent import fixture
        old=baseline_package(SNAPSHOT)
        info,calibration,samples=fixture();ctx=build_context(info,calibration,samples=samples)
        for version in ('r8-v4','r8-v5','r8-v6'):
            ctx.prompt_version=version
            self.assertEqual(render_capabilities(version),old('capabilities').render_capabilities(version))
            self.assertEqual(prompts.build_system_prompt(ctx),old('planners.prompts').build_system_prompt(ctx))
            self.assertEqual(prompts.initial_message(ctx),old('planners.prompts').initial_message(ctx))


if __name__=='__main__':unittest.main()
