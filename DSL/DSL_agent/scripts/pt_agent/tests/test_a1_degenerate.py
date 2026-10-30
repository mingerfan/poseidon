"""A1 §1.7: bounded rejection, full context and implementation-error propagation."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.kernels import approx,poly
from pt_agent.kernels.layout import KernelError
from pt_agent.planners import transformer_rule as rule
from pt_agent.planners import approximation_failures as failures
from pt_agent.planners.transformer_session import ModelBudgetClient,run_session
from pt_agent.tests._artifacts import COMPILER,require
from pt_agent.tests.test_a1_agent import SOURCE,RecordedClient,fixture


class CandidateFailureTests(unittest.TestCase):
    def tearDown(self):rule._select_approximation.cache_clear()

    def test_01_only_declared_exceptions_are_rejected(self):
        for error in (KernelError('constant'),FloatingPointError('overflow'),
                      approx.ApproximationError('exchange',iterations=50,max_error=2.,
                                                error_amplitude=1.,history=[])):
            rule._select_approximation.cache_clear()
            with patch.object(rule,'_relative_profile',side_effect=error) as fit:
                report=rule.select_approximation('exp',(-100,100),exp_ladder=True,
                                                 context=dict(role='exp',k=0,rho=.25))
            self.assertEqual(fit.call_count,8)
            self.assertEqual(report['status'],'infeasible')
            for candidate in report['candidates']:
                self.assertEqual(candidate['exception_type'],type(error).__name__)
                self.assertIn(candidate['reason'],('degenerate','nonfinite'))
                self.assertTrue({'role','fn','fit','method','degree','domain','target','k','rho','message'}<=candidate.keys())
        for error in (TypeError('implementation'),KeyError('implementation'),IndexError('implementation')):
            rule._select_approximation.cache_clear()
            with patch.object(rule,'_relative_profile',side_effect=error),self.assertRaises(type(error)):
                rule.select_approximation('exp',(-1,1),context=dict(role='exp',k=1,rho=.25))

    def test_02_nonfinite_coefficients_and_errors_are_not_used(self):
        with patch.object(approx,'remez',return_value=np.array([1.,np.nan])), \
             patch.object(poly,'cheb_eval',side_effect=AssertionError('nonfinite coefficients used')):
            report=rule.select_approximation('exp',(-1,1),exp_ladder=True,
                                            context=dict(role='exp',k=0,rho=.25))
        self.assertEqual(report['status'],'infeasible')
        self.assertTrue(all(c['reason']=='nonfinite' for c in report['candidates']))
        rule._select_approximation.cache_clear()
        with patch.object(approx,'remez',return_value=np.array([1.,1.])), \
             patch.object(poly,'cheb_eval',return_value=np.array([np.inf])):
            report=rule.select_approximation('exp',(-1,1),exp_ladder=True,
                                            context=dict(role='exp',k=0,rho=.25))
        self.assertTrue(all(c['reason']=='nonfinite' for c in report['candidates']))

    def test_03_search_visits_all_k_after_degenerate_fits(self):
        def profile(fn,degree,domain,fit):
            if (fn=='exp' and domain[1]-domain[0]>3) or (fn=='inv' and domain[1]/domain[0]>100):
                raise KernelError('Chebyshev coefficients reduce to a constant')
            return dict(depth=2,relative_error=0.,degree=degree,fit=fit),np.array([1.])
        visited=[]
        def latency(attrs,*args,**kwargs):
            visited.append(attrs['steps']);return 100+attrs['steps']
        with patch.object(rule,'_relative_profile',side_effect=profile), \
             patch.object(rule,'_softmax_latency',side_effect=latency):
            report=rule.select_softmax_budget(dict(min=-2,max=2),(1,3,3),rho=.25)
        self.assertEqual([c['steps'] for c in report['candidates']],list(range(9)))
        self.assertFalse(report['candidates'][0]['feasible'])
        self.assertEqual(visited,list(range(1,9)))
        self.assertEqual(report['selected']['steps'],1)
        errors=[c for point in report['candidates'][0]['reports'].values() for c in point['candidates']]
        for candidate in errors:
            self.assertEqual(candidate['reason'],'degenerate');self.assertEqual(candidate['k'],0)
            self.assertEqual(candidate['rho'],.25);self.assertIn('role',candidate)

    def test_04_program_feedback_has_role_domain_and_exception(self):
        text=SOURCE.replace('method="cheb"','method="remez"')
        with patch.object(approx,'fitted_coefficients',return_value=np.array([1.,np.nan])):
            errors=failures.program_failures(parse(text))
        self.assertEqual(len(errors),1)
        self.assertEqual(errors[0]['role'],'act');self.assertEqual(errors[0]['domain'],(-1.,1.))
        self.assertEqual(errors[0]['reason'],'nonfinite')
        with patch.object(approx,'fitted_coefficients',side_effect=TypeError('implementation')):
            with self.assertRaises(TypeError):failures.program_failures(parse(text))

    def test_99_actual_extreme_exp_and_inverse_are_infeasible(self):
        # The calibration values from the retained stop, without loading model
        # data or running a numerical network, define both extreme domains.
        lo,hi=-232.51723682006593,171.9213304311596
        cfg=rule.budget_softmax_adjustments(dict(min=lo,max=hi),32,0,.25)
        for role,fn in (('exp','exp'),('inv','inv')):
            report=rule.select_approximation(fn,cfg[role+'_domain'],exp_ladder=fn=='exp',
                                             context=dict(role=role,k=0,rho=.25))
            self.assertEqual(report['status'],'infeasible')
            self.assertTrue(all(not c['feasible'] for c in report['candidates']))
            numeric=[c for c in report['candidates'] if c.get('reason') in ('degenerate','nonfinite')]
            self.assertTrue(numeric)
            for candidate in numeric:
                self.assertTrue({'role','fn','fit','method','domain','target','k','rho','exception_type','message'}<=candidate.keys())


@require(COMPILER)
class AgentFailureTests(unittest.TestCase):
    def test_rejected_program_keeps_call_and_continues_session(self):
        info,calibration,samples=fixture();original=poly.cheb_fit
        bad=SOURCE.replace('domain=(-1.,1.)','domain=(-1000.,1000.)')
        def fit(fn,degree,lo,hi):
            if lo==-1000.:raise KernelError('Chebyshev coefficients reduce to a constant')
            return original(fn,degree,lo,hi)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=RecordedClient([json.dumps(dict(program=bad)),json.dumps(dict(program=SOURCE))])
            client=ModelBudgetClient(raw,root/'calls')
            with patch.object(poly,'cheb_fit',side_effect=fit):
                result=run_session(info,calibration,samples,client,root/'session',max_iters=2,tuner=False)
            self.assertEqual(result['status'],'feasible');self.assertEqual(client.calls,2)
            first=json.loads((root/'session/agent_trace.jsonl').read_text().splitlines()[0])['fb']
            self.assertEqual(first['stage'],'approximation');self.assertFalse(first['feasible'])
            self.assertEqual(first['errors'][0]['reason'],'degenerate')
            self.assertEqual(first['errors'][0]['role'],'act');self.assertIn('line',first['errors'][0])
            self.assertNotIn('report',samples.reads)


if __name__=='__main__':unittest.main()
