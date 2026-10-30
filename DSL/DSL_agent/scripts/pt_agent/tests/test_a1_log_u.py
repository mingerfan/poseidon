"""A1 §6.18: logarithmic coverage is confined to positive calibrated inverses."""

import json
import math
from types import SimpleNamespace
import unittest

import numpy as np

from pt_agent import domain_criteria as criteria
from pt_agent.dsl.parser import parse
from pt_agent.planners import agent_tools, prompts
from pt_agent.tests.test_a1_agent import fixture, SOURCE
from pt_agent.transformer_validation import validate


class LogCoverageTests(unittest.TestCase):
    def u(self, values, domain=(1., 100.), **kwargs):
        return criteria.maximum_u(values, domain, version='r8-v10', fn='rsqrt', **kwargs)

    def test_endpoints_and_geometric_midpoint(self):
        self.assertEqual(self.u([1.]), 1.)
        self.assertAlmostEqual(self.u([10.]), 0., places=15)
        self.assertEqual(self.u([100.]), 1.)
        self.assertGreater(self.u([.5]), 1.)

    def test_nonpositive_and_nonfinite(self):
        for value in (0., -1., np.inf, -np.inf, np.nan):
            with self.subTest(value=value):
                self.assertEqual(self.u([10., value]), float('inf'))

    def test_formula_and_extrema(self):
        values = np.geomspace(2., 80., 100)
        expected = float(np.max(np.abs(2*(np.log(values)-math.log(1.))/math.log(100.)-1)))
        self.assertEqual(self.u(values), expected)
        self.assertEqual(self.u(values), self.u([values.min(), values.max()]))
        self.assertEqual(criteria.maximum_u(values, (1.,100.),fn='inv',version='r8-v10'), expected)

    def test_other_functions_and_analytic_points_keep_linear(self):
        data = np.array([2., 10., 80.])
        for fn in ('exp', 'tanh', 'relu', 'silu', None):
            for source in ('calibrated', 'analytic'):
                args = dict(fn=fn, domain_source=source)
                self.assertEqual(criteria.maximum_u(data,(1.,100.),version='r8-v10',**args),
                                 criteria.maximum_u(data,(1.,100.),version='r8-v9',**args))
        self.assertEqual(self.u(data,domain_source='analytic'),
                         criteria.maximum_u(data,(1.,100.),fn='rsqrt',version='r8-v9'))
        self.assertEqual(self.u(data,domain=(-1.,100.)),
                         criteria.maximum_u(data,(-1.,100.),fn='rsqrt',version='r8-v9'))

    def test_all_older_versions_use_original_arithmetic(self):
        data = np.array([2., 10., 80.])
        expected = float(np.max(np.abs(data*(2/99.)-101/99.)))
        for version in ('r8-v4','r8-v5','r8-v6','r8-v7','r8-v8','r8-v9'):
            self.assertEqual(criteria.maximum_u(data,(1.,100.),fn='rsqrt',version=version),expected)

    def test_linear_preserves_callers_numpy_exception_context(self):
        with np.errstate(over='raise',invalid='raise'):
            with self.assertRaises(FloatingPointError):
                criteria.maximum_u([1e308],(0.,1.),version='r8-v9')
            self.assertEqual(self.u([np.nan]),float('inf'))

    def test_v10_keeps_v617_and_older_defaults_are_unchanged(self):
        from pt_agent.planners.transformer_tuner import validate_tune_rule
        self.assertEqual(validate_tune_rule('r8-v10',None),'v8-6.17')
        self.assertEqual(validate_tune_rule('r8-v9',None),'v8-6.14')
        self.assertEqual(validate_tune_rule('r8-v8',None),'v8-6.14')
        self.assertEqual(validate_tune_rule('r8-v7',None),'v8-6.11')

    def test_violation_order_uses_same_coordinate(self):
        point = SimpleNamespace(op=SimpleNamespace(id=0),role='rsqrt',attrs=dict(fn='rsqrt'))
        row = dict(program_line=4,graph_tensor='a::rsqrt',domain=(1.,100.),domain_source='calibrated')
        tracker = criteria.ViolationOrder([point],[row],1,version='r8-v10')
        tracker.observe(0,'rsqrt',np.array([[0.,10.]]))
        first = tracker.report(engine='tensor',split='heldout')['samples'][0]['first_violation']
        self.assertEqual(first['max_u'],float('inf'))
        self.assertFalse(tracker.analytic_domain_violation)

    def test_prompt_and_graph_differ_only_in_u_explanation(self):
        info,calibration,samples = fixture()
        ctx = agent_tools.build_context(info,calibration,samples=samples)
        ctx.prompt_version = 'r8-v9'
        before,graph = prompts.build_system_prompt(ctx),prompts.initial_message(ctx)
        ctx.prompt_version = 'r8-v10'
        after = prompts.build_system_prompt(ctx)
        self.assertIn(prompts.LOG_U_EXPLANATION,after)
        self.assertEqual(after.replace(prompts.LOG_U_EXPLANATION,prompts.LINEAR_U_EXPLANATION),before)
        self.assertEqual(prompts.initial_message(ctx),graph)

    def test_non_inverse_validation_preserves_values_and_feasibility(self):
        info,calibration,samples = fixture()
        ctx = agent_tools.build_context(info,calibration,samples=samples)
        ctx.prompt_version = 'r8-v9'
        before = validate(parse(SOURCE),ctx)
        ctx.prompt_version = 'r8-v10'
        after = validate(parse(SOURCE),ctx)
        self.assertEqual(after,before)

    def test_saved_extrema_regrade_preserves_other_measurements(self):
        from pt_agent.experiments.a1_v10_replay import regrade_validation
        info,calibration,samples = fixture()
        ctx = agent_tools.build_context(info,calibration,samples=samples)
        ctx.prompt_version = 'r8-v9'
        saved = validate(parse(SOURCE),ctx)
        ctx.prompt_version = 'r8-v10'
        result = regrade_validation(saved,parse(SOURCE),ctx)
        self.assertEqual(result['max_u_calibrated'],saved['max_u_calibrated'])
        self.assertEqual(result['feasible'],saved['feasible'])
        self.assertEqual(result['agree'],saved['agree'])
        self.assertNotIn('u_scale',saved['rows'][0])
        saved['diverged'] = True
        self.assertFalse(regrade_validation(saved,parse(SOURCE),ctx)['feasible'])


if __name__ == '__main__':
    unittest.main()
