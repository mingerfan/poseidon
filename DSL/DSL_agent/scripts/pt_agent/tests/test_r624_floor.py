"""A1 §6.24: refit settings bound later descent (tune_rule v8-6.24)."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import approximation_failures, transformer_tuner as tuner


SOURCE = '''@fhe.program(slots=1024)
def token_rule(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 v=act(a,fn="tanh",method="cheb",deg=15,domain=(-1.,1.))
 z=act(v,fn="tanh",method="cheb",deg=15,domain=(-1.,1.))
 return z
'''


def measured(feasible):
    u = .5 if feasible else .95
    return dict(feasible=feasible, diverged=False, agree=6, total=6, agreement_pct=100.,
                max_u=u, max_u_calibrated=u, max_u_analytic=0., analytic_domain_violation=False,
                worst_acts=[], logit_rel_err=1e-3)


def run(rule, *, aligned=True):
    program = parse(SOURCE)
    ctx = SimpleNamespace(prompt_version='r8-v8', weights={}, profile='unused',
                          model_info=SimpleNamespace(task='causal_lm',
                                                     threshold=dict(agreement_pct=99., logit_rel_err=1e-2)),
                          calibration={key: dict(min=-.9, max=.9) for key in ('first', 'second')})
    costs, validations = [], []

    def cost(candidate, *args, **kwargs):
        costs.append(print_program(candidate))
        return dict(latency_pred_us=200-10*len(costs), boots_pred=1)

    def validate(candidate, *args, **kwargs):
        validations.append(kwargs.get('engine'))
        if len(validations) == 2:
            return measured(aligned)
        return measured(True)

    with ExitStack() as stack:
        stack.enter_context(tuner.tune_rule_scope(rule))
        stack.enter_context(patch('pt_agent.dsl.interface.require_interface'))
        stack.enter_context(patch.object(tuner, 'point_names', return_value=['first', 'second']))
        stack.enter_context(patch.object(tuner, 'cached_estimate', side_effect=cost))
        stack.enter_context(patch.object(tuner, 'validate', side_effect=validate))
        stack.enter_context(patch.object(tuner, 'exact_check', return_value=dict(ok=True)))
        stack.enter_context(patch.object(tuner.time, 'monotonic', return_value=0.))
        stack.enter_context(patch.object(approximation_failures, 'program_failures', return_value=[]))
        stack.enter_context(patch.object(tuner, 'softmax_moves', side_effect=AssertionError('Unexpected softmax scan')))
        result = tuner.tune(program, ctx, budget_s=1, max_sweeps=8, engine='tensor', engine_profile='tensor')
    return result, costs


class FloorTests(unittest.TestCase):
    def test_below_floor_by_component(self):
        self.assertTrue(tuner.below_floor(dict(deg=15), dict(deg=31)))
        self.assertFalse(tuner.below_floor(dict(deg=31), dict(deg=31)))
        self.assertFalse(tuner.below_floor(dict(deg=63), dict(deg=31)))
        self.assertTrue(tuner.below_floor(dict(iters=1), dict(deg0=63, iters=2)))
        self.assertTrue(tuner.below_floor(dict(deg0=31), dict(deg0=63, iters=1)))
        self.assertFalse(tuner.below_floor(dict(iters=1), dict(deg0=63, iters=1)))
        self.assertFalse(tuner.below_floor(dict(deg=7), None))

    def test_floor_from_refit_status(self):
        self.assertEqual(tuner.refit_floor(dict(status='within', setting=dict(deg=31))), dict(deg=31))
        self.assertEqual(tuner.refit_floor(dict(status='closest', setting=dict(deg0=63, iters=2))),
                         dict(deg0=63, iters=2))
        self.assertIsNone(tuner.refit_floor(dict(status='kept', setting=None)))

    def test_rule(self):
        self.assertIn('v8-6.24', tuner.REFIT_RULES)
        self.assertIn('v8-6.24', tuner.FLOOR_RULES)
        self.assertNotIn('v8-6.23', tuner.FLOOR_RULES)
        self.assertEqual(tuner.MARGIN_DOMAIN_FACTORS['v8-6.24'], 2.0)
        self.assertEqual(tuner.validate_tune_rule('r8-v10', 'v8-6.24'), 'v8-6.24')
        self.assertTrue(tuner.margin_rule('v8-6.24') and tuner.accuracy_rule('v8-6.24')
                        and tuner.convergent_rule('v8-6.24'))

    def test_descent_below_refit_is_skipped(self):
        result, costs = run('v8-6.24')
        domain = result.history[0]
        self.assertEqual(domain['phase'], 'domain')
        self.assertTrue(domain['accepted'])
        refits = [r['setting'] for r in domain['refits']]
        self.assertEqual([f['setting'] for f in domain['floors']], refits)
        self.assertTrue(all(s['deg'] > 15 for s in refits))
        skipped = [r for r in result.history if r.get('rejected') == 'refit_floor']
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(r['attrs']['deg'] < 31 for r in skipped))
        self.assertEqual(result.changes, [])
        self.assertEqual(len(costs), 2)
        self.assertEqual([p.attrs['deg'] for p in tuner.points(result.program)], [s['deg'] for s in refits])

    def test_same_case_descends_under_v8_623(self):
        result, costs = run('v8-6.23')
        self.assertNotIn('floors', result.history[0])
        self.assertFalse(any(r.get('rejected') == 'refit_floor' for r in result.history))
        self.assertTrue(result.changes)
        self.assertGreater(len(costs), 2)

    def test_rejected_alignment_has_no_floor(self):
        result, costs = run('v8-6.24', aligned=False)
        self.assertFalse(result.history[0]['accepted'])
        self.assertNotIn('floors', result.history[0])
        self.assertFalse(any(r.get('rejected') == 'refit_floor' for r in result.history))
        self.assertTrue(any(r.get('phase') == 'coarse' and r['attrs'] == dict(deg=7) for r in result.history))


if __name__ == '__main__':
    unittest.main()
