"""E8 §2: v8-6.25 alignment records, staged refit and per-point fallback."""

from contextlib import ExitStack
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import approximation_failures, margin_refit as refit, transformer_tuner as tuner


SOURCE = '''@fhe.program(slots=1024)
def token_rule(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 v=act(a,fn="tanh",method="cheb",deg=15,domain=(-1.,1.))
 z=act(v,fn="tanh",method="cheb",deg=15,domain=(-1.,1.))
 return z
'''


def measured(feasible=True, *, u=None, logit=1e-3):
    u = (.5 if feasible else .95) if u is None else u
    return dict(feasible=feasible, diverged=False, agree=6, total=6, agreement_pct=100.,
                max_u=u, max_u_calibrated=u, max_u_analytic=0., analytic_domain_violation=False,
                worst_acts=[], logit_rel_err=logit)


def run(rule, judge, *, budget_s=1):
    """`judge(domains, call)` returns the validation result for a candidate with these point domains."""
    program = parse(SOURCE)
    ctx = SimpleNamespace(prompt_version='r8-v8', weights={}, profile='unused',
                          model_info=SimpleNamespace(task='causal_lm',
                                                     threshold=dict(agreement_pct=99., logit_rel_err=1e-2)),
                          calibration={key: dict(min=-.9, max=.9) for key in ('first', 'second')})
    costs, calls = [], []

    def cost(candidate, *args, **kwargs):
        costs.append(print_program(candidate))
        return dict(latency_pred_us=200-10*len(costs), boots_pred=1)

    def validate(candidate, *args, **kwargs):
        calls.append(kwargs.get('engine'))
        return judge([tuple(p.attrs['domain']) for p in tuner.points(candidate)], len(calls))

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
        result = tuner.tune(program, ctx, budget_s=budget_s, max_sweeps=8, engine='tensor', engine_profile='tensor')
    return result, costs


WIDE = (-1.8, 1.8)


def second_point_fails(domains, call):
    """Widening the second point fails the domain criterion; everything else passes."""
    return measured(not (domains[1] == WIDE))


class RuleTests(unittest.TestCase):
    def test_registration(self):
        rule = 'v8-6.25'
        self.assertEqual(tuner.ALIGN_FALLBACK_TUNE_RULE, rule)
        for group in (tuner.REFIT_RULES, tuner.FLOOR_RULES, tuner.EXTENDED_REFIT_RULES, tuner.FALLBACK_RULES):
            self.assertIn(rule, group)
        self.assertNotIn('v8-6.24', tuner.EXTENDED_REFIT_RULES)
        self.assertNotIn('v8-6.24', tuner.FALLBACK_RULES)
        self.assertEqual(tuner.MARGIN_DOMAIN_FACTORS[rule], 2.0)
        self.assertEqual(tuner.MARGIN_RULE_FACTORS[rule], tuner.MARGIN_ACCURACY_FACTOR)
        self.assertEqual(tuner.validate_tune_rule('r8-v11', rule), rule)
        self.assertTrue(tuner.margin_rule(rule) and tuner.accuracy_rule(rule) and tuner.convergent_rule(rule))
        source = (Path(tuner.__file__).parents[1]/'experiments/a1_transformer_agent.py').read_text()
        self.assertIn("'v8-6.24','v8-6.25'))", source)

    def test_floor_drops_method_and_none(self):
        floor = tuner.refit_floor(dict(status='within', setting=dict(method='newton', deg=None, deg0=63, iters=4,
                                                                     fit='relative')))
        self.assertEqual(floor, dict(deg0=63, iters=4))
        self.assertEqual(tuner.refit_floor(dict(status='within', setting=dict(deg=31))), dict(deg=31))
        self.assertIsNone(tuner.refit_floor(dict(status='kept', setting=None)))


class RecordTests(unittest.TestCase):
    def test_accepted_alignment_has_empty_reasons(self):
        for rule in ('v8-6.24', 'v8-6.25'):
            result, _ = run(rule, lambda domains, call: measured(True))
            row = result.history[0]
            self.assertEqual(row['phase'], 'domain')
            self.assertTrue(row['accepted'])
            self.assertEqual(row['reasons'], [])
            self.assertEqual(row['metrics']['move_logit_limit'], 5e-3)
            self.assertNotIn('fallback', row)

    def test_each_reason(self):
        cases = {('infeasible', 'domain'): measured(False),
                 ('accuracy',): measured(True, logit=2e-2),
                 ('domain',): dict(measured(True), max_u_calibrated=.95),
                 ('diverged',): dict(measured(True), diverged=True)}
        for expected, aligned in cases.items():
            result, _ = run('v8-6.24', lambda domains, call, aligned=aligned: aligned if call == 2 else measured(True))
            row = result.history[0]
            self.assertFalse(row['accepted'])
            self.assertEqual(tuple(row['reasons']), expected)
            self.assertEqual(row['metrics']['max_u_calibrated'], aligned['max_u_calibrated'])

    def test_numerical_failure_reason(self):
        program = parse(SOURCE)
        calls = []

        def failures(candidate):
            calls.append(1)
            return [dict(code='E_NUM')] if len(calls) == 1 else []
        with patch.object(approximation_failures, 'program_failures', side_effect=failures):
            ctx = SimpleNamespace(prompt_version='r8-v8', weights={}, profile='unused',
                                  model_info=SimpleNamespace(task='causal_lm',
                                                             threshold=dict(agreement_pct=99., logit_rel_err=1e-2)),
                                  calibration={key: dict(min=-.9, max=.9) for key in ('first', 'second')})
            with ExitStack() as stack:
                stack.enter_context(tuner.tune_rule_scope('v8-6.24'))
                stack.enter_context(patch('pt_agent.dsl.interface.require_interface'))
                stack.enter_context(patch.object(tuner, 'point_names', return_value=['first', 'second']))
                stack.enter_context(patch.object(tuner, 'cached_estimate',
                                                 return_value=dict(latency_pred_us=100, boots_pred=1)))
                stack.enter_context(patch.object(tuner, 'validate', return_value=measured(True)))
                stack.enter_context(patch.object(tuner, 'exact_check', return_value=dict(ok=True)))
                stack.enter_context(patch.object(tuner.time, 'monotonic', return_value=0.))
                result = tuner.tune(program, ctx, budget_s=0, max_sweeps=8, engine='tensor', engine_profile='tensor')
        row = result.history[0]
        self.assertEqual(row['reasons'], ['numerical_failure'])
        self.assertNotIn('metrics', row)


class FallbackTests(unittest.TestCase):
    def test_v824_keeps_rejection(self):
        result, _ = run('v8-6.24', second_point_fails)
        row = result.history[0]
        self.assertFalse(row['accepted'])
        self.assertNotIn('fallback', row)
        self.assertEqual([tuple(p.attrs['domain']) for p in tuner.points(result.program)], [(-1., 1.), (-1., 1.)])

    def test_responsible_point_stays(self):
        result, _ = run('v8-6.25', second_point_fails)
        row = result.history[0]
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reasons'], ['infeasible', 'domain'])
        self.assertEqual([e['line'] for e in row['fallback']], [4, 5])
        self.assertEqual([e['accepted'] for e in row['fallback']], [True, False])
        self.assertEqual(row['fallback'][1]['reasons'], ['infeasible', 'domain'])
        self.assertEqual(row['accepted_lines'], [4])
        self.assertTrue(row['fallback_accepted'])
        self.assertEqual([f['line'] for f in row['fallback_floors']], [4])
        domains = [tuple(p.attrs['domain']) for p in tuner.points(result.program)]
        self.assertEqual(domains, [WIDE, (-1., 1.)])
        # Only the accepted point is floored: descent below its refit is skipped, the other point may descend.
        floored = [r for r in result.history if r.get('rejected') == 'refit_floor']
        self.assertTrue(floored and all(r['line'] == 4 for r in floored))

    def test_fallback_counts_measurements(self):
        base, _ = run('v8-6.24', second_point_fails, budget_s=0)
        result, _ = run('v8-6.25', second_point_fails, budget_s=0)
        self.assertEqual(result.full_evals, base.full_evals+2)

    def test_infeasible_initial_has_no_fallback(self):
        result, _ = run('v8-6.25', lambda domains, call: measured(False), budget_s=0)
        self.assertFalse(result.history[0]['accepted'])
        self.assertNotIn('fallback', result.history[0])

    def test_align_points_restricts_alignment(self):
        program = parse(SOURCE)
        ctx = SimpleNamespace(prompt_version='r8-v8', weights={}, profile='unused',
                              model_info=SimpleNamespace(task='causal_lm',
                                                         threshold=dict(agreement_pct=99., logit_rel_err=1e-2)),
                              calibration={key: dict(min=-.9, max=.9) for key in ('first', 'second')})
        with ExitStack() as stack:
            stack.enter_context(tuner.tune_rule_scope('v8-6.25'))
            stack.enter_context(patch('pt_agent.dsl.interface.require_interface'))
            stack.enter_context(patch.object(tuner, 'point_names', return_value=['first', 'second']))
            stack.enter_context(patch.object(tuner, 'cached_estimate',
                                             return_value=dict(latency_pred_us=100, boots_pred=1)))
            stack.enter_context(patch.object(tuner, 'validate', side_effect=lambda c, *a, **k: measured(
                [tuple(p.attrs['domain']) for p in tuner.points(c)][1] != WIDE)))
            stack.enter_context(patch.object(tuner, 'exact_check', return_value=dict(ok=True)))
            stack.enter_context(patch.object(tuner.time, 'monotonic', return_value=0.))
            stack.enter_context(patch.object(approximation_failures, 'program_failures', return_value=[]))
            result = tuner.tune(program, ctx, budget_s=0, max_sweeps=8, engine='tensor', engine_profile='tensor',
                                align_points={1})
        row = result.history[0]
        self.assertEqual([r['line'] for r in row.get('refits', [])], [5])
        self.assertFalse(row['accepted'])
        self.assertNotIn('fallback', row)


class StagedRefitTests(unittest.TestCase):
    def test_stage0_within_matches_v824(self):
        attrs = dict(fn='rsqrt', method='remez', deg=15, domain=(1.0, 10.0), fit='relative')
        old_setting, old_record = refit.refit(attrs, (0.5, 20.0))
        new_setting, new_record = refit.refit(attrs, (0.5, 20.0), extended=True)
        self.assertEqual(old_setting, new_setting)
        self.assertEqual(new_record['stage'], 0)
        self.assertEqual(new_record['status'], 'within')
        self.assertEqual(new_record['method'], 'remez')
        self.assertEqual({k: v for k, v in new_record.items() if k not in ('stage', 'method', 'stage_counts', 'seconds')},
                         old_record)
        self.assertNotIn('stage', old_record)

    def test_line32_rsqrt_switches_to_newton(self):
        # E7b q1 line-32 rsqrt: remez 63 misses the original error after widening, the remez fitter rejects
        # degrees above 63 (stage 1 unusable), and stage 2 reaches the target with newton.
        attrs = dict(fn='rsqrt', method='remez', deg=63, domain=(444.171921261918, 271807.57509245997), fit='relative')
        domain = (444.171921261918, 362410.10012327996)
        _, old_record = refit.refit(attrs, domain)
        self.assertEqual(old_record['status'], 'closest')
        for degree in (127, 255):
            row = refit._evaluate(attrs, domain, dict(deg=degree))
            self.assertTrue(math.isinf(row['error']))
        setting, record = refit.refit(attrs, domain, extended=True)
        self.assertEqual((record['status'], record['stage'], record['method']), ('within', 2, 'newton'))
        self.assertEqual(setting['fit'], 'relative')
        self.assertLessEqual(refit.max_error(dict(attrs, domain=domain, **setting)), record['target']*(1+refit.SLACK))

    def test_stage1_newton_above_63_is_unusable(self):
        # The newton initial polynomial uses the same fitter, so deg0 above 63 is unusable as well.
        attrs = dict(fn='inv', method='newton', deg0=63, iters=0, domain=(0.1, 2.0))
        row = refit._evaluate(attrs, (0.05, 4.0), dict(deg0=127, iters=0))
        self.assertTrue(math.isinf(row['error']))

    def test_stage2_switches_method(self):
        attrs = dict(fn='rsqrt', method='remez', deg=63, domain=(1.0, 10.0), fit='relative')

        def error(candidate):
            if candidate['domain'] == (1.0, 10.0):
                return 1e-6
            return 1e-7 if candidate['method'] == 'newton' and candidate['deg0'] >= 31 else 1e-3
        with patch.object(refit, 'max_error', side_effect=error):
            setting, record = refit.refit(attrs, (0.5, 20.0), extended=True)
        self.assertEqual(record['stage'], 2)
        self.assertEqual(record['status'], 'within')
        self.assertEqual(setting['method'], 'newton')
        self.assertEqual(setting['deg0'], 31)
        self.assertIsNone(setting['deg'])
        self.assertEqual(setting['fit'], 'relative')
        self.assertEqual(record['stage_counts'], [4, 2, 36])
        self.assertEqual(tuner.refit_floor(record), dict(deg0=31, iters=setting['iters']))

    def test_closest_over_all_stages(self):
        attrs = dict(fn='silu', method='remez', deg=63, domain=(-8.0, 8.0))

        def error(candidate):
            if candidate['domain'] == (-8.0, 8.0):
                return 1e-9
            return 1e-3/(candidate['deg'] or 1)/(10 if candidate['method'] == 'cheb' else 1)
        with patch.object(refit, 'max_error', side_effect=error):
            setting, record = refit.refit(attrs, (-16.0, 16.0), extended=True)
        self.assertEqual(record['status'], 'closest')
        self.assertEqual((setting['method'], setting['deg']), ('cheb', 255))
        self.assertEqual(record['stage'], 2)

    def test_dsl_illegal_combination_is_unusable(self):
        row = refit._evaluate(dict(fn='silu', method='remez', deg=63, domain=(-8.0, 8.0), degrees=None),
                              (-16.0, 16.0), dict(method='newton', deg=None, deg0=31, iters=2, fit=None))
        self.assertTrue(math.isinf(row['error']))
        self.assertTrue(row['failure'].startswith('dsl: '))
        methods = {s['method'] for s in refit.alternative_settings(dict(fn='silu', method='remez', deg=63))}
        self.assertEqual(methods, {'cheb'})
        methods = {s['method'] for s in refit.alternative_settings(dict(fn='inv', method='newton', deg0=7, iters=2))}
        self.assertEqual(methods, {'remez'})

    def test_extended_ladders(self):
        self.assertEqual(refit.extended_settings(dict(method='remez', deg=63)), [dict(deg=127), dict(deg=255)])
        self.assertEqual(refit.extended_settings(dict(method='remez', deg=127)), [dict(deg=255)])
        self.assertEqual(refit.extended_settings(dict(method='cheb', deg=15)), [])
        self.assertEqual(len(refit.extended_settings(dict(method='newton', deg0=31, iters=2))), 12)

    def test_deterministic(self):
        attrs = dict(fn='rsqrt', method='remez', deg=63, domain=(444.171921261918, 271807.57509245997), fit='relative')
        domain = (444.171921261918, 362410.10012327996)
        first = refit.refit(attrs, domain, extended=True)
        second = refit.refit(attrs, domain, extended=True)
        self.assertEqual(first[0], second[0])
        self.assertEqual({k: v for k, v in first[1].items() if k != 'seconds'},
                         {k: v for k, v in second[1].items() if k != 'seconds'})


if __name__ == '__main__':
    unittest.main()
