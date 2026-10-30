"""A1 §6.11: one alignment attempt and a separately selectable old rule."""

from contextlib import ExitStack
from dataclasses import asdict
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


def measured(feasible, *, diverged=False):
    u = .5 if feasible else .95
    return dict(feasible=feasible, diverged=diverged, agree=6, total=6,
                agreement_pct=100., max_u=u, max_u_calibrated=u,
                max_u_analytic=0., analytic_domain_violation=False, worst_acts=[])


class TuneRuleTests(unittest.TestCase):
    def run_case(self, *, initial=False, aligned=False, rule=tuner.DEFAULT_TUNE_RULE,
                 alignment_cost=100, calibration=None, precheck_errors=None,
                 count_budget=None, version='r8-v8', diverged=False, exact=True,
                 candidate_cost=80):
        program = parse(SOURCE)
        ctx = SimpleNamespace(prompt_version=version, weights={}, profile='unused',
                              model_info=SimpleNamespace(threshold=dict(agreement_pct=99.)),
                              calibration={key: dict(min=-.4, max=.4)
                                           for key in ('first', 'second')})
        if calibration is not None:
            ctx.calibration['first'] = calibration
        costs, validations, exact_calls = [], [], []

        def cost(candidate, *args, **kwargs):
            costs.append(print_program(candidate))
            value = (100 if len(costs) == 1 else alignment_cost if len(costs) == 2
                     else candidate_cost-20*(len(costs)-3))
            return dict(latency_pred_us=value, boots_pred=1)

        def validate(candidate, *args, **kwargs):
            validations.append(dict(source=print_program(candidate), args=args,
                                    engine=kwargs.get('engine')))
            if kwargs.get('engine') == 'mock':
                # All accepted test candidates have the same feasibility.
                accepted_alignment = aligned and (not initial or alignment_cost <= 100)
                return measured(initial or accepted_alignment, diverged=diverged)
            return measured(initial if len(validations) == 1 else aligned,
                            diverged=diverged if len(validations) == 1 else False)

        def check_exact(*args, **kwargs):
            exact_calls.append(True)
            return dict(ok=exact)

        # The test budget counts calls, including any estimator cache hits.
        def clock():
            return float(count_budget is not None and len(costs) >= count_budget)

        with ExitStack() as stack:
            stack.enter_context(tuner.tune_rule_scope(rule))
            stack.enter_context(patch('pt_agent.dsl.interface.require_interface'))
            stack.enter_context(patch.object(tuner, 'point_names', return_value=['first', 'second']))
            stack.enter_context(patch.object(tuner, 'cached_estimate', side_effect=cost))
            stack.enter_context(patch.object(tuner, 'validate', side_effect=validate))
            stack.enter_context(patch.object(tuner, 'exact_check', side_effect=check_exact))
            stack.enter_context(patch.object(tuner.time, 'monotonic', side_effect=clock))
            precheck = stack.enter_context(patch.object(approximation_failures, 'program_failures',
                return_value=[] if precheck_errors is None else precheck_errors))
            moves = stack.enter_context(patch.object(tuner, 'softmax_moves',
                                                    side_effect=AssertionError('Unexpected softmax scan')))
            result = tuner.tune(program, ctx, budget_s=1, max_sweeps=1,
                                engine='tensor', engine_profile='tensor')
        return result, costs, validations, exact_calls, precheck.call_count, moves.call_count

    def test_infeasible_alignment_returns_original_after_final_mock(self):
        result, costs, validations, exact, prechecks, moves = self.run_case()
        self.assertEqual(len(costs), 2)  # Original plus exactly one alignment.
        self.assertEqual([row['engine'] for row in validations], ['tensor', 'tensor', 'mock'])
        self.assertEqual(len(exact), 1)
        self.assertEqual(prechecks, 1)
        self.assertEqual(moves, 0)
        self.assertFalse(result.feasible)
        self.assertEqual(result.source, print_program(parse(SOURCE)))
        self.assertEqual(result.changes, [])
        self.assertEqual((result.coarse_evals, result.full_evals), (0, 2))
        self.assertEqual([row['phase'] for row in result.history], ['domain', 'result'])
        self.assertFalse(result.history[0]['accepted'])

    def test_degenerate_alignment_skips_cost_validation_and_fit(self):
        result, costs, validations, exact, prechecks, _ = self.run_case(
            calibration=dict(min=0., max=0.))
        self.assertEqual(len(costs), 1)
        self.assertEqual([row['engine'] for row in validations], ['tensor', 'mock'])
        self.assertEqual(len(exact), 1)
        self.assertEqual(prechecks, 0)
        self.assertEqual(result.changes, [])
        self.assertEqual(result.source, print_program(parse(SOURCE)))
        self.assertEqual((result.coarse_evals, result.full_evals), (0, 1))
        self.assertEqual([row['phase'] for row in result.history], ['domain', 'result'])
        error = result.history[0]['numerical_errors'][0]
        self.assertEqual(error['reason'], 'degenerate')
        self.assertEqual(error['domain'], (0., 0.))
        self.assertEqual(error['role'], 'act')

    def test_precheck_failure_skips_alignment_cost_and_validation(self):
        errors = [dict(reason='degenerate', role='act', message='retained numerical failure')]
        result, costs, validations, _, prechecks, _ = self.run_case(precheck_errors=errors)
        self.assertEqual(len(costs), 1)
        self.assertEqual([row['engine'] for row in validations], ['tensor', 'mock'])
        self.assertEqual(prechecks, 1)
        self.assertEqual((result.coarse_evals, result.full_evals), (0, 1))
        self.assertEqual([row['phase'] for row in result.history], ['domain', 'result'])
        self.assertEqual(result.history[0]['numerical_errors'], errors)
        self.assertEqual(result.changes, [])
        self.assertEqual(result.source, print_program(parse(SOURCE)))

    def test_infeasible_start_accepts_feasible_more_expensive_alignment(self):
        result, costs, validations, _, _, _ = self.run_case(
            aligned=True, alignment_cost=120, count_budget=2)
        self.assertTrue(result.history[0]['accepted'])
        self.assertTrue(result.feasible)
        self.assertEqual(result.cost_before['latency_pred_us'], 100)
        self.assertEqual(result.cost_after['latency_pred_us'], 120)
        self.assertNotEqual(result.source, print_program(parse(SOURCE)))
        self.assertEqual(validations[-1]['source'], result.source)
        self.assertEqual(validations[-1]['engine'], 'mock')
        self.assertEqual(len(costs), 2)

    def test_accepted_alignment_continues_normal_scan(self):
        result, costs, validations, _, _, _ = self.run_case(
            aligned=True, alignment_cost=120, candidate_cost=110, count_budget=3)
        self.assertEqual(len(costs), 3)
        self.assertEqual(len(result.changes), 1)
        self.assertEqual((result.coarse_evals, result.full_evals), (1, 3))
        self.assertEqual([row['phase'] for row in result.history], ['domain', 'coarse', 'result'])
        self.assertEqual(result.cost_after['latency_pred_us'], 110)
        self.assertTrue(result.feasible)
        self.assertEqual(validations[-1]['engine'], 'mock')

    def test_feasible_start_history_changes_and_result_match_at_call_budget(self):
        results = []
        for rule in (tuner.PRE_TUNE_RULE, tuner.DEFAULT_TUNE_RULE):
            result, costs, validations, _, _, _ = self.run_case(
                initial=True, aligned=True, alignment_cost=90, count_budget=3, rule=rule)
            self.assertEqual(len(costs), 3)
            self.assertEqual((result.coarse_evals, result.full_evals), (1, 3))
            results.append((asdict(result), costs, validations))
        self.assertEqual(results[0], results[1])

    def test_feasible_start_still_rejects_cost_increase(self):
        result, _, _, _, _, _ = self.run_case(
            initial=True, aligned=True, alignment_cost=120, count_budget=2)
        self.assertFalse(result.history[0]['accepted'])
        self.assertEqual(result.source, print_program(parse(SOURCE)))
        self.assertEqual(result.cost_after['latency_pred_us'], 100)

    def test_pre_rule_keeps_scanning_infeasible_start(self):
        result, costs, _, _, _, _ = self.run_case(rule=tuner.PRE_TUNE_RULE, count_budget=3)
        self.assertEqual(len(costs), 3)
        self.assertEqual(result.coarse_evals, 1)
        self.assertEqual([row['phase'] for row in result.history], ['domain', 'coarse', 'result'])
        self.assertFalse(result.feasible)

    def test_v7_default_keeps_scanning_infeasible_start(self):
        result, costs, _, _, _, _ = self.run_case(version='r8-v7', count_budget=3)
        self.assertEqual(len(costs), 3)
        self.assertEqual(result.coarse_evals, 1)
        self.assertFalse(result.feasible)

    def test_initial_divergence_or_exact_failure_keeps_existing_finish(self):
        for options in (dict(diverged=True), dict(exact=False)):
            with self.subTest(options=options):
                result, costs, validations, _, prechecks, _ = self.run_case(**options)
                self.assertEqual(len(costs), 1)
                self.assertEqual(len(validations), 2)
                self.assertEqual(prechecks, 0)
                self.assertEqual([row['phase'] for row in result.history], ['result'])

    def test_rule_validation_and_nested_scope_restore(self):
        for version in ('r8-v4', 'r8-v5', 'r8-v6', 'r8-v7', 'r8-v8'):
            self.assertEqual(tuner.validate_tune_rule(version, tuner.DEFAULT_TUNE_RULE),
                             tuner.DEFAULT_TUNE_RULE)
        for version in ('r8-v4', 'r8-v5', 'r8-v6', 'r8-v7'):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'requires r8-v8'):
                tuner.validate_tune_rule(version, tuner.PRE_TUNE_RULE)
        with self.assertRaisesRegex(ValueError, 'Unknown tune_rule'):
            tuner.validate_tune_rule('r8-v8', 'unknown')
        self.assertEqual(tuner._tune_rule.get(), tuner.DEFAULT_TUNE_RULE)
        with tuner.tune_rule_scope(tuner.PRE_TUNE_RULE):
            with self.assertRaisesRegex(RuntimeError, 'test scope'):
                with tuner.tune_rule_scope(tuner.DEFAULT_TUNE_RULE):
                    self.assertEqual(tuner._tune_rule.get(), tuner.DEFAULT_TUNE_RULE)
                    raise RuntimeError('test scope')
            self.assertEqual(tuner._tune_rule.get(), tuner.PRE_TUNE_RULE)
        self.assertEqual(tuner._tune_rule.get(), tuner.DEFAULT_TUNE_RULE)


if __name__ == '__main__':
    unittest.main()
