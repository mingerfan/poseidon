import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from pt_agent.quality import model_metric, quality_fields, quality_matches, strict_tuner_quality
from pt_agent.tests._artifacts import COMPILER, require


class TaskQualityTests(unittest.TestCase):
    def test_classification_history_keeps_the_original_count(self):
        result = dict(agree=997, top1=91.3, total=1000)
        self.assertEqual(quality_fields(result, 'classification_top1'), dict(agree=997))
        self.assertTrue(quality_matches(result, dict(result)))
        self.assertFalse(quality_matches(result, dict(result, agree=996)))

    def test_regression_has_no_classification_field(self):
        result = dict(relative_error=0.004, total=256)
        self.assertEqual(quality_fields(result), dict(relative_error=0.004))
        self.assertEqual(quality_fields(dict(result, agree=None)), dict(relative_error=0.004))
        self.assertEqual(model_metric(SimpleNamespace(task='regression')), 'regression_relative')

    def test_regression_comparison_preserves_the_math_tolerance(self):
        result = dict(feasible=True, relative_error=0.004)
        self.assertTrue(quality_matches(result, dict(result, relative_error=0.004+5e-13)))
        self.assertFalse(quality_matches(result, dict(result, relative_error=0.004+2e-12)))
        self.assertFalse(quality_matches(result, dict(result, relative_error=float('nan'))))

    def test_missing_regression_error_is_only_comparable_for_rejected_results(self):
        result = dict(feasible=False, relative_error=None)
        self.assertTrue(quality_matches(result, dict(result)))
        self.assertFalse(quality_matches(result, dict(result, feasible=True)))

    def test_generic_tuner_preserves_task_feasibility_and_classifier_margin(self):
        self.assertTrue(strict_tuner_quality(dict(feasible=True, relative_error=0.004), 0.3))
        self.assertFalse(strict_tuner_quality(dict(feasible=False, relative_error=0.011), 0.3))
        self.assertTrue(strict_tuner_quality(dict(feasible=True, acc_drop_pp=0.3), 0.3))
        self.assertFalse(strict_tuner_quality(dict(feasible=True, acc_drop_pp=0.30001), 0.3))


@require(COMPILER)
class TunerQualityTests(unittest.TestCase):
    def context(self, regression):
        return SimpleNamespace(model_info=SimpleNamespace(task='regression' if regression else 'classification',
            metric='regression_relative' if regression else 'classification_top1',
            threshold={'relative_error': 0.01} if regression else {'acc_drop_pp': 0.5}),
            prompt_version='r8-v11', placement='program-plan', weights={}, profile={})

    def test_move_quality_uses_the_task_field_and_original_margin_factor(self):
        from pt_agent.planners import transformer_tuner as tuner
        for regression, key in [(True, 'relative_error'), (False, 'logit_rel_err')]:
            ctx = self.context(regression)
            self.assertEqual(tuner.move_logit_limit(ctx, tuner.ALIGN_FALLBACK_TUNE_RULE), 0.005)
            self.assertTrue(tuner.move_accuracy_ok({key: 0.005}, ctx, tuner.ALIGN_FALLBACK_TUNE_RULE))
            self.assertFalse(tuner.move_accuracy_ok({key: 0.005000001}, ctx, tuner.ALIGN_FALLBACK_TUNE_RULE))

    def finish(self, regression, *, changed=False):
        from pt_agent.planners import transformer_tuner as tuner
        from pt_agent import domain_criteria, margin
        from pt_agent.dsl import interface
        quality = dict(relative_error=0.004) if regression else dict(agree=999, logit_rel_err=0.004)
        first = dict(feasible=True, diverged=False, max_u=0.7, max_u_calibrated=0.7,
                     max_u_analytic=0.9, total=1000, worst_acts=[], **quality)
        second = dict(first, **(dict(relative_error=0.006) if changed else {}))
        costs = dict(latency_pred_us=1., boots_pred=1)
        with ExitStack() as stack:
            for target, name, value in [
                (tuner, 'cached_estimate', lambda *a, **k: costs),
                (tuner, 'validate', unittest.mock.Mock(side_effect=[first, second])),
                (tuner, 'exact_check', lambda *a, **k: {'ok': False}),
                (tuner, 'estimate_calls', lambda: 0),
                (tuner, 'print_program', lambda p: 'trusted-test-program'),
                (domain_criteria, 'require_analytic_coverage', lambda p: None),
                (interface, 'require_interface', lambda *a: None),
                (margin, 'require_margin', lambda *a: None)]:
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(tuner.tune_rule_scope(tuner.ALIGN_FALLBACK_TUNE_RULE))
            return tuner.tune(SimpleNamespace(ops=[]), self.context(regression), budget_s=1)

    def test_regression_reaches_final_mock_verification_without_agree(self):
        result = self.finish(True)
        self.assertEqual(result.history[-1]['relative_error'], 0.004)
        self.assertNotIn('agree', result.history[-1])
        generic = self.generic_finish(True)
        self.assertEqual(generic.history[-1]['relative_error'], 0.004)
        self.assertNotIn('agree', generic.history[-1])

    def test_classification_final_record_stays_unchanged(self):
        result = self.finish(False)
        self.assertEqual(result.history[-1]['agree'], 999)
        self.assertNotIn('relative_error', result.history[-1])
        generic = self.generic_finish(False)
        self.assertEqual(generic.history[-1]['agree'], 999)
        self.assertEqual(list(generic.history[-1])[:8], ['phase', 'latency_pred_us', 'agree', 'total',
            'max_u', 'top1', 'exact_top1', 'acc_drop_pp'])

    def generic_finish(self, regression):
        from pt_agent.planners import tuner
        from pt_agent.dsl import interface
        from pt_agent import transformer_lower
        quality = dict(relative_error=0.004) if regression else dict(agree=999, top1=91.3)
        measured = dict(feasible=True, diverged=False, max_u=0.7, total=1000, **quality)
        cost = dict(latency_pred_us=1., boots_pred=1, boot_nodes=[])
        with ExitStack() as stack:
            for target, name, value in [
                (tuner, 'cached_estimate', lambda *a, **k: cost),
                (tuner, 'validate', lambda *a, **k: dict(measured)),
                (tuner, '_exact_check', lambda *a, **k: {'ok': False}),
                (tuner, 'print_program', lambda p: 'trusted-test-program'),
                (transformer_lower, 'uses_transformer', lambda p: False),
                (interface, 'require_interface', lambda *a: None)]:
                stack.enter_context(patch.object(target, name, value))
            return tuner.tune(SimpleNamespace(ops=[]), self.context(regression), budget_s=1,
                              engine_profile='tensor')

    def test_disagreeing_regression_verification_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'Final Mock verification differs'):
            self.finish(True, changed=True)


if __name__ == '__main__':
    unittest.main()
