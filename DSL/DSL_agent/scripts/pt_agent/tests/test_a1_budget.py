"""A1 §6.12: the upstream inverse budget conditions an analytical bound."""

import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

from pt_agent import domain_criteria as criteria, transformer_tensor as tensor
from pt_agent import transformer_validation as validation
from pt_agent.dsl.parser import parse
from pt_agent.kernels import transformer as kernel
from pt_agent.backends.mock import MockExpr
from pt_agent.tests.test_a1_violation_order import context
from pt_agent.tests._artifacts import RESULTS_DIR, require


SOURCE = '''@fhe.program(slots=1024)
def sm(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 p=softmax(a,exp_method="cheb",exp_domain=(-1.,1.),exp_deg=7,inv_method="newton",inv_domain=(.3,6.),inv_deg0=7,inv_iters=1,steps=1,step_inv_method="newton",step_inv_domain=(.08333333333333333,3.375),step_inv_deg0=15,step_inv_iters=3,final_inv_method="newton",final_inv_domain=(.08333333333333333,3.375),final_inv_deg0=15,final_inv_iters=3,inv_budget=.5)
 return p
'''


def fake_polynomial(value, attrs, exact=False):
    if attrs['fn'] == 'exp':
        return np.exp(value)
    return np.full_like(value, .01) if attrs['deg0'] == 7 else 1/value


class BudgetTests(unittest.TestCase):
    def test_failed_upstream_budget_rejects_without_stopping_in_both_engines(self):
        program = parse(SOURCE)
        def packed(b, value, attrs, **kwargs):
            return MockExpr(b, fake_polynomial(value.value, attrs))
        for engine in ('tensor', 'mock'):
            with self.subTest(engine=engine), patch.object(tensor, 'polynomial', side_effect=fake_polynomial), \
                    patch.object(kernel, 'apply', side_effect=packed), criteria.capture_violations() as reports:
                result = validation.validate(program, context([[0., 0.]]), engine=engine)
            self.assertFalse(result['feasible'])
            self.assertGreater(result['max_u_analytic'], 1)
            self.assertFalse(result['analytic_domain_violation'])
            criteria.require_analytic_coverage(result)
            first = reports[0]['samples'][0]['first_violation']
            self.assertEqual((first['kind'], first['role']), ('budget', 'inv'))
            self.assertEqual(first['program_line'], 4)
            self.assertAlmostEqual(first['c_min'], .02)
            self.assertAlmostEqual(first['c_max'], .02)
            self.assertEqual((first['rho'], first['R'], first['target']), (.5, (.5, 1.5), .5))
            self.assertAlmostEqual(first['max_relative_error'], .997)
            self.assertEqual(reports[0]['samples'][0]['chain'][1]['role'], 'final_inv')
            self.assertEqual(result['budget_violations'][0]['kind'], 'budget')

    def test_valid_upstream_budget_keeps_true_analytic_stop(self):
        program = parse(SOURCE)
        def execute(program, weights, inputs, *, observer, **kwargs):
            op = program.ops[-1]
            for role, data in [('exp', np.zeros((1, 1, 2))),
                               ('inv', np.full((1, 1, 1), .5)),
                               ('final_inv', np.full((1, 1, 1), .001))]:
                observer.sample_event(op.id, role, data, None)
                observer('point', data, ())
                if role == 'inv':
                    observer.inverse_event(op.id, role, data, np.full_like(data, 2.))
            return np.zeros((1, 1, 2))
        with patch.object(validation, 'execute', side_effect=execute):
            result = validation.validate(program, context([[0., 0.]]))
        self.assertTrue(result['analytic_domain_violation'])
        self.assertFalse(result['feasible'])
        self.assertNotIn('budget_violations', result)
        with self.assertRaisesRegex(RuntimeError, '§6.7'):
            criteria.require_analytic_coverage(result)

    def test_endpoint_roundoff_is_relative_and_does_not_accept_narrow_domains(self):
        contains = criteria.contains_analytic_domain
        self.assertTrue(contains((.01, 1.), (.01, 1+3e-13)))
        self.assertFalse(contains((.01, .999), (.01, 1.)))
        self.assertTrue(contains((.01*(1+3e-13), 1.), (.01, 1.)))
        self.assertFalse(contains((.01001, 1.), (.01, 1.)))
        self.assertFalse(contains((1e-20*(1+1e-6), 1.), (1e-20, 1.)))

    def test_no_rho_uses_the_existing_one_point_five_margin(self):
        program = parse(SOURCE.replace(',final_inv_method="newton",final_inv_domain=(.08333333333333333,3.375),final_inv_deg0=15,final_inv_iters=3,inv_budget=.5', '')
                        .replace('step_inv_domain=(.08333333333333333,3.375)',
                                 'step_inv_domain=(.3333333333333333,1.5)'))
        with patch.object(tensor, 'polynomial', side_effect=fake_polynomial):
            result = validation.validate(program, context([[0., 0.]]))
        row = result['budget_violations'][0]
        self.assertIsNone(row['rho'])
        np.testing.assert_array_equal(row['R'], (np.sqrt(2/3), np.sqrt(3/2)))
        self.assertFalse(result['analytic_domain_violation'])


MODEL = RESULTS_DIR/'A1-bert-prepare-20261027T123333-7967/model'
SESSION = RESULTS_DIR/'A1-bert-session-20261028T110734-9ce8'
DIAGNOSTIC = RESULTS_DIR/'A1-bert-budget-diagnostic-20261028T124955-9239/DIAGNOSTIC.json'


@require(MODEL/'model_info.json', MODEL/'calibration.json', MODEL/'samples/heldout.npz',
         SESSION/'sources/iteration-5.ptdsl', DIAGNOSTIC)
class RetainedBudgetTests(unittest.TestCase):
    def test_call21_is_budget_with_the_diagnosed_c_minimum(self):
        from pt_agent.data.samples import StoredSplits
        from pt_agent.models.info import ModelInfo
        info = ModelInfo.load(MODEL)
        ctx = validation.context(info, json.loads((MODEL/'calibration.json').read_text()),
                                 samples=StoredSplits(MODEL/'samples', info))
        ctx.prompt_version = 'r8-v8'
        program = parse((SESSION/'sources/iteration-5.ptdsl').read_text())
        with criteria.capture_violations() as reports:
            result = validation.validate(program, ctx)
        self.assertFalse(result['feasible'])
        self.assertFalse(result['analytic_domain_violation'])
        actual = [s for s in reports[0]['samples'] if s['first_violation'] is not None]
        self.assertEqual([s['sample_index'] for s in actual], [7, 10, 11, 14, 15])
        self.assertTrue(all((s['first_violation']['program_line'], s['first_violation']['role'],
                             s['first_violation']['kind']) == (12, 'inv', 'budget') for s in actual))
        expected = json.loads(DIAGNOSTIC.read_text())['c_min']
        self.assertEqual(min(s['first_violation']['c_min'] for s in actual), expected)
        inv = next(r for r in result['rows'] if r['program_line'] == 12 and r['role'] == 'inv')
        self.assertEqual(inv['domain_source'], 'analytic')
        self.assertLessEqual(inv['max_u'], 1)
        self.assertAlmostEqual(inv['max_u'], .9992093231654837)

    def test_recovery_reuses_four_tunes_and_preserves_the_six_candidates(self):
        import tempfile
        from pt_agent.data.samples import StoredSplits
        from pt_agent.models.info import ModelInfo
        from pt_agent.planners import agent
        from pt_agent.planners.transformer_session import recover_session, load_state
        info = ModelInfo.load(MODEL)
        ctx = validation.context(info, json.loads((MODEL/'calibration.json').read_text()),
                                 samples=StoredSplits(MODEL/'samples', info))
        ctx.prompt_version = 'r8-v8'
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(agent, 'evaluate', side_effect=AssertionError('do not reevaluate prefix')), \
                patch.object(agent, 'tune', side_effect=AssertionError('do not retune prefix')):
            directory = Path(temporary)
            result = recover_session(ctx, SESSION, directory)
            state = load_state(directory/'SESSION.json')
        self.assertTrue(result['complete'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual((state['calls'], state['iteration'], state['phase']), (4, 5, 'evaluate'))
        self.assertEqual(len(state['pool']), 6)
        self.assertEqual(result['recovery']['reused_tune_rounds'], [1, 2, 3, 4])
        self.assertEqual(state['tune_rule'], 'v8-6.11')


REPLAY = RESULTS_DIR/'A1-bert-budget-replay-20261028T131528-8fa2/REPLAY.json'


@require(REPLAY)
class RetainedBudgetReplayTests(unittest.TestCase):
    def test_twenty_unchanged_feedbacks_and_only_call21_changes(self):
        result = json.loads(REPLAY.read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(result['changed_calls'], [21])
        self.assertTrue(all(row['feedback_equal'] for row in result['rows'][:20]))
        self.assertEqual([row['model_call'] for row in result['rows'] if row['first_changed']], [21])
        last = result['rows'][-1]
        self.assertEqual(last['old_stop'], 'analytic_domain_violation')
        self.assertIsNone(last['new_stop'])
        self.assertFalse(last['new_feasible'])
        self.assertEqual([x['role'] for x in last['new_first'] if x is not None], ['inv']*5)
        self.assertTrue(all(x['kind'] == 'budget' for x in last['new_first'] if x is not None))
        self.assertEqual((result['llm_calls'], result['tune_calls'], result['report_images']), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
