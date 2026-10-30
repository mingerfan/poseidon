"""A1 §6.15: calibration-only margin and the retained model allowance."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import margin
from pt_agent.dsl.parser import parse
from pt_agent.planners import agent_tools, prompts, transformer_tuner
from pt_agent.planners.transformer_session import ModelBudgetClient, output_tokens, model_calls
from pt_agent.tests.test_a1_agent import fixture, SOURCE, RecordedClient
from pt_agent.llm import LLMError


def context(version='r8-v9'):
    info, calibration, samples = fixture()
    ctx = agent_tools.build_context(info, calibration, samples=samples)
    ctx.prompt_version = version
    return ctx


class MarginTests(unittest.TestCase):
    def test_signed_and_rsqrt_domains(self):
        for lo, hi, expected in [(-2, 3, (-3, 4.5)), (2, 3, (2/1.5, 4.5)),
                                 (-3, -2, (-4.5, -2/1.5)), (.03, 12, (.02, 18))]:
            self.assertEqual(margin.minimum_domain(lo, hi), expected)

    def test_relative_roundoff_both_signs_and_zero(self):
        for required in [(-3., 4.5), (2., 3.), (-3., -2.), (0., 1.)]:
            lo, hi = required
            self.assertTrue(margin.contains((lo+abs(lo)*3e-13, hi-abs(hi)*3e-13), required))
            self.assertFalse(margin.contains((lo, hi-1e-5), required))
        self.assertFalse(margin.contains((1e-30, 1), (0, 1)))

    def test_all_violations_are_listed(self):
        source = SOURCE.replace('domain=(-1.,1.)', 'domain=(-.1,.1)')
        source = source.replace(' y=flatten', ' q=act(z,fn="tanh",method="cheb",deg=15,domain=(-.1,.1))\n y=flatten')
        ctx = context()
        ctx.act_tensors = ['v', 'z']
        ctx.calibration['z'] = dict(min=-.3, max=.3)
        errors = margin.check_margin(parse(source), ctx)['errors']
        self.assertEqual([row['line'] for row in errors], [5, 6])

    def test_analyzer_exp_calibration_uses_raw_scores(self):
        from pt_agent.experiments.a1_v9_margin import score_calibration_proof
        self.assertTrue(score_calibration_proof()['raw_scores'])

    def test_exp_margin_is_transformed_from_raw_scores(self):
        source = '''@fhe.program(slots=1024)
def f(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 y=softmax(a,steps=3,shift=4.,exp_method="cheb",exp_deg=7,exp_domain=(-2.,0.),inv_method="newton",inv_deg0=7,inv_iters=2,inv_domain=(.01,3.),step_inv_method="newton",step_inv_deg0=7,step_inv_iters=2,step_inv_domain=(.01,3.))
 return y
'''
        ctx = SimpleNamespace(prompt_version='r8-v9', act_tensors=['s::exp', 's::inv'],
                              calibration={'s::exp':dict(min=-2., max=4.)})
        result = margin.check_margin(parse(source), ctx)
        self.assertFalse(result['ok'])
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['errors'][0]['min_domain'], ((1.5*-2-4)/8, (1.5*4-4)/8))
        self.assertTrue(margin.check_margin(parse(source.replace('exp_domain=(-2.,0.)',
                                'exp_domain=(-.875,.25)')), ctx)['ok'])

    def test_margin_stage_lists_all_and_precedes_numerics(self):
        ctx = context()
        source = SOURCE.replace('domain=(-1.,1.)', 'domain=(-.1,.1)')
        with (patch.object(agent_tools, '_exact_check', side_effect=AssertionError('execute')),
              patch('pt_agent.planners.approximation_failures.program_failures',
                    side_effect=AssertionError('fit'))):
            result = agent_tools.evaluate(source, ctx, engine_profile='tensor')
        self.assertEqual(result['stage'], 'margin')
        self.assertFalse(result['feasible'])
        self.assertIsNone(result['exact'])
        self.assertEqual(result['errors'][0]['role'], 'act')
        self.assertEqual(result['errors'][0]['line'], 5)

    def test_interface_still_precedes_margin(self):
        with patch('pt_agent.margin.check_margin', side_effect=AssertionError('margin')):
            result = agent_tools.evaluate(SOURCE.replace('x:', 'v:').replace('pack(x,', 'pack(v,'),
                                          context(), engine_profile='tensor')
        self.assertEqual(result['stage'], 'interface')

    def test_v8_does_not_check_margin(self):
        ctx = context('r8-v8')
        self.assertTrue(margin.check_margin(parse(SOURCE.replace('domain=(-1.,1.)','domain=(-.1,.1)')),ctx)['ok'])
        self.assertTrue(agent_tools.evaluate(SOURCE, ctx, engine_profile='tensor')['feasible'])

    def test_prompt_changes_only_calibration_sentence_and_summary_domains(self):
        ctx = context('r8-v8')
        before = prompts.build_system_prompt(ctx)
        original = json.loads(prompts.initial_message(ctx))
        ctx.prompt_version = 'r8-v9'
        after = prompts.build_system_prompt(ctx)
        summary = json.loads(prompts.initial_message(ctx))
        old = 'calib = [min, max] observed on calibration samples. Derive approximation domains only from calibration and the registered analytical formulas.'
        start = after.index('calib = [lo, hi]')
        end = after.index(' Heldout samples judge feasibility', start)
        self.assertEqual(after[:start]+old+after[end:], before)
        count = 0
        for node in summary['nodes']:
            for point in node.get('approximation_points', []):
                self.assertEqual(point.pop('min_domain'), list(margin.minimum_domain(
                    point['calib']['min'], point['calib']['max'])))
                count += 1
        self.assertGreater(count, 0)
        self.assertEqual(summary, original)

    def test_tuner_aligns_to_M_and_keeps_all_candidates_covered(self):
        ctx = context()
        from pt_agent.transformer_validation import points
        seen = []
        from pt_agent.planners import transformer_tuner as tuner
        original = tuner.validate
        def validate(program, *args, **kwargs):
            self.assertTrue(margin.check_margin(program, ctx)['ok'])
            seen.append(copy.deepcopy(points(program)[0].attrs['domain']))
            return original(program, *args, **kwargs)
        with patch.object(tuner, 'validate', side_effect=validate):
            result = tuner.tune(parse(SOURCE), ctx, max_sweeps=0, engine='tensor', engine_profile='tensor')
        expected = margin.minimum_domain(ctx.calibration['v']['min'], ctx.calibration['v']['max'])
        self.assertEqual(seen[1], expected)
        self.assertTrue(margin.check_margin(result.program, ctx)['ok'])
        self.assertEqual(transformer_tuner.validate_tune_rule('r8-v9', None), 'v8-6.14')

    def test_v9_retains_six_bert_calls_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = RecordedClient(['{}']*6)
            client = ModelBudgetClient(raw, tmp, model='bert', prompt_version='r8-v9')
            self.assertEqual(client.calls, 34)
            for _ in range(6): client.chat([], purpose='test')
            self.assertTrue((client.directory/'call-035.response.json').is_file())
            restored = ModelBudgetClient(raw, tmp, model='bert', prompt_version='r8-v9')
            with self.assertRaisesRegex(LLMError, 'budget_exhausted'):
                restored.chat([], purpose='extra')
            self.assertEqual(model_calls(tmp, 'bert', 'r8-v9'), 40)
            self.assertEqual(ModelBudgetClient(raw, tmp, model='qwen', prompt_version='r8-v9').calls, 0)
            self.assertEqual(len(raw.messages), 6)
            self.assertEqual(output_tokens('r8-v9'), 65536)

    def test_invalid_tuner_move_is_an_implementation_error(self):
        source = SOURCE.replace('method="cheb"', 'method="remez"')
        with patch('pt_agent.planners.tuner._new_settings',
                   return_value=[dict(attrs=dict(domain=(-.01,.01)))]):
            with self.assertRaisesRegex(ValueError, 'minimum margin'):
                transformer_tuner.tune(parse(source), context(), engine='tensor', engine_profile='tensor')

    def test_fresh_v9_session_uses_only_six_remaining_calls_and_empty_pool(self):
        from pt_agent.planners.transformer_session import run_session
        info, calibration, samples = fixture()
        source = SOURCE.replace('domain=(-1.,1.)', 'domain=(-.1,.1)')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ModelBudgetClient(RecordedClient([json.dumps(dict(program=source))]*6),
                                       root/'calls', model='bert', prompt_version='r8-v9')
            result = run_session(info, calibration, samples, client, root/'session',
                                 max_iters=6, prompt_version='r8-v9')
            rows = [json.loads(line) for line in (root/'session/agent_trace.jsonl').read_text().splitlines()]
            self.assertEqual([row['fb']['stage'] for row in rows], ['margin']*6)
            self.assertEqual([row['model_call'] for row in rows], list(range(35, 41)))
            self.assertEqual(result['pool_size'], 0)
            self.assertEqual(result['llm_calls'], 6)
            self.assertEqual(client.calls, 40)
            self.assertEqual(result['confirm']['evals'], [])
