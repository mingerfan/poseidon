"""E5c: offline placement integration; all clients are local fixtures."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.cost import estimate, estimate_fast, estimate_normalized_fast, PROFILES
from pt_agent.dsl.parser import parse
from pt_agent.experiments.a1_v10_replay import recost
from pt_agent.planners.agent import _run_agent_once, new_agent_state, agent_step
from pt_agent.planners.agent_tools import apply_placement, cost_feedback, evaluate
from pt_agent.planners.prompts import build_system_prompt, PLACEMENT_EXPLANATION
from pt_agent.planners.tuner import tune
from pt_agent.tensor_sim import cached_estimate, clear_caches, reset_timing, timing_snapshot
from pt_agent.tests.test_r8c_feedback import SOURCE, context


FIXTURES = Path(__file__).with_name('fixtures')


class PlacementTests(unittest.TestCase):
    def setUp(self):
        clear_caches()
        reset_timing()
        self.ctx = context()

    def test_rewrite_idempotent_and_preserves_body_keywords_lines(self):
        self.assertEqual(apply_placement(SOURCE, self.ctx), SOURCE)
        self.ctx.placement = 'program-plan'
        source = SOURCE.replace('slots=16384', 'slots=8192, manage="compiler"')
        result = apply_placement(source, self.ctx)
        p = parse(result)
        self.assertEqual((p.header.slots, p.header.refresh, p.header.placement), (8192, 'agent', 'program-plan'))
        self.assertEqual((p.header.l_max, p.header.boot_out_level, p.header.manage), (5, 5, 'compiler'))
        self.assertEqual(result.splitlines()[1:], source.splitlines()[1:])
        self.assertEqual(apply_placement(result, self.ctx), result)
        self.assertEqual([op.lineno for op in p.ops], [op.lineno for op in parse(source).ops])

    def test_unparseable_or_nonunique_header_is_retained(self):
        self.ctx.placement = 'program-plan'
        for source in ('bad', SOURCE.replace('slots=16384', 'slots='), SOURCE+SOURCE,
                       SOURCE.replace('slots=16384', 'slots=16384, slots=8192')):
            self.assertEqual(apply_placement(source, self.ctx), source)

    def test_default_feedback_and_mlp_tune_match_prechange_snapshot(self):
        saved = json.loads((FIXTURES/'e5c_default.json').read_text())
        self.assertEqual(json.loads(json.dumps(evaluate(SOURCE, self.ctx))), saved['feedback'])
        result = asdict(tune(parse(SOURCE), self.ctx, budget_s=2400, max_sweeps=1))
        result.pop('accepted_states')  # E7 compares the frozen fields separately from the new record.
        result.pop('program')
        result.pop('seconds')
        self.assertEqual(json.loads(json.dumps(result)), saved['tune'])

    def test_fast_estimate_only_omits_replay(self):
        self.ctx.placement = 'program-plan'
        program = parse(apply_placement(SOURCE, self.ctx))
        with patch('time.monotonic', return_value=0):
            expected = estimate(program, self.ctx.weights)
            actual = estimate(program, self.ctx.weights, verify_plan=False)
        self.assertIn('replay', expected['refresh_plan'])
        del expected['refresh_plan']['replay']
        self.assertEqual(actual, expected)

    def test_fast_estimator_identity_hits_cache_and_timing_counts_miss(self):
        self.ctx.placement = 'program-plan'
        program = parse(apply_placement(SOURCE, self.ctx))
        first = cached_estimate(program, self.ctx.weights, self.ctx.profile, estimate_fast)
        before = timing_snapshot()
        self.assertGreater(before['estimate_s'], 0)
        second = cached_estimate(program, self.ctx.weights, self.ctx.profile, estimate_fast)
        self.assertEqual(first, second)
        self.assertEqual(timing_snapshot(), before)

    def test_validation_timer_and_reset(self):
        evaluate(SOURCE, self.ctx)
        self.assertGreater(timing_snapshot()['validation_s'], 0)
        reset_timing()
        self.assertTrue(all(value == 0 for value in timing_snapshot().values()))

    def test_recost_two_saved_calls_preserves_validation_and_sources(self):
        self.ctx.placement = 'program-plan'
        feedback = json.loads(json.dumps(evaluate(SOURCE, context())))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.ptdsl'
            source.write_text(SOURCE)
            old = root/'old.json'
            old.write_text(json.dumps(feedback))
            manifest = root/'before.json'
            manifest.write_text(json.dumps(dict(cases=[dict(call=i, session='old-session', source=str(source),
                                                          feedback=str(old)) for i in (1, 2)])))
            out = root/'recost'
            out.mkdir()
            args = SimpleNamespace(model_dir=root, out=out, placement='program-plan', response_file=manifest)
            with patch('pt_agent.experiments.a1_v10_replay.context', return_value=self.ctx):
                result = recost(args)
            self.assertTrue(result['complete'])
            after = json.loads((out/'REEVALUATION.json').read_text())
            for row in after['cases']:
                fb = json.loads(Path(row['feedback']).read_text())
                self.assertEqual(fb['validation'], feedback['validation'])
                self.assertEqual(fb['cost']['placement_model'], 'program-plan')
            self.assertEqual(source.read_text(), SOURCE)
            self.assertEqual(json.loads(old.read_text()), feedback)

    def test_recost_rechecks_resources_and_boot_input_bound(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source').write_text(SOURCE)
            old = json.loads(json.dumps(evaluate(SOURCE, self.ctx)))
            (root/'feedback').write_text(json.dumps(old))
            (root/'review').write_text(json.dumps(dict(cases=[dict(call=1, session='x', source=str(root/'source'),
                                                                 feedback=str(root/'feedback'))])))
            args = SimpleNamespace(model_dir=root, out=root, placement='program-plan', response_file=root/'review')
            with patch('pt_agent.experiments.a1_v10_replay.context', return_value=self.ctx), \
                    patch.dict(PROFILES['qwen48']['limits'], const_bytes=1):
                recost(args)
            fb = json.loads((root/'call-001-feedback.json').read_text())
            self.assertEqual(fb['stage'], 'resources')
            self.assertFalse(fb['feasible'])
            self.assertEqual(fb['validation'], old['validation'])
            with patch('pt_agent.experiments.a1_v10_replay.context', return_value=self.ctx), \
                    patch.dict(PROFILES['qwen48'], boot_input_bound=1):
                with self.assertRaises(AssertionError):
                    recost(args)

    def test_both_agent_loops_save_original_and_evaluate_rewritten_source(self):
        reply = dict(content=json.dumps(dict(program=SOURCE)), seconds=1., usage={})
        class Client:
            def chat(self, *args, **kwargs):
                return reply
        feedback = dict(stage='interface', feasible=False, exact=None, cost=None, validation=None)
        self.ctx.placement = 'program-plan'
        with TemporaryDirectory() as directory, \
                patch('pt_agent.planners.agent.build_system_prompt', return_value='system'), \
                patch('pt_agent.planners.agent.initial_message', return_value='initial'), \
                patch('pt_agent.planners.agent.evaluate', return_value=feedback) as evaluate_mock:
            for version in ('r8-v4', 'r8-v7'):
                self.ctx.prompt_version = version
                trace = Path(directory)/('legacy-'+version)/'trace.jsonl'
                _run_agent_once(self.ctx, Client(), max_iters=1, tuner=False, trace_path=trace)
                self.assertEqual((trace.parent/'sources/iteration-1.ptdsl').read_text(), SOURCE)
                self.assertEqual(parse(evaluate_mock.call_args.args[0]).header.placement, 'program-plan')
                self.assertEqual(json.loads(trace.read_text())['placement'], 'program-plan')
            for version in ('r8-v10', 'r8-v11'):
                self.ctx.prompt_version = version
                state = new_agent_state(self.ctx, max_iters=1, tuner=False)
                trace = Path(directory)/('stepped-'+version)/'trace.jsonl'
                trace.parent.mkdir()
                agent_step(self.ctx, state, trace_path=trace, reply=reply)
                self.assertEqual((trace.parent/'sources/iteration-1.ptdsl').read_text(), SOURCE)
                self.assertEqual(parse(evaluate_mock.call_args.args[0]).header.placement, 'program-plan')
                self.assertEqual(state['placement'], 'program-plan')
                self.assertEqual(json.loads(trace.read_text())['placement'], 'program-plan')

    def test_v11_prompt_snapshot_keeps_v10_and_adds_only_registered_text(self):
        self.ctx.prompt_version = 'r8-v10'
        before = (FIXTURES/'e5c_prompt_v10.txt').read_text()
        self.assertEqual(build_system_prompt(self.ctx), before)
        self.ctx.prompt_version = 'r8-v11'
        expected = before.replace('@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")',
            '@fhe.program(slots=16384, refresh="agent", placement="program-plan", l_max=5, boot_out_level=5)')
        expected = expected.replace('an unmodified compiler then inserts\nrescale/modswitch/bootstrap.',
            'a placement planner inserts bootstraps and\nan unmodified compiler then inserts rescale/modswitch.')
        expected = expected.replace('bootstraps, computed with the same bootstrap placement as the compiler, and',
            'bootstraps, computed with the program-level placement used for compilation, and')
        self.assertEqual(build_system_prompt(self.ctx), expected+PLACEMENT_EXPLANATION)

    def test_v11_cost_feedback_snapshot_contains_variant_and_simulated_levels(self):
        self.ctx.prompt_version = 'r8-v11'
        self.ctx.placement = 'program-plan'
        text = apply_placement(SOURCE, self.ctx)
        cost, feedback = cost_feedback(parse(text), self.ctx, estimate_normalized_fast, lambda line: '')
        self.assertEqual(set(feedback), {'latency_pred_us','boots_pred','max_depth','ops_total','rot_keys',
            'placement_model','placement_exact','barriers','boot_share','hotspots','placement_variant'})
        self.assertEqual(feedback['placement_variant'], cost['refresh_plan']['variant'])
        levels = {(row['line'], tuple(row['call_lines'])): row['mean']
                  for row in cost['refresh_plan']['levels_left_at_refresh']}
        for row in feedback['hotspots']:
            self.assertEqual(set(row), {'line','call_lines','latency_share','boots','line_text','call_texts',
                                       'levels_left_at_refresh'})
            self.assertEqual(row['levels_left_at_refresh'], levels.get((row['line'], tuple(row['call_lines']))))
        self.assertTrue(any(row['levels_left_at_refresh'] is not None for row in feedback['hotspots']))


if __name__ == '__main__':
    unittest.main()
