"""Small synthetic checks of the explicit numerical-tuner ablation option."""
from argparse import Namespace
import ast
from contextlib import ExitStack, nullcontext, redirect_stderr
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import r28_common_agent as common
from pt_agent.experiments import r30_common_model_e2e as entry
# Load the state serializer before any scoped sys.modules replacement. Its
# deferred import otherwise leaves a stale package attribute after restoration.
from pt_agent.planners import transformer_session


def _isolated_agent():
    """Execute unchanged trusted state functions with numerical imports isolated.

    The tested function bodies come directly from agent.py. Only their external
    evaluators, prompts and rule lookup are stubs, as in the orchestration tests;
    importing a model runtime is unnecessary for this bool/state transition.
    """
    package = Path(entry.__file__).parents[1]
    source = package/'planners/agent.py'
    names = {'new_agent_state', '_evaluate_step', '_finish_iteration', '_selected_record', 'agent_step'}
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise AssertionError('The trusted Agent state interface changed')
    candidate_source = package/'planners/confirm.py'
    candidate = next(node for node in ast.parse(candidate_source.read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == 'Candidate')
    isolated = ModuleType('pt_agent.tests._r30_no_tuner_agent')
    isolated.__package__ = 'pt_agent.planners'
    isolated.__dict__.update(json=json, Path=Path, time=time, asdict=asdict, dataclass=dataclass,
        LLMError=RuntimeError, PRE_TUNE_RULE='v8-pre-6.11', STRUCTURE_NOTE='',
        validate_tune_rule=lambda version, rule: rule,
        build_system_prompt=None, initial_message=None, feedback_message=None,
        confirmation_rule=None, apply_placement=None, parse=None, print_program=None,
        evaluate=None, tune=None, confirm_select=None)
    with patch.dict(sys.modules, {isolated.__name__: isolated}):
        exec(compile(ast.Module(body=[candidate, *definitions], type_ignores=[]),
                     str(source), 'exec'), isolated.__dict__)
    return isolated


agent = _isolated_agent()


class NoTunerChecks(unittest.TestCase):
    def setUp(self):
        guard=patch('pt_agent.llm.subprocess.run',side_effect=AssertionError('Provider transport is blocked in synthetic tests'))
        guard.start();self.addCleanup(guard.stop)

    def _phase_args(self, **options):
        values = dict(out=Path('/fresh/run'), model='resnet20_silu', inputs=Path('/original/records'),
            selections=Path('/original/selections'), agent_seconds=7200, trace_seconds=1800,
            compile_seconds=7200, ledger=Path('/budget/calls'), allowance=Path('/budget/allowance'),
            profile=Path('/profile'), frontend=Path('/frontend'), metadata_template=Path('/metadata'),
            calls=6, tuner_seconds=600, memory_gib=6, case_gib=8)
        values.update(options)
        return Namespace(**values)

    def _agent_case(self, directory, *, tuner, feasible=True, confirm=True):
        ctx = SimpleNamespace(prompt_version='r8-v11', placement='program-plan',
            tune_search='sweep', model_info=SimpleNamespace(split={}))
        feedback = dict(stage='complete', feasible=feasible, exact=dict(ok=True),
            cost=dict(latency_pred_us=123., boots_pred=1),
            validation=dict(diverged=False, agree=1000, max_u=.5,
                max_u_calibrated=.5, max_u_analytic=0.))
        reply = dict(content=json.dumps(dict(program='def candidate(x):\n    return x\n')),
                     seconds=.1, usage=dict(prompt_tokens=2, completion_tokens=3))
        path = directory/'agent_trace.jsonl'
        with ExitStack() as stack:
            lookup = ModuleType('pt_agent.planners.transformer_tuner')
            lookup.validate_tune_search = lambda ctx: getattr(ctx, 'tune_search', 'sweep')
            stack.enter_context(patch.dict(sys.modules, {lookup.__name__: lookup}))
            stack.enter_context(patch.object(agent, 'build_system_prompt', return_value='synthetic grammar'))
            stack.enter_context(patch.object(agent, 'initial_message', return_value='synthetic task'))
            stack.enter_context(patch.object(agent, 'confirmation_rule', return_value='c1'))
            stack.enter_context(patch.object(agent, 'apply_placement', side_effect=lambda source, ctx: source))
            stack.enter_context(patch.object(agent, 'parse', return_value=object()))
            stack.enter_context(patch.object(agent, 'print_program', return_value='normalized fixture'))
            evaluate = stack.enter_context(patch.object(agent, 'evaluate', return_value=feedback))
            tune = stack.enter_context(patch.object(agent, 'tune', side_effect=AssertionError('No numerical work allowed')))
            shown = stack.enter_context(patch.object(agent, 'feedback_message', return_value='synthetic feedback'))
            confirmation = stack.enter_context(patch.object(agent, 'confirm_select',
                side_effect=lambda pool, ctx, **kwargs: dict(selected=pool[0] if pool else None,
                    evals=[], duplicates=[], deduplicated=dict(before=len(pool), after=len(pool)))))
            state = agent.new_agent_state(ctx, **common.agent_options(1, 600, tuner=tuner))
            agent.agent_step(ctx, state, trace_path=path, reply=reply)
            evaluate.assert_called_once()
            tune.assert_not_called()
            if not tuner and confirm:
                self.assertEqual(state['phase'], 'confirm')
                agent.agent_step(ctx, state, trace_path=path)
                confirmation.assert_called_once()
                self.assertIsNone(shown.call_args.kwargs['tuned'])
            return state

    def test_default_options_and_variant_preserve_the_previous_policy(self):
        expected = dict(max_iters=6, tuner=True, tuner_budget_s=600, tune_policy='all',
            engine_profile='tensor', stale_limit=3, tune_rule='v8-6.25', focus_retune=True,
            numeric_lock=False)
        self.assertEqual(common.agent_options(6, 600), expected)
        disabled = common.agent_options(6, 600, tuner=False)
        self.assertEqual({key for key in expected if expected[key] != disabled[key]}, {'tuner'})
        self.assertEqual(entry.agent_variant(Namespace()), entry.VARIANT)
        self.assertNotEqual(entry.agent_variant(Namespace(tuner=False)), entry.VARIANT)
        for invalid in (None, 0, 'false'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                common.agent_options(6, 600, tuner=invalid)

    def test_cli_and_every_phase_forward_the_option_with_the_same_finite_limits(self):
        for phase in entry.PHASES:
            original, original_wall = entry.phase_command(self._phase_args(), phase,
                Path('/fresh/run/work')/phase, remaining_seconds=28000)
            disabled, disabled_wall = entry.phase_command(self._phase_args(tuner=False), phase,
                Path('/fresh/run/work')/phase, remaining_seconds=28000)
            self.assertNotIn('--no-tuner', original)
            self.assertEqual(disabled[:-1], original)
            self.assertEqual(disabled[-1], '--no-tuner')
            self.assertEqual(disabled_wall, original_wall)
        for extra, enabled in (([], True), (['--no-tuner'], False)):
            with patch.object(entry, 'execute', return_value=dict(status='prepared')) as execute:
                self.assertEqual(entry.main(['--model', 'resnet20_silu', '--out', '/fresh/new', *extra]), 0)
            self.assertIs(execute.call_args.args[0].tuner, enabled)
            self.assertEqual(execute.call_args.args[0].calls, 6)
            self.assertEqual(execute.call_args.args[0].tuner_seconds, 600)
        for invalid in (['--calls', '0'], ['--calls', '7'], ['--agent-seconds', '0'],
                        ['--seconds', '180'], ['--tuner-seconds', '0']):
            with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.main(['--model', 'resnet20_silu', '--out', '/fresh/new', '--no-tuner', *invalid])

    def test_eligible_candidate_skips_tuning_and_retains_final_confirmation(self):
        with TemporaryDirectory() as directory:
            state = self._agent_case(Path(directory), tuner=False)
        self.assertEqual(state['phase'], 'done')
        self.assertEqual(state['calls'], 1)
        self.assertFalse(state['has_tuned'])
        self.assertEqual(state['tune_cache'], {})
        self.assertEqual([candidate['origin'] for candidate in state['pool']], ['llm'])
        self.assertIsNone(state['history'][0]['tuned'])
        self.assertEqual(state['confirmation']['selected']['origin'], 'llm')
        with TemporaryDirectory() as directory:
            enabled = self._agent_case(Path(directory), tuner=True, confirm=False)
        self.assertEqual(enabled['phase'], 'tune')
        self.assertTrue(enabled['has_tuned'])

    def test_disabled_tuning_preserves_rejection_and_original_quality_policy(self):
        with TemporaryDirectory() as directory:
            state = self._agent_case(Path(directory), tuner=False, feasible=False)
        self.assertEqual(state['phase'], 'done')
        self.assertIsNone(state['confirmation']['selected'])
        self.assertEqual(state['pool'], [])
        row = dict(iteration=1, origin='llm', passed=True, diverged=False,
                   max_u=.5, max_u_calibrated=.5, max_u_analytic=0., acc_drop_pp=.500001)
        selected = dict(iteration=1, origin='llm')
        self.assertFalse(common.final_quality_acceptance(dict(selected=selected, evals=[row]))['passed'])
        row['acc_drop_pp'] = .5
        self.assertTrue(common.final_quality_acceptance(dict(selected=selected, evals=[row]))['passed'])

    def test_r30_agent_adapter_forwards_the_effective_bool_to_r28(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                args = Namespace(model='resnet20_silu', model_info=Path('/fresh/info'),
                    calibration=Path('/fresh/cal'), work_dir=Path('/fresh/agent'),
                    ledger=Path('/budget/calls'), allowance=Path('/budget/allowance'),
                    calls=6, tuner_seconds=600, memory_gib=6, tuner=enabled)
                result = {}
                with patch.object(common, 'worker', return_value=0) as worker:
                    entry._agent(args, None, result)
                self.assertIs(worker.call_args.args[0].tuner, enabled)
                self.assertIs(result['tuner_enabled'], enabled)
                self.assertEqual(result['variant'], entry.VARIANT if enabled else entry.NO_TUNER_VARIANT)

    def test_phase_result_records_the_new_variant_without_reusing_an_existing_phase(self):
        from pt_agent.experiments import r29_local_compile_matrix
        for enabled in (True, False):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                root = Path(directory)
                metadata = root/'metadata.json';metadata.write_text('{}')
                args = Namespace(work_dir=root/'phase', model='resnet20_silu',
                    worker_phase='agent', metadata_template=metadata, tuner=enabled)
                with patch.object(entry, '_threads'), patch.object(entry, '_agent'), \
                        patch.object(r29_local_compile_matrix, 'native_metadata_commands', return_value=nullcontext()):
                    self.assertEqual(entry.worker(args), 0)
                    with self.assertRaisesRegex(ValueError, 'Preserve'):
                        entry.worker(args)
                result = json.loads((args.work_dir/'phase_result.json').read_text())
                self.assertIs(result['tuner_enabled'], enabled)
                self.assertEqual(result['variant'], entry.VARIANT if enabled else entry.NO_TUNER_VARIANT)
                for field in ('report_samples', 'keygen_runs', 'real_ckks_runs', 'extra_integrity_checks'):
                    self.assertEqual(result[field], 0)


if __name__ == '__main__':
    unittest.main()
