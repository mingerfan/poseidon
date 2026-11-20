"""Stdlib fake replay/state tests: no model, provider, fitting or compiler."""

from argparse import Namespace
import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent.experiments import qwen_guarded_replay as entry
from pt_agent.experiments.r10_paper_metrics import StageRecorder
from pt_agent.planners.transformer_session import load_state, save_state


def resource_fixture(args):
    scope = 'Cold continuation from saved original import; historical import/reference clocks excluded'
    available = args.available
    return available < GIB, scope


class GuardedReplayChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    def fixture(self):
        case = self.root/'partial'
        session = case/'session'
        messages = [dict(role='system', content='Immutable project schema'), dict(role='user', content='Immutable public model')]
        for number in (*range(1, 9), 10, 11, 12):
            fb = dict(stage='approximation' if number == 8 else 'parse', feasible=False)
            if number == 8:
                fb['errors'] = [dict(code='E_APPROX_NUMERICAL', method='remez', degree=31, domain=[1., 2.]) for _ in range(43)]
            messages.extend([dict(role='assistant', content='Original candidate'),
                dict(role='user', content=json.dumps(dict(iteration=number, feedback=fb)))])
        state = dict(phase='evaluate', iteration=13, calls=12, max_iters=15, pending=None, failed=None,
            tuner=False, prompt_version='r8-v11', engine_profile='tensor', placement='program-plan',
            confirm_rule='c2', tune_rule='v8-6.25', tune_search='sweep', tune_policy='all',
            prompt_variant_id=entry.bridge.PROMPT_ID, algorithm_variant=entry.bridge.VARIANT,
            presentation_variant=entry.bounded.PRESENTATION_ID, method_guidance_version=entry.bounded.METHOD_GUIDANCE_VERSION,
            recovery=dict(inherited_attempts=9, pinned_error_counts={'8': 43}),
            history=[dict(iteration=i, stage='parse', feasible=False) for i in range(1, 13)],
            pool=[], best=None, best_program=None, first_feasible=None, stale=0, has_tuned=False, llm_min=None,
            records={}, messages=messages, prompt_tokens=100, completion_tokens=50, elapsed_s=10.)
        source = dict(model_info=str(self.root/'original-model-info'), split_indices=dict(heldout=list(range(20))),
            recovery=dict(inherited_attempts=9, pinned_error_counts={'8': 43}))
        self.write(case/'result.json', dict(round=entry.bounded.ROUND, mode=entry.bounded.MODE,
            model=entry.original.MODEL, status='error', candidate_frozen=False, source=source,
            termination=dict(cause='own_budget_guard'), api_registration=dict(registered=[508, 509, 510, 511])))
        self.write(case/'completion.json', dict(status='error', registry_persisted=True, required_artifacts_persisted=True))
        self.write(case/'guard-failure.json', dict(cause='own_budget_guard'))
        save_state(session/'SESSION.json', state)
        source_text = '@fhe.program(slots=16384)\ndef candidate(x):\n    return x\n'
        reply = dict(content=json.dumps(dict(program=source_text)), finish_reason='stop',
            usage=dict(prompt_tokens=11, completion_tokens=7), seconds=42.)
        for number in range(1, 14):
            request = dict(purpose=f'agent-iter-{number}', messages=[])
            if number == 13:
                request['messages'] = entry.bounded.project_messages(messages, pinned_errors={8: 43})[0]
            self.write(session/'calls'/f'call-{number:04d}.request.json', request)
            if number != 9:
                self.write(session/'calls'/f'call-{number:04d}.response.json', reply)
                path = session/'sources'/f'iteration-{number}.ptdsl'
                path.parent.mkdir(exist_ok=True)
                path.write_text(source_text)
        self.write(session/'calls/call-0009.error.json', dict(error='transport_worker_failed', diagnostics=dict(stage='setup')))
        trace = [dict(iteration=i, fb=dict(stage='parse')) for i in (10, 11, 12)]
        (session/'agent_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))
        for name in ('calibration.json', 'unit-width-bridge.json', 'generation-input.json'):
            self.write(case/name, dict(original23calibration=True))
        out = self.root/'new'
        out.mkdir()
        work = out/'phases/003-step'
        work.mkdir(parents=True)
        args = Namespace(partial_from=case, resume_from=self.root/'original-nine', out=out, work_dir=work,
            session=out/'session', calibration=out/'calibration.json', codegen=out/'codegen',
            memory_gib=24., rss_gib=16., minimum_available_gib=1.25, case_gib=32, seconds=10800,
            server_mode=False, make_server_job=None, worker_phase='step')
        return args, state, reply, source

    def original_step(self, *, failure=False):
        """Exercise original evaluate/finish control flow with a pure fake evaluator."""
        path = Path(entry.original.__file__).parents[1]/'planners/agent.py'
        tree = ast.parse(path.read_text())
        wanted = {'_evaluate_step', '_finish_iteration'}
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
        namespace = dict(__package__='pt_agent.planners', json=json, Path=Path,
            LLMError=type('UncalledProviderError', (Exception,), {}),
            apply_placement=lambda text, ctx: text, PRE_TUNE_RULE='prior-rule',
            feedback_message=lambda fb, iteration, best, **kw: json.dumps(dict(iteration=iteration, feedback=fb)))
        def evaluate(source, ctx, **kwargs):
            if failure:
                raise RuntimeError('local-resource-stop')
            return dict(stage='approximation', feasible=False, exact=None, cost=None, validation=None,
                        errors=[dict(reason='original numerical rejection')])
        namespace['evaluate'] = evaluate
        exec(compile(selected, str(path), 'exec'), namespace)
        def step(ctx, state, *, trace_path, client=None, reply=None):
            self.assertIsNone(client)
            self.assertIsNotNone(reply)
            namespace['_evaluate_step'](ctx, state, client, trace_path, reply=reply)
        return step

    def replay(self, args, *, failure=False):
        result = {}
        with patch.object(entry.bridge, '_context', return_value=SimpleNamespace()), \
                patch.object(entry.bounded, '_agent_functions', return_value={'agent_step': self.original_step(failure=failure)}):
            changed = entry.replay_once(args, StageRecorder(args.work_dir/'events.jsonl'), result)
        return changed, result

    def test_inspect_exact_partial_and_reject_changed_source_or_checkpoint(self):
        args, _, _, source = self.fixture()
        metadata, state, _ = entry.inspect_partial(args.partial_from, source=source)
        self.assertEqual(metadata['remaining_possible_HTTP'], [14, 15])
        self.assertEqual(state['calls'], 12)
        with self.assertRaisesRegex(ValueError, 'source field'):
            entry.inspect_partial(args.partial_from, source=dict(source, model_info='foreign'))
        state['calls'] = 13
        save_state(args.partial_from/'session/SESSION.json', state)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            entry.inspect_partial(args.partial_from)

    def test_inspect_rejects_missing_reply_frozen_or_foreign_request(self):
        args, _, _, _ = self.fixture()
        path = args.partial_from/'session/calls/call-0013.request.json'
        saved = read = json.loads(path.read_text())
        self.write(path, dict(read, messages=[]))
        with self.assertRaisesRegex(ValueError, 'projection'):
            entry.inspect_partial(args.partial_from)
        self.write(path, saved)
        self.write(args.partial_from/'session/FROZEN.json', {})
        with self.assertRaisesRegex(ValueError, 'unfrozen'):
            entry.inspect_partial(args.partial_from)

    def test_seed_full_inherited_evidence_and_committed_only_live_sources(self):
        args, state, reply, _ = self.fixture()
        entry.seed_partial(args)
        self.assertEqual(load_state(args.session/'SESSION.json'), state)
        self.assertFalse((args.session/'sources/iteration-13.ptdsl').exists())
        self.assertEqual(json.loads((args.out/'inherited/session/calls/call-0013.response.json').read_text()), reply)
        self.assertTrue((args.out/'inherited/session/sources/iteration-13.ptdsl').is_file())
        self.assertEqual(len(list((args.session/'calls').glob('call-*.request.json'))), 13)
        self.assertEqual(len(list((args.session/'sources').glob('*.ptdsl'))), 11)
        entry.seed_partial(args)
        self.assertEqual(load_state(args.partial_from/'session/SESSION.json'), state)

    def test_saved_reply_original_agent_flow_commits13_without_HTTP(self):
        args, initial, _, _ = self.fixture()
        entry.seed_partial(args)
        changed, result = self.replay(args)
        state = load_state(args.session/'SESSION.json')
        self.assertTrue(changed)
        self.assertEqual((state['calls'], state['iteration'], state['phase']), (13, 14, 'evaluate'))
        self.assertEqual((state['prompt_tokens'], state['completion_tokens']), (111, 57))
        self.assertEqual(state['history'][-1]['stage'], 'approximation')
        self.assertEqual(result['api_calls'], 0)
        self.assertEqual(result['new_provider_requests_for_13'], 0)
        self.assertEqual(len(list((args.session/'calls').glob('call-*.request.json'))), 13)
        self.assertEqual(load_state(args.partial_from/'session/SESSION.json'), initial)
        self.assertTrue((args.session/'sources/iteration-13.ptdsl').is_file())
        self.assertEqual(len(state['messages']), len(initial['messages'])+2)

    def test_partial_calls13_never_replaces_authoritative_calls12(self):
        args, initial, _, _ = self.fixture()
        entry.seed_partial(args)
        with self.assertRaisesRegex(RuntimeError, 'local-resource-stop'):
            self.replay(args, failure=True)
        self.assertEqual(load_state(args.session/'SESSION.json'), initial)
        partial = load_state(args.work_dir/'partial-local-evaluate-state.json')
        self.assertEqual(partial['calls'], 13)
        self.assertEqual(partial['iteration'], 13)
        self.assertEqual(read_json(args.work_dir/'local-replay-error.json')['new_HTTP'], 0)
        with self.assertRaisesRegex(ValueError, 'isolated new case'):
            self.replay(args)

    def test_reconcile_commit_marker_never_repeats_agent_or_usage(self):
        args, _, _, _ = self.fixture()
        entry.seed_partial(args)
        self.replay(args)
        marker = read_json(args.session/'guarded-replay.json')
        self.write(args.session/'guarded-replay.json', dict(marker, status='pending_local_replay'))
        state = load_state(args.session/'SESSION.json')
        with patch.object(entry.bridge, '_context') as context:
            self.assertFalse(entry.replay_once(args, StageRecorder(args.work_dir/'reconcile.jsonl'), {}))
            context.assert_not_called()
        self.assertEqual(load_state(args.session/'SESSION.json'), state)
        self.assertEqual(read_json(args.session/'guarded-replay.json')['status'], 'committed')

    def test_new_provider_view_numbering14_then15_and16_rejection(self):
        args, _, _, _ = self.fixture()
        entry.seed_partial(args)
        self.replay(args)
        state = load_state(args.session/'SESSION.json')
        class FakeProvider:
            calls = 0
            def chat(self, messages, *, purpose):
                self.calls += 1
                return {'content': 'saved fake reply'}
        provider = FakeProvider()
        client = entry.bounded.ProjectedClient(provider, args.work_dir/'future-presentation',
            pinned_errors={8: 43}, attempt_offset=state['calls'])
        client.chat(state['messages'], purpose='agent-iter-14')
        client.chat(state['messages'], purpose='agent-iter-15')
        with self.assertRaises(entry.bounded.PresentationError):
            client.chat(state['messages'], purpose='agent-iter-16')
        self.assertEqual(provider.calls, 2)
        self.assertTrue((args.work_dir/'future-presentation/attempt-0014.full-messages.json').is_file())
        self.assertTrue((args.work_dir/'future-presentation/attempt-0015.full-messages.json').is_file())

    def test_prompt_has_no_API_and_next_phase_delegates_only_after_commit(self):
        args, _, _, _ = self.fixture()
        args.worker_phase = 'prompt'
        with patch.object(entry.bounded, '_agent_phase') as delegated:
            result = {}
            entry._agent_phase(args, StageRecorder(args.work_dir/'prompt.jsonl'), result)
            self.assertFalse(result['api_required'])
            delegated.assert_called_once()
        args.worker_phase = 'step'
        self.replay(args)
        with patch.object(entry.bounded, '_agent_phase') as delegated:
            entry._agent_phase(args, StageRecorder(args.work_dir/'next.jsonl'), {})
            delegated.assert_called_once()

    def test_original23_calibration_is_copied_without_context_or_provider(self):
        args, _, _, _ = self.fixture()
        with patch.object(entry.bridge, '_context') as context:
            result = {}
            entry._calibrate(args, None, result)
        context.assert_not_called()
        self.assertEqual(result['calibration_samples'], 23)
        self.assertEqual(result['fresh_calibration_forwards'], 0)
        for name in ('calibration.json', 'unit-width-bridge.json', 'generation-input.json'):
            self.assertEqual(read_json(args.out/name), read_json(args.partial_from/name))

    def test_resources_default_explicit_and_invalid_values(self):
        parser = entry.parser()
        args = parser.parse_args(['--resume-from', 'old', '--partial-from', 'partial'])
        self.assertEqual(entry.limits(args)['minimum_available_gib'], 1.25)
        for field, value in (('rss_gib', 14), ('rss_gib', True), ('rss_gib', float('nan')),
                ('memory_gib', 32), ('minimum_available_gib', 1.), ('case_gib', True), ('seconds', 10801)):
            altered = copy.copy(args)
            setattr(altered, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                entry.limits(altered)

    def test_private_monitor_reserve_and_clock_only_delta(self):
        namespace = dict(GIB=1024)
        delegated = entry._delegate(resource_fixture, namespace)
        self.assertEqual(delegated(SimpleNamespace(available=1200, minimum_available_gib=1.25)), (True, entry.CLOCK_SCOPE))
        self.assertEqual(resource_fixture(SimpleNamespace(available=1200))[0], False)
        self.assertEqual(delegated(SimpleNamespace(available=1300, minimum_available_gib=1.25))[0], False)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            entry._delegate(entry.read, {})

    def test_namespace_exports_and_main_resource_validation_without_execution(self):
        functions = entry._namespace()
        self.assertIs(functions['_program'], entry.bounded._program)
        self.assertIs(entry.session_outcome, entry.bounded.session_outcome)
        execute = Mock(return_value=0)
        functions['execute'] = execute
        argv = ['--out', str(self.root/'fake'), '--through', 'metadata', '--resume-from', str(self.root/'old'),
                '--partial-from', str(self.root/'partial')]
        self.assertEqual(functions['main'](argv), 0)
        self.assertEqual(execute.call_args.args[0].rss_gib, 16)
        with self.assertRaises(SystemExit):
            functions['main'](argv+['--rss-gib', 'nan'])

    def test_phase_argv_propagates_partial_identity_and_exact_resources(self):
        args, _, _, _ = self.fixture()
        for name in ('model', 'model_info', 'inputs', 'source_rows', 'metadata_template', 'frontend', 'profile',
                     'linked_result', 'coefficients', 'ledger', 'allowance', 'intake_python'):
            setattr(args, name, self.root/name)
        argv = entry.phase_command(args, 'step', args.work_dir)
        worker_argv = argv[argv.index('--')+1:]
        for key, value in (('--partial-from', str(args.partial_from)), ('--resume-from', str(args.resume_from)),
                ('--memory-gib', '24.0'), ('--rss-gib', '16.0'), ('--minimum-available-gib', '1.25'), ('--seconds', '10800')):
            self.assertEqual(worker_argv[worker_argv.index(key)+1], value)
        self.assertEqual(argv[argv.index('--seconds')+1], str(entry.original.PHASE_SECONDS['step']+60))
        self.assertEqual(argv.count(entry.MODULE), 1)
        self.assertEqual(argv[argv.index('--worker-phase')+1], 'step')
        with self.assertRaises(SystemExit):
            entry.parser().parse_args(['--resume-from', 'a', '--partial-from', 'b', '--through', 'compile'])

    def test_path_protection_precedes_any_write(self):
        args, _, _, _ = self.fixture()
        args.out = args.partial_from/'forbidden'
        with self.assertRaisesRegex(ValueError, 'outside both'):
            entry._namespace()['execute'](args)
        self.assertFalse(args.out.exists())
        args.out = args.resume_from/'forbidden'
        with self.assertRaises(ValueError):
            entry.check_write_paths(args)


def read_json(path):
    return json.loads(path.read_text())


GIB = 1024


if __name__ == '__main__':
    unittest.main()
