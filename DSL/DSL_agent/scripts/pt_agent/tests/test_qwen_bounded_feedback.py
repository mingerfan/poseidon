"""Public JSON/synthetic grammar tests; zero HTTP, fitting or trained models."""
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pt_agent import paths
from pt_agent.experiments import qwen_bounded_feedback as entry
from pt_agent.tests._artifacts import require


def format_feedback(feedback,iteration,best,*,tuned=None):
    return json.dumps(dict(iteration=iteration,feedback=feedback,best_feasible_so_far=best,tuned=tuned))


def numerical_feedback(count=43):
    history=[dict(iteration=i+1,max_error=.25/(i+1),E=-.1/(i+1),criterion='same original convergence rule') for i in range(50)]
    return dict(stage='approximation',feasible=False,exact=None,cost=None,validation=None,
        errors=[dict(code='E_APPROX_NUMERICAL',line=29,call_lines=[100+i],op_id=i,role='act',
            fn='silu',fit='absolute',method='remez',degree=127,deg0=None,iters=None,
            domain=[-19.,20.],target=None,k=None,rho=None,reason='degenerate',
            exception_type='ApproximationError',message='Remez did not converge',iterations=50,
            max_error=.001+i*.0001,abs_E=.0001,history=copy.deepcopy(history),line_text='y = act(x,...)')
            for i in range(count)])


def conversation(*,response_chars=30):
    result=[dict(role='system',content='Original immutable schema'),dict(role='user',content='Original immutable model structure')]
    for i in range(1,9):
        fb=numerical_feedback() if i==8 else dict(stage='parse',feasible=False,error='Original syntax error')
        result.extend([dict(role='assistant',content='x'*response_chars),
            dict(role='user',content=format_feedback(fb,i,None))])
    return result


class BoundedFeedbackChecks(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    @staticmethod
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

    def old_case(self):
        case=self.root/'old';session=case/'session'
        state=dict(phase='done',failed='transport_worker_failed',calls=9,iteration=9,max_iters=10,
            prompt_variant_id=entry.bridge.PROMPT_ID,algorithm_variant=entry.bridge.VARIANT,
            prompt_version='r8-v11',placement='program-plan',confirm_rule='c2',tuner=False,
            pool=[],best=None,pending=None,confirmation=dict(selected=None),messages=conversation(),
            history=[dict(iteration=i,stage='llm' if i==9 else 'parse',feasible=False) for i in range(1,10)],records={})
        source=dict(model_info=str(self.root/'model/model_info.json'),original_model=str(self.root/'original'),
            samples=str(self.root/'samples'),source_rows=str(self.root/'rows'),previous_linked_math=str(self.root/'linked'),
            split_indices=dict(calibration=list(range(23))),algorithm_variant=entry.bridge.VARIANT,
            maximum_square_steps=11,maximum_newton_iterations=7)
        self.write(case/'result.json',dict(status='error',round=entry.bridge.ROUND,mode=entry.bridge.MODE,source=source,
            phases=[dict(phase='calibration',worker=dict(calibration_completed=True,stable_softmax_completed=552,
                bridge_calibration_completed=True))]))
        self.write(session/'SESSION.json',state)
        rows=[dict(iteration=i,fb=numerical_feedback() if i==8 else dict(stage='llm' if i==9 else 'parse')) for i in range(1,10)]
        (session/'agent_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for i in range(1,10):
            self.write(session/'calls'/f'call-{i:04d}.request.json',dict(messages=[],purpose=f'agent-iter-{i}'))
            if i<9:
                self.write(session/'calls'/f'call-{i:04d}.response.json',dict(content='{"program":"original saved candidate"}'))
                path=session/'sources'/f'iteration-{i}.ptdsl';path.parent.mkdir(exist_ok=True);path.write_text('original saved syntax\n')
        self.write(session/'calls/call-0009.error.json',dict(error='transport_worker_failed',diagnostics=dict(stage='setup')))
        for name in ('calibration.json','unit-width-bridge.json','generation-input.json'):self.write(case/name,dict(source=name))
        return case,state,source

    def test_all43_key_diagnostics_and_original_rejection_survive_history_summary(self):
        original=numerical_feedback();saved=copy.deepcopy(original);shown=entry.compact_feedback(original)
        self.assertEqual(original,saved);self.assertFalse(shown['feasible']);self.assertEqual(shown['stage'],'approximation')
        self.assertEqual(len(shown['errors']),43)
        for full,row in zip(original['errors'],shown['errors']):
            for key in entry.KEY_FIELDS:
                if key in full:self.assertEqual(row[key],full[key])
            self.assertEqual(row['history_count'],50)
            self.assertEqual(row['history_head'],full['history'][:2]);self.assertEqual(row['history_tail'],full['history'][-2:])
        self.assertLess(entry._bytes(shown),entry.FEEDBACK_BYTES)

    def test_nonfinite_classification_and_nulls_remain_unmodified(self):
        fb=numerical_feedback(1);row=fb['errors'][0]
        row.update(reason='nonfinite',max_error='inf',abs_E=None,message='nonfinite coefficients')
        shown=entry.compact_feedback(fb)['errors'][0]
        self.assertEqual(shown['reason'],'nonfinite');self.assertEqual(shown['max_error'],'inf');self.assertIsNone(shown['abs_E'])
        self.assertFalse(entry.compact_feedback(fb)['feasible'])

    def test_successful_feedback_semantics_and_best_tuned_values_are_preserved(self):
        fb=dict(stage='complete',feasible=True,exact=dict(ok=True),cost=dict(boots_pred=8),
            validation=dict(agree=100.,max_u=.8),domain=dict(passed=True))
        presenter=entry.FeedbackPresenter(self.root,format_feedback);best=dict(iteration=2);tuned=dict(cost_after=dict(boots_pred=7))
        shown=json.loads(presenter(fb,3,best,tuned=tuned))
        self.assertEqual(shown['feedback'],fb);self.assertEqual(shown['best_feasible_so_far'],best);self.assertEqual(shown['tuned'],tuned)
        self.assertEqual(json.loads((self.root/'iteration-0003.full.json').read_text())['feedback'],fb)

    def test_full_local_diagnostic_is_saved_before_formatter_and_large_output_failure(self):
        fb=numerical_feedback();path=self.root/'iteration-0008.full.json'
        def formatter(*args,**kwargs):
            self.assertEqual(json.loads(path.read_text())['feedback'],fb)
            return 'x'*(entry.FEEDBACK_BYTES+1)
        with self.assertRaises(entry.PresentationError):entry.FeedbackPresenter(self.root,formatter)(fb,8,None)
        self.assertEqual(len(json.loads(path.read_text())['feedback']['errors'][0]['history']),50)

    def test_local_artifact_failure_prevents_formatter_or_client_fallback(self):
        formatter=Mock()
        with patch.object(entry,'_write_new',side_effect=OSError('fixture disk full')):
            with self.assertRaises(OSError):entry.FeedbackPresenter(self.root,formatter)(numerical_feedback(),8,None)
        formatter.assert_not_called()

    def test_historical_feedback_projection_is_idempotent_and_keeps_all_pinned43(self):
        messages=conversation();saved=copy.deepcopy(messages)
        shown,record=entry.project_messages(messages,pinned_errors={8:43})
        again,_=entry.project_messages(shown,pinned_errors={8:43})
        self.assertEqual(messages,saved);self.assertEqual(shown,again);self.assertEqual(shown[:2],messages[:2])
        errors=json.loads(shown[-1]['content'])['feedback']['errors'];self.assertEqual(len(errors),43)
        self.assertEqual(record['removed_pairs'],[]);self.assertLessEqual(record['public_messages_JSON_bytes'],entry.REQUEST_BYTES)

    def test_future_response_growth_drops_only_recorded_complete_old_pairs(self):
        messages=conversation(response_chars=65000)
        for i in range(9,15):messages.extend([dict(role='assistant',content='y'*65000),
            dict(role='user',content=format_feedback(dict(stage='parse',feasible=False,error='invalid'),i,None))])
        shown,record=entry.project_messages(messages,pinned_errors={8:43})
        self.assertTrue(record['removed_pairs']);self.assertEqual(shown[:2],messages[:2])
        kept=record['retained_message_indices'];self.assertTrue(all(i in kept for i in (16,17,len(messages)-4,len(messages)-3,len(messages)-2,len(messages)-1)))
        for row in record['removed_pairs']:
            self.assertEqual(row['message_indices'][1],row['message_indices'][0]+1)
            self.assertEqual(messages[row['message_indices'][0]]['role'],'assistant')
            self.assertTrue(row['reason'])
        pinned=[entry._feedback_message(message) for message in shown if entry._feedback_message(message)]
        self.assertEqual(len(next(row for row in pinned if row['iteration']==8)['feedback']['errors']),43)

    def test_missing_pinned_error_or_partial_pairs_are_explicit_failures(self):
        for messages,pinned in ((conversation(),{8:44}),(conversation()[:-1],{8:43})):
            with self.assertRaises(entry.PresentationError):entry.project_messages(messages,pinned_errors=pinned)

    def test_budget_includes_existing_settings_and_maximum_loader_key_without_reading_credentials(self):
        from pt_agent import llm
        with patch.object(llm,'load_api_key',side_effect=AssertionError('Credentials must remain unread')):
            shown,record=entry.project_messages(conversation(),pinned_errors={8:43})
        self.assertEqual(record['wire_size_upper_bound_bytes'],entry.wire_size_upper_bound(shown))
        self.assertGreater(record['wire_size_upper_bound_bytes'],record['public_messages_JSON_bytes']+512)
        self.assertLessEqual(record['wire_size_upper_bound_bytes'],entry.REQUEST_BYTES)

    def test_required_oversize_context_stops_before_underlying_client(self):
        client=SimpleNamespace(calls=0,chat=Mock());messages=conversation();messages[0]['content']='s'*entry.REQUEST_BYTES
        wrapped=entry.ProjectedClient(client,self.root,pinned_errors={8:43})
        with self.assertRaises(entry.PresentationError):wrapped.chat(messages,purpose='fixture')
        client.chat.assert_not_called()
        self.assertFalse(json.loads((self.root/'attempt-0010.projection-error.json').read_text())['HTTP_sent'])
        self.assertEqual(json.loads((self.root/'attempt-0010.full-messages.json').read_text())['messages'],messages)

    def test_client_projects_once_and_records_cumulative_attempt11_not_replayed9(self):
        client=SimpleNamespace(calls=0)
        def chat(messages,*,purpose):
            client.calls+=1;self.assertEqual(len(json.loads(messages[-1]['content'])['feedback']['errors']),43)
            return dict(content='original provider reply')
        client.chat=Mock(side_effect=chat)
        wrapped=entry.ProjectedClient(client,self.root,pinned_errors={8:43},attempt_offset=10)
        self.assertEqual(wrapped.chat(conversation(),purpose='agent-iter-11')['content'],'original provider reply')
        record=json.loads((self.root/'attempt-0011.projection.json').read_text())
        self.assertEqual(record['cumulative_attempt'],11);self.assertEqual(wrapped.calls,1);client.chat.assert_called_once()
        self.assertFalse((self.root/'attempt-0009.projection.json').exists())

    def test_recovery_attempt_allowance_is_explicitly_capped_at15(self):
        client=SimpleNamespace(calls=0,chat=Mock())
        with self.assertRaises(entry.PresentationError):
            entry.ProjectedClient(client,self.root,pinned_errors={8:43},attempt_offset=15).chat(conversation(),purpose='no16')
        client.chat.assert_not_called()

    def test_source_gate_requires_original_terminal_and_all_provider_history(self):
        case,state,source=self.old_case();metadata,loaded=entry.inspect_recovery(case,source=source)
        self.assertEqual(loaded,state);self.assertEqual(metadata['inherited_attempts'],9)
        self.assertEqual(metadata['inherited_provider_responses'],8);self.assertEqual(metadata['maximum_new_attempts'],6)
        self.assertEqual(metadata['pinned_error_counts'],{8:43})
        source['model_info']=str(self.root/'foreign/model_info.json')
        with self.assertRaises(ValueError):entry.inspect_recovery(case,source=source)
        self.write(case/'session/calls/call-0009.response.json',dict(content='forbidden replay'))
        with self.assertRaises(ValueError):entry.inspect_recovery(case)

    def test_seed_retains_original_done_failure_calls_and_creates_explicit_new_branch(self):
        case,state,_=self.old_case();args=Namespace(resume_from=case,session=self.root/'new/session')
        entry.seed_recovery(args);new=json.loads((args.session/'SESSION.json').read_text())
        self.assertEqual(json.loads((case/'session/SESSION.json').read_text()),state)
        self.assertEqual(json.loads((args.session/'inherited-SESSION.json').read_text()),state)
        self.assertEqual(new['calls'],9);self.assertEqual(new['history'],state['history'])
        self.assertEqual((new['phase'],new['failed'],new['iteration'],new['max_iters']),('evaluate',None,10,15))
        self.assertTrue(new['messages'][0]['content'].startswith(state['messages'][0]['content']))
        self.assertEqual(new['messages'][1:],state['messages'][1:]);self.assertIn(entry.SYNTAX_GUIDANCE,new['messages'][0]['content'])
        self.assertIn(entry.METHOD_GUIDANCE,new['messages'][0]['content'])
        self.assertEqual(new['method_guidance_version'],entry.METHOD_GUIDANCE_VERSION)
        self.assertEqual(len(list((args.session/'calls').glob('*.request.json'))),9)
        self.assertEqual(len(list((args.session/'calls').glob('*.response.json'))),8)
        outcome=entry.session_outcome(args.session)
        self.assertEqual(outcome['new_attempts'],0);self.assertFalse(outcome['done'])

    def test_supported_method_guidance_preserves_original_diagnostics_and_policy(self):
        case,state,_=self.old_case();args=Namespace(resume_from=case,session=self.root/'new/session')
        entry.seed_recovery(args)
        original=json.loads((case/'session/SESSION.json').read_text())
        saved=json.loads((args.session/'SESSION.json').read_text())
        record=json.loads((args.session/'recovery.json').read_text())
        self.assertEqual(original,state);self.assertEqual(saved['history'],state['history'])
        self.assertEqual(saved['messages'][1:],state['messages'][1:])
        self.assertEqual((saved['calls'],saved['max_iters']),(9,15))
        self.assertEqual(record['method_guidance_version'],entry.METHOD_GUIDANCE_VERSION)
        for contract in ('Remez','fit="relative"','lower explicit deg0','up to7','method="cheb"',
                         'max|u|<=0.90','minimum domain','confirmation or report'):
            self.assertIn(contract,entry.METHOD_GUIDANCE)
        projected,observation=entry.project_messages(saved['messages'],pinned_errors={8:43})
        self.assertEqual(projected[:2],saved['messages'][:2])
        self.assertLessEqual(observation['wire_size_upper_bound_bytes'],entry.REQUEST_BYTES)

    def test_cached_calibration_reports_reuse_with_zero_new_forwards(self):
        case,_,_=self.old_case();target=self.root/'new';target.mkdir()
        result={};args=Namespace(resume_from=case,calibration=target/'calibration.json')
        entry._calibrate(args,None,result)
        for name in ('calibration.json','unit-width-bridge.json','generation-input.json'):
            self.assertEqual(json.loads((target/name).read_text()),json.loads((case/name).read_text()))
        self.assertTrue(result['calibration_cache']['calibration_cache_reused']);self.assertEqual(result['fresh_calibration_forwards'],0)
        self.assertEqual(result['calibration_forward_completed'],0)

    def test_private_namespace_and_phase_command_leave_existing_modules_unchanged(self):
        old=(entry.bridge.MODULE,entry.original.MODULE,entry.bridge._agent_phase,entry.original.parser)
        functions=entry._namespace()
        self.assertIs(functions['main'].__globals__,functions);self.assertIs(functions['parser'],entry.parser)
        names=('model','model_info','inputs','source_rows','metadata_template','frontend','profile',
            'linked_result','coefficients','calibration','session','codegen','ledger','allowance','intake_python','resume_from')
        args=Namespace(**{name:self.root/name for name in names},memory_gib=24,case_gib=32,server_mode=False)
        argv=entry.phase_command(args,'step',self.root/'phase')
        self.assertIn(entry.MODULE,argv);self.assertEqual(argv[argv.index('--resume-from')+1],str(args.resume_from))
        self.assertEqual(old,(entry.bridge.MODULE,entry.original.MODULE,entry.bridge._agent_phase,entry.original.parser))

    def test_outer_and_worker_writes_cannot_overlap_inherited_source_case(self):
        source=self.root/'old'
        for output in (source,source/'nested',self.root):
            with self.subTest(output=output),self.assertRaises(ValueError):
                entry.check_write_paths(Namespace(resume_from=source,out=output,worker_phase=None))
        with self.assertRaises(ValueError):
            entry.check_write_paths(Namespace(resume_from=source,worker_phase='step',
                work_dir=self.root/'new/work',session=source/'session',codegen=None,calibration=None))
        with self.assertRaises(ValueError):entry.seed_recovery(Namespace(resume_from=source,session=source/'session'))
        entry.check_write_paths(Namespace(resume_from=source,out=self.root/'fresh',worker_phase=None))

    def test_private_feedback_hook_reuses_existing_evaluator_and_formatter_namespace(self):
        evaluate=object();functions={'evaluate':evaluate,'feedback_message':format_feedback}
        with patch.object(entry.bridge,'_agent_functions',return_value=functions),patch.object(
                entry.resource_policy,'adapt_agent_functions',side_effect=lambda value:value):
            observed=entry._agent_functions(self.root)
        self.assertIs(observed['evaluate'],evaluate);self.assertIsInstance(observed['feedback_message'],entry.FeedbackPresenter)
        self.assertIs(observed['feedback_message'].formatter,format_feedback)

    def test_private_outer_clock_label_records_cached_scope_without_changing_old_entry(self):
        module=ModuleType('trusted_clock_fixture')
        exec('def label(): return '+repr(entry.OLD_CLOCK_SCOPE),module.__dict__)
        delegated=entry._recovery_clock_label(module.label)
        self.assertEqual(delegated(),entry.CLOCK_SCOPE);self.assertEqual(module.label(),entry.OLD_CLOCK_SCOPE)
        self.assertIs(delegated.__globals__,module.label.__globals__)
        with self.assertRaises(ValueError):entry._recovery_clock_label(lambda:'Unexpected upstream label')


GRAMMAR_PYTHON=paths.WORKSPACE/'baselines/orbit-venv/bin/python'


@require(GRAMMAR_PYTHON)
class GrammarInterpreterChecks(unittest.TestCase):
    def test_original_parser_accepts_guidance_and_keeps_invalid_forms_rejected(self):
        macro='def signed_example(x):\n    y=shuffle(x,index=(2,3,0,1),sign=(-1,-1,1,1))\n    return y\n'
        main='@fhe.program(slots=16384)\ndef tiny(x: Cipher[1,1,4]):\n    p=pack(x,layout="token")\n    z=signed_example(p)\n    return z\n'
        invalid=(macro.replace('y=shuffle','return shuffle').replace('    return y\n',''),
            macro.replace('    y=shuffle','    index=(2,3,0,1)\n    y=shuffle'),
            macro.replace('sign=(-1,-1,1,1)','sign=(-1,1)'))
        code=('import json,sys\nfrom pt_agent.dsl.parser import parse\n'
            'from pt_agent.dsl.errors import PTDSLError\n'
            'sources=json.loads(sys.argv[1])\nassert parse(sources[0]).ops\n'
            'for source in sources[1:]:\n'
            ' try: parse(source)\n'
            ' except PTDSLError: pass\n'
            ' else: raise AssertionError("Invalid syntax unexpectedly accepted")\n'
            'print(json.dumps(dict(grammar_checks=4,passed=True)))\n')
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            observed=entry.original.run_process([str(GRAMMAR_PYTHON),'-B','-c',code,
                json.dumps([macro+main]+[source+main for source in invalid])],
                cwd=Path(__file__).resolve().parents[2],log=directory/'grammar.log',resources=directory/'resources.jsonl',
                timeout=30,mem_gib=2,env=env)
            self.assertEqual(observed['exit_code'],0,(directory/'grammar.log').read_text())
            self.assertFalse(observed['timed_out']);self.assertEqual(observed['remaining_pids'],[])
            self.assertIn('"grammar_checks": 4',(directory/'grammar.log').read_text())


SAVED=paths.RESULTS_DIR/'qwen-unit-width-newton7-agent-20261116'


@require(SAVED/'session/SESSION.json',SAVED/'session/agent_trace.jsonl')
class SavedPublicMetadataChecks(unittest.TestCase):
    def test_saved_nine_attempts_and_all43_errors_fit_without_HTTP_or_numerical_work(self):
        metadata,state=entry.inspect_recovery(SAVED)
        messages=copy.deepcopy(state['messages']);messages[0]['content']+='\n\n'+entry.recovery_guidance()
        projected,record=entry.project_messages(messages,pinned_errors=metadata['pinned_error_counts'])
        self.assertLessEqual(record['public_messages_JSON_bytes'],entry.REQUEST_BYTES)
        self.assertEqual(len(json.loads(projected[-1]['content'])['feedback']['errors']),43)
        self.assertEqual(metadata['inherited_attempts'],9);self.assertEqual(metadata['inherited_provider_responses'],8)
        self.assertEqual(messages[:2],projected[:2])


if __name__=='__main__':unittest.main()
