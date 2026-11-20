"""Retained13 transaction and original Agent dispatch with tiny metadata callbacks."""
import ast
import copy
from contextlib import contextmanager
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_retained_component_replay as entry
from pt_agent.experiments.r10_paper_metrics import StageRecorder
from pt_agent.planners.transformer_session import load_state,save_state


@dataclass
class Candidate:
    source:str
    origin:str
    iteration:int
    latency_pred_us:float
    boots_pred:int


def state():
    return dict(phase='evaluate',iteration=13,calls=12,max_iters=13,tuner=False,
        failed=None,pending=None,engine_profile='tensor',tune_search='sweep',
        tune_rule='v8-6.25',prompt_version='r8-v11',placement='program-plan',confirm_rule='c2',
        presentation_variant=entry.guarded.bounded.PRESENTATION_ID,
        method_guidance_version=entry.guarded.bounded.METHOD_GUIDANCE_VERSION,
        prompt_variant_id=entry.guarded.bridge.PROMPT_ID,algorithm_variant=entry.guarded.bridge.VARIANT,
        recovery=dict(inherited_attempts=9),prompt_tokens=100,completion_tokens=200,
        llm_min=None,best=None,best_program=None,best_iteration=None,best_llm_lines=None,
        first_feasible=None,has_tuned=False,history=[dict(iteration=i,stage='parse')for i in range(1,13)],
        messages=[dict(role='system',content='original schema'),dict(role='user',content='original model')],
        pool=[],records={},elapsed_s=0.,stale=0,tune_policy='all',tune_cache={},
        component_cost_policy=entry.POLICY,new_provider_calls_allowed=0)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class RetainedComponentChecks(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.session=self.root/'live/session';self.session.mkdir(parents=True)
        self.work=self.root/'live/work';self.work.mkdir()
        self.args=SimpleNamespace(session=self.session,work_dir=self.work,worker_phase='step')
        self.recorder=StageRecorder(self.work/'events.jsonl')

    def original_functions(self,seen):
        """Execute original trusted function ASTs, with deterministic I/O seams."""
        base=Path(entry.__file__).parents[1]
        agent=ModuleType('component_original_agent_fixture');agent.__package__='pt_agent.planners'
        ns=agent.__dict__
        ns.update(json=json,Path=Path,time=time,asdict=asdict,Candidate=Candidate,LLMError=RuntimeError,
            validate_tune_rule=lambda *a:None,apply_placement=lambda source,ctx:source,
            PRE_TUNE_RULE='v8-pre-6.11',feedback_message=lambda *a,**k:'original metadata feedback')
        names={'agent_step','_evaluate_step','_finish_iteration','_selected_record','agent_result'}
        tree=ast.parse((base/'planners/agent.py').read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],
            type_ignores=[]),str(base/'planners/agent.py'),'exec'),ns)
        tools=ModuleType('pt_agent.planners.agent_tools');tools.__package__='pt_agent.planners'
        def exact(*a,**k):seen.append('exact');return dict(ok=True)
        def cost_feedback(program,ctx,estimator,line_text):
            seen.append('cost');value=estimator(program,ctx.weights,profile=ctx.profile)
            return value,value
        def validate(*a,**k):
            seen.append('heldout');return dict(feasible=True,diverged=False,agree=1.,max_u=.5,
                max_u_calibrated=.5,max_u_analytic=0.,rows=[],boot_input_bound=None)
        tools.__dict__.update(AgentContext=object,parse=lambda text:SimpleNamespace(ops=[]),
            check=lambda *a,**k:SimpleNamespace(ok=True),_exact_check=exact,cost_feedback=cost_feedback,
            validate=validate,PROFILES={'qwen48':dict(limits={})},PTDSLError=SyntaxError,
            LoweringError=ArithmeticError,BackendError=FloatingPointError,PlacementError=ValueError)
        tree=ast.parse((base/'planners/agent_tools.py').read_text())
        evaluator=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='evaluate')
        exec(compile(ast.Module(body=[evaluator],type_ignores=[]),str(base/'planners/agent_tools.py'),'exec'),tools.__dict__)
        def bridge(original_evaluate):
            def evaluate(source,ctx,**kwargs):
                seen.append('candidate_gate');return original_evaluate(source,ctx,**kwargs)
            return evaluate
        ns['evaluate']=bridge(tools.evaluate)
        def confirm(candidates,ctx,**kwargs):
            self.assertEqual(kwargs['confirm_rule'],'c2');self.assertEqual(len(candidates),1)
            self.assertEqual(candidates[0].iteration,13);seen.append('C2')
            return dict(selected=candidates[0],evals=[dict(iteration=13,origin='llm',passed=True,tail=dict(passed=True))])
        ns['confirm_select']=confirm
        return entry.guarded.bridge._clone_namespace(agent,{})

    @contextmanager
    def numeric_seams(self):
        tuner=ModuleType('pt_agent.planners.transformer_tuner');tuner.validate_tune_search=lambda ctx:'sweep'
        interface=ModuleType('pt_agent.dsl.interface');interface.check_interface=lambda *a:dict(ok=True)
        margin=ModuleType('pt_agent.margin');margin.check_margin=lambda *a:dict(ok=True)
        failures=ModuleType('pt_agent.planners.approximation_failures');failures.program_failures=lambda *a:[]
        with patch.dict('sys.modules',{m.__name__:m for m in (tuner,interface,margin,failures)}):yield

    def prepared(self):
        save_state(self.session/'SESSION.json',state())
        inherited=self.root/'retained';inherited.mkdir()
        reply=dict(content=json.dumps(dict(program='original_paid_reply13 = 1')),usage=dict(prompt_tokens=7,completion_tokens=11),seconds=2)
        write(inherited/'calls/call-0013.response.json',reply)
        write(self.session/'guarded-replay.json',dict(status='pending_local_replay',inherited_session=str(inherited)))
        return reply

    def test_original_dispatch_component_cost_full_feedback_rank_and_one_C2(self):
        self.prepared();seen=[];functions=self.original_functions(seen)
        def cost(program,weights,**kwargs):
            seen.append('component_plan');self.assertEqual(kwargs['profile'],'qwen48')
            return dict(placement_model='component-program-plan',latency_pred_us=55.,boots_pred=4,boot_nodes=[])
        ctx=SimpleNamespace(prompt_version='r8-v11',placement='program-plan',profile='qwen48',weights={},weight_shapes={},graph=object(),model_info=object())
        with self.numeric_seams(),patch.object(entry.guarded.bridge,'_agent_functions',return_value=functions), \
                patch.object(entry.component,'estimate',side_effect=cost),patch.object(entry,'seed_partial'), \
                patch.object(entry.guarded.bridge,'_context',return_value=ctx):
            result={};entry._agent_phase(self.args,self.recorder,result)
            checkpoint=load_state(self.session/'SESSION.json')
            self.assertEqual(checkpoint['phase'],'confirm');self.assertEqual(checkpoint['calls'],13)
            self.assertEqual(checkpoint['prompt_tokens'],107);self.assertEqual(checkpoint['completion_tokens'],211)
            self.assertEqual([c['iteration']for c in checkpoint['pool']],[13]);self.assertEqual(len(checkpoint['records']),1)
            self.assertEqual([*seen],['candidate_gate','exact','cost','component_plan','heldout'])
            self.assertTrue((self.session/'full-feedback/iteration-0013.full.json').is_file())
            self.assertEqual(result['new_provider_requests_for_13'],0)
            entry._agent_phase(self.args,self.recorder,{})
        self.assertEqual(seen[-1],'C2');self.assertEqual(seen.count('C2'),1)
        self.assertEqual(entry.session_outcome(self.session)['status'],'ok')
        frozen=json.loads((self.session/'FROZEN.json').read_text());self.assertEqual(frozen['iteration'],13)
        self.assertEqual(load_state(self.session/'SESSION.json')['phase'],'done')
        self.assertEqual(json.loads((self.session/'component-c2-access.json').read_text())['status'],'completed')

    def test_cost_interruption_keeps_authoritative12_and_same_exception(self):
        self.prepared();seen=[];functions=self.original_functions(seen);error=InterruptedError('finite fixture')
        ctx=SimpleNamespace(prompt_version='r8-v11',placement='program-plan',profile='qwen48',weights={},weight_shapes={},graph=object(),model_info=object())
        with self.numeric_seams(),patch.object(entry.guarded.bridge,'_agent_functions',return_value=functions), \
                patch.object(entry.component,'estimate',side_effect=error),patch.object(entry,'seed_partial'), \
                patch.object(entry.guarded.bridge,'_context',return_value=ctx):
            with self.assertRaises(InterruptedError)as caught:entry._agent_phase(self.args,self.recorder,{})
        self.assertIs(caught.exception,error)
        self.assertEqual(load_state(self.session/'SESSION.json')['calls'],12)
        self.assertEqual(load_state(self.work/'partial-local-evaluate-state.json')['calls'],13)
        self.assertFalse((self.session/'component-c2-access.json').exists())
        events=[json.loads(line)for line in (self.work/'evaluation-stage-events.jsonl').read_text().splitlines()]
        self.assertTrue(any(row.get('status')=='error'for row in events))
        self.assertEqual(json.loads((self.work/'local-replay-error.json').read_text())['new_HTTP'],0)

    def test_original_quality_rejection_remains_rejection_without_freezing(self):
        self.prepared();seen=[];functions=self.original_functions(seen)
        original=functions['evaluate'].__closure__[0].cell_contents
        original.__globals__['validate']=lambda *a,**k:dict(feasible=False,diverged=False,agree=.1,
            max_u=1.1,max_u_calibrated=1.1,max_u_analytic=0.,rows=[],boot_input_bound=None)
        def confirm(candidates,ctx,**kwargs):
            self.assertEqual(candidates,[]);seen.append('C2-empty')
            return dict(selected=None,evals=[])
        functions['confirm_select']=confirm
        ctx=SimpleNamespace(prompt_version='r8-v11',placement='program-plan',profile='qwen48',weights={},weight_shapes={},graph=object(),model_info=object())
        cost=dict(placement_model='component-program-plan',latency_pred_us=55.,boots_pred=4,boot_nodes=[])
        with self.numeric_seams(),patch.object(entry.guarded.bridge,'_agent_functions',return_value=functions), \
                patch.object(entry.component,'estimate',return_value=cost),patch.object(entry,'seed_partial'), \
                patch.object(entry.guarded.bridge,'_context',return_value=ctx):
            entry._agent_phase(self.args,self.recorder,{})
            self.assertEqual(load_state(self.session/'SESSION.json')['pool'],[])
            entry._agent_phase(self.args,self.recorder,{})
        self.assertEqual(entry.session_outcome(self.session)['status'],'quality_rejected')
        self.assertFalse((self.session/'FROZEN.json').exists());self.assertEqual(seen.count('C2-empty'),1)

    def test_prompt_phase_does_not_construct_context_or_provider(self):
        save_state(self.session/'SESSION.json',state());self.args.worker_phase='prompt';result={}
        with patch.object(entry,'seed_partial'),patch.object(entry.guarded.bridge,'_context',side_effect=AssertionError('context')):
            entry._agent_phase(self.args,self.recorder,result)
        self.assertFalse(result['api_required']);self.assertEqual(result['new_provider_instances'],0)

    def test_incomplete_C2_claim_prevents_context_retry(self):
        value=state();value.update(phase='confirm',calls=13,iteration=14);save_state(self.session/'SESSION.json',value)
        write(self.session/'component-c2-access.json',dict(status='started'))
        with patch.object(entry,'seed_partial'),patch.object(entry,'agent_functions',return_value={}), \
                patch.object(entry.guarded.bridge,'_context',side_effect=AssertionError('context')):
            with self.assertRaises(FileExistsError):entry._agent_phase(self.args,self.recorder,{})

    def test_foreign_cost_pool_rejected_before_context(self):
        value=state();value['pool']=[dict(iteration=12)];save_state(self.session/'SESSION.json',value)
        with self.assertRaisesRegex(ValueError,'only candidate13'):entry.session_outcome(self.session)

    def test_forbidden_future_phase_rejected_without_context(self):
        value=state();value.update(phase='tune',calls=13,iteration=14);save_state(self.session/'SESSION.json',value)
        with patch.object(entry,'seed_partial'),patch.object(entry,'agent_functions',return_value={}), \
                patch.object(entry.guarded.bridge,'_context',side_effect=AssertionError('context')):
            with self.assertRaisesRegex(ValueError,'No tune/evaluate/provider'):entry._agent_phase(self.args,self.recorder,{})

    def test_phase_command_uses_new_module_explicit_proof_and_original_limits(self):
        args=entry.parser().parse_args(['--resume-from',str(self.root/'resume'),'--partial-from',str(self.root/'partial'),
            '--grouped-from',str(self.root/'grouped')])
        for name in ('model','model_info','inputs','source_rows','metadata_template','frontend','profile','linked_result',
                'coefficients','calibration','session','codegen','ledger','allowance','intake_python'):
            setattr(args,name,self.root/name)
        argv=entry.phase_command(args,'step',self.work);start=argv.index(entry.MODULE)
        inner=entry.parser().parse_args(argv[start+1:])
        self.assertEqual(inner.worker_phase,'step');self.assertEqual(inner.grouped_from,args.grouped_from)
        self.assertEqual(inner.source_admission,args.calibration.parent/'component-source-admission.json')
        self.assertEqual((inner.memory_gib,inner.rss_gib,inner.minimum_available_gib,inner.seconds),(24.,16.,1.25,10800))
        namespace=entry._namespace()
        self.assertIs(namespace['source_gate'],entry.source_gate)
        self.assertEqual(namespace['MODULE'],entry.MODULE)
        self.assertIs(namespace['session_outcome'],entry.session_outcome)
        self.assertEqual(namespace['_program'].__globals__['session_outcome'],entry.session_outcome)


if __name__=='__main__':unittest.main()
