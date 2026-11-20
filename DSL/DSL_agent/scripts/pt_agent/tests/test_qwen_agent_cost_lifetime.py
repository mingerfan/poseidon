"""Small deterministic ownership, original-plan and original Agent dispatch checks."""
import ast
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from pt_agent.experiments import qwen_agent_cost_lifetime as entry
from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands


TEMPLATE = ['{hecate_opt}', '{input}', '--pars', '--ckks-config={package}/config/compiler-config.json',
    '--waterline=43', '--output-val=10', '--runtime-plan-context-id=qwen-bootstrap48',
    '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
    '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64, '-o', '{output_dir}/lowered.mlir']


@contextmanager
def no_cyclic_gc():
    was = gc.isenabled(); gc.disable()
    try:
        yield
    finally:
        if was: gc.enable()


def toy(backend):
    x, y = backend.input(0), backend.input(1)
    backend.op_context = 0; backend.stage('start')
    a, duplicate = x*x, x*x
    z = a-duplicate+y
    vector = np.linspace(.125, .25, backend.slots); vector.flags.writeable = False
    z = z-backend.const(vector)
    backend.op_context = 1; backend.stage('chain')
    for _ in range(8): z = z*z+x*.125
    z = z.rotate(1)+z.rotate(-1)+(-y)
    dead = z
    for _ in range(3): dead = dead*dead
    backend.output([z, z*.5])


PROGRAM = '''@fhe.program(slots=8192, refresh="agent", placement="program-plan", l_max=5, boot_out_level=5)
def small(x: Cipher[1,8]):
    x=pack(x,layout="flat")
    a=mul(x,x)
    b=mul(a,a)
    c=mul(b,b)
    d=mul(c,c)
    e=mul(d,d)
    f=mul(e,e)
    g=add(f,x)
    return g
'''


class NumericLifetimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the original profile reader, with a syntactically valid recorded
        # metadata command.  No new identifier digest is computed on import.
        cls.metadata = native_metadata_commands(TEMPLATE); cls.metadata.__enter__()
        from pt_agent import cost, plan
        from pt_agent.backends.cost_dag import CostDag
        from pt_agent.backends.dag import DagExpr
        from pt_agent.backends import CountBackend, MockBackend, RefreshBackend
        cls.cost, cls.plan, cls.CostDag, cls.DagExpr = cost, plan, CostDag, DagExpr
        cls.CountBackend, cls.MockBackend, cls.RefreshBackend = CountBackend, MockBackend, RefreshBackend

    @classmethod
    def tearDownClass(cls):
        cls.metadata.__exit__(None, None, None)

    def scalar(self, lifetime=None):
        lifetime = lifetime or entry.Lifetime()
        return entry.scalar_cost_dag(self.CostDag, np, lifetime)(64)

    def test_live_cse_identity_and_dead_handle_reconstruction_preserve_node(self):
        with no_cyclic_gc():
            backend = self.scalar(); x = backend.input(0)
            first = x*x; nid = first.nid
            self.assertIs(x*x, first)
            saved = weakref.ref(first); del first
            self.assertIsNone(saved())
            recreated = x*x
            self.assertEqual(recreated.nid, nid); self.assertEqual(len(backend.kind), 2)
            self.assertIs(x*x, recreated); self.assertIs(x.rotate(0), x)
            mapping = backend.numbering
            self.assertTrue(all(type(value) is int for value in mapping.ids.values()))
            del recreated, x
            backend_ref = weakref.ref(backend); entry.release_backend(backend); del backend
            self.assertIsNone(backend_ref()); self.assertIsNone(mapping.owner())

    def test_original_and_new_dag_arrays_normalization_and_record_calls_match(self):
        old, old_record, _ = self.plan.record_dag(toy, 64, frontend_normalize=True)
        lifetime = entry.Lifetime(); planner = entry.private_planner(lifetime)
        new, new_record, _ = planner['record_dag'](toy, 64, frontend_normalize=True)
        try:
            for name in ('kind','args','prog_op','operands','steps','_depth','_total_depth','normalization'):
                self.assertEqual(getattr(new, name), getattr(old, name), name)
            self.assertEqual(new.stats(), old.stats())
            self.assertEqual(new_record.calls, old_record.calls)
            self.assertEqual(new_record.node_stage, old_record.node_stage)
            self.assertEqual(new_record.output_nodes, old_record.output_nodes)
            self.assertEqual(len(new.numbering), 0); self.assertEqual(new._constant_objects, {})
        finally:
            lifetime.finish_records(); entry.release_record(old, old_record)

    def outputs(self, run, plan=None):
        backend = self.MockBackend(64)
        arrays = [np.linspace(-.125, .125, 64), np.linspace(.25, .375, 64)]
        backend.bind(arrays)
        active = self.RefreshBackend(backend, policy='plan', plan=plan, **self.plan.levels()) if plan else backend
        run(active)
        return [value.value.copy() for value in backend._outputs]

    def test_original_plan_and_new_plan_strict_mock_and_count_arithmetic(self):
        old, old_record, _ = self.plan.record_dag(toy, 64, frontend_normalize=True)
        lifetime = entry.Lifetime(); planner = entry.private_planner(lifetime)
        new, new_record, _ = planner['record_dag'](toy, 64, frontend_normalize=True)
        try:
            old_plan = self.plan.express(old, old_record, self.cost.depth_dp_estimate(old))
            new_plan = planner['express'](new, new_record, self.cost.depth_dp_estimate(new))
            self.assertEqual(new_plan, old_plan)
            old_counts = self.plan.verify(toy, 64, old_plan)
            new_counts = planner['verify'](toy, 64, new_plan)
            self.assertEqual({k:v for k,v in old_counts.items() if k!='apply_s'},
                             {k:v for k,v in new_counts.items() if k!='apply_s'})
            self.assertGreater(new_counts['boots'], 0)
            for reference, observed in zip(self.outputs(toy), self.outputs(toy, new_plan)):
                self.assertTrue(np.all(np.abs(reference-observed) <= 1e-12+1e-12*np.abs(reference)))
        finally:
            lifetime.finish_records(); entry.release_record(old, old_record)

    def test_real_program_plan_and_cost_dict_keep_values_after_private_release(self):
        from pt_agent.dsl.parser import parse
        program = parse(PROGRAM)
        old = self.cost.estimate_normalized_fast(program, {}, profile='qwen48')
        lifetime = entry.Lifetime(); planner = entry.private_planner(lifetime)
        references = []
        original_track = lifetime.track_backend
        def track(backend): references.append(weakref.ref(backend)); original_track(backend)
        lifetime.track_backend = track
        costs = entry.private_cost_namespace(lifetime, planner=planner)
        with no_cyclic_gc():
            observed = costs['estimate_normalized_fast'](program, {}, profile='qwen48')
            # Timings describe distinct real invocations. All other original
            # placement/normalization/cost values must agree exactly.
            expected = copy.deepcopy(old); actual = copy.deepcopy(observed)
            expected['refresh_plan'].pop('timing'); actual['refresh_plan'].pop('timing')
            for value in (expected,actual):
                for row in value['refresh_plan']['candidates'].values():
                    for key in ('seconds','prune_s','total_s','dp_s'):
                        row.pop(key,None)
            self.assertEqual(actual, expected)
            self.assertTrue(references); self.assertTrue(all(r() is None for r in references))
            retained = copy.deepcopy(observed)
            with lifetime.estimating(): pass
            self.assertEqual(observed, retained)
        self.assertEqual(lifetime.records, []); self.assertFalse(lifetime.active)

    def test_plan_keep_recorder_true_preserves_until_explicit_release(self):
        from pt_agent.dsl.parser import parse
        lifetime = entry.Lifetime(); planner = entry.private_planner(lifetime)
        program = parse(PROGRAM)
        plan, backend, _, recorder = planner['plan_program'](program, {}, keep_recorder=True)
        snapshot = copy.deepcopy(plan)
        self.assertTrue(recorder.calls); self.assertTrue(recorder.node_stage)
        self.assertIn('_apply', backend.__dict__)
        entry.release_record(backend, recorder); entry.release_backend(backend)
        self.assertNotIn('_apply', backend.__dict__); self.assertEqual(plan, snapshot)
        self.assertEqual(lifetime.records, [])

    def test_cleanup_restores_class_hook_and_prior_instance_hook_without_gc(self):
        for custom in (False, True):
            with self.subTest(custom=custom), no_cyclic_gc():
                backend = self.scalar()
                if custom:
                    old_apply = backend._apply
                    def before(kind, left, right=None, *, step=None):
                        return old_apply(kind, left, right, step=step)
                    backend._apply = before
                previous = backend.__dict__.get('_apply', entry._MISSING)
                active = self.RefreshBackend(backend, policy='record', **self.plan.levels())
                x = active.input(0); active.output([x*x])
                recorder_ref = weakref.ref(active)
                entry.release_record(backend, active, previous)
                if custom: self.assertIs(backend._apply, before)
                else: self.assertNotIn('_apply', backend.__dict__)
                del active, x
                self.assertIsNone(recorder_ref())
                entry.release_backend(backend)
                if custom:
                    del before, old_apply, previous

    def test_failed_lower_releases_record_backend_and_propagates_original_error(self):
        lifetime = entry.Lifetime(); planner = entry.private_planner(lifetime)
        refs = []
        def failure(backend):
            refs.extend([weakref.ref(backend), weakref.ref(backend.inner)])
            x=backend.input(0); x=x*x
            raise ValueError('deliberate original lower failure')
        with no_cyclic_gc():
            with self.assertRaisesRegex(ValueError, 'original lower failure'):
                with lifetime.estimating(): planner['record_dag'](failure, 64, frontend_normalize=True)
            self.assertTrue(all(r() is None for r in refs))
        self.assertEqual(lifetime.records, []); self.assertFalse(lifetime.active)

    def test_readonly_original_array_and_original_globals_remain_unchanged(self):
        original = {name:getattr(self.plan,name) for name in ('CostDag','RefreshBackend','plan_program','record_dag')}
        vector=np.arange(64,dtype=np.float64); vector.flags.writeable=False; expected=vector.copy()
        backend=self.scalar(); plain=backend.const(vector)
        np.testing.assert_array_equal(plain.data,expected); np.testing.assert_array_equal(vector,expected)
        self.assertFalse(vector.flags.writeable); self.assertEqual(backend._constant_objects,{})
        self.assertEqual(original,{name:getattr(self.plan,name)for name in original})
        entry.release_backend(backend)


class DispatchAndEntryChecks(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.directory=Path(temporary.name)

    def functions(self, *, error=False):
        from pt_agent.experiments.qwen_bridge_agent import _clone_namespace
        source=Path(entry.__file__).parents[1]/'planners/agent.py'
        tree=ast.parse(source.read_text()); names={'agent_step','_evaluate_step','_finish_iteration'}
        module=ModuleType('original_agent_fixture'); ns=module.__dict__
        ns.update(__package__='pt_agent.planners',json=json,Path=Path,time=time,LLMError=RuntimeError,
            validate_tune_rule=lambda *a:None,apply_placement=lambda source,ctx:source,json_safe=lambda value:value,
            PRE_TUNE_RULE='v8-pre-6.11',feedback_message=lambda *a,**k:'feedback')
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),str(source),'exec'),ns)
        tools=ModuleType('pt_agent.planners.agent_tools')
        original=Path(entry.__file__).parents[1]/'planners/agent_tools.py'
        evaluate=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='evaluate')
        calls=[]
        def exact(*a,**k): calls.append('exact'); return dict(ok=True)
        def cost(program,ctx,estimator,line):
            calls.append('cost')
            if error: raise ValueError('deliberate exception, not payload feedback')
            value=estimator(program,ctx)
            return value,value
        def validate(*a,**k):
            calls.append('validation');return dict(rows=[],feasible=False,diverged=False,agree=.5,max_u=.8,total=1)
        tools.__dict__.update(parse=lambda source:SimpleNamespace(ops=[]),PTDSLError=TypeError,AgentContext=object,
            check=lambda *a,**k:SimpleNamespace(ok=True),_exact_check=exact,cost_feedback=cost,validate=validate,
            PROFILES={'tiny':dict(limits={})},LoweringError=LookupError,BackendError=ArithmeticError,PlacementError=RuntimeError)
        exec(compile(ast.Module(body=[evaluate],type_ignores=[]),str(original),'exec'),tools.__dict__)
        def bridge(original_evaluate):
            def evaluate(source,ctx,**kwargs):return original_evaluate(source,ctx,**kwargs)
            return evaluate
        ns['evaluate']=bridge(tools.evaluate)
        functions=_clone_namespace(module,{})
        state=dict(phase='evaluate',iteration=13,calls=12,max_iters=15,tuner=False,failed=None,
            engine_profile='tensor',tune_search='sweep',tune_rule='v8-6.25',prompt_version='r8-v7',
            prompt_tokens=0,completion_tokens=0,llm_min=None,best=None,best_program=None,has_tuned=False,
            history=[],messages=[],pool=[],pending=None,elapsed_s=0.,stale=0,tune_policy='all')
        ctx=SimpleNamespace(prompt_version='r8-v7',placement='program-plan',profile='tiny',weight_shapes={},graph=None)
        return functions,state,ctx,calls

    @contextmanager
    def tune_import(self):
        module=ModuleType('pt_agent.planners.transformer_tuner');module.validate_tune_search=lambda ctx:'sweep'
        with mock.patch.dict('sys.modules',{module.__name__:module}):yield

    def adapted(self,error=False):
        functions,state,ctx,calls=self.functions(error=error)
        def fits(program):calls.append('fit');return []
        def estimate(*a,**k):calls.append('estimate');return dict(boot_nodes=[],ops_total=3)
        @contextmanager
        def scope():yield
        diagnostics=entry.Diagnostics(self.directory/'evaluation.jsonl')
        adapted=entry.adapt_agent_functions(functions,diagnostics=diagnostics,
            costs=dict.fromkeys(('estimate','estimate_fast','estimate_normalized','estimate_normalized_fast'),estimate),
            scope=scope,program_failures=fits)
        return functions,adapted,state,ctx,calls,diagnostics

    def test_actual_original_agent_step_reaches_boundaries_and_feedback_presenter(self):
        from pt_agent.experiments.qwen_bounded_feedback import FeedbackPresenter
        functions,adapted,state,ctx,calls,diagnostics=self.adapted()
        adapted['feedback_message']=FeedbackPresenter(self.directory/'full-feedback',adapted['feedback_message'])
        with self.tune_import():adapted['agent_step'](ctx,state,trace_path=self.directory/'trace',client=None,
            reply=dict(content=json.dumps(dict(program='deterministic metadata')),usage={},seconds=0))
        self.assertEqual(calls,['fit','exact','cost','estimate','validation'])
        self.assertTrue((self.directory/'full-feedback/iteration-0013.full.json').is_file())
        self.assertEqual(state['calls'],13);self.assertEqual(state['iteration'],14)
        events=[json.loads(s)for s in(self.directory/'evaluation.jsonl').read_text().splitlines()]
        self.assertEqual([s['stage_id']for s in events if s['type']=='begin'],
            ['approximation_preflight','independent_exact_math','cost','heldout_validation'])
        self.assertTrue(all(s['status']=='ok'for s in events if s['type']=='end'))
        self.assertEqual(diagnostics.recorder.stack,[])
        for name in ('agent_step','_evaluate_step','_finish_iteration'):
            self.assertIs(adapted[name].__code__,functions[name].__code__)
            self.assertIs(adapted[name].__globals__,adapted)

    def test_actual_dispatch_error_preserves_error_and_durable_partial_boundaries(self):
        functions,adapted,state,ctx,calls,diagnostics=self.adapted(error=True)
        with self.tune_import(),self.assertRaisesRegex(ValueError,'deliberate exception'):
            adapted['agent_step'](ctx,state,trace_path=self.directory/'trace',client=None,
                reply=dict(content=json.dumps(dict(program='metadata')),usage={},seconds=0))
        events=[json.loads(s)for s in(self.directory/'evaluation.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['stage_id'],'cost');self.assertEqual(events[-1]['status'],'error')
        self.assertEqual(events[-1]['error'],'ValueError')
        self.assertNotIn('payload feedback',(self.directory/'evaluation.jsonl').read_text())
        self.assertEqual(state['history'],[]);self.assertEqual(diagnostics.recorder.stack,[])
        self.assertFalse((self.directory/'trace').exists())

    def test_interrupted_begin_has_no_invented_end_and_dependency_drift_rejects(self):
        diagnostics=entry.Diagnostics(self.directory/'partial.jsonl')
        span=diagnostics.span('record_lower');span.__enter__()
        try:
            events=[json.loads(s)for s in(self.directory/'partial.jsonl').read_text().splitlines()]
            self.assertEqual([s['type']for s in events],['begin'])
        finally:span.__exit__(None,None,None)
        with self.assertRaisesRegex(ValueError,'preflight import changed'):
            entry._evaluation_delegate(entry.main,{})

    def test_entry_keeps_original_limits_parser_and_original_module_unmodified(self):
        from pt_agent.experiments import qwen_guarded_replay as original
        original_phase=original._agent_phase;original_bounded=original.bounded._agent_functions
        namespace=entry._namespace()
        self.assertEqual(namespace['MODULE'],entry.MODULE)
        self.assertEqual(namespace['ROUND'],entry.ROUND);self.assertEqual(namespace['MODE'],entry.MODE)
        self.assertIs(original._agent_phase,original_phase);self.assertIs(original.bounded._agent_functions,original_bounded)
        defaults=entry.parser().parse_args(['--out',str(self.directory/'out'),'--partial-from',str(self.directory/'partial'),
            '--resume-from',str(self.directory/'resume'),'--through','agent'])
        self.assertEqual((defaults.memory_gib,defaults.rss_gib,defaults.minimum_available_gib),(24,16,1.25))
        self.assertEqual((defaults.case_gib,defaults.seconds),(32,10800))
        self.assertEqual(namespace['source_gate'],entry.source_gate)
        argv=namespace['phase_command'](defaults,'step',self.directory/'step')
        worker=argv[argv.index('--')+1:]
        self.assertEqual(worker[worker.index('-m')+1],entry.MODULE)
        self.assertEqual(worker[worker.index('--rss-gib')+1],'16.0')
        self.assertEqual(worker[worker.index('--minimum-available-gib')+1],'1.25')

    def test_new_entry_original_transaction_invokes_private_functions_without_provider(self):
        from pt_agent.experiments import qwen_guarded_replay as original
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        session=self.directory/'session';session.mkdir();inherited=self.directory/'inherited';(inherited/'calls').mkdir(parents=True)
        state=dict(phase='evaluate',iteration=13,calls=12,history=[],pending=None,prompt_tokens=0,completion_tokens=0)
        (session/'SESSION.json').write_text(json.dumps(state))
        (session/'guarded-replay.json').write_text(json.dumps(dict(status='pending_local_replay',inherited_session=str(inherited))))
        reply=dict(content='retained synthetic metadata',usage=dict(prompt_tokens=2,completion_tokens=3))
        (inherited/'calls/call-0013.response.json').write_text(json.dumps(reply))
        called=[]
        def step(ctx,working,*,trace_path,client,reply):
            self.assertIsNone(client);self.assertEqual(reply['content'],'retained synthetic metadata')
            called.append(ctx)
            working.update(calls=13,iteration=14,phase='evaluate',pending=None,prompt_tokens=2,completion_tokens=3)
            working['history'].append(dict(iteration=13));trace_path.write_text(json.dumps(dict(iteration=13))+'\n')
        expected=dict(agent_step=step,feedback_message=lambda *a,**k:'feedback')
        with mock.patch.object(original,'seed_partial'),mock.patch.object(original.bridge,'_context',return_value='fixed context'), \
                mock.patch.object(original.bridge,'_agent_functions',return_value={'original':'functions'}), \
                mock.patch.object(original.bounded,'session_outcome',return_value=dict(frozen=False)), \
                mock.patch.object(entry,'adapt_agent_functions',return_value=expected) as adapter:
            namespace=entry._namespace();work=self.directory/'step';work.mkdir()
            args=SimpleNamespace(session=session,work_dir=work,worker_phase='step')
            result={};namespace['_agent_phase'](args,StageRecorder(work/'events.jsonl'),result)
        self.assertEqual(called,['fixed context']);adapter.assert_called_once()
        self.assertEqual(adapter.call_args.args,({'original':'functions'},))
        self.assertEqual(json.loads((session/'SESSION.json').read_text())['calls'],13)
        self.assertEqual(json.loads((session/'guarded-replay.json').read_text())['HTTP_added'],0)
        self.assertEqual(result['cost_lifetime_variant'],entry.VARIANT)
        self.assertEqual(result['new_provider_requests_for_13'],0)
        self.assertEqual(state['calls'],12)

    def test_source_variant_is_added_after_original_partial_source_validation(self):
        from pt_agent.experiments import qwen_guarded_replay as original
        args=SimpleNamespace(out=None);calls=[]
        def gate(received):
            self.assertIs(received,args);calls.append('original complete gate')
            return dict(original_layers=24,weights_archive=dict(values_read=False))
        with mock.patch.object(original,'source_gate',side_effect=gate):
            checked=entry.source_gate(args)
        self.assertEqual(calls,['original complete gate'])
        self.assertEqual(checked['original_layers'],24)
        self.assertEqual(checked['cost_lifetime_variant'],entry.VARIANT)
        self.assertEqual(checked['extra_numeric_or_Count_passes'],0)


if __name__=='__main__':unittest.main()
