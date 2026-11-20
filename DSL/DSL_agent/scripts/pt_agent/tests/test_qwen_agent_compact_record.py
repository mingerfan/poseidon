"""Small exact-record and private-entry checks; no model or provider runs."""
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from pt_agent.experiments import qwen_agent_compact_record as entry
from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands


TEMPLATE = ['{hecate_opt}', '{input}', '--pars', '--ckks-config={package}/config/compiler-config.json',
    '--waterline=43', '--output-val=10', '--runtime-plan-context-id=qwen-bootstrap48',
    '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
    '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64, '-o', '{output_dir}/lowered.mlir']

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


def toy(backend):
    x, y = backend.input(0), backend.input(1)
    backend.op_context = 0; backend.stage('start')
    a, duplicate = x*x, x*x
    z = a-duplicate+y
    vector = np.linspace(.125, .25, backend.slots); vector.flags.writeable = False
    z = z-backend.const(vector)
    backend.op_context = 1; backend.stage('chain')
    for _ in range(8):
        z = z*z+x*.125
    z = z.rotate(1)+z.rotate(-1)+(-y)
    dead = z
    for _ in range(3):
        dead = dead*dead
    backend.output([z, z*.5])


@contextmanager
def no_cyclic_gc():
    enabled = gc.isenabled(); gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def cost_values(value):
    value = copy.deepcopy(value)
    plan = value['refresh_plan']; plan.pop('timing')
    for candidate in plan['candidates'].values():
        for key in ('seconds', 'prune_s', 'total_s', 'dp_s'):
            candidate.pop(key, None)
    return value


class CompactRecordChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = native_metadata_commands(TEMPLATE); cls.metadata.__enter__()
        from pt_agent import cost, plan
        from pt_agent.backends import MockBackend, RefreshBackend
        cls.cost, cls.plan = cost, plan
        cls.MockBackend, cls.RefreshBackend = MockBackend, RefreshBackend

    @classmethod
    def tearDownClass(cls):
        cls.metadata.__exit__(None, None, None)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_factory_preserves_constructor_seed_and_nonrecord_result(self):
        seed_calls = [((1, None, 1), 2, (3, 4, 3)), ((), 0, ())]
        seed_stages = [(-1, 0), (2, 4), (2, 4)]
        created = []
        def original(inner, **options):
            active = SimpleNamespace(policy=options['policy'], calls=seed_calls.copy(),
                node_stage=seed_stages.copy(), call_count=2, output_nodes=[2], inner=inner)
            created.append(active); return active
        namespace = {'RefreshBackend':original}
        with patch.object(entry.base, 'private_planner', return_value=namespace):
            planner = entry.private_planner(entry.base.Lifetime())
        active = planner['RefreshBackend'](object(), policy='record')
        self.assertIs(active, created[0]); self.assertEqual(active.calls, seed_calls)
        self.assertEqual(active.node_stage, seed_stages)
        self.assertIs(active.node_stage[1], active.node_stage[2])
        self.assertEqual((active.call_count, active.output_nodes), (2, [2]))
        plain = planner['RefreshBackend'](object(), policy='plan')
        self.assertIs(type(plain.calls), list); self.assertIs(type(plain.node_stage), list)

    def test_actual_record_arrays_order_express_count_and_strict_mock_match(self):
        old, old_recorder, _ = self.plan.record_dag(toy, 64, frontend_normalize=True)
        lifetime = entry.base.Lifetime(); planner = entry.private_planner(lifetime)
        new, new_recorder, _ = planner['record_dag'](toy, 64, frontend_normalize=True)
        try:
            self.assertIs(planner['record_dag'].__wrapped__.__globals__, planner)
            self.assertIsInstance(new_recorder.calls, entry.PackedCalls)
            self.assertIsInstance(new_recorder.node_stage, entry.InternedStages)
            for name in ('kind','args','prog_op','operands','steps','_depth','_total_depth','normalization'):
                self.assertEqual(getattr(new, name), getattr(old, name), name)
            self.assertEqual(new.stats(), old.stats())
            self.assertEqual(new_recorder.calls, old_recorder.calls)
            self.assertEqual(old_recorder.calls, new_recorder.calls)
            self.assertEqual(new_recorder.node_stage, old_recorder.node_stage)
            old_plan = self.plan.express(old, old_recorder, self.cost.depth_dp_estimate(old))
            new_plan = planner['express'](new, new_recorder, self.cost.depth_dp_estimate(new))
            self.assertEqual(new_plan, old_plan)
            old_count, new_count = self.plan.verify(toy,64,old_plan), planner['verify'](toy,64,new_plan)
            self.assertEqual({k:v for k,v in old_count.items() if k!='apply_s'},
                             {k:v for k,v in new_count.items() if k!='apply_s'})
            self.assertGreater(new_count['boots'], 0)
            outputs = []
            for plan in (None, new_plan):
                backend = self.MockBackend(64)
                backend.bind([np.linspace(-.125,.125,64),np.linspace(.25,.375,64)])
                active = self.RefreshBackend(backend, policy='plan', plan=plan,
                    **self.plan.levels()) if plan else backend
                toy(active); outputs.append([v.value.copy() for v in backend._outputs])
            for reference, observed in zip(*outputs):
                self.assertTrue(np.all(np.abs(reference-observed)<=1e-12+1e-12*np.abs(reference)))
        finally:
            lifetime.finish_records(); entry.base.release_record(old, old_recorder)
            entry.base.release_backend(new); entry.base.release_backend(old)

    def test_original_plan_cost_values_and_retained_aliases_survive_cleanup(self):
        from pt_agent.dsl.parser import parse
        program = parse(PROGRAM)
        old = self.cost.estimate_normalized_fast(program, {}, profile='qwen48')
        lifetime = entry.base.Lifetime(); planner = entry.private_planner(lifetime)
        costs = entry.private_cost_namespace(lifetime, planner=planner)
        references = []; original_track = lifetime.track_backend
        def track(backend):
            references.append(weakref.ref(backend)); original_track(backend)
        lifetime.track_backend = track
        with no_cyclic_gc():
            observed = costs['estimate_normalized_fast'](program, {}, profile='qwen48')
            self.assertEqual(cost_values(observed), cost_values(old))
            self.assertTrue(references); self.assertTrue(all(r() is None for r in references))
            retained = copy.deepcopy(observed)
            with lifetime.estimating():
                pass
            self.assertEqual(observed, retained)
        plan, backend, _, recorder = planner['plan_program'](program, {}, keep_recorder=True)
        snapshot = copy.deepcopy(plan); calls, stages = recorder.calls, recorder.node_stage
        self.assertTrue(calls); self.assertTrue(stages)
        entry.base.release_record(backend, recorder); entry.base.release_backend(backend)
        self.assertEqual(len(calls), 0); self.assertEqual(stages, [])
        self.assertEqual(plan, snapshot); self.assertEqual(lifetime.records, [])

    def test_failed_factory_install_releases_original_hook_with_gc_disabled(self):
        lifetime = entry.base.Lifetime(); planner = entry.private_planner(lifetime)
        owners = []; original_track = lifetime.track_backend
        def track(backend):
            owners.append(weakref.ref(backend)); original_track(backend)
        lifetime.track_backend = track
        with no_cyclic_gc(), patch.object(entry, 'PackedCalls', side_effect=ValueError('fixture install error')):
            with self.assertRaisesRegex(ValueError, 'fixture install error'):
                with lifetime.estimating():
                    planner['record_dag'](toy, 64, frontend_normalize=True)
            self.assertTrue(owners); self.assertTrue(all(r() is None for r in owners))
        self.assertEqual(lifetime.records, []); self.assertFalse(lifetime.active)

    def test_source_gate_preserves_base_and_adds_only_storage_provenance_snapshots(self):
        out = self.directory/'metadata'; out.mkdir()
        args = SimpleNamespace(out=out); old = dict(original_layers=24, maximum_calls=15,
            cost_lifetime_variant=entry.base.VARIANT, payload_values_read=False)
        with patch.object(entry.base, 'source_gate', return_value=copy.deepcopy(old)) as gate:
            actual = entry.source_gate(args)
        gate.assert_called_once_with(args)
        self.assertEqual({k:actual[k] for k in old}, old)
        self.assertEqual(actual['compact_record_variant'], entry.VARIANT)
        self.assertEqual(actual['compact_record_extra_passes'], 0)
        self.assertTrue((out/'variant-source/qwen_agent_compact_record.py').is_file())
        self.assertTrue((out/'variant-source/qwen_record_storage.py').is_file())

    def test_entry_namespace_worker_argv_limits_and_base_globals_preserved(self):
        from pt_agent.experiments import qwen_guarded_replay as guarded
        original_factory, original_phase = entry.base.private_planner, guarded._agent_phase
        namespace = entry._namespace()
        self.assertEqual((namespace['MODULE'],namespace['ROUND'],namespace['MODE']),
                         (entry.MODULE,entry.ROUND,entry.MODE))
        self.assertIs(namespace['source_gate'],entry.source_gate)
        args = namespace['parser']().parse_args(['--out',str(self.directory/'out'),
            '--partial-from',str(self.directory/'partial'),'--resume-from',str(self.directory/'resume'),
            '--through','agent'])
        self.assertEqual((args.memory_gib,args.rss_gib,args.minimum_available_gib,args.case_gib,args.seconds),
                         (24,16,1.25,32,10800))
        argv = namespace['phase_command'](args,'step',self.directory/'step')
        worker = argv[argv.index('--')+1:]
        self.assertEqual(worker[worker.index('-m')+1],entry.MODULE)
        self.assertIs(entry.base.private_planner,original_factory); self.assertIs(guarded._agent_phase,original_phase)

    def test_actual_entry_reply_transaction_reaches_new_record_factory_without_provider(self):
        from pt_agent.dsl.parser import parse
        from pt_agent.experiments import qwen_guarded_replay as guarded
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        session = self.directory/'session'; session.mkdir()
        inherited = self.directory/'inherited'; (inherited/'calls').mkdir(parents=True)
        state = dict(phase='evaluate',iteration=13,calls=12,max_iters=15,history=[],pending=None,
            prompt_tokens=0,completion_tokens=0)
        (session/'SESSION.json').write_text(json.dumps(state))
        (session/'guarded-replay.json').write_text(json.dumps(dict(
            status='pending_local_replay',inherited_session=str(inherited))))
        (inherited/'calls/call-0013.response.json').write_text(json.dumps(dict(
            content='trusted tiny fixture',usage=dict(prompt_tokens=2,completion_tokens=3))))
        program = parse(PROGRAM); observed = []
        tools = ModuleType('pt_agent.planners.agent_tools')
        def evaluate(source, ctx, **kwargs):
            from .approximation_failures import program_failures
            return estimate_normalized_fast(ctx.program, {}, profile='qwen48')
        original_evaluate = FunctionType(evaluate.__code__, tools.__dict__, 'evaluate')
        tools.__dict__.update(evaluate=original_evaluate,_exact_check=lambda:None,
            cost_feedback=lambda:None,validate=lambda:None)
        def bridge(original_evaluate):
            def evaluate(source, ctx, **kwargs):
                return original_evaluate(source,ctx,**kwargs)
            return evaluate
        module = ModuleType('trusted_compact_fixture'); functions = module.__dict__
        def step(ctx, working, *, trace_path, client, reply):
            if client is not None:
                raise AssertionError('The retained-reply fixture requires client=None')
            value = globals()['evaluate'](reply['content'],ctx)
            globals()['observed'].append(value)
            working.update(calls=13,iteration=14,pending=None,prompt_tokens=2,completion_tokens=3)
            working['history'].append(dict(iteration=13))
            trace_path.write_text(json.dumps(dict(iteration=13))+'\n')
        functions.update(evaluate=bridge(original_evaluate),observed=observed,json=json,
            feedback_message=lambda *a,**k:'metadata',agent_step=FunctionType(step.__code__,functions,'agent_step'))
        # Closure-free step globals are rebound by the same original dispatcher helper.
        calls = []; original_packed = entry.PackedCalls
        class TrackedPacked(original_packed):
            def __init__(self):
                super().__init__(); calls.append(True)
        work = self.directory/'step'; work.mkdir()
        args = SimpleNamespace(session=session,work_dir=work,worker_phase='step')
        result = {}
        with patch.object(guarded,'seed_partial'), patch.object(guarded.bridge,'_context',
                return_value=SimpleNamespace(program=program)), \
             patch.object(guarded.bridge,'_agent_functions',return_value=functions), \
             patch.object(guarded.bounded,'session_outcome',return_value=dict(frozen=False)), \
             patch.object(entry,'PackedCalls',TrackedPacked):
            namespace = entry._namespace()
            namespace['_agent_phase'](args,StageRecorder(work/'events.jsonl'),result)
        saved = json.loads((session/'SESSION.json').read_text())
        self.assertTrue(calls); self.assertEqual(len(observed),1)
        self.assertEqual(saved['calls'],13); self.assertEqual(saved['max_iters'],15)
        self.assertEqual(state['calls'],12)
        self.assertEqual(result['new_provider_requests_for_13'],0)
        self.assertEqual(json.loads((session/'guarded-replay.json').read_text())['HTTP_added'],0)
        self.assertEqual(result['cost_lifetime_variant'],entry.base.VARIANT)

    def test_parser_export_delegates_base_and_retains_original_policy(self):
        marker = object()
        with patch.object(entry.base,'parser',return_value=marker) as delegated:
            self.assertIs(entry.parser(),marker)
        delegated.assert_called_once_with()
        argv = ['--out',str(self.directory/'out'),'--partial-from',str(self.directory/'partial'),
            '--resume-from',str(self.directory/'resume'),'--through','agent']
        current, original = entry.parser().parse_args(argv), entry.base.parser().parse_args(argv)
        self.assertEqual(vars(current),vars(original))
        self.assertEqual((current.memory_gib,current.rss_gib,current.minimum_available_gib,
            current.case_gib,current.seconds),(24,16,1.25,32,10800))

    def test_session_outcome_export_delegates_directory_and_preserves_result(self):
        directory = self.directory/'synthetic-session'
        marker = dict(status='metadata',frozen=False,calls=12)
        with patch.object(entry.base,'session_outcome',return_value=marker) as delegated:
            self.assertIs(entry.session_outcome(directory),marker)
        delegated.assert_called_once_with(directory)


if __name__ == '__main__':
    unittest.main()
