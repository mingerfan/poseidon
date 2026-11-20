"""New grouped entry synthetic exact arithmetic/dispatch checks; no model runs."""
import copy
import json
from pathlib import Path
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from pt_agent.experiments import qwen_agent_grouped_record as entry
from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands
from pt_agent.tests.test_qwen_agent_compact_record import PROGRAM, TEMPLATE, toy, cost_values, no_cyclic_gc


class GroupedRecordChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = native_metadata_commands(TEMPLATE); cls.metadata.__enter__()
        from pt_agent import cost, plan
        from pt_agent.backends import MockBackend, RefreshBackend
        from pt_agent.backends.cost_dag import CostDag
        cls.cost, cls.plan, cls.MockBackend, cls.RefreshBackend, cls.CostDag = cost, plan, MockBackend, RefreshBackend, CostDag

    @classmethod
    def tearDownClass(cls):
        cls.metadata.__exit__(None, None, None)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_factory_returns_original_scalar_class_and_keeps_live_and_dead_nids(self):
        lifetime = entry.compact.base.Lifetime(); planner = entry.private_planner(lifetime)
        with no_cyclic_gc():
            backend = planner['CostDag'](64)
            self.assertIsInstance(backend, self.CostDag)
            self.assertEqual(type(backend).__name__, 'ScalarCostDag')
            self.assertIsInstance(backend.numbering.ids, entry.GroupedIds)
            x = backend.input(0); first = x*x; nid = first.nid
            self.assertIs(x*x, first)
            key = next(iter(backend.numbering.ids))
            self.assertIs(key[1], backend.operands[nid])
            held = weakref.ref(first); del first
            self.assertIsNone(held())
            recreated = x*x
            self.assertEqual(recreated.nid, nid); self.assertEqual(len(backend.kind), 2)
            self.assertIs(x*x, recreated)
            self.assertEqual(list(backend.numbering.ids.values()), [nid])
            owner = weakref.ref(backend); mapping = backend.numbering
            del x, recreated; entry.compact.base.release_backend(backend); del backend
            self.assertIsNone(owner()); self.assertIsNone(mapping.owner())

    def test_constructor_rejects_seeded_mapping_without_replacing_it(self):
        class Owner: pass
        owner = Owner(); numbering = entry.compact.base.ScalarNumbering(owner, object)
        original_ids = numbering.ids; original_ids[('add', (), None)] = 7
        active = SimpleNamespace(numbering=numbering, original_field='retained')
        namespace = {'CostDag':lambda *a,**k:active}
        with patch.object(entry.compact,'private_planner',return_value=namespace):
            planner = entry.private_planner(entry.compact.base.Lifetime())
        with self.assertRaisesRegex(ValueError,'original empty ScalarNumbering'):
            planner['CostDag'](64)
        self.assertIs(numbering.ids, original_ids); self.assertEqual(len(original_ids), 1)
        original_ids.clear()
        self.assertIs(planner['CostDag'](64), active)
        self.assertEqual(active.original_field, 'retained')

    def test_actual_original_dag_plan_count_mock_and_packed_record_match(self):
        old, old_record, _ = self.plan.record_dag(toy, 64, frontend_normalize=True)
        lifetime = entry.compact.base.Lifetime(); planner = entry.private_planner(lifetime)
        new, record, _ = planner['record_dag'](toy, 64, frontend_normalize=True)
        try:
            self.assertIs(planner['record_dag'].__wrapped__.__globals__, planner)
            self.assertIsInstance(new.numbering.ids, entry.GroupedIds)
            self.assertIsInstance(record.calls, entry.compact.PackedCalls)
            self.assertIsInstance(record.node_stage, entry.compact.InternedStages)
            for name in ('kind','args','prog_op','operands','steps','_depth','_total_depth','normalization'):
                self.assertEqual(getattr(new,name),getattr(old,name),name)
            self.assertEqual(new.stats(), old.stats()); self.assertEqual(len(new.numbering), 0)
            self.assertEqual(record.calls, old_record.calls); self.assertEqual(record.node_stage, old_record.node_stage)
            original_plan = self.plan.express(old,old_record,self.cost.depth_dp_estimate(old))
            grouped_plan = planner['express'](new,record,self.cost.depth_dp_estimate(new))
            self.assertEqual(grouped_plan, original_plan)
            counts = [self.plan.verify(toy,64,original_plan), planner['verify'](toy,64,grouped_plan)]
            self.assertEqual(*({k:v for k,v in row.items()if k!='apply_s'}for row in counts))
            self.assertGreater(counts[1]['boots'], 0)
            outputs = []
            for plan in (None, grouped_plan):
                backend = self.MockBackend(64)
                backend.bind([np.linspace(-.125,.125,64),np.linspace(.25,.375,64)])
                active = self.RefreshBackend(backend,policy='plan',plan=plan,**self.plan.levels())if plan else backend
                toy(active); outputs.append([value.value.copy()for value in backend._outputs])
            for reference, observed in zip(*outputs):
                self.assertTrue(np.all(np.abs(reference-observed)<=1e-12+1e-12*np.abs(reference)))
        finally:
            lifetime.finish_records(); entry.compact.base.release_record(old,old_record)
            entry.compact.base.release_backend(new); entry.compact.base.release_backend(old)

    def test_cost_values_alias_cleanup_and_install_failure_with_gc_disabled(self):
        from pt_agent.dsl.parser import parse
        program = parse(PROGRAM); expected = self.cost.estimate_normalized_fast(program,{})
        lifetime = entry.compact.base.Lifetime(); planner = entry.private_planner(lifetime)
        costs = entry.private_cost_namespace(lifetime,planner=planner)
        references=[]; original_track=lifetime.track_backend
        def track(backend):
            references.append(weakref.ref(backend)); original_track(backend)
        lifetime.track_backend=track
        with no_cyclic_gc():
            observed=costs['estimate_normalized_fast'](program,{})
            self.assertEqual(cost_values(observed),cost_values(expected))
            self.assertTrue(references); self.assertTrue(all(ref()is None for ref in references))
            snapshot=copy.deepcopy(observed)
            with lifetime.estimating(): pass
            self.assertEqual(observed,snapshot)
            with patch.object(entry,'GroupedIds',side_effect=ValueError('fixture group install')):
                with self.assertRaisesRegex(ValueError,'fixture group install'):
                    costs['estimate_normalized_fast'](program,{})
            self.assertTrue(all(ref()is None for ref in references))
            self.assertEqual(lifetime.records,[]); self.assertFalse(lifetime.active)
        plan,backend,_,record=planner['plan_program'](program,{},keep_recorder=True)
        calls,stages=record.calls,record.node_stage; saved=copy.deepcopy(plan)
        self.assertTrue(calls); entry.compact.base.release_record(backend,record)
        entry.compact.base.release_backend(backend)
        self.assertEqual(len(calls),0); self.assertEqual(stages,[]); self.assertEqual(plan,saved)

    def test_real_parent_source_gate_exports_and_worker_policies_preserved(self):
        out=self.directory/'metadata';out.mkdir();args=SimpleNamespace(out=out)
        base=dict(original_layers=24,maximum_calls=15,cost_lifetime_variant=entry.compact.base.VARIANT)
        with patch.object(entry.compact.base,'source_gate',return_value=copy.deepcopy(base))as original:
            actual=entry.source_gate(args)
        original.assert_called_once_with(args)
        self.assertEqual({k:actual[k]for k in base},base)
        self.assertEqual(actual['compact_record_variant'],entry.compact.VARIANT)
        self.assertEqual(actual['grouped_record_variant'],entry.VARIANT)
        self.assertEqual(actual['grouped_record_extra_passes'],0)
        for name in ('qwen_agent_compact_record.py','qwen_record_storage.py','qwen_agent_grouped_record.py','qwen_grouped_ids.py'):
            self.assertTrue((out/'variant-source'/name).is_file())
        namespace=entry._namespace()
        self.assertEqual((namespace['MODULE'],namespace['ROUND'],namespace['MODE']),(entry.MODULE,entry.ROUND,entry.MODE))
        self.assertIs(namespace['source_gate'],entry.source_gate)
        argv=['--out',str(out),'--partial-from',str(self.directory/'partial'),'--resume-from',str(self.directory/'resume'),'--through','agent']
        current=entry.parser().parse_args(argv);parent=entry.compact.parser().parse_args(argv)
        self.assertEqual(vars(current),vars(parent))
        self.assertEqual((current.memory_gib,current.rss_gib,current.minimum_available_gib,current.case_gib,current.seconds),(24,16,1.25,32,10800))
        command=namespace['phase_command'](current,'step',self.directory/'step')
        worker=command[command.index('--')+1:]
        self.assertEqual(worker[worker.index('-m')+1],entry.MODULE)
        marker=object()
        with patch.object(entry.compact,'session_outcome',return_value=marker)as delegate:
            self.assertIs(entry.session_outcome(self.directory/'session'),marker)
        delegate.assert_called_once_with(self.directory/'session')

    def test_actual_retained_reply_transaction_dispatches_grouped_factory(self):
        from pt_agent.dsl.parser import parse
        from pt_agent.experiments import qwen_guarded_replay as guarded
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        session=self.directory/'session';session.mkdir();inherited=self.directory/'inherited'
        (inherited/'calls').mkdir(parents=True)
        state=dict(phase='evaluate',iteration=13,calls=12,max_iters=15,history=[],pending=None,prompt_tokens=0,completion_tokens=0)
        (session/'SESSION.json').write_text(json.dumps(state))
        (session/'guarded-replay.json').write_text(json.dumps(dict(status='pending_local_replay',inherited_session=str(inherited))))
        (inherited/'calls/call-0013.response.json').write_text(json.dumps(dict(content='trusted tiny fixture',usage=dict(prompt_tokens=2,completion_tokens=3))))
        tools=ModuleType('pt_agent.planners.agent_tools')
        def evaluate(source,ctx,**kwargs):
            from .approximation_failures import program_failures
            return estimate_normalized_fast(ctx.program,{},profile='qwen48')
        original_evaluate=FunctionType(evaluate.__code__,tools.__dict__,'evaluate')
        tools.__dict__.update(evaluate=original_evaluate,_exact_check=lambda:None,cost_feedback=lambda:None,validate=lambda:None)
        def bridge(original_evaluate):
            def evaluate(source,ctx,**kwargs):return original_evaluate(source,ctx,**kwargs)
            return evaluate
        module=ModuleType('trusted_grouped_fixture');functions=module.__dict__
        def step(ctx,working,*,trace_path,client,reply):
            if client is not None:raise AssertionError('client must remain None')
            globals()['observed'].append(globals()['evaluate'](reply['content'],ctx))
            working.update(calls=13,iteration=14,pending=None,prompt_tokens=2,completion_tokens=3)
            working['history'].append(dict(iteration=13));trace_path.write_text(json.dumps(dict(iteration=13))+'\n')
        observed=[]
        functions.update(evaluate=bridge(original_evaluate),observed=observed,json=json,
            feedback_message=lambda *a,**k:'metadata',agent_step=FunctionType(step.__code__,functions,'agent_step'))
        installed=[];original_grouped=entry.GroupedIds
        class TrackedGrouped(original_grouped):
            def __init__(self):super().__init__();installed.append(True)
        work=self.directory/'step';work.mkdir();args=SimpleNamespace(session=session,work_dir=work,worker_phase='step');result={}
        with patch.object(guarded,'seed_partial'),patch.object(guarded.bridge,'_context',return_value=SimpleNamespace(program=parse(PROGRAM))), \
             patch.object(guarded.bridge,'_agent_functions',return_value=functions), \
             patch.object(guarded.bounded,'session_outcome',return_value=dict(frozen=False)),patch.object(entry,'GroupedIds',TrackedGrouped):
            entry._namespace()['_agent_phase'](args,StageRecorder(work/'events.jsonl'),result)
        saved=json.loads((session/'SESSION.json').read_text())
        self.assertTrue(installed);self.assertEqual(len(observed),1)
        self.assertEqual((saved['calls'],saved['iteration'],saved['max_iters']),(13,14,15))
        self.assertEqual(state['calls'],12);self.assertEqual(result['new_provider_requests_for_13'],0)
        self.assertEqual(json.loads((session/'guarded-replay.json').read_text())['HTTP_added'],0)
        self.assertEqual(result['cost_lifetime_variant'],entry.compact.base.VARIANT)


if __name__ == '__main__':
    unittest.main()
