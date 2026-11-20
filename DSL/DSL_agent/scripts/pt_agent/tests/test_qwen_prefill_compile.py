"""Small fake protocol tests; no original model, data, fitting or compiler."""
from argparse import Namespace
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType,SimpleNamespace
import sys
import unittest
import weakref
from unittest.mock import patch

from pt_agent.experiments import qwen_prefill_compile as entry


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def _streaming_projection_fixture(backend,xs=None,layout=None,weight=None):
    if not isinstance(backend,(CountBackend,StreamingMockBackend,StreamingEmitBackend)):
        raise ValueError('Unsupported backend fixture')
    return backend,xs,layout,weight


class QwenPrefillCompileChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def arguments(self):
        return Namespace(source_case=self.root/'frozen',out=self.root/'compile',intake_python=self.root/'python',
            memory_gib=32.,case_gib=128,seconds=10800,through='compile',diagnostic_elide_final_ir=True)

    def source_fixture(self):
        case=self.root/'frozen';session=case/'session'
        argv=['python3','-B','-m',entry.bounded.MODULE,'--resume-from',str(self.root/'old'),
            '--worker-phase','calibration','--session',str(session),'--model-info',str(self.root/'info'),
            '--model',str(self.root/'original'),'--calibration',str(case/'calibration.json')]
        save(case/'command-1.json',argv)
        source=dict(original_layers=24,graph_operation_counts=dict(softmax=24,shuffle=48),
            model_info=str(self.root/'info/model_info.json'))
        save(case/'result.json',dict(round=entry.bounded.ROUND,mode=entry.bounded.MODE,model=entry.MODEL,
            status='ok',candidate_frozen=True,source=source))
        save(case/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
        save(session/'SESSION.json',dict(failed=None,calls=12));save(case/'calibration.json',{})
        save(case/'unit-width-bridge.json',{})
        # Native paths are inherited from the saved argv, never user descriptors.
        argv+=['--metadata-template',str(self.root/'template.json'),'--frontend',str(self.root/'expr.py'),
            '--profile',str(entry.paths.PROFILES_DIR/'qwen48/compiler-config.json')];save(case/'command-1.json',argv)
        outcome=dict(done=True,frozen=True,status='ok',selected=dict(iteration=11,origin='llm'))
        return case,source,outcome

    def test_saved_argv_is_parsed_without_invoking_worker(self):
        case,_,_=self.source_fixture();module,args=entry.source_arguments(case)
        self.assertIs(module,entry.bounded);self.assertEqual(args.session,case/'session')
        self.assertIsNone(args.out);self.assertEqual(args.resume_from,self.root/'old')

    def test_saved_argv_list_and_trusted_module_are_required(self):
        case,_,_=self.source_fixture();save(case/'command-1.json',dict(argv=[]))
        with self.assertRaisesRegex(ValueError,'argv list'):entry.source_arguments(case)
        save(case/'command-1.json',['python3','-m','untrusted.module'])
        with self.assertRaisesRegex(ValueError,'trusted'):entry.source_arguments(case)

    def test_frozen_source_keeps_full_graph_and_has_no_report_selector(self):
        case,source,outcome=self.source_fixture()
        with patch.object(entry.bounded,'session_outcome',return_value=outcome),patch.object(entry.bounded,'source_gate',return_value=source):
            gate=entry.frozen_source(case)
        self.assertEqual(gate['full_model_layers'],24);self.assertEqual(gate['inherited_calls'],12)
        self.assertFalse(gate['encrypted_selector_included']);self.assertFalse(gate['token_embedding_feedback_included'])
        self.assertFalse(gate['report_outcome_used_for_selection']);self.assertEqual(gate['forward_calls'],1)

    def test_pending_or_error_or_unconfirmed_source_rejected(self):
        case,source,outcome=self.source_fixture()
        for changed in (dict(done=False),dict(frozen=False),dict(status='error')):
            with self.subTest(changed=changed),patch.object(entry.bounded,'session_outcome',return_value=dict(outcome,**changed)),patch.object(entry.bounded,'source_gate',return_value=source):
                with self.assertRaisesRegex(ValueError,'C2-confirmed'):entry.frozen_source(case)

    def test_saved_integer_feedback_keys_compare_by_json_semantics(self):
        case,source,outcome=self.source_fixture();source['recovery']=dict(pinned_error_counts={8:43})
        saved=json.loads((case/'result.json').read_text());saved['source']=source;save(case/'result.json',saved)
        self.assertEqual(json.loads((case/'result.json').read_text())['source']['recovery']['pinned_error_counts'],{'8':43})
        with patch.object(entry.bounded,'session_outcome',return_value=outcome),patch.object(entry.bounded,'source_gate',return_value=source):
            self.assertTrue(entry.frozen_source(case)['candidate_frozen'])
        self.assertEqual(source['recovery']['pinned_error_counts'],{8:43})

    def test_source_graph_or_original_metadata_drift_rejected(self):
        case,source,outcome=self.source_fixture()
        bad=copy.deepcopy(source);bad['graph_operation_counts']['shuffle']=47
        with patch.object(entry.bounded,'session_outcome',return_value=outcome),patch.object(entry.bounded,'source_gate',return_value=bad):
            with self.assertRaisesRegex(ValueError,'24 decoder'):entry.frozen_source(case)
        with patch.object(entry.bounded,'session_outcome',return_value=outcome),patch.object(entry.bounded,'source_gate',return_value=dict(source,changed=True)):
            with self.assertRaisesRegex(ValueError,'metadata differs'):entry.frozen_source(case)

    def test_explicit_phase_argv_preserves_budgets_and_has_no_agent_phase(self):
        args=self.arguments()
        for phase,seconds in entry.PHASE_SECONDS.items():
            command=entry.phase_command(args,phase,self.root/phase)
            self.assertEqual(command[command.index('--seconds')+1],str(seconds+60))
            self.assertEqual(command[command.index('--worker-phase')+1],phase)
            self.assertIn('--diagnostic-elide-final-ir',command);self.assertNotIn('--ledger',command)
            self.assertNotIn('--through',command)
        self.assertEqual(entry.PHASE_SECONDS['compile'],7200);self.assertEqual(entry.PHASE_SECONDS['trace'],1800)
        shortened=entry.phase_command(args,'compile',self.root/'shortened',remaining=900)
        self.assertEqual(shortened[shortened.index('--phase-seconds')+1],'780')
        self.assertEqual(shortened[shortened.index('--seconds')+1],'840')

    def test_resource_controls_reject_nonfinite_bool_and_out_of_range(self):
        args=self.arguments();limits=entry.resource_limits(args)
        self.assertEqual(limits['subtree_rss_gib'],14);self.assertEqual(limits['minimum_available_gib'],1.25)
        self.assertEqual(limits['file_gib'],64)
        for value in (True,float('nan'),float('inf'),11,65):
            args.memory_gib=value
            with self.subTest(value=value),self.assertRaises(ValueError):entry.resource_limits(args)
        args.memory_gib=32
        for value in (True,31,129,64.):
            args.case_gib=value
            with self.subTest(case=value),self.assertRaises(ValueError):entry.resource_limits(args)

    def test_same_lowering_uses_compact_scope_and_restores_on_failure(self):
        active=[];compact=ModuleType('pt_agent.kernels.compact_linear');lower=ModuleType('pt_agent.lower')
        @contextmanager
        def scope():
            active.append(True)
            try:yield
            finally:active.pop()
        compact.compact_transformer_projections=scope
        def apply(program,weights,backend,**kwargs):
            self.assertEqual(active,[True]);self.assertEqual(kwargs,dict(act_mode='program'))
            if program=='fail':raise RuntimeError('fixture')
            return ['output'],['layout']
        lower.lower=apply
        streaming=ModuleType('pt_agent.kernels.streaming_linear');streaming.streaming_projection=_streaming_projection_fixture
        with patch.dict(sys.modules,{'pt_agent.kernels.compact_linear':compact,'pt_agent.lower':lower,
                                    'pt_agent.kernels.streaming_linear':streaming}):
            self.assertEqual(entry._same_lowering('program','original weights','backend'),(['output'],['layout']))
            with self.assertRaisesRegex(RuntimeError,'fixture'):entry._same_lowering('fail','weights','backend')
        self.assertEqual(active,[])

    def test_private_projection_guard_preserves_wrapper_and_restores_source_on_error(self):
        class CostDag:pass
        class Count:pass
        class Emit:pass
        class Refresh:
            def __init__(self,inner):self.inner=inner
        def allowed(value):
            return isinstance(value.inner if isinstance(value,Refresh)else value,(CostDag,Count,Emit))
        module=SimpleNamespace(streaming_projection=_streaming_projection_fixture)
        original=module.streaming_projection
        for inner in (CostDag(),Count(),Emit()):
            wrapper=Refresh(inner)
            with entry._projection_backend_scope(module,allowed):
                returned=module.streaming_projection(wrapper,'original values','layout','weight')
                self.assertIs(returned[0],wrapper);self.assertIs(returned[0].inner,inner)
            self.assertIs(module.streaming_projection,original)
        with self.assertRaisesRegex(ValueError,'Unsupported backend'):
            with entry._projection_backend_scope(module,allowed):module.streaming_projection(Refresh(object()))
        self.assertIs(module.streaming_projection,original)

    def test_projection_delegate_rejects_original_guard_drift(self):
        with self.assertRaisesRegex(ValueError,'type guard changed'):
            entry._projection_delegate(save,lambda _:True)

    def test_compiler_argv_uses_existing_identifier_and_complete_payload(self):
        args=self.arguments();args.codegen=self.root/'codegen';args.worker_phase='compile'
        args.metadata_template=self.root/'template';save(args.metadata_template,['original argv'])
        args.profile=self.root/'profile'
        builder='pt_agent.experiments.r29_local_compile_matrix.compiler_argv'
        printer='pt_agent.experiments.r30_common_model_e2e.diagnostic_compile_argv'
        with patch(builder,return_value=['compiler','original.mlir'])as compiled,patch(printer,side_effect=lambda argv,**kw:argv+['diagnostic'])as diagnostic:
            command=entry.native_command(args,dict(hecate_opt=self.root/'compiler'))
        self.assertEqual(command[-1],'diagnostic');self.assertEqual(compiled.call_args.args,(['original argv'],'pars'))
        self.assertEqual(compiled.call_args.kwargs['profile'],args.profile)
        diagnostic.assert_called_once_with(['compiler','original.mlir'],elide_final_ir=True)

    def test_compiled_inventory_streams_all_operations_and_rejects_incomplete_outputs(self):
        path=self.root/'plan.json'
        plan=dict(values=[dict(id='fixture')],external_inputs=['x'],final_outputs=[str(i)for i in range(19)],
            execution=[dict(kind='compute',op=op)for op in ('add_cc','boot','rotate','boot')])
        save(path,plan);observed=entry.compiled_inventory(path,2)
        self.assertEqual(observed['compiled_operation_count'],4);self.assertEqual(observed['compiled_bootstraps'],2)
        self.assertEqual(observed['compiled_outputs'],19)
        with self.assertRaisesRegex(ValueError,'planned bootstraps'):entry.compiled_inventory(path,1)
        plan['final_outputs'].pop();save(path,plan)
        with self.assertRaisesRegex(ValueError,'19 vocabulary'):entry.compiled_inventory(path,2)

    def codegen_fixture(self):
        active=[];observed=[];created=[]
        compact=ModuleType('pt_agent.kernels.compact_linear');lower=ModuleType('pt_agent.lower')
        plan_module=ModuleType('pt_agent.plan');backends=ModuleType('pt_agent.backends')
        stream=ModuleType('pt_agent.backends.streaming_emit')
        count=ModuleType('pt_agent.backends.streaming_count')
        @contextmanager
        def scope():
            active.append(True)
            try:yield
            finally:active.pop()
        compact.compact_transformer_projections=scope
        stats=dict(counts=dict(boot=1),ops_total=7,distinct_consts=2)
        class Counter:
            def __init__(self,*a,**kw):pass
            def stats(self):return copy.deepcopy(stats)
        class Refresh:
            def __init__(self,inner,**kw):self.inner=inner
            def stats(self):return dict(boots=1,calls=42)
        class Stream(Counter):
            def __init__(self,*a,**kw):
                self.directory=kw['directory'];self.finished=False;self.closed=False;created.append(self)
            def write(self):self.finished=True;return self.directory/'candidate_trace.py'
            def close(self):self.closed=True
        backends.CountBackend=Counter;backends.RefreshBackend=Refresh;stream.StreamingEmitBackend=Stream
        count.StreamingCountBackend=Counter
        plan=dict(boots=1,placement='original program-plan',levels=dict(boot_out_level=5,boot_in_level=1,input_level=8))
        def placement(*a,**kw):
            self.assertEqual(active,[True]);self.assertEqual(kw,dict(profile='qwen48',frontend_normalize=True))
            observed.append('place');return plan,object(),object()
        def verify(*a,**kw):
            self.assertEqual(active,[True]);observed.append('verify');return dict(boots=1,inner_boots=1)
        def apply(program,weights,backend,**kw):
            self.assertEqual(active,[True]);self.assertIs(weights,original_weights)
            self.assertEqual(kw,dict(act_mode='program'));observed.append('lower')
            return ['one symbolic block']*19,['original full vocabulary layout']
        plan_module.plan_program=placement;plan_module.verify_program=verify;lower.lower=apply
        original_weights=object();info=SimpleNamespace(graph=SimpleNamespace(weights=original_weights))
        program=SimpleNamespace(header=SimpleNamespace(slots=16384))
        modules={'pt_agent.kernels.compact_linear':compact,'pt_agent.lower':lower,'pt_agent.plan':plan_module,
            'pt_agent.backends':backends,'pt_agent.backends.streaming_emit':stream,'pt_agent.backends.streaming_count':count}
        streaming=ModuleType('pt_agent.kernels.streaming_linear');streaming.streaming_projection=_streaming_projection_fixture
        modules['pt_agent.kernels.streaming_linear']=streaming
        return modules,info,program,observed,created,stats

    def test_placement_replay_and_emit_use_same_compact_lower_and_original_weights(self):
        args=self.arguments();args.codegen=self.root/'codegen';args.codegen.mkdir();args.frontend=self.root/'expr.py'
        modules,info,program,observed,created,_=self.codegen_fixture()
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        recorder=StageRecorder(self.root/'events.jsonl');placed={};emitted={}
        planner=vars(modules['pt_agent.plan'])
        with patch.dict(sys.modules,modules),patch.object(entry,'_planner_namespace',return_value=planner),patch.object(entry,'_load_program',return_value=(info,program)):
            entry._placement(args,placed,recorder);entry._emit(args,emitted,recorder)
        self.assertEqual(observed,['place','verify','lower','lower'])
        self.assertEqual(placed['output_blocks'],19);self.assertEqual(emitted['output_blocks'],19)
        self.assertTrue(placed['placement_completed']);self.assertTrue(emitted['emit_completed'])
        self.assertTrue(created[0].finished);self.assertFalse(created[0].closed)

    def test_private_planner_replaces_dag_and_count_without_changing_original_globals(self):
        module=ModuleType('fixture_qwen_plan');original_dag=type('OriginalDag',(),{})
        original_count=type('OriginalCount',(),{});stream_count=type('StreamCount',(),{})
        module.CostDag=original_dag;module.CountBackend=original_count
        exec('def selected_classes():\n    return CostDag, CountBackend\n',module.__dict__)
        numeric=SimpleNamespace(ndarray=type('Array',(),{}),array=lambda x,copy:x)
        private=entry._planner_namespace(module,original_dag,stream_count,numeric)
        dag,count=private['selected_classes']()
        self.assertTrue(issubclass(dag,original_dag));self.assertIs(count,stream_count)
        self.assertEqual(module.selected_classes(),(original_dag,original_count))

    def test_readonly_dag_constant_values_are_ephemeral_and_original_array_unchanged(self):
        class Array:
            def __init__(self,values,writeable=False):self.values=tuple(values);self.flags=SimpleNamespace(writeable=writeable)
        class Dag:
            def __init__(self):self.identity_cache={};self.stored=[];self.calls=0
            def _const(self,value,*,keep_data):
                self.calls+=1
                if not value.flags.writeable:self.identity_cache[id(value)]=value
                self.stored.append(None)
                return SimpleNamespace(data=value,keep_data=keep_data)
        copies=[]
        def copy_array(value,*,copy):
            self.assertTrue(copy);array=Array(value.values,True);copies.append(weakref.ref(array));return array
        numeric=SimpleNamespace(ndarray=Array,array=copy_array);subclass=entry._ephemeral_cost_dag(Dag,numeric)
        original=Array([1.,-0.,2.]);backend=subclass();operand=backend.const(original)
        self.assertEqual(operand.data.values,original.values);self.assertTrue(operand.keep_data)
        self.assertFalse(original.flags.writeable);self.assertEqual(backend.identity_cache,{})
        self.assertEqual(backend.stored,[None]);self.assertEqual(backend.calls,1)
        del operand;gc.collect();self.assertIsNone(copies[0]())

    def test_count_emit_mismatch_stops_before_source_write_and_closes_store(self):
        args=self.arguments();args.codegen=self.root/'codegen';args.codegen.mkdir();args.frontend=self.root/'expr.py'
        modules,info,program,_,created,stats=self.codegen_fixture()
        save(args.codegen/'refresh-plan.json',dict(boots=1,levels=dict(boot_out_level=5,boot_in_level=1,input_level=8)))
        save(args.codegen/'count-replay.json',dict(stats=dict(stats,distinct_consts=3)))
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        with patch.dict(sys.modules,modules),patch.object(entry,'_load_program',return_value=(info,program)):
            with self.assertRaisesRegex(ValueError,'Count/Emit schedule'):entry._emit(args,{},StageRecorder(self.root/'events'))
        self.assertFalse(created[0].finished);self.assertTrue(created[0].closed)

    def fake_execute(self,*,failed_phase=None):
        args=self.arguments();calls=[]
        class NoThread:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def is_alive(self):return False
        def run(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1]);calls.append((phase,kwargs))
            child=dict(status='error'if phase==failed_phase else'ok',**{phase+'_completed':phase!=failed_phase})
            if phase=='compile':child.update(compiled_bootstraps=1234,runtime_plan='fixture-plan')
            save(work/'worker-result.json',child)
            return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],started_ns=1,ended_ns=2)
        with patch.object(entry,'frozen_source',return_value=dict(candidate_frozen=True)),patch.object(entry.paths,'RESULTS_DIR',self.root),\
                patch.object(entry,'budget',return_value={}),patch.object(entry.threading,'Thread',NoThread),\
                patch.object(entry,'run_process',side_effect=run),patch('pt_agent.registry.git_rev',return_value='fixture'):
            result=entry.execute(args)
        return args,result,calls

    def test_all_four_phases_run_once_with_native7200_and_complete_registry(self):
        args,result,calls=self.fake_execute()
        self.assertEqual(result['status'],'ok');self.assertEqual([x[0]for x in calls],list(entry.PHASE_SECONDS))
        self.assertEqual(result['completed'],dict.fromkeys(entry.PHASE_SECONDS,1))
        self.assertEqual(calls[-1][1]['timeout'],7320);self.assertEqual(calls[-1][1]['fsize_gib'],64)
        registry=[json.loads(line)for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry),1);self.assertEqual(len(registry[0]),8)
        self.assertTrue(read_complete(args.out)['required_artifacts_persisted'])
        self.assertEqual(result['artifact_bytes'],sum(p.stat().st_size for p in args.out.rglob('*')if p.is_file()and p.name!='completion.json'))
        self.assertTrue(all(result[k]==0 for k in entry.ZERO));self.assertTrue(all(result[k]is None for k in entry.FORMAL))

    def test_first_phase_error_preserves_partial_and_stops_following_actions(self):
        args,result,calls=self.fake_execute(failed_phase='emit')
        self.assertEqual(result['status'],'error');self.assertEqual([x[0]for x in calls],['placement','emit'])
        self.assertEqual(result['failed_phase'],'emit');self.assertEqual(result['completed']['placement'],1)
        self.assertEqual(result['completed']['emit'],0);self.assertEqual(result['attempted']['trace'],0)
        self.assertEqual(result['phase_results']['emit']['status'],'error')

    def test_quality_pending_prevents_output_creation_and_any_child(self):
        args=self.arguments()
        with patch.object(entry,'frozen_source',side_effect=ValueError('not frozen')),patch.object(entry,'run_process')as child:
            with self.assertRaisesRegex(ValueError,'not frozen'):entry.execute(args)
        child.assert_not_called();self.assertFalse(args.out.exists())

    def test_case_budget_reserves_completion_and_storage_guard(self):
        args=self.arguments()
        with patch.object(entry,'artifact_bytes',return_value=args.case_gib*entry.GIB-entry.TERMINAL_RESERVE+1),patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.GIB)):
            with self.assertRaises(RuntimeError):entry.budget(args,self.root)
        with patch.object(entry,'artifact_bytes',return_value=0),patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB-1)):
            with self.assertRaises(RuntimeError):entry.budget(args,self.root)


def read_complete(out):return json.loads((out/'completion.json').read_text())


if __name__=='__main__':unittest.main()
