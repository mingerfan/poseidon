"""Bounded new fixtures; no real model, data, frontend, native tool or API."""
from argparse import Namespace
import ast
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_remaining_prefill_compile as entry
from pt_agent.tests.test_qwen_layered_structure import fixture
from pt_agent.tests.test_qwen_layer_native_probe import generic_trace, native_plan


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def clean():return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])


def head_plan():
    data=native_plan();data['execution']=data['execution'][:38]
    data['values']=data['values'][:72];data['final_outputs']=data['final_outputs'][:19]
    return data


def head_trace():
    source=generic_trace();tensor='tensor<1x!earth.ci<0 * 0>>'
    return source.replace(' -> ('+', '.join([tensor]*32)+'), sym_name',
                          ' -> ('+', '.join([tensor]*19)+'), sym_name',1)


class RemainingPrefillChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.source=self.root/'frozen';self.first=self.root/'probe';self.results=self.root/'results';self.results.mkdir()
        self.program,self.shapes=fixture();self.partition=entry.layered.split_program(self.program,self.shapes)
        self.bindings=dict(layers=[entry.layered.source_binding(x)for x in self.partition.layers],head=entry.layered.source_binding(self.partition.head))
        self.args=Namespace(source_case=self.source,probe_case=self.first,out=self.results/'continuation',
            component=None,worker_phase=None,through='compile',work_dir=None,codegen=None,intake_python=self.root/'python',
            memory_gib=24.,case_gib=128,seconds=43200,phase_seconds=None,diagnostic_elide_final_ir=True)
        self.frozen=dict(source_case=str(self.source.resolve()),program=str(self.source/'program.ptdsl'),
            metadata_template=str(self.root/'metadata.json'),frontend=str(self.root/'frontend.py'),profile=str(self.root/'profile.json'))
        self.proof=dict(frozen_source=self.frozen,source_case=str(self.source.resolve()),probe_case=str(self.first.resolve()),
            bindings=self.bindings,first_layer_runtime_plan=str(self.first/'codegen/emit/lowered.fixture.runtime-plan.json'),
            first_layer_native=dict(compiled_bootstraps=32),first_layer_phase_clocks={},first_layer_timing={},
            full_prefill_verified=False,whole_generation_verified=False)

    def source_fixture(self):
        self.source.mkdir();(self.source/'program.ptdsl').write_text(self.fixture_source())
        save(self.first/'codegen/source-bindings.json',dict(self.bindings,planned_layers=[0],remaining_plans='not_run'))
        save(self.first/'codegen/planner-return.json',dict(source=self.bindings['layers'][0],planner_returned=True,
            verify_completed=True,planner='original.pt_agent.plan.plan_program'))
        save(self.first/'codegen/refresh-plan.json',dict(boots=0))
        path=self.first/'codegen/emit/lowered.fixture.runtime-plan.json';save(path,native_plan())
        accepted=entry.probe.inspect_runtime_plan(path,32)
        phases={}
        for phase in entry.PHASE_SECONDS:
            child=dict(status='ok',**{phase+'_completed':True})
            if phase=='compile':child.update(accepted)
            phases[phase]=child;save(self.first/'phases'/phase/'worker-result.json',child)
        record=dict(round=entry.probe.ROUND,mode=entry.probe.MODE,model=entry.MODEL,status='ok',probe_accepted=True,
            full_model_compile_verified=False,probe_layer_index=0,source=dict(self.frozen,probe=True),runtime_plan=str(path),
            phase_results=phases,attempted=dict.fromkeys(entry.PHASE_SECONDS,1),completed=dict.fromkeys(entry.PHASE_SECONDS,1),
            processes=[dict(phase=p,observation=clean())for p in entry.PHASE_SECONDS],
            **dict.fromkeys(entry.original.ZERO,0),**dict.fromkeys(entry.original.FORMAL))
        save(self.first/'result.json',record)
        save(self.first/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
        return record

    def fixture_source(self):
        # Retrieve the public shape-only DSL fixture without running a model.
        import inspect
        module=inspect.getmodule(fixture)
        # Canonical source is captured from the parser call inside this fixture.
        with patch.object(module,'parse',side_effect=lambda source:source):return fixture()[0]

    def source_patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(entry.original,'frozen_source',return_value=self.frozen))
        stack.enter_context(patch.object(entry.probe,'admission',return_value=dict(self.frozen,probe=True)))
        return stack

    def execute_fake(self,runner=None):
        self.calls=[]
        def run(argv,**kwargs):
            self.calls.append((argv,kwargs));return runner(argv,**kwargs)if runner else clean()
        def accepted(case,name,checked):
            return dict(status='ok',component=entry.spec(name),case=str(case),source_binding=checked['bindings']['head']if name=='head'else checked['bindings']['layers'][entry.spec(name)['index']],
                native=dict(compiled_bootstraps=entry.spec(name)['outputs']),timing={},phase_results={})
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results),patch('pt_agent.registry.git_rev',return_value='fixture'), \
                patch.object(entry,'admission',return_value=self.proof),patch.object(entry,'accept_component',side_effect=accepted), \
                patch.object(entry,'_available_ram_bytes',return_value=32*entry.GIB), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=200*entry.GIB)), \
                patch.object(entry,'run_process',side_effect=run):
            return entry.execute(self.args)

    def test_head_and_layer_spec_are_explicit_finite_and_distinct(self):
        self.assertEqual(entry.spec('head')['outputs'],19);self.assertEqual(entry.spec('layer23')['outputs'],32)
        self.assertEqual(len(entry.COMPONENTS),24);self.assertNotIn('layer0',entry.COMPONENTS)
        for name in ('layer24','layer01','selector'):
            with self.assertRaises(ValueError):entry.spec(name)

    def test_complete_actual_first_layer_admission_rechecks_AST_and_native(self):
        self.source_fixture()
        with self.source_patches():checked=entry.admission(self.source,self.first)
        self.assertEqual(checked['bindings'],self.bindings)
        self.assertTrue(checked['first_layer_native']['native_ABI_verified'])
        self.assertFalse(checked['full_prefill_verified'])

    def test_unfrozen_or_failed_probe_rejects_before_component_or_weights(self):
        record=self.source_fixture();record['probe_accepted']=False;save(self.first/'result.json',record)
        with self.source_patches(),patch.object(entry.original,'_load_program')as load:
            with self.assertRaises(ValueError):entry.admission(self.source,self.first)
            load.assert_not_called()
        with patch.object(entry.original,'frozen_source',side_effect=ValueError('unfrozen')),patch.object(entry.original,'_load_program')as load:
            with self.assertRaises(ValueError):entry.admission(self.source,self.first)
            load.assert_not_called()

    def test_source_AST_weight_shapes_and_probe_runtime_mutations_reject(self):
        record=self.source_fixture()
        saved=entry.original.read(self.first/'codegen/source-bindings.json');saved['layers'][2]['original_ops'][-1]['kind']='negate'
        save(self.first/'codegen/source-bindings.json',saved)
        with self.source_patches():
            with self.assertRaises(ValueError):entry.admission(self.source,self.first)
        saved['layers'][2]=self.bindings['layers'][2];save(self.first/'codegen/source-bindings.json',saved)
        path=Path(record['runtime_plan']);data=entry.original.read(path);data['values'][0]['level']=4;save(path,data)
        with self.source_patches():
            with self.assertRaises(ValueError):entry.admission(self.source,self.first)

    def test_head_generic_header_retains_raw34inputs19outputs_with_only_attributes(self):
        raw=self.root/'raw.mlir';marked=self.root/'marked.mlir';text=head_trace();raw.write_text(text)
        result=entry.mark_trace(raw,marked,19)
        attrs=' {is_mid_segment = true, segment_returnBypasses = ['+', '.join(['true']*19)+']}'
        self.assertEqual(marked.read_text().replace(attrs,'',1),text);self.assertEqual(raw.read_text(),text)
        self.assertEqual(result['physical_outputs'],19);self.assertEqual(len(result['return_bypasses']),19)
        self.assertFalse(result['native_ABI_verified'])
        wrong=self.root/'wrong.mlir';wrong.write_text(generic_trace())
        with self.assertRaises(ValueError):entry.mark_trace(wrong,self.root/'wrong-copy.mlir',19)

    def test_header_conflicting_existing_attrs_and_multiple_functions_reject(self):
        for i,text in enumerate((head_trace().replace('  }) :','  }) {is_mid_segment=false} :',1),head_trace()+head_trace())):
            raw=self.root/str(i);raw.write_text(text)
            with self.assertRaises(ValueError):entry.mark_trace(raw,self.root/(str(i)+'-copy'),19)

    def test_real_head_runtime_metadata19_with_actual_final_boots_and_SSA(self):
        path=self.root/'head.runtime-plan.json';save(path,head_plan())
        proof=entry.inspect_plan(path,19,'head')
        self.assertEqual(proof['physical_inputs'],34);self.assertEqual(proof['physical_outputs'],19)
        self.assertEqual(proof['compiled_bootstraps'],19);self.assertTrue(proof['complete_native_SSA_verified'])
        self.assertFalse(proof['full_prefill_verified']);self.assertFalse(proof['whole_generation_verified'])
        self.assertEqual(entry.probe.inspect_runtime_plan.__code__.co_consts.count(32),1)
        with self.assertRaises(ValueError):entry.probe.inspect_runtime_plan(path,19)

    def test_head_runtime_wrong_arity_level_scale_context_boot_and_SSA_reject(self):
        original=head_plan()
        for label,change in (
                ('arity',lambda d:d['final_outputs'].pop()),('level',lambda d:d['values'][0].update(level=4)),
                ('scale',lambda d:d['values'][-1].update(scale_log2=42)),
                ('context',lambda d:d['values'][0].update(context='other')),
                ('boot',lambda d:d['execution'][-1]['attrs'].update(target_level=4)),
                ('ssa',lambda d:d['execution'][0].update(inputs=['9999']))):
            with self.subTest(label=label):
                data=deepcopy(original);change(data);path=self.root/(label+'.json');save(path,data)
                with self.assertRaises(ValueError):entry.inspect_plan(path,19,'head')

    def test_small_original_planner_verify_Count_Emit_for_layer_and_head(self):
        from pt_agent.backends.streaming_count import StreamingCountBackend
        from pt_agent.backends.streaming_emit import StreamingEmitBackend
        for role,segment,outputs in (('layer',self.partition.layers[1],32),('head',self.partition.head,19)):
            with self.subTest(role=role):
                weights={name:SimpleNamespace(shape=tuple(shape))for name,shape in self.shapes.items()}
                callback=entry.layered.SegmentLowering(segment,weights)
                def lower(active,outputs=outputs,role=role):
                    inputs=[active.input(i)for i in range(34)];values=[x*.5 for x in inputs[:outputs]]
                    active.output(values+inputs[32:]if role=='layer'else values)
                planner=entry.probe.planner_namespace()if role=='layer'else entry.original._planner_namespace()
                placed=SimpleNamespace(refreshed_uses={},boot_nodes=[],boots=0,barriers=0,cuts=[],latency_us=0.)
                with patch.object(entry.layered.SegmentLowering,'__call__',lambda cb,backend:lower(backend)), \
                        patch.object(entry.original,'_projection_backend_scope',return_value=ExitStack()):
                    dag,recorder,_=planner['record_dag'](callback,16384,frontend_normalize=True)
                    plan=planner['express'](dag,recorder,placed)
                    self.assertEqual(planner['verify'](callback,16384,plan)['boots'],0)
                    bound=entry.layered.bind_plan(segment,plan,planned_source=entry.layered.source_binding(segment))
                    count=StreamingCountBackend(16384,input_level=5,boot_out_level=5)
                    state=entry.export_component(count,callback,bound)
                    self.assertEqual(count.stats()['counts']['boot'],outputs)
                    frontend=self.root/(role+'-frontend.py');frontend.write_text('trusted fixture pathname only\n')
                    emit=StreamingEmitBackend(16384,directory=self.root/(role+'-emit'),frontend=frontend,max_constant_bytes=1024**2)
                    try:
                        emitted=entry.export_component(emit,callback,bound);source=emit.write();tree=ast.parse(source.read_text())
                        golden=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='golden')
                        self.assertEqual(len(golden.args.args),34)
                        self.assertEqual(golden.decorator_list[0].args[0].value,','.join(['c']*34))
                        self.assertEqual(len(next(n for n in ast.walk(golden)if isinstance(n,ast.Return)).value.elts),outputs)
                        self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='bootstrap'for n in ast.walk(golden)),outputs)
                        self.assertEqual(emit.stats()['counts'],count.stats()['counts'])
                        self.assertEqual(emitted.metadata['explicit_boundary_bootstrap_calls'],outputs)
                        self.assertEqual(state.metadata['physical_output_blocks'],outputs)
                    finally:
                        if not emit.finished:emit.close()

    def test_head_export_exception_restores_backend_hooks(self):
        from pt_agent.backends.streaming_count import StreamingCountBackend
        backend=StreamingCountBackend(16384,input_level=5,boot_out_level=5)
        callback=entry.layered.SegmentLowering(self.partition.head,{})
        plan=dict(format='pt-agent-refresh-plan-v1',levels=dict(entry.layered.staged.LEVELS),calls=34,boots=0,substitutions=[],outputs=[],dead=[])
        bound=entry.layered.bind_plan(callback.segment,plan,planned_source=entry.layered.source_binding(callback.segment))
        with patch.object(entry.layered.SegmentLowering,'__call__',side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError,'fixture'):entry.export_component(backend,callback,bound)
        self.assertNotIn('input',backend.__dict__);self.assertNotIn('output',backend.__dict__)

    def test_phase_commands_keep_pure_paths_roles_AS24_and_finite_time(self):
        self.args.component='head';self.args.seconds=10800
        command=entry.phase_command(self.args,'compile',self.root/'worker',remaining=900)
        for flag,value in (('--component','head'),('--probe-case',str(self.first)),('--memory-gib','24')):
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertEqual(int(command[command.index('--phase-seconds')+1]),780)
        self.assertIn('pt_agent.tools.nixrun',command);self.assertNotIn('--registry',command)

    def component_fixture(self,name='head'):
        case=self.results/name;target=entry.spec(name);source=entry.component_source(self.proof,name)
        path=case/'codegen/emit/lowered.fixture.runtime-plan.json';save(path,head_plan()if name=='head'else native_plan())
        native=entry.inspect_plan(path,target['outputs'],name)
        plan=dict(boots=0);counted=dict(counts=dict(boot=target['outputs']),ops_total=target['outputs']*2,distinct_consts=1)
        save(case/'codegen/refresh-plan.json',plan)
        save(case/'codegen/planner-return.json',dict(source=source['component_source'],planner_returned=True,
            verify_completed=True,planner='original.pt_agent.plan.plan_program'))
        save(case/'codegen/count-replay.json',dict(stats=counted))
        save(case/'codegen/component-interface.json',dict(physical_inputs=34,physical_output_blocks=target['outputs'],
            explicit_boundary_bootstrap_calls=target['outputs'],internal_plan_bootstrap_calls=0))
        phases={}
        for phase in entry.PHASE_SECONDS:
            child=dict(status='ok',component=target,source=source,**{phase+'_completed':True})
            if phase=='compile':child.update(native)
            if phase=='emit':child['emit_stats']=counted
            phases[phase]=child;save(case/'phases'/phase/'worker-result.json',child)
        record=dict(status='ok',round=entry.ROUND,mode=entry.COMPONENT_MODE,model=entry.MODEL,component=target,
            component_native_accepted=True,whole_generation_verified=False,full_prefill_verified=False,source=source,
            linked_native_program_verified=False,prefill_compilation_scope=entry.PREFILL_SCOPE,
            attempted=dict.fromkeys(entry.PHASE_SECONDS,1),completed=dict.fromkeys(entry.PHASE_SECONDS,1),phase_results=phases,
            processes=[dict(phase=p,observation=clean())for p in entry.PHASE_SECONDS],runtime_plan=str(path),
            compiled_bootstraps=target['outputs'],**dict.fromkeys(entry.original.ZERO,0),**dict.fromkeys(entry.original.FORMAL))
        save(case/'result.json',record);save(case/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
        return case,record

    def test_component_acceptance_replays_actual_native_and_source_Count_Emit(self):
        for name in ('layer1','head'):
            with self.subTest(name=name):
                case,record=self.component_fixture(name)
                proof=entry.accept_component(case,name,self.proof)
                self.assertEqual(proof['component']['outputs'],entry.spec(name)['outputs'])
                self.assertTrue(proof['native']['native_ABI_verified'])

    def test_component_source_and_Count_Emit_drift_rejected_after_saved_success(self):
        case,record=self.component_fixture()
        record['source']['selected_iteration']=999;save(case/'result.json',record)
        with self.assertRaises(ValueError):entry.accept_component(case,'head',self.proof)
        record['source'].pop('selected_iteration');save(case/'result.json',record)
        counted=entry.original.read(case/'codegen/count-replay.json');counted['stats']['counts']['boot']=18
        save(case/'codegen/count-replay.json',counted)
        with self.assertRaises(ValueError):entry.accept_component(case,'head',self.proof)

    def test_finite_limits_reject_unbounded_values(self):
        self.assertEqual(entry.limits(self.args)['complete_batch_seconds'],43200)
        for key,value in (('seconds',43201),('memory_gib',32),('case_gib',129)):
            previous=getattr(self.args,key);setattr(self.args,key,value)
            with self.assertRaises(ValueError):entry.limits(self.args)
            setattr(self.args,key,previous)

    def test_master_all24_continuations_required_and_preserves_source_clocks(self):
        result=self.execute_fake()
        self.assertEqual(result['status'],'ok');self.assertEqual(result['accepted_components'],25)
        self.assertEqual(len(self.calls),24);self.assertTrue(result['full_prefill_verified'])
        self.assertFalse(result['whole_generation_verified'])
        self.assertFalse(result['linked_native_program_verified'])
        self.assertEqual(result['prefill_compilation_scope'],entry.PREFILL_SCOPE)
        self.assertEqual(result['source']['first_layer_timing'],{})
        self.assertEqual(result['compiled_bootstraps'],24*32+19)
        for argv,kw in self.calls:
            self.assertEqual(kw['mem_gib'],24);self.assertEqual(kw['fsize_gib'],64)
            self.assertLessEqual(int(argv[argv.index('--seconds')+1]),10800)
        registry=entry.original.read(self.results/'registry.jsonl')
        self.assertEqual(set(registry),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertTrue(entry.original.read(self.args.out/'completion.json')['registry_persisted'])

    def test_first_component_error_stops_and_retains_not_run_later_items(self):
        result=self.execute_fake(lambda *a,**kw:dict(clean(),exit_code=1))
        self.assertEqual(result['status'],'error');self.assertFalse(result['full_prefill_verified'])
        self.assertEqual(len(self.calls),1);self.assertEqual(result['components']['layer1']['status'],'error')
        self.assertEqual(result['components']['layer2']['status'],'not_run');self.assertEqual(result['components']['head']['status'],'not_run')
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_master_source_failure_persists_no_component_launch(self):
        self.calls=[]
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results),patch('pt_agent.registry.git_rev',return_value='fixture'), \
                patch.object(entry,'admission',side_effect=ValueError('probe not accepted')),patch.object(entry,'run_process')as run:
            result=entry.execute(self.args)
        run.assert_not_called();self.assertFalse(result['full_prefill_verified'])
        self.assertEqual(result['accepted_components'],0);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_master_budget_counts_existing_probe_and_revokes_terminal_acceptance(self):
        self.first.mkdir();(self.first/'metadata').write_text('bounded fixture')
        with patch.object(entry,'artifact_bytes',side_effect=lambda p:128*entry.GIB if Path(p)==self.first else 0), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)):
            with self.assertRaises(RuntimeError):entry._budget(self.args,self.results)
        self.args.out.mkdir()
        result=dict(status='ok',accepted_components=25,compiled_bootstraps=787,components={'layer0':dict(status='ok')})
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results), \
                patch.object(entry,'artifact_bytes',side_effect=lambda p:128*entry.GIB if Path(p)==self.first else 0), \
                patch.object(entry,'persist_case_completion'):
            entry._close(self.args,self.args.out,result,1,entry.StageRecorder(self.args.out/'events.jsonl'))
        self.assertEqual(result['status'],'error');self.assertFalse(result['full_prefill_verified'])
        self.assertEqual(result['termination']['cause'],'accumulated_artifact_allowance')
        self.assertFalse(result['whole_generation_verified'])

    def test_original_component_supervision_runs_four_distinct_fake_workers_and_closes(self):
        self.args.component='head';self.args.seconds=10800
        source=entry.component_source(self.proof,'head');calls=[]
        def launch(argv,**kwargs):
            calls.append((argv,kwargs));phase=argv[argv.index('--worker-phase')+1]
            work=Path(argv[argv.index('--work-dir')+1])
            child=dict(status='ok',component=entry.spec('head'),source=source,**{phase+'_completed':True})
            if phase=='compile':child.update(component_native_accepted=True,compiled_bootstraps=19,runtime_plan='synthetic-native-observation')
            save(work/'worker-result.json',child);return clean()
        from pt_agent.experiments import r27_orbit_lazy_backend
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results),patch('pt_agent.registry.git_rev',return_value='fixture'), \
                patch.object(entry,'_component_source',return_value=source),patch.object(entry.original,'run_process',side_effect=launch), \
                patch.object(r27_orbit_lazy_backend,'_available_ram_bytes',return_value=32*entry.GIB), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=200*entry.GIB)):
            result=entry.component_execute(self.args)
        self.assertEqual(result['status'],'ok');self.assertTrue(result['component_native_accepted'])
        self.assertFalse(result['full_prefill_verified']);self.assertFalse(result['whole_generation_verified'])
        self.assertFalse(result['linked_native_program_verified'])
        self.assertEqual([argv[argv.index('--worker-phase')+1]for argv,_ in calls],list(entry.PHASE_SECONDS))
        self.assertEqual(len({argv[argv.index('--work-dir')+1]for argv,_ in calls}),4)
        self.assertTrue(all(kw['mem_gib']==24 and kw['fsize_gib']==64 for _,kw in calls))
        self.assertTrue(entry.original.read(self.args.out/'completion.json')['registry_persisted'])

    def test_component_terminal_reserve_failure_clears_native_acceptance(self):
        self.args.component='head';self.args.seconds=10800
        source=entry.component_source(self.proof,'head')
        def launch(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1])
            child=dict(status='ok',component=entry.spec('head'),source=source,**{phase+'_completed':True})
            if phase=='compile':child.update(component_native_accepted=True,compiled_bootstraps=19,runtime_plan='synthetic')
            save(work/'worker-result.json',child);return clean()
        from pt_agent.experiments import r27_orbit_lazy_backend
        # Only terminal accounting uses the original artifact_bytes binding;
        # actual finite stage guards still inspect this tiny public fixture.
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results),patch('pt_agent.registry.git_rev',return_value='fixture'), \
                patch.object(entry,'_component_source',return_value=source),patch.object(entry.original,'run_process',side_effect=launch), \
                patch.object(entry.original,'artifact_bytes',return_value=128*entry.GIB), \
                patch.object(r27_orbit_lazy_backend,'_available_ram_bytes',return_value=32*entry.GIB), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=200*entry.GIB)):
            result=entry.component_execute(self.args)
        self.assertEqual(result['status'],'error');self.assertFalse(result['component_native_accepted'])
        self.assertFalse(entry.original.read(self.args.out/'result.json')['component_native_accepted'])

    def test_component_nonclean_outer_observation_stops_before_next_phase_and_saves_actual(self):
        self.args.component='head';self.args.seconds=10800
        source=entry.component_source(self.proof,'head');calls=[]
        observed=dict(clean(),signals_sent=[15])
        def launch(argv,**kwargs):calls.append(argv);return observed
        from pt_agent.experiments import r27_orbit_lazy_backend
        with patch.object(entry.original.paths,'RESULTS_DIR',self.results),patch('pt_agent.registry.git_rev',return_value='fixture'), \
                patch.object(entry,'_component_source',return_value=source),patch.object(entry.original,'run_process',side_effect=launch), \
                patch.object(r27_orbit_lazy_backend,'_available_ram_bytes',return_value=32*entry.GIB), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=200*entry.GIB)):
            result=entry.component_execute(self.args)
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'error')
        self.assertEqual(result['attempted']['placement'],1);self.assertEqual(result['attempted']['emit'],0)
        self.assertEqual(entry.original.read(self.args.out/'placement-supervision-observation.json'),observed)
        self.assertFalse(result['component_native_accepted'])


if __name__=='__main__':unittest.main()
