"""New small metadata/fake supervision and tiny trusted Count/Emit only."""
from argparse import Namespace
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pt_agent.experiments import qwen_generation_kernel_compile as entry


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def observed(**changes):return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],**changes)


def state():return dict(step=0,prefix_length=9,prefix_token_ids=list(range(32)),attention_mask=[[1.]*9+[0.]*23],sel=[[0.]*8+[1.]+[0.]*23])


def source_fixture(root):
    case=root/'source';model=root/'original-model';info=root/'model/model_info.json'
    generation=dict(index=0,sample_id='saved-inputs.jsonl:0',partition='calibration',prompt_length=9,sequence_length=32)
    put(case/'generation-input.json',generation);put(info,dict(split=dict(calibration=[0],sample_ids=[generation['sample_id']])))
    linked=root/'linked.json';own=root/'own.json';reference=root/'reference.json'
    put(linked,dict(status='ok',linked_packed_math_verified=True,source=dict(cached_sources=dict(own_output_result=str(own)))))
    put(own,dict(status='ok',worker=dict(status='ok',phase='own-output-tensor',own_output_generation_completed=True,
        source=dict(model_info=str(info.parent),original_hf_reference=str(reference)),
        sample_rows=[dict(sample_id='train:0',step=0,input_state=state())])))
    put(reference,dict(status='ok',hf_reference_completed=True,hf_forward_completed=6,original_hf_source=str(model),
        original_model_identity=dict(layers=24,hidden_size=896,vocabulary=151936,use_cache=False,sequence_length=32,dtype='float64',attention='eager'),
        trajectories=[dict(sample_id='train:0',steps=[state()])]))
    frozen=dict(source_case=str(case),model_info=str(info),original_model=str(model),program=str(case/'session/program.ptdsl'),
        original_source=dict(previous_linked_math=str(linked)),candidate_frozen=True,source_program_ast_verified=True,
        full_model_layers=24,sequence_capacity=32)
    return case,frozen,own,reference


def interface(kind):
    target=entry.spec(kind);levels=[1+i%5 for i in range(19)]+([2,4,4]if kind=='first'else [])
    return dict(kind=kind,inputs=target['inputs'],outputs=target['outputs'],symbolic_output_levels=levels,
        source=dict(kind=kind),explicit_boundary_bootstraps_in_this_adapter=0,
        caller_required_boundary_positions=target['future_boundary_positions'])


def native_plan(kind):
    target=entry.spec(kind);contract=interface(kind);values=[];execution=[];outputs=[]
    def value(level):
        name=str(len(values));values.append(dict(id=name,place=dict(kind='host',rank=0),kind='ciphertext',level=level,
            scale_log2=43,components=2,context='qwen-bootstrap48',ntt=True));return name
    inputs=[value(5)for _ in range(target['inputs'])]
    for index,level in enumerate(contract['symbolic_output_levels']):
        out=value(level);outputs.append(out)
        execution.append(dict(kind='compute',op='negate'if level==5 else'mod_switch',inputs=[inputs[index%len(inputs)]],
            output=out,ordinal=index,attrs=dict(target_level=level)))
    return dict(format_version=1,values=values,external_inputs=inputs,initialization=[],execution=execution,
        finalization=[],final_outputs=outputs),contract


def generic_trace(inputs,outputs):
    tensor='tensor<1x!earth.ci<0 * 0>>'
    return ('"builtin.module"() ({\n  "func.func"() <{function_type = ('+', '.join([tensor]*inputs)+') -> ('+
        ', '.join([tensor]*outputs)+'), sym_name = "_hecate_golden"}> ({\n'+
        '  ^bb0('+', '.join('%arg'+str(i)+': '+tensor for i in range(inputs))+'):\n'+
        '    %0 = "earth.negate"(%arg0) : ('+tensor+') -> '+tensor+'\n'+
        '    "func.return"('+', '.join(['%0']*outputs)+') : ('+', '.join([tensor]*outputs)+') -> ()\n'+
        '  }) : () -> ()\n}) : () -> ()\n')


def template():
    return ['{hecate_opt}','{input}','--pars','--ckks-config={package}/config/compiler-config.json',
        '--waterline=43','--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
        '--runtime-plan-context-id=qwen-bootstrap48','--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
        '--runtime-plan-ntt=true','--runtime-plan-boot-implementation=native',
        '--runtime-plan-format=json','--runtime-plan-bundle-format=files','-o','{output_dir}/lowered.mlir']


class AdmissionChecks(unittest.TestCase):
    def test_original_nested_integer_keys_and_tuples_keep_JSON_same_source_after_save(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);case,frozen,own,reference=source_fixture(root)
            frozen['original_source']['recovery']=dict(pinned_error_counts={8:43},original_shape=(1,32,896))
            with patch.object(entry.original,'frozen_source',return_value=frozen):checked=entry.admission(case)
            put(root/'source.json',checked);restored=entry.original.read(root/'source.json')
            self.assertEqual(checked,restored);self.assertEqual(checked['binding']['frozen'],checked['frozen_source'])
            self.assertEqual(checked['frozen_source']['original_source']['recovery']['pinned_error_counts'],{'8':43})
            self.assertEqual(checked['frozen_source']['original_source']['recovery']['original_shape'],[1,32,896])
            self.assertEqual(entry.component_source(restored,'first'),entry.component_source(checked,'first'))

    def test_actual_model_info_directory_and_compatible_json_file_both_resolve(self):
        for directory in (True,False):
            with self.subTest(directory=directory),TemporaryDirectory()as tmp:
                case,frozen,own,reference=source_fixture(Path(tmp))
                if directory:frozen['model_info']=str(Path(frozen['model_info']).parent)
                with patch.object(entry.original,'frozen_source',return_value=frozen):checked=entry.admission(case)
                self.assertTrue(checked['saved_state_metadata_verified'])
                self.assertEqual(checked['initial_state']['prefix_length'],9)

    def test_saved_original_step0_exactly_bound_without_sample_or_model_read(self):
        with TemporaryDirectory()as tmp:
            case,frozen,own,reference=source_fixture(Path(tmp))
            with patch.object(entry.original,'frozen_source',return_value=frozen):checked=entry.admission(case)
            self.assertEqual(checked['initial_state']['prefix_token_ids'],list(range(32)))
            self.assertTrue(checked['saved_state_metadata_verified']);self.assertFalse(checked['initial_ciphertext_payload_verified'])
            self.assertEqual(checked['binding']['initial_state']['sample_id'],'saved-inputs.jsonl:0')
            self.assertEqual(checked['lineage']['original_sample_id'],'train:0')

    def test_original_C2_rejection_occurs_before_any_state_read(self):
        with patch.object(entry.original,'frozen_source',side_effect=ValueError('Original C2 pending')):
            with self.assertRaisesRegex(ValueError,'C2 pending'):entry.admission(Path('/absent-original-case'))

    def test_foreign_saved_state_model_or_sample_is_rejected(self):
        for change in ('sample','state','model'):
            with self.subTest(change=change),TemporaryDirectory()as tmp:
                case,frozen,own,reference=source_fixture(Path(tmp))
                if change=='sample':
                    p=case/'generation-input.json';d=entry.original.read(p);d['sample_id']='foreign:1';put(p,d)
                elif change=='state':
                    d=entry.original.read(own);d['worker']['sample_rows'][0]['input_state']['prefix_token_ids'][0]=55;put(own,d)
                else:
                    d=entry.original.read(reference);d['original_hf_source']='/foreign-model';put(reference,d)
                with patch.object(entry.original,'frozen_source',return_value=frozen):
                    with self.assertRaises(ValueError):entry.admission(case)

    def test_interface_keeps_all_raw_levels_and_external_Go_three_boots(self):
        for kind in entry.KINDS:
            raw=interface(kind);record=entry.interface_record(raw)
            self.assertEqual(record['symbolic_output_levels'],raw['symbolic_output_levels'])
            self.assertEqual(record['external_boundary_bootstraps_in_component'],0)
            self.assertEqual(record['future_boundary_bootstraps'],3 if kind=='first'else 0)
            self.assertEqual(entry.spec(kind)['boundary_bootstraps'],0)
        bad=interface('first');bad['explicit_boundary_bootstraps_in_this_adapter']=3
        with self.assertRaises(ValueError):entry.interface_record(bad)


class NativeMetadataChecks(unittest.TestCase):
    def test_both_variable_output_level_ABIs_are_verified_without_forcing_five(self):
        for kind in entry.KINDS:
            with self.subTest(kind=kind),TemporaryDirectory()as tmp:
                plan,contract=native_plan(kind);path=Path(tmp)/'lowered.golden.runtime-plan.json';put(path,plan)
                result=entry.inspect_plan(path,0,contract,kind)
                self.assertTrue(result['native_ABI_verified']);self.assertEqual(result['output_levels'],contract['symbolic_output_levels'])
                self.assertEqual(result['compiled_bootstraps'],0);self.assertFalse(result['whole_generation_verified'])

    def test_native_level_scale_context_components_NTT_or_SSA_changes_rejected(self):
        for key,value in [('level',5),('scale_log2',50),('context','foreign'),('components',3),('ntt',False),('id','900')]:
            with self.subTest(key=key),TemporaryDirectory()as tmp:
                plan,contract=native_plan('first');plan['values'][20][key]=value
                path=Path(tmp)/'p.json';put(path,plan)
                with self.assertRaises(ValueError):entry.inspect_plan(path,0,contract,'first')

    def test_native_boot_inventory_and_supported_arity_are_enforced(self):
        for change in ('missing','arity','duplicate','finalization'):
            with self.subTest(change=change),TemporaryDirectory()as tmp:
                plan,contract=native_plan('terminal');expected=0
                if change=='missing':expected=1
                elif change=='arity':plan['execution'][0]['inputs']=[]
                elif change=='duplicate':plan['execution'][1]['output']=plan['execution'][0]['output']
                else:plan['finalization']=[dict(kind='decode')]
                path=Path(tmp)/'p.json';put(path,plan)
                with self.assertRaises(ValueError):entry.inspect_plan(path,expected,contract,'terminal')

    def test_actual_boot_input_and_profile_metadata_checked(self):
        for change in ('valid','profile','inputlevel'):
            with self.subTest(change=change),TemporaryDirectory()as tmp:
                plan,contract=native_plan('terminal');out=plan['final_outputs'][0];contract['symbolic_output_levels'][0]=5
                plan['values'][int(out)]['level']=5;extra=str(len(plan['values']))
                plan['values'].append(dict(plan['values'][0],id=extra,level=1))
                plan['execution'][0]=dict(kind='compute',op='boot',inputs=[extra],output=out,ordinal=1,
                    attrs=dict(target_level=5,target_scale_log2=43,target_components=2,operator_profile='cpu-bootstrap48-level1-to5',implementation='native'))
                plan['execution'].insert(0,dict(kind='compute',op='mod_switch',inputs=['0'],output=extra,ordinal=0,attrs=dict(target_level=1)))
                for i,row in enumerate(plan['execution']):row['ordinal']=i
                if change=='profile':plan['execution'][1]['attrs']['operator_profile']='foreign'
                if change=='inputlevel':plan['values'][-1]['level']=2
                path=Path(tmp)/'p.json';put(path,plan)
                if change=='valid':self.assertEqual(entry.inspect_plan(path,1,contract,'terminal')['compiled_bootstraps'],1)
                else:
                    with self.assertRaises(ValueError):entry.inspect_plan(path,1,contract,'terminal')

    def test_native_bundle_closure_stats_without_payload_read(self):
        with TemporaryDirectory()as tmp:
            plan,contract=native_plan('terminal');locator='sha256:'+'0'*64;name=str(len(plan['values']))
            plan['values'].append(dict(plan['values'][0],id=name,kind='plaintext',components=1))
            plan['initialization']=[dict(kind='encode',output=name,payload=dict(kind='bundle',content=locator))]
            plan['plaintext_bundle']=dict(id='original',version=1);path=Path(tmp)/'lowered.golden.runtime-plan.json';put(path,plan)
            bundle=Path(tmp)/'lowered.golden.bundle';put(bundle/'manifest.json',dict(bundle_format_version=1,version=1,bundle_id='original',blobs=[dict(content=locator,byte_length=1)]))
            data=bundle/'data'/('0'*64+'.bin');data.parent.mkdir();data.write_bytes(b'\0')
            checked=entry.inspect_plan(path,0,contract,'terminal');self.assertEqual(checked['bundle_closure']['payload_reads'],0)
            data.unlink()
            with self.assertRaises(ValueError):entry.inspect_plan(path,0,contract,'terminal')

    def test_generic_markers_keep_raw_program_and_actual_first_terminal_arities(self):
        for kind in entry.KINDS:
            with self.subTest(kind=kind),TemporaryDirectory()as tmp:
                target=entry.spec(kind);raw=Path(tmp)/'raw.mlir';marked=Path(tmp)/'marked.mlir'
                text=generic_trace(target['inputs'],target['outputs']);raw.write_text(text)
                result=entry.mark_trace(raw,marked,kind);new=marked.read_text()
                self.assertEqual(raw.read_text(),text);self.assertEqual(result['physical_inputs'],target['inputs'])
                self.assertEqual(result['physical_outputs'],target['outputs']);self.assertEqual(len(result['return_bypasses']),target['outputs'])
                self.assertIn('is_mid_segment = true',new)
                with self.assertRaises(ValueError):entry.mark_trace(raw,marked,kind)

    def test_generic_wrong_arity_conflicting_attrs_and_multiple_functions_rejected(self):
        for change in ('arity','attrs','multiple'):
            with self.subTest(change=change),TemporaryDirectory()as tmp:
                text=generic_trace(19,19)
                if change=='arity':text=generic_trace(20,22)
                if change=='attrs':text=text.replace('  }) :','  }) {is_mid_segment = true} :',1)
                if change=='multiple':text+=generic_trace(19,19)
                raw=Path(tmp)/'raw.mlir';raw.write_text(text)
                with self.assertRaises(ValueError):entry.mark_trace(raw,Path(tmp)/'marked.mlir','terminal')


class TinyOriginalCountEmitChecks(unittest.TestCase):
    def test_original_helpers_keep_raw_levels_no_component_boundary_boots(self):
        from pt_agent.backends import RefreshBackend
        from pt_agent.backends.streaming_count import StreamingCountBackend
        from pt_agent.backends.streaming_emit import StreamingEmitBackend
        class TinyKernel:
            kind='first';input_count=3;geometry=dict(slots=64,vocabulary_blocks=2)
            def source(self):return dict(tiny=True,kind=self.kind)
            def output_roles(self):return ['one_hot:0','one_hot:1','mixed_embedding','attention_mask','sel']
            def __call__(self,active):
                x=[active.input(i)for i in range(3)]
                a=x[0]*active.const([.5]);b=x[1]+x[0];mixed=x[0]*active.const([.25]);zero=x[2]*active.const([0.])
                active.output([a,b,mixed,zero+active.const([1.]),zero+active.const([2.])])
        kernel=TinyKernel();plan=dict(format='pt-agent-refresh-plan-v1',levels=dict(entry.kernels.LEVELS),calls=9,boots=0,substitutions=[],outputs=[],dead=[])
        planned=entry.kernels.PlannedKernel(kernel.source(),plan,dict(calls=9,boots=0,inner_boots=0))
        api=dict(Count=StreamingCountBackend,Emit=StreamingEmitBackend,Refresh=RefreshBackend)
        with TemporaryDirectory()as tmp,patch.object(entry.kernels,'_api',return_value=api),patch.object(entry.kernels,'_metadata',lambda value:nullcontext()):
            counted=entry.kernels.count_kernel(kernel,planned,metadata_template=[])
            emitted=entry.kernels.emit_kernel(kernel,planned,metadata_template=[],directory=Path(tmp)/'emit',frontend='/unused-native-frontend',max_constant_bytes=1024**2,counted=counted)
            self.assertEqual(counted['interface']['symbolic_output_levels'],[4,5,4,4,4])
            self.assertEqual(counted['interface'],emitted['interface']);self.assertEqual(counted['stats']['counts']['boot'],0)
            tree=ast.parse(Path(emitted['trace_source']).read_text());golden=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='golden')
            self.assertEqual(len(golden.args.args),3);returned=next(n for n in ast.walk(golden)if isinstance(n,ast.Return))
            self.assertEqual(len(returned.value.elts),5)


class SupervisionChecks(unittest.TestCase):
    def args(self,root):
        results=root/'results';results.mkdir();return Namespace(source_case=root/'source',out=results/'new',kind=None,batch_out=None,
            memory_gib=24.,case_gib=128,seconds=21600,intake_python=Path('/unused-intake'),diagnostic_elide_final_ir=True)

    def test_finite_resource_and_worker_argv_exact_original_limits(self):
        with TemporaryDirectory()as tmp:
            args=self.args(Path(tmp));args.kind='first';limits=entry.limits(args)
            self.assertEqual(limits['subtree_rss_gib'],14);self.assertEqual(limits['minimum_available_gib'],1.25)
            self.assertEqual(limits['phases'],dict(placement=7200,emit=3600,trace=1800,compile=7200))
            argv=entry.phase_command(args,'compile',Path(tmp)/'worker',remaining=400)
            self.assertEqual(argv[argv.index('--phase-seconds')+1],'280');self.assertEqual(argv[argv.index('--memory-gib')+1],'24')
            self.assertNotIn('--report',argv);self.assertNotIn('--reference',argv)
            for key,value in [('memory_gib',32),('seconds',21601),('case_gib',129)]:
                bad=Namespace(**vars(args));setattr(bad,key,value)
                with self.assertRaises(ValueError):entry.limits(bad)

    def fake_outer(self,root,*,fail=None,admission_error=False,terminal_error=False):
        args=self.args(root);calls=[];source=dict(binding=dict(original='fixture'))
        def child(argv,**kwargs):
            kind=argv[argv.index('--kind')+1];calls.append((kind,kwargs))
            if kind==fail:return dict(observed(),exit_code=1)
            return observed()
        def accept(case,kind,checked):return dict(status='ok',kernel=entry.spec(kind),native=dict(compiled_bootstraps=7 if kind=='first'else 5))
        with (patch.object(entry.original.paths,'RESULTS_DIR',args.out.parent),
                patch.object(entry,'admission',side_effect=ValueError('C2 pending')if admission_error else None,return_value=source),
                patch.object(entry,'run_process',side_effect=child),patch.object(entry,'accept_component',side_effect=accept),
                patch.object(entry,'_available_ram_bytes',return_value=50*entry.GIB),
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.GIB)),
                patch('pt_agent.registry.git_rev',return_value='fixture')):
            if terminal_error:
                original_close=entry.close
                def late(a,r,s,rec,deadline,failures):return original_close(a,r,s,rec,deadline,['recorded late resource failure'])
                with patch.object(entry,'close',side_effect=late):result=entry.execute(args)
            else:result=entry.execute(args)
        return args,result,calls

    def test_fake_full_two_component_chain_and_registry_keep_global_scope_pending(self):
        with TemporaryDirectory()as tmp:
            args,result,calls=self.fake_outer(Path(tmp));self.assertEqual(result['status'],'ok',result['error'])
            self.assertEqual([k for k,_ in calls],['first','terminal']);self.assertEqual(result['accepted_components'],2)
            self.assertTrue(result['two_kernel_native_components_verified']);self.assertEqual(result['compiled_bootstraps'],12)
            self.assertFalse(result['linked_native_program_verified']);self.assertFalse(result['whole_generation_verified'])
            self.assertTrue(all(x['mem_gib']==24 and x['fsize_gib']==64 for _,x in calls))
            row=json.loads((args.out.parent/'registry.jsonl').read_text());self.assertEqual(len(row),8)
            self.assertTrue(entry.original.read(args.out/'completion.json')['registry_persisted'])

    def test_first_error_stops_unrun_terminal_and_retains_failure(self):
        with TemporaryDirectory()as tmp:
            args,result,calls=self.fake_outer(Path(tmp),fail='first');self.assertEqual(result['status'],'error')
            self.assertEqual([k for k,_ in calls],['first']);self.assertEqual(result['components']['terminal']['status'],'not_run')
            self.assertFalse(result['two_kernel_native_components_verified']);self.assertTrue((args.out/'completion.json').is_file())

    def test_unfrozen_source_has_zero_child_and_durable_admission_error(self):
        with TemporaryDirectory()as tmp:
            args,result,calls=self.fake_outer(Path(tmp),admission_error=True)
            self.assertEqual(result['status'],'error');self.assertEqual(calls,[]);self.assertEqual(result['accepted_components'],0)
            self.assertTrue((args.out/'result.json').is_file());self.assertFalse(result['two_kernel_native_components_verified'])

    def test_terminal_resource_failure_revokes_acceptance_preserves_component_facts(self):
        with TemporaryDirectory()as tmp:
            args,result,calls=self.fake_outer(Path(tmp),terminal_error=True)
            self.assertEqual(result['status'],'error');self.assertEqual(result['accepted_components'],2)
            self.assertFalse(result['two_kernel_native_components_verified'])
            self.assertTrue(all(r['status']=='ok'for r in result['components'].values()))


class ActualDelegationAndAcceptanceChecks(unittest.TestCase):
    def test_private_original_supervision_runs_four_fake_phases_with_matching_limits_and_scope(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);args=SupervisionChecks().args(root);args.kind='first';args.through='compile'
            checked=dict(binding=dict(exact='frozen fixture'));source=entry.component_source(checked,'first');calls=[]
            def child(argv,**kwargs):
                phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1]);calls.append((phase,argv,kwargs))
                record=dict(round=entry.ROUND,model=entry.MODEL,mode=entry.COMPONENT_MODE,status='ok',phase=phase,source=source,
                    kernel=entry.spec('first'),**{phase+'_completed':True},**dict.fromkeys(entry.original.ZERO,0),**dict.fromkeys(entry.original.FORMAL))
                if phase=='compile':record.update(kernel_native_accepted=True,native_ABI_verified=True,compiled_bootstraps=0,runtime_plan='/synthetic-plan')
                put(work/'worker-result.json',record);return observed()
            with (patch.object(entry.original.paths,'RESULTS_DIR',args.out.parent),patch.object(entry,'admission',return_value=checked),
                    patch.object(entry,'run_process',side_effect=child),patch.object(entry,'_available_ram_bytes',return_value=50*entry.GIB),
                    patch.object(entry.original.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.GIB)),
                    patch('pt_agent.registry.git_rev',return_value='fixture')):
                # Original supervision's monitor imports its own RAM function;
                # patch only that lookup, leaving real original run control.
                with patch('pt_agent.experiments.r27_orbit_lazy_backend._available_ram_bytes',return_value=50*entry.GIB):
                    result=entry.component_execute(args)
            self.assertEqual(result['status'],'ok',result['error']);self.assertTrue(result['kernel_native_accepted'])
            self.assertEqual([p for p,_,_ in calls],list(entry.PHASE_SECONDS))
            for phase,argv,kw in calls:
                self.assertEqual(kw['mem_gib'],24);self.assertEqual(kw['fsize_gib'],64)
                self.assertEqual(argv[argv.index('--memory-gib')+1],'24');self.assertLessEqual(kw['timeout'],entry.PHASE_SECONDS[phase]+120)
            row=json.loads((args.out.parent/'registry.jsonl').read_text())
            self.assertTrue(row['metrics']['kernel_native_accepted']);self.assertFalse(row['metrics']['linked_native_program_verified'])

    def kernel_source(self,checked,kind):
        from pt_agent.experiments.qwen_monotone_selector import config
        return dict(binding=checked['binding'],kind=kind,
            geometry=entry.kernels._geometry(16384,151936,896,32),selector=entry.original.json_value(config()),
            output_roles=['one_hot:'+str(i)for i in range(19)]+(['mixed_embedding','attention_mask','sel']if kind=='first'else []),
            embedding_weight=entry.kernels.WEIGHT if kind=='first'else None,
            original_token0_input=19 if kind=='first'else None,future_boundary_bootstraps=3 if kind=='first'else 0)

    def test_placement_and_emit_call_original_bound_helper_APIs_without_extra_boot(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);checked=dict(binding=dict(exact='frozen-fixture'));kind='first';source=self.kernel_source(checked,kind)
            class Kernel:
                closed=False
                def source(self):return deepcopy(source)
                def close(self):self.closed=True
            kernel=Kernel();plan=dict(format='pt-agent-refresh-plan-v1',levels=dict(entry.kernels.LEVELS),calls=9,boots=0,outputs=[],substitutions=[],dead=[])
            replay=dict(calls=9,boots=0,inner_boots=0);planned=entry.kernels.PlannedKernel(source,plan,replay)
            metadata=[[entry.kernels.WEIGHT,[151936,896],'float64',False]]+[['w'+str(i),[1],'float64',False]for i in range(558)]
            raw=interface(kind);raw['source']=source;stats=dict(counts=dict(boot=0),ops_total=9,distinct_consts=2)
            args=Namespace(kind=kind,codegen=root/'codegen',case_gib=128,frontend=root/'unused-frontend',metadata_template=root/'template.json')
            args.codegen.mkdir();put(args.metadata_template,[]);result={};recorder=entry.StageRecorder(root/'events.jsonl')
            with (patch.object(entry,'_load',return_value=(SimpleNamespace(),kernel,metadata)),
                    patch.object(entry.kernels,'plan_kernel',return_value=(planned,dict(planner='original fixture')))as planner,
                    patch.object(entry.kernels,'count_kernel',return_value=dict(stats=stats,interface=raw))as count):
                entry.placement(args,result,recorder)
            planner.assert_called_once();count.assert_called_once();self.assertEqual(result['expected_compiled_bootstraps'],0)
            kernel.closed=False;emitted=dict(stats=stats,interface=raw,trace_source=str(root/'trusted-trace.py'))
            with patch.object(entry,'_load',return_value=(SimpleNamespace(),kernel,metadata)),patch.object(entry.kernels,'emit_kernel',return_value=emitted)as emit:
                entry.emit(args,result,recorder)
            emit.assert_called_once();self.assertTrue(kernel.closed);self.assertTrue(result['emit_completed'])
            self.assertEqual(result['interface']['future_boundary_bootstraps'],3)
            self.assertEqual(result['interface']['external_boundary_bootstraps_in_component'],0)

    def acceptance_fixture(self,root,kind='first'):
        case=root/kind;checked=dict(binding=dict(exact='frozen-fixture'));source=entry.component_source(checked,kind)
        ks=self.kernel_source(checked,kind);plan,contract=native_plan(kind);contract['source']=ks;contract=entry.interface_record(contract)
        path=case/'codegen/emit/lowered.golden.runtime-plan.json';put(path,plan);native=entry.inspect_plan(path,0,contract,kind)
        refresh=dict(format='pt-agent-refresh-plan-v1',levels=dict(entry.kernels.LEVELS),calls=9,boots=0,substitutions=[],outputs=[],dead=[])
        replay=dict(calls=9,boots=0,inner_boots=0);stats=dict(counts=dict(boot=0),ops_total=9,distinct_consts=2)
        weights=[[entry.kernels.WEIGHT,[151936,896],'float64',False]]+[['w'+str(i),[1],'float64',False]for i in range(558)]
        put(case/'codegen/planned-kernel.json',dict(source=ks,refresh_plan=refresh,replay=replay))
        put(case/'codegen/planner-return.json',dict(source=ks,weights=weights,planner_returned=True,verify_completed=True,verification=replay))
        put(case/'codegen/count-replay.json',dict(stats=stats,interface=contract));put(case/'codegen/kernel-interface.json',contract)
        record=dict(round=entry.ROUND,model=entry.MODEL,mode=entry.COMPONENT_MODE,status='ok',kernel=entry.spec(kind),source=source,
            kernel_native_accepted=True,compiled_bootstraps=0,runtime_plan=str(path),termination=None,terminal_budget_error=None,failed_phase=None,
            attempted=dict.fromkeys(entry.PHASE_SECONDS,1),completed=dict.fromkeys(entry.PHASE_SECONDS,1),processes=[],phase_results={},
            **dict.fromkeys(entry.original.ZERO,0),**dict.fromkeys(entry.original.FORMAL))
        for phase in entry.PHASE_SECONDS:
            child=dict(round=entry.ROUND,model=entry.MODEL,mode=entry.COMPONENT_MODE,status='ok',phase=phase,source=source,
                kernel=entry.spec(kind),**{phase+'_completed':True},**dict.fromkeys(entry.original.ZERO,0),**dict.fromkeys(entry.original.FORMAL))
            if phase=='compile':child.update(native)
            if phase=='emit':child['emit_stats']=stats
            put(case/'phases'/phase/'worker-result.json',child);record['phase_results'][phase]=child
            record['processes'].append(dict(phase=phase,observation=observed()))
        put(case/'result.json',record);put(case/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
        return case,checked

    def test_complete_accept_gate_replays_all_phases_same_source_and_actual_native_metadata(self):
        for kind in entry.KINDS:
            with self.subTest(kind=kind),TemporaryDirectory()as tmp:
                case,checked=self.acceptance_fixture(Path(tmp),kind)
                result=entry.accept_component(case,kind,checked);self.assertEqual(result['status'],'ok')
                self.assertEqual(result['native']['output_levels'],interface(kind)['symbolic_output_levels'])

    def test_accept_gate_rejects_foreign_planner_binding_or_readonly_weight_inventory(self):
        for change in ('binding','weights','replay'):
            with self.subTest(change=change),TemporaryDirectory()as tmp:
                case,checked=self.acceptance_fixture(Path(tmp));p=case/'codegen/planned-kernel.json';plan=entry.original.read(p)
                if change=='binding':plan['source']['binding']={'foreign':'other-source'};put(p,plan)
                elif change=='replay':plan['replay']['inner_boots']=1;put(p,plan)
                else:
                    p=case/'codegen/planner-return.json';record=entry.original.read(p);record['weights'][0][-1]=True;put(p,record)
                with self.assertRaises(ValueError):entry.accept_component(case,'first',checked)


if __name__=='__main__':unittest.main()
