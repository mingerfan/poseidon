"""New tiny planner/metadata fixtures; no model, frontend, compiler or API."""
from argparse import Namespace
import ast
from contextlib import ExitStack
from copy import deepcopy
import io
import json
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_layer_native_probe as entry
from pt_agent.tests.test_qwen_layered_structure import fixture


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data)+'\n')


def template():
    # Opaque required metadata string; no digest is computed or compared.
    return ['{hecate_opt}','{input}','--pars','--ckks-config={package}/config/compiler-config.json',
        '--waterline=43','--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
        '--runtime-plan-context-id=qwen-bootstrap48','--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
        '--runtime-plan-ntt=true','--runtime-plan-boot-implementation=native',
        '--runtime-plan-format=json','--runtime-plan-bundle-format=files','-o','{output_dir}/lowered.mlir']


def generic_trace(body=None):
    tensor='tensor<1x!earth.ci<0 * 0>>'
    args=', '.join([tensor]*34);outputs=', '.join([tensor]*32)
    block=', '.join(f'%arg{i}: {tensor} loc(#loc0)'for i in range(34))
    if body is None:body='    %0 = "earth.negate"(%arg0) : ('+tensor+') -> '+tensor+' loc(#loc0)\n'
    return ('"builtin.module"() ({\n'
        '  "func.func"() <{function_type = ('+args+') -> ('+outputs+'), sym_name = "_hecate_golden"}> ({\n'
        '  ^bb0('+block+'):\n'+body+'    "func.return"(%0) : ('+tensor+') -> ()\n'
        '  }) : () -> () loc(#loc0)\n}) : () -> ()\n#loc0 = loc("public-fixture":1:0)\n')


def native_plan():
    values=[dict(id=str(i),place=dict(kind='host',rank=0),**entry.EXPECTED)for i in range(98)]
    execution=[];outputs=[]
    for i in range(32):
        lower=str(34+2*i);out=str(35+2*i);values[int(lower)]['level']=1
        execution.append(dict(kind='compute',op='mod_switch',inputs=[str(i)],output=lower,
            ordinal=2*i,attrs=dict(target_level=1)))
        execution.append(dict(kind='compute',op='boot',inputs=[lower],output=out,ordinal=2*i+1,
            attrs=dict(target_level=5,target_scale_log2=43,target_components=2,
                operator_profile='cpu-bootstrap48-level1-to5',implementation='native')))
        outputs.append(out)
    return dict(format_version=1,values=values,external_inputs=[str(i)for i in range(34)],
        initialization=[],execution=execution,finalization=[],final_outputs=outputs)


class ProbeChecks(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.args=Namespace(source_case=self.root/'frozen',out=self.root/'results/probe',
            codegen=self.root/'codegen',work_dir=self.root/'work',memory_gib=24.,case_gib=128,
            seconds=10800,phase_seconds=7200,worker_phase='compile',through='compile',
            intake_python=self.root/'unused-python',metadata_template=self.root/'template.json',
            profile=entry.original.paths.PROFILES_DIR/'qwen48/compiler-config.json',frontend=self.root/'unused-frontend',
            diagnostic_elide_final_ir=True)
        save(self.args.metadata_template,template())

    def test_original_record_dag_express_verify_keep34_mask_outputs_without_extra_arithmetic(self):
        planner=entry.planner_namespace()
        def callback(active):
            inputs=[active.input(i)for i in range(34)]
            active.output([*[value*.5 for value in inputs[:32]],*inputs[32:]])
        with patch.object(entry.original,'_projection_backend_scope',return_value=ExitStack()):
            dag,recorder,_=planner['record_dag'](callback,16384,frontend_normalize=True)
        self.assertEqual(len(dag._inputs),34)
        self.assertEqual(len(dag._outputs),34)
        self.assertEqual(dag.stats()['counts']['mul_cp'],32)
        self.assertEqual(dag.stats()['counts']['boot'],0)
        placed=SimpleNamespace(refreshed_uses={},boot_nodes=[],boots=0,barriers=0,cuts=[],latency_us=0.)
        plan=planner['express'](dag,recorder,placed)
        verified=planner['verify'](callback,16384,plan)
        self.assertEqual(plan['calls'],66)
        self.assertEqual(verified['calls'],66)
        self.assertEqual(verified['boots'],0)
        self.assertEqual(verified['inner_boots'],0)

    def test_logical_output_backend_rejects_computed_mask_and_preserves_original_base(self):
        from pt_agent.backends.streaming_count import StreamingCountBackend
        adapted=entry.logical_mask_backend(StreamingCountBackend)(16384,input_level=5,boot_out_level=5)
        inputs=[adapted.input(i)for i in range(34)]
        with self.assertRaisesRegex(ValueError,'exact original mask'):
            adapted.output([*[x*.5 for x in inputs[:32]],inputs[32]+0.,inputs[33]])
        self.assertIsNone(adapted._outputs)
        ordinary=StreamingCountBackend(16384,input_level=5,boot_out_level=5)
        ordinary_inputs=[ordinary.input(i)for i in range(34)]
        with self.assertRaisesRegex(Exception,'computed ciphertext'):
            ordinary.output([*[x*.5 for x in ordinary_inputs[:32]],*ordinary_inputs[32:]])

    def test_original_generic_trace_gets_only_two_function_discardable_attributes(self):
        raw=self.root/'raw.mlir';marked=self.root/'marked.mlir';source=generic_trace();raw.write_text(source)
        result=entry.mark_segment_trace(raw,marked)
        attrs=' {is_mid_segment = true, segment_returnBypasses = ['+', '.join(['true']*32)+']}'
        self.assertEqual(raw.read_text(),source)
        self.assertEqual(marked.read_text().replace(attrs,'',1),source)
        self.assertEqual(marked.read_text().count('is_mid_segment'),1)
        self.assertFalse(result['segment_inputType_added'])
        self.assertFalse(result['arithmetic_rewritten'])
        self.assertFalse(result['native_ABI_verified'])
        self.assertEqual(result['expected_native_remaining_input_level'],5)
        self.assertEqual(result['Earth_consumed_input_level'],0)
        self.assertIn('})'+attrs+' : () -> ()',marked.read_text())

    def test_conflicting_attributes_bad_arity_types_and_nested_regions_reject(self):
        source=generic_trace()
        changed=[source.replace('  }) :','  }) {is_mid_segment = false} :',1),
            source.replace('sym_name = "_hecate_golden"','sym_name = "other"'),
            source.replace('!earth.ci<0 * 0>','!earth.ci<43 * 5>',1),
            source.replace('function_type = (','function_type = (tensor<1x!earth.ci<0 * 0>>, ',1),
            generic_trace('    "scf.if"() ({\n    }) : () -> ()\n')]
        for i,value in enumerate(changed):
            with self.subTest(i=i):
                raw=self.root/f'bad{i}.mlir';raw.write_text(value)
                with self.assertRaises(ValueError):entry.mark_segment_trace(raw,self.root/f'bad{i}-marked.mlir')

    def test_quoted_comments_cannot_spoof_closure_or_function_and_multiple_functions_reject(self):
        # Original-format operation names and location strings remain quoted.
        source=generic_trace('    // fake }) : () -> ()\n'
            '    %0 = "earth.negate"(%arg0) {note = "fake }) : () -> ()"} : () -> ()\n')
        raw=self.root/'quoted.mlir';raw.write_text(source)
        entry.mark_segment_trace(raw,self.root/'quoted-marked.mlir')
        self.assertIn('note = "fake }) : () -> ()"',(self.root/'quoted-marked.mlir').read_text())
        for i,value in enumerate(('// '+generic_trace().replace('\n','\n// '),
                                   generic_trace()+generic_trace(),generic_trace().replace('  }) :','  }) BAD :',1))):
            raw=self.root/f'spoof{i}.mlir';raw.write_text(value)
            with self.assertRaises(ValueError):entry.mark_segment_trace(raw,self.root/f'spoof{i}-marked.mlir')

    def test_large_dense_line_is_copied_in_bounded_reads_without_whole_file_read_text(self):
        source=generic_trace('    %0 = "earth.constant"() {value = dense<['+'0.0,'*600000+'0.0]>} : () -> ()\n')
        raw=self.root/'dense.mlir';raw.write_text(source)
        with patch.object(Path,'read_text',side_effect=AssertionError('No whole-trace read_text')):
            entry.mark_segment_trace(raw,self.root/'dense-marked.mlir')
        self.assertGreater((self.root/'dense-marked.mlir').stat().st_size,raw.stat().st_size)
        with self.assertRaisesRegex(ValueError,'fresh marked trace'):
            entry.mark_segment_trace(raw,self.root/'dense-marked.mlir')

    def test_actual_native34_32_ABI_and_full_SSA_are_required_without_full_model_claim(self):
        plan=self.root/'lowered.golden.runtime-plan.json';save(plan,native_plan())
        report=entry.inspect_runtime_plan(plan,32)
        self.assertTrue(report['probe_accepted'])
        self.assertTrue(report['native_ABI_verified'])
        self.assertTrue(report['selector_input_retained'])
        self.assertFalse(report['full_model_compile_verified'])
        self.assertEqual(report['compiled_bootstraps'],32)
        self.assertEqual(report['input_output_metadata'],entry.EXPECTED)
        self.assertEqual(report['remaining23_layers_and_head'],'not_run')

    def test_native_IO_boot_target_context_and_final_producer_mismatches_reject(self):
        for i,(field,value)in enumerate((('level',4),('scale_log2',42),('context','new-context'),('ntt',False),('components',3))):
            plan=native_plan();plan['values'][33][field]=value;path=self.root/f'bad{i}.json';save(path,plan)
            with self.assertRaises(ValueError):entry.inspect_runtime_plan(path,32)
        variations=[]
        plan=native_plan();plan['values'][34]['level']=2;variations.append(plan)
        plan=native_plan();plan['execution'][1]['attrs']['operator_profile']='different';variations.append(plan)
        plan=native_plan();plan['final_outputs'][0]='0';variations.append(plan)
        plan=native_plan();plan['execution'][0]['inputs']=['97'];variations.append(plan)
        plan=native_plan();plan['execution'][2]['output']='35';variations.append(plan)
        plan=native_plan();plan['execution'][0]['op']='unknown';variations.append(plan)
        plan=native_plan();plan['external_inputs'].pop();variations.append(plan)
        for i,plan in enumerate(variations):
            path=self.root/f'var{i}.json';save(path,plan)
            with self.subTest(i=i),self.assertRaises(ValueError):entry.inspect_runtime_plan(path,32)

    def test_bundle_closure_uses_manifest_and_stat_without_opening_payload(self):
        plan=native_plan();plan['values'].append(dict(id='98',kind='plaintext',level=5,scale_log2=43,components=1,
            context='qwen-bootstrap48',ntt=True,place=dict(kind='host',rank=0)))
        content='sha256:'+'a'*64
        plan['initialization']=[dict(kind='encode',output='98',payload=dict(kind='bundle',content=content))]
        plan['plaintext_bundle']=dict(id='public-bundle',version=1)
        path=self.root/'lowered.golden.runtime-plan.json';save(path,plan)
        bundle=self.root/'lowered.golden.bundle';save(bundle/'manifest.json',dict(bundle_id='public-bundle',
            version=1,bundle_format_version=1,blobs=[dict(content=content,byte_length=8)]))
        payload=bundle/'data'/('a'*64+'.bin');payload.parent.mkdir();payload.write_bytes(b'public!!')
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path==payload:raise AssertionError('No payload read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):report=entry.inspect_runtime_plan(path,32)
        self.assertEqual(report['bundle_closure']['blob_payload_reads'],0)
        payload.unlink()
        with self.assertRaisesRegex(ValueError,'file closure'):entry.inspect_runtime_plan(path,32)

    def test_original_native_argv_reuses_pars_profile_boot_contract_and_only_new_paths(self):
        command=entry.native_command(self.args,dict(hecate_opt=self.root/'unused-hecate-opt'))
        self.assertIn('--pars',command);self.assertIn('--waterline=43',command)
        self.assertIn('--runtime-plan-context-id=qwen-bootstrap48',command)
        self.assertIn('--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',command)
        self.assertEqual(command[1],str(self.args.codegen/'emit/candidate_trace.segment.earth.mlir'))
        self.assertIn('--ckks-config='+str(self.args.profile),command)
        bad=template()+['--runtime-plan-levels-per-logical-level=2'];save(self.args.metadata_template,bad)
        with self.assertRaisesRegex(ValueError,'without physical remapping'):entry.native_command(self.args,dict(hecate_opt='unused'))

    def test_C2_rejection_happens_before_case_creation_model_or_process(self):
        with patch.object(entry.original,'frozen_source',side_effect=ValueError('C2 not frozen')), \
                patch.object(entry,'_load',side_effect=AssertionError('No model')), \
                patch.object(entry,'run_process',side_effect=AssertionError('No process')):
            with self.assertRaisesRegex(ValueError,'C2 not frozen'):entry.execute(self.args)
        self.assertFalse(self.args.out.exists())

    def test_finite_resources_and_original_ordinary_supervision_namespace(self):
        limits=entry.resource_limits(self.args)
        self.assertEqual(limits['address_space_gib'],24)
        self.assertEqual(limits['subtree_rss_gib'],14)
        self.assertEqual(limits['minimum_available_gib'],1.25)
        self.assertEqual(limits['minimum_free_bytes'],10*1024**3)
        self.assertEqual(limits['file_gib'],64)
        self.assertEqual(limits['case_bytes'],128*1024**3)
        namespace=entry._namespace()
        self.assertIs(namespace['_projection_backend_scope'],entry.original._projection_backend_scope)
        self.assertIsNot(namespace['execute'].__globals__,entry.original.execute.__globals__)
        self.assertIs(namespace['execute'].__code__,entry.original.execute.__code__)
        command=namespace['phase_command'](self.args,'placement',self.root/'new',200)
        self.assertIn(entry.MODULE,command)
        self.assertEqual(command[command.index('--memory-gib')+1],'24.0')
        self.assertEqual(command[command.index('--phase-seconds')+1],'80')
        changed=deepcopy(self.args);changed.seconds=10801
        with self.assertRaises(ValueError):entry.resource_limits(changed)

    def test_short_complete_budget_also_caps_original_source_admission_alarm(self):
        self.args.seconds=1
        def source(case):
            remaining,_=signal.getitimer(signal.ITIMER_REAL)
            self.assertGreater(remaining,0.)
            self.assertLessEqual(remaining,1.)
            raise ValueError('source admission stopped within complete budget')
        with patch.object(entry.original,'frozen_source',side_effect=source):
            with self.assertRaisesRegex(ValueError,'source admission stopped'):
                entry.execute(self.args)
        self.assertFalse(self.args.out.exists())

    def test_independent_Count_and_trusted_Emit_share_complete32_boundary_inventory(self):
        program,shapes=fixture();partition=entry.layered.split_program(program,shapes)
        weights={name:SimpleNamespace(shape=shape)for name,shape in shapes.items()}
        callback=entry.layered.callbacks(partition,weights)[0][0]
        plan=dict(format='pt-agent-refresh-plan-v1',levels=dict(input_level=5,boot_in_level=1,boot_out_level=5),
            calls=66,boots=0,substitutions=[],outputs=[],dead=[])
        bound=entry.layered.bind_plan(callback.segment,plan,planned_source=entry.layered.source_binding(callback.segment))
        def lower(program,weights,active):
            inputs=[active.input(i)for i in range(34)]
            active.output([*[x*.5 for x in inputs[:32]],*inputs[32:]])
        self.args.codegen.mkdir()
        save(self.args.codegen/'planner-return.json',dict(source=entry.layered.source_binding(callback.segment),
            planner='original.pt_agent.plan.plan_program',planner_returned=True,verify_completed=True))
        save(self.args.codegen/'refresh-plan.json',plan)
        with patch.object(entry.original,'_same_lowering',side_effect=lower):
            counted=entry._count(callback,bound)
            save(self.args.codegen/'count-replay.json',counted)
            result={}
            with patch.object(entry,'_load',return_value=(partition,callback)):
                entry.emit(self.args,result,entry.StageRecorder(self.root/'emit-events.jsonl'))
        self.assertEqual(result['emit_stats']['counts'],counted['stats']['counts'])
        self.assertEqual(result['emit_stats']['counts']['boot'],32)
        self.assertTrue(result['selector_input_retained'])
        source=self.args.codegen/'emit/candidate_trace.py'
        functions=[n for n in ast.walk(ast.parse(source.read_text()))if isinstance(n,ast.FunctionDef)]
        self.assertEqual(len(functions[0].args.args),34)

    def test_failed_first_outer_phase_closes_registry_and_never_claims_full_model(self):
        self.args.out.parent.mkdir()
        source=dict(metadata_template=str(self.args.metadata_template),frontend=str(self.args.frontend),profile=str(self.args.profile))
        observed=dict(exit_code=1,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
        namespace=entry._namespace();namespace['run_process']=lambda *args,**kwargs:observed
        namespace['frozen_source']=lambda case:source
        with patch.object(entry.original.paths,'RESULTS_DIR',self.args.out.parent), \
                patch('pt_agent.registry.git_rev',return_value='public-fixture-revision'), \
                patch.object(entry,'_namespace',return_value=namespace):
            result=entry.execute(self.args)
        self.assertEqual(result['status'],'error')
        self.assertFalse(result['probe_accepted'])
        self.assertFalse(result['full_model_compile_verified'])
        self.assertEqual(result['attempted']['placement'],1)
        self.assertEqual(result['attempted']['emit'],0)
        rows=[json.loads(line)for line in (self.args.out.parent/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(rows[0]['mode'],entry.MODE)
        complete=json.loads((self.args.out/'completion.json').read_text())
        self.assertEqual(complete['status'],'error')
        self.assertTrue(complete['registry_persisted'])

    def test_terminal_reserve_failure_clears_probe_acceptance_before_persistence(self):
        self.args.out.mkdir(parents=True);(self.args.out/'source').mkdir()
        result=dict(round=entry.ROUND,mode=entry.MODE,model=entry.MODEL,status='ok',
            completed=dict(compile=1),attempted=dict(compile=1),
            phase_results=dict(compile=dict(native_ABI_verified=True)),probe_accepted=True)
        namespace=entry._namespace()
        # A fixed fake inventory triggers the original terminal reserve branch;
        # no large file, model, subprocess or real resource exhaustion is used.
        namespace['artifact_bytes']=lambda out:self.args.case_gib*entry.GIB
        namespace['persist_case_completion']=lambda *a,**kw:None
        with patch.object(entry.original.paths,'RESULTS_DIR',self.args.out.parent):
            namespace['_terminal'](self.args,self.args.out,result,1,'public-fixture-revision',
                entry.StageRecorder(self.args.out/'events.jsonl'))
        saved=json.loads((self.args.out/'result.json').read_text())
        self.assertEqual(saved['status'],'error')
        self.assertFalse(saved['probe_accepted'])
        self.assertFalse(saved['full_model_compile_verified'])
        self.assertEqual(saved['metrics']['status'],'error')


if __name__=='__main__':unittest.main()
