"""Finite N17 wrapper protocol using only synthetic metadata and fake children."""
from argparse import Namespace
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_native_execute as execute


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def observation(**changes):
    result=dict(exit_code=0,timed_out=False,signal=None,started_ns=1,ended_ns=2,
        termination_observed_ns=2,peak_rss_kib_by_pid={},outer_peak_rss_kib=0,
        limits=dict(memory_gib=24,file_gib=8,wall_seconds=10800),signals_sent=[],remaining_pids=[])
    result.update(changes);return result


def parameter_fixture():
    q=list(range(101,118));p=[211,223,227]
    return dict(residual_parameters=dict(n=131072,logical_log_slots=16,max_level=16,
        log_default_scale=51,q=q,p=p),bootstrap_parameter_summary=dict(n=131072,logical_log_slots=16,
        max_level=31,log_default_scale=51,q=q+list(range(301,316)),p=p+[229]))


def fixtures(root):
    prepared=root/'prepared';input_prepare=root/'input-prepare';original=root/'original'
    binary=prepared/'work/dacapo-client';binary.parent.mkdir(parents=True)
    binary.write_text('prepared binary placeholder, never executed');binary.chmod(0o700)
    source_go=prepared/'work/module/main.go';source_go.parent.mkdir();source_go.write_text('// actual prepared source\n')
    plan=original/'model.runtime-plan.json';save(plan,{'execution':[]})
    bundle=original/'model.bundle';save(bundle/'manifest.json',dict(blobs=[]))
    scalar=original/'scalar.json';proof=original/'math-proof.json';save(scalar,{});save(proof,{})
    archive=original/'archive.json';save(archive,{})
    math_result=original/'math/result.json';save(math_result,dict(selected_paths=dict(archive_manifest=str(archive))))
    source=dict(math_result=str(math_result),plan=str(plan),bundle=str(bundle),scalar_source_proof=str(scalar),
        source_math_proof=str(proof),reference=[float(i)-5 for i in range(10)],
        expected_reference_source='independent synthetic formula',sample_id='train:0')
    client=dict(status='ok',mode='parameters',variant=execute.binding.VARIANT,precision_bits=51,
        bootstrap_cache_capacity=1,requested_checkpoint_interval=1000,bootstrap_input_divisor=0,
        bootstrap_normalization_log2=0,final_decoder_protocol=1,supported_final_output_multipliers=[1,32],
        original_final_decoder=execute.reference.DECODER)
    client.update(dict.fromkeys(execute.NATIVE_COUNTS,0))
    client.update(parameter_fixture())
    prior=dict(status='ok',variant=execute.PREPARE_VARIANT,tests_passed=11,actual_model_ckks_runs=0,
        extra_integrity_checks=0,downloads=0,installations=0,llm_calls=0,
        cache_selftest=dict(cache_selftest_passed=True),checkpoint_selftest=dict(checkpoint_selftest_passed=True),
        ready_binary=str(binary),source_go=str(source_go),parameters=client,
        steps=[dict(stage=s,process=observation()) for s in ('build','n17_tests','cache_selftest','checkpoint_selftest','parameters')])
    save(prepared/'result.json',prior);save(prepared/'parameters.json',client)
    inp=input_prepare/'input.bin';inp.parent.mkdir()
    with inp.open('wb') as stream:stream.truncate(execute.reference.INPUT_BYTES)
    expected=input_prepare/'expected.json';save(expected,execute.reference.make_expected(source))
    inputs=dict(round=execute.reference.ROUND,model=execute.MODEL,mode=execute.reference.MODE,status='ok',
        reference_prepared=True,source_plan_modified=False,saved_input_copy_attempts=1,
        saved_input_copy_completed=1,source=source,input=str(inp),expected=str(expected))
    inputs.update(dict.fromkeys(execute.reference.ZERO_FIELDS,0))
    inputs['native_arguments']=execute.native_arguments(source,inp,expected)
    save(input_prepare/'result.json',inputs)
    inventory=dict(variant=execute.binding.VARIANT,execution_count=12661,bootstrap_count=19,
        inline_scalar_count=122,original_plan_modified=False,execution_scope_ready=True,
        initialization_count=1702,op_counts=dict(add_cc=12661))
    return prepared,input_prepare,source,inventory,prior,inputs


def selected(root):
    prepared,inputs,source,inventory,_,_=fixtures(root)
    return dict(prepared=str(prepared),input_prepare=str(inputs),binary=str(prepared/'work/dacapo-client'),
        compiled_source=str(prepared/'work/module/main.go'),source=source,inventory=inventory,
        input=str(inputs/'input.bin'),expected=str(inputs/'expected.json'),
        native_parameter_metadata=parameter_fixture(),
        native_arguments=execute.native_arguments(source,inputs/'input.bin',inputs/'expected.json'))


def completed_client(source, inventory):
    values=list(source['reference'])
    result=dict(status='ok',mode='execute',variant=execute.binding.VARIANT,encrypted_plan_completed=True,
        source_math_and_scalar_sites_verified=True,public_fixture_only=False,get_log_enabled=False,
        compute_attempts=inventory['execution_count'],compute_completed=inventory['execution_count'],
        bootstrap_attempts=inventory['bootstrap_count'],bootstrap_completed=inventory['bootstrap_count'],
        encryption_attempts=1,encryption_completed=1,final_decryption_attempts=1,final_decryption_completed=1,
        intermediate_decryptions=0,output_multiplier=32,final_selected_values=values,
        final_selected_raw_values=[x/32 for x in values],final_selected_scaled_values=values,
        final_decoder_contract=dict(decoder=execute.reference.DECODER,source_plan=source['plan'],
            source_math_proof=source['source_math_proof'],source_factory=execute.binding.SOURCE_FACTORY,
            sample_id='train:0',public_fixture=False),
        comparison=dict(passed=True,elements=10,absolute_tolerance=1e-5,relative_tolerance=1e-4))
    result.update(parameter_fixture())
    result.update(input_ssa_check_enabled=True,public_key_request_trace_enabled=True,
        input_ssa_observations=[dict(ordinal=i,op='add_cc',metadata_preserved=True)
            for i in range(inventory['initialization_count'],inventory['initialization_count']+inventory['execution_count'])],
        native_bootstrap_api_attempts=inventory['bootstrap_count'],native_bootstrap_api_completed=inventory['bootstrap_count'],
        bootstrap_observations=[dict(native_completed=True,input_ssa_preserved=True,input_value_divisor=1)
            for _ in range(inventory['bootstrap_count'])])
    return result


class OriginalN17ExecutionChecks(unittest.TestCase):
    def gate_stack(self,source,inventory):
        stack=ExitStack();stack.enter_context(patch.object(execute.reference,'source_gate',return_value=source))
        inv=stack.enter_context(patch.object(execute.binding,'plan_inventory',return_value=inventory))
        stack.enter_context(patch.object(execute,'check_restored_bundle',return_value=dict(status='metadata_checked')))
        return stack,inv

    def test_gate_recomputes_original_inventory_and_preserves_actual_build_source(self):
        with TemporaryDirectory() as temp:
            prepared,inputs,source,inventory,_,_=fixtures(Path(temp));stack,inv=self.gate_stack(source,inventory)
            with stack:actual=execute.source_gate(prepared,inputs)
            self.assertEqual(actual['compiled_source'],str(prepared/'work/module/main.go'))
            self.assertEqual(actual['inventory'],inventory)
            self.assertEqual(inv.call_args.kwargs['plan_path'],Path(source['plan']))
            self.assertIn('source_proof',inv.call_args.kwargs);self.assertIn('source_math',inv.call_args.kwargs)
            self.assertEqual(actual['native_arguments'][-2:],['-checkpoint-interval','1000'])

    def test_gate_rejects_foreign_binary_old_decoder_dirty_prepare_and_added_argument(self):
        for change in ('foreign','old_decoder','dirty','flag','expected','source','partial_inventory'):
            with TemporaryDirectory() as temp:
                prepared,inputs,source,inventory,prior,record=fixtures(Path(temp))
                if change=='foreign':prior['ready_binary']=str(prepared/'work/module/main.go')
                elif change=='old_decoder':prior['parameters'].pop('final_decoder_protocol');save(prepared/'parameters.json',prior['parameters'])
                elif change=='dirty':prior['steps'][2]['process']['remaining_pids']=[123]
                elif change=='flag':record['native_arguments']+=['-public-scalar-fixture']
                elif change=='expected':
                    path=inputs/'expected.json';bad=json.loads(path.read_text());bad['output_multiplier']=1;save(path,bad)
                elif change=='source':record['source']=dict(source,sample_id='test:0')
                elif change=='partial_inventory':inventory['execution_count']=12660
                save(prepared/'result.json',prior);save(inputs/'result.json',record)
                stack,_=self.gate_stack(source,inventory)
                with self.subTest(change=change),stack,self.assertRaises(ValueError):execute.source_gate(prepared,inputs)

    def test_final_physical32_is_checked_against_original_reference_without_changing_tolerance(self):
        source=dict(plan='/original/plan',source_math_proof='/original/proof',reference=[float(i)-5 for i in range(10)])
        inventory=dict(execution_count=12661,bootstrap_count=19,initialization_count=1702,
            op_counts=dict(add_cc=12661));expected=execute.reference.make_expected(dict(source,expected_reference_source='formula'))
        client=completed_client(source,inventory);execute.check_client(client,inventory,expected,parameter_fixture())
        for change in ('unscaled','nonfinite','wrong_source','partial','mismatch','intermediate'):
            bad=copy.deepcopy(client)
            if change=='unscaled':bad['output_multiplier']=1
            elif change=='nonfinite':bad['final_selected_raw_values'][0]=float('inf')
            elif change=='wrong_source':bad['final_decoder_contract']['source_plan']='/foreign'
            elif change=='partial':bad['bootstrap_completed']=18
            elif change=='mismatch':
                bad['final_selected_raw_values'][0]+=.1;bad['final_selected_scaled_values'][0]+=3.2
                bad['final_selected_values']=bad['final_selected_scaled_values']
            else:bad['intermediate_decryptions']=1
            with self.subTest(change=change),self.assertRaises(ValueError):execute.check_client(bad,inventory,expected,parameter_fixture())

    def test_input_ssa_and_bootstrap_observations_require_every_original_instruction(self):
        source=dict(plan='/original/plan',source_math_proof='/original/proof',reference=[float(i) for i in range(10)])
        inventory=dict(execution_count=3,bootstrap_count=1,initialization_count=2,op_counts=dict(add_cc=3))
        client=completed_client(source,inventory)
        expected=execute.reference.make_expected(dict(source,expected_reference_source='formula'))
        execute.check_client(client,inventory,expected,parameter_fixture())
        for change in ('disabled','missing','mutated','ordinal','boot','parameters'):
            bad=copy.deepcopy(client)
            if change=='disabled':bad['input_ssa_check_enabled']=False
            elif change=='missing':bad['input_ssa_observations'].pop()
            elif change=='mutated':bad['input_ssa_observations'][0]['metadata_preserved']=False
            elif change=='ordinal':bad['input_ssa_observations'][0]['ordinal']=99
            elif change=='boot':bad['bootstrap_observations'][0]['input_ssa_preserved']=False
            else:bad['residual_parameters']['q'][0]+=1
            with self.subTest(change=change),self.assertRaises(ValueError):execute.check_client(bad,inventory,expected,parameter_fixture())

    def test_public_parameter_group_requires_n17_scale51_and_shared_qp_prefix(self):
        correct=parameter_fixture();self.assertEqual(execute.check_n17_parameters(correct),correct)
        for change in ('n16','scale','prefix','p_count'):
            bad=copy.deepcopy(correct)
            if change=='n16':bad['residual_parameters']['n']=65536
            elif change=='scale':bad['bootstrap_parameter_summary']['log_default_scale']=50
            elif change=='prefix':bad['bootstrap_parameter_summary']['q'][0]+=1
            else:bad['residual_parameters']['p'].append(239)
            with self.subTest(change=change),self.assertRaises(ValueError):execute.check_n17_parameters(bad)

    def execute_fixture(self,root,*,child=None,available=20*execute.GIB):
        chosen=selected(root);results=root/'results';results.mkdir();args=Namespace(out=results/'native',
            prepared=Path(chosen['prepared']),input_prepare=Path(chosen['input_prepare']))
        def run(argv,**kwargs):
            self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['env']['GOMEMLIMIT'],'11GiB')
            self.assertLessEqual(kwargs['timeout'],10800);self.assertEqual(kwargs['env']['GOMAXPROCS'],'1')
            save(args.out/'client.json',completed_client(chosen['source'],chosen['inventory']))
            return observation() if child is None else child(args.out)
        with (patch.object(execute.paths,'RESULTS_DIR',results),patch.object(execute,'source_gate',return_value=chosen),
              patch.object(execute,'run_process',side_effect=run) as call,
              patch.object(execute,'_available_ram_bytes',return_value=available),
              patch.object(execute.shutil,'disk_usage',return_value=Namespace(free=20*execute.GIB)),
              patch('pt_agent.registry.git_rev',return_value='fixture')):
            observed=execute.execute(args)
        return args,observed,call

    def test_single_child_unique_registry_full_endpoint_and_exclusive_timing(self):
        with TemporaryDirectory() as temp:
            args,record,call=self.execute_fixture(Path(temp))
            self.assertEqual(record['status'],'ok',record['error']);self.assertEqual(call.call_count,1)
            self.assertEqual((record['native_attempts'],record['native_completed']),(1,1))
            self.assertEqual((args.out/'actual-compiled-client.go').read_text(),'// actual prepared source\n')
            rows=[json.loads(s) for s in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            completion=json.loads((args.out/'completion.json').read_text());self.assertTrue(completion['registry_persisted'])
            self.assertEqual(completion['case_artifact_bytes'],execute.artifact_bytes(args.out))
            spans=record['timing']['spans'];roots=[r for r in spans if r['parent_id'] is None]
            self.assertEqual(len(roots),1);self.assertEqual(sum(r['exclusive_ns'] for r in spans),roots[0]['inclusive_ns'])
            self.assertTrue(all(record[k]==0 for k in execute.ZERO_FIELDS))
            for key in ('security_acceptance','accuracy_acceptance','CKKS_latency','compiled_bootstraps'):self.assertIsNone(record[key])

    def test_external_signal_retains_partial_client_and_does_not_retry(self):
        with TemporaryDirectory() as temp:
            args,record,call=self.execute_fixture(Path(temp),child=lambda out:observation(exit_code=-15,signal=15,
                signals_sent=[dict(pid=123,signal=15)]))
            self.assertEqual(call.call_count,1);self.assertEqual(record['status'],'error')
            self.assertEqual(record['termination'],dict(cause='incoming_signal',signal=15))
            self.assertEqual(record['native_completed'],0);self.assertIsNotNone(record['client'])
            self.assertTrue((args.out/'completion.json').exists())

    def test_admission_failure_starts_zero_children_and_keeps_unknown_native_counts_null(self):
        with TemporaryDirectory() as temp:
            _,record,call=self.execute_fixture(Path(temp),available=execute.GIB)
            self.assertEqual(call.call_count,0);self.assertEqual(record['native_attempts'],0)
            self.assertEqual(record['termination']['cause'],'physical_memory_admission')
            self.assertTrue(all(v is None for v in record['metrics']['native_counts'].values()))

    def test_resource_guard_observations_distinguish_rss_and_available_ram(self):
        observed=execute.physical_memory_observation(dict(subtree_rss_bytes=15*execute.GIB,sample_ts_ns=1),
            20*execute.GIB,rss_gib=14,minimum_available_gib=1.25)
        self.assertTrue(observed['stop']);self.assertEqual(observed['reasons'],['child subtree RSS allowance exceeded'])
        self.assertTrue(execute.physical_memory_observation(None,execute.GIB,rss_gib=14,minimum_available_gib=1.25)['stop'])

    def test_guard_notification_write_error_still_signals_own_supervisor(self):
        with (patch.object(execute,'write_json',side_effect=OSError('notification failure')),
              patch.object(execute.os,'kill') as kill):
            with self.assertRaises(OSError):execute._notify_guard(Path('/synthetic'),dict(cause='physical_memory_guard'))
        kill.assert_called_once_with(execute.os.getpid(),signal.SIGTERM)

    def test_terminal_case_includes_result_metadata_and_records_budget_failure(self):
        with TemporaryDirectory() as temp:
            out=Path(temp);record=dict(status='ok',error=None,artifact_bytes=0,metrics=dict(status='ok'))
            with patch.object(execute,'CASE_BYTES',65537):execute._terminal(out,record)
            self.assertEqual(record['status'],'error');self.assertEqual(record['metrics']['status'],'error')
            self.assertEqual(record['artifact_bytes'],execute.artifact_bytes(out))

    def test_existing_or_source_output_is_rejected_before_writes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);prepared,inputs,*_=fixtures(root)
            with patch.object(execute.paths,'RESULTS_DIR',root):
                with self.assertRaises(ValueError):execute.execute(Namespace(out=prepared,prepared=prepared,input_prepare=inputs))


if __name__=='__main__':unittest.main()
