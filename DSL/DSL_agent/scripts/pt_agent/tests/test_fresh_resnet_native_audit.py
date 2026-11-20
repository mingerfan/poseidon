"""Saved-record audits use finite synthetic metadata and public numeric fixtures."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import fresh_resnet_native_audit as audit


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def write_events(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def events_fixture():
    rows=[]
    def begin(event_id,stage,at,metadata=None):
        rows.append(dict(type='begin',event_id=event_id,stage=stage,elapsed_ns=at,
            metadata=metadata or {}))
    def end(event_id,at,duration,status='ok'):
        rows.append(dict(type='end',event_id=event_id,elapsed_ns=at,duration_ns=duration,status=status))
    begin(1,'parameters.construct',0);end(1,10,10)
    begin(2,'residual.secret_key',10);end(2,20,10)
    begin(3,'bootstrap.evaluator_constructor',20)
    begin(4,'bootstrap_lazy.galois',25);end(4,40,15)
    # This callback intentionally has no parent_event_id, matching the native
    # DFT observer. The actual begin/end stack supplies its nesting.
    begin(5,'bootstrap.dft_matrix',40);end(5,65,25)
    end(3,100,80)
    begin(6,'runtime.compute',100,dict(ordinal=7,op='rotate',index=0))
    begin(7,'residual_lazy.galois',110);end(7,130,20)
    end(6,200,100)
    begin(8,'plaintext.read_encode',200);end(8,210,10)
    begin(9,'final.decrypt_decode_compare',210);end(9,220,10)
    return rows


def recovery_fixture(root):
    case=root/'raw';observed=root/'observer';case.mkdir();observed.mkdir()
    command=dict(native_pid=77,native_starttime_ticks=123,executable=str(root/'r31-client'),
        capture_perf_counter_ns=400,environment_read=False,credentials_read=False,extra_integrity_checks=0,
        argv=[str(root/'r31-client'),'-output',str(case/'client.json'),'-events',str(case/'go-events.jsonl')])
    recovery=dict(round='R31-SURVIVOR-OBSERVATION',mode='existing-native-process-after-supervisor-interruption',
        status='native_completed_observed',source_case=str(case),native_pid=77,native_starttime_ticks=123,
        native_executable=str(root/'r31-client'),initial_identity=dict(pid=77,starttime_ticks=123,ppid=1),
        native_client=str(case/'client.json'),native_events=str(case/'go-events.jsonl'),
        original_resources=str(case/'execute-resources.jsonl'),resources=str(observed/'resources.jsonl'),
        original_supervisor_exit_code=143,original_supervisor_signal=15,original_supervisor_cause='unknown',
        original_supervisor_clean=False,original_continuous_outer_completion=None,native_exit_code=None,waitpid_available=False,
        original_execute_start_ns=100,last_original_resource_ns=200,observation_start_ns=300,observation_end_ns=900,
        resource_observation_gap_seconds=100/1e9,observed_total_since_original_execute_seconds=800/1e9,
        timing=dict(attribution_ready=True,spans=[dict(parent_id=None,inclusive_ns=600,exclusive_ns=600)]),
        native_alive_after_observation=False,guard_stop=None,termination_signals_sent=[],error=None)
    recovery.update(dict.fromkeys(('native_spawned','native_restarted','additional_encryption','additional_keygen',
        'additional_CKKS_runs','llm_calls','extra_integrity_checks'),0))
    completion=dict(required_artifacts_persisted=True,registry_persisted=True,status=recovery['status'],case_artifact_bytes=50)
    row=dict(run_id=observed.name,round=recovery['round'],model='resnet20_silu',mode=recovery['mode'],git_rev=None,
        metrics=dict(status=recovery['status']),artifacts_dir=str(observed),ts='synthetic')
    return case,observed,recovery,command,completion,[row]


def numerical_fixture():
    execution=[dict(op='rotate',ordinal=5,inputs=['0']),dict(op='boot',ordinal=6,inputs=['1'])]
    counts=dict(rotate=1,boot=1)
    residual=dict(n=65536,logical_log_slots=14,log_default_scale=50,q=list(range(9)),p=[0,1,3])
    boot=dict(n=65536,q=list(range(28)),p=list(range(4)))
    client=dict(status='ok',encrypted_plan_completed=True,compute_attempts=2,compute_completed=2,
        bootstrap_attempts=1,bootstrap_completed=1,op_attempts_by_kind=counts,op_completed_by_kind=counts,
        input_ssa_check_enabled=True,input_ssa_observations=[dict(row,metadata_preserved=True) for row in execution],
        intermediate_decryptions=0,get_log_enabled=False,variant=audit.VARIANT,precision_bits=50,c2s_bits=52,
        auxiliary_prime_bits=61,bootstrap_cache_capacity=32,bootstrap_normalization_log2=3,bootstrap_input_divisor=8,
        bootstrap_precision_policy=dict(coeffs_to_slots_log_scales=[[52]]*4,slots_to_coeffs_log_scales=[[39]]*3,
            mod1_inverse_degree=15,log_message_ratio=2),native_bootstrap_api_attempts=1,native_bootstrap_api_completed=1,
        bootstrap_input_value_divisions=1,bootstrap_output_integer_restores=1,
        bootstrap_observations=[dict(native_completed=True,input_ssa_preserved=True,input_value_divisor=8,
            output_integer_multiplier=8,restored_output_level=5)],residual_parameters=residual,bootstrap_parameter_summary=boot,
        keygen_api_attempts=0,keygen_api_completed=0,final_selected_values=[0.0]*10,
        comparison=dict(passed=True,absolute_tolerance=1e-5,relative_tolerance=1e-4),
        security_acceptance=None,accuracy_acceptance=None,concrete_prime_binding_verified=False,extra_integrity_checks=0)
    client.update(dict.fromkeys(('encryption_attempts','encryption_completed','final_decryption_attempts','final_decryption_completed'),1))
    go=dict(compute_rows=[dict(ordinal=row['ordinal'],op=row['op'],index=i) for i,row in enumerate(execution)],
        issues=[],error_stages=[],key_generation_stages=0)
    return dict(execution=execution),client,dict(values=[0.0]*10),go


class FreshNativeAuditChecks(unittest.TestCase):
    def test_original_ckks_numeric_criterion_is_recomputed_without_native_rerun(self):
        result=audit.recompute_ckks([0.0,2.0001],[0.0,2.0])
        self.assertTrue(result['passed']);self.assertEqual(result['elements'],2)
        self.assertEqual(result['absolute_tolerance'],1e-5)
        self.assertEqual(result['relative_tolerance'],1e-4)
        rejected=audit.recompute_ckks([.00002],[0.0])
        self.assertFalse(rejected['passed']);self.assertEqual(rejected['max_error_ratio'],2)
        for values,wants in [([float('nan')],[0.0]),([0.0],[]),([True],[1.0])]:
            with self.subTest(values=values),self.assertRaises(ValueError):audit.recompute_ckks(values,wants)

    def test_native_stage_decomposition_counts_nested_keys_and_dft_once(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'native.jsonl';write_events(path,events_fixture());record=audit.native_events(path)
        self.assertEqual(record['issues'],[])
        self.assertEqual(record['key_generation_stages'],3)
        self.assertEqual(record['compute_rows'],[dict(ordinal=7,op='rotate',index=0)])
        expected=dict(covered_operator_core_seconds=80e-9,bootstrap_galois_generation_seconds=15e-9,
            residual_galois_generation_seconds=20e-9,fixed_key_generation_seconds=10e-9,
            constructor_dft_seconds=25e-9,constructor_other_seconds=40e-9,
            plaintext_read_encode_transfer_seconds=10e-9,cold_parameters_input_encryption_final_seconds=20e-9)
        self.assertEqual(record['components'],expected)
        self.assertAlmostEqual(record['same_component_sum_seconds'],220e-9)
        self.assertEqual(record['native_event_elapsed_endpoint_seconds'],220e-9)

    def test_unclosed_malformed_and_failed_native_stages_are_visible(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl';rows=events_fixture()
            rows[-1]['status']='error';rows.append(dict(type='begin',event_id=10,stage='unfinished',elapsed_ns=221))
            write_events(path,rows)
            with path.open('a') as stream:stream.write('{"type":')
            record=audit.native_events(path)
        self.assertIn('Unclosed native stages',record['issues'])
        self.assertIn('Incomplete/invalid native event JSON',record['issues'])
        self.assertEqual(record['error_stages'],[dict(stage='final.decrypt_decode_compare',event_id=9)])

    def test_resource_tree_rss_and_single_pid_hwm_are_separate(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'resources.jsonl'
            write_events(path,[dict(ts_ns=1,processes=[dict(vmrss_kib=10,vmhwm_kib=90,vmsize_kib=200),
                dict(vmrss_kib=20,vmhwm_kib=50,vmsize_kib=100)],outer_process=dict(vmhwm_kib=80)),
                dict(ts_ns=2,processes=[dict(vmrss_kib=25,vmhwm_kib=100,vmsize_kib=220)])])
            record=audit.resource_summary(path)
        self.assertEqual(record['subtree_sampled_peak_rss_bytes'],30*1024)
        self.assertEqual(record['single_pid_sampled_hwm_kib'],100)
        self.assertEqual(record['single_pid_sampled_vmsize_kib'],220)
        self.assertEqual(record['outer_sampled_hwm_kib'],80)

    def test_recovery_resource_format_keeps_observed_process_metadata(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'resources.jsonl'
            write_events(path,[dict(ts_ns=3,observed=dict(pid=123,starttime_ticks=70,vmrss_kib=33,vmhwm_kib=55)),
                dict(ts_ns=4,observed=None)])
            record=audit.resource_summary(path)
        self.assertEqual(record['samples'],2)
        self.assertEqual(record['subtree_sampled_peak_rss_bytes'],33*1024)
        self.assertEqual(record['single_pid_sampled_hwm_kib'],55)
        self.assertEqual(record['outer_sampled_hwm_kib'],0)

    def test_pending_original_case_never_loads_models_or_writes_terminal_page(self):
        with TemporaryDirectory() as temp:
            case=Path(temp)/'case';save(case/'client.json',dict(status='running',compute_completed=6,bootstrap_completed=2))
            args=Namespace(case=case)
            with patch.object(audit,'validate_sources',side_effect=AssertionError('No terminal source replay')):
                record=audit.audit_case(args)
            self.assertEqual(record['status'],'pending');self.assertFalse(record['terminal'])
            self.assertFalse(record['terminal_success_claim']);self.assertEqual(record['bootstrap_completed'],2)
            with self.assertRaises(ValueError):audit.write_report(Path(temp),record)
            self.assertFalse((Path(temp)/'FRESH_RESNET20_FULL_NATIVE_RESULTS.json').exists())

    def test_recovery_preserves_unknown_exit_and_original_observation_gap(self):
        with TemporaryDirectory() as temp:
            case,observed,recovery,command,completion,registry=recovery_fixture(Path(temp))
            checks=audit.check_recovery_observation(recovery,command,completion,registry,case,observed,case_bytes=50)
        self.assertTrue(all(row['passed'] for row in checks))
        self.assertIsNone(recovery['native_exit_code']);self.assertFalse(recovery['original_supervisor_clean'])
        self.assertIsNone(recovery['original_continuous_outer_completion'])

    def test_recovery_rejects_pid_reuse_invented_exit_duplicate_registry_and_gap_change(self):
        for change in ('identity','exit','registry','gap','restarted','command_time'):
            with TemporaryDirectory() as temp:
                case,observed,recovery,command,completion,registry=recovery_fixture(Path(temp))
                if change=='identity':command['native_starttime_ticks']+=1
                elif change=='exit':recovery['native_exit_code']=0
                elif change=='registry':registry.append(copy.deepcopy(registry[0]))
                elif change=='gap':recovery['resource_observation_gap_seconds']=0.0
                elif change=='restarted':recovery['native_restarted']=1
                else:command['capture_perf_counter_ns']=1
                checks=audit.check_recovery_observation(recovery,command,completion,registry,case,observed,case_bytes=50)
            with self.subTest(change=change):self.assertFalse(all(row['passed'] for row in checks))

    def test_observing_recovery_remains_pending_without_original_terminal_record(self):
        with TemporaryDirectory() as temp:
            case,observed,recovery,_,_,_=recovery_fixture(Path(temp));recovery['status']='observing'
            save(observed/'result.json',recovery);save(case/'client.json',dict(status='running',compute_completed=4))
            args=Namespace(case=case,recovery_case=observed)
            with patch.object(audit,'validate_sources',side_effect=AssertionError('Terminal provenance only')):
                record=audit.audit_case(args)
        self.assertEqual(record['status'],'pending');self.assertFalse(record['terminal'])
        self.assertIsNone(record['original_continuous_outer_completion'])
        self.assertFalse(record['original_supervisor_clean'])

    def test_native_math_and_metadata_audit_separates_normal_numeric_rejection(self):
        plan,client,expected,go=numerical_fixture()
        with patch.object(audit,'EXECUTIONS',2),patch.object(audit,'BOOTSTRAPS',1):
            result=audit.audit_records(None,client,plan,expected,None,[],go,
                dict(subtree_sampled_peak_rss_bytes=10),case_bytes=0)
            self.assertTrue(result['evidence_consistent']);self.assertTrue(result['independent_ckks']['passed'])
            client['status']='error';client['encrypted_plan_completed']=False
            client['final_selected_values'][0]=1.0;client['comparison']['passed']=False
            go['error_stages']=[dict(stage='final.decrypt_decode_compare',event_id=9)]
            rejected=audit.audit_records(None,client,plan,expected,None,[],go,
                dict(subtree_sampled_peak_rss_bytes=10),case_bytes=0)
        self.assertTrue(rejected['evidence_consistent']);self.assertFalse(rejected['independent_ckks']['passed'])

    def test_c2s_auxiliary_basis_uses_same_first_prime_and_subset_without_requiring_prefix(self):
        plan,client,expected,go=numerical_fixture()
        with patch.object(audit,'EXECUTIONS',2),patch.object(audit,'BOOTSTRAPS',1):
            valid=audit.audit_records(None,client,plan,expected,None,[],go,
                dict(subtree_sampled_peak_rss_bytes=10),case_bytes=0)
            self.assertTrue(valid['evidence_consistent'])
            for primes in ([0,1,99],[1,0,3]):
                client['residual_parameters']['p']=primes
                rejected=audit.audit_records(None,client,plan,expected,None,[],go,
                    dict(subtree_sampled_peak_rss_bytes=10),case_bytes=0)
                with self.subTest(primes=primes):self.assertFalse(rejected['evidence_consistent'])

    def test_recovery_source_gate_binds_current_program_and_exact_observed_command(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);math_result=root/'math/result.json';sourcecase=root/'current-case'
            source=dict(input_prepare=str(root/'input-prepare/result.json'),source_case=str(sourcecase),
                plan=str(sourcecase/'original43.runtime-plan.json'),bundle=str(sourcecase/'original43.bundle'),
                input=str(root/'input.bin'),expected=str(root/'expected.json'),program=str(sourcecase/'program.ptdsl'),
                model_info=str(sourcecase/'model_info.json'),reference_values=[0.0]*10)
            proof_path=math_result.parent/'worker/source-math-proof.json'
            proof=dict(status='strict_math_passed',sample_ids=['train:0'],operation_count=audit.EXECUTIONS,
                ideal_bootstrap_count=audit.BOOTSTRAPS,criterion='1e-12+1e-12*abs(reference)',
                source_plan=source['plan'],current_program=source['program'],input_prepare=source['input_prepare'],
                input=source['input'],reference=source['expected'],source_case=source['source_case'])
            save(proof_path,proof);save(math_result,dict(status='ok',source=dict(source,archive_manifest=str(root/'archive.json')),
                worker=dict(mathematical_equivalence_verified=True,source_math_proof=str(proof_path))))
            save(Path(source['plan']),dict(original=True));mapped=root/'mapped.json';save(mapped,dict(mapped=True))
            save(Path(source['expected']),dict(sample_id='train:0',indices=list(range(10)),values=[0.0]*10))
            argv=['native','-mode','execute','-plan',str(mapped),'-bundle',source['bundle'],'-input',source['input'],
                '-expected',source['expected'],'-precision-bits','50','-bootstrap-cache','32','-mod1-inv-degree','15',
                '-log-message-ratio','2','-bootstrap-normalization-log2','3','-auxiliary-prime-bits','61',
                '-c2s-bits','52','-checkpoint-interval','1000','-allow-bootstrap','-trace-key-requests','-check-input-ssa']
            with patch.object(audit.source_math,'source_gate',return_value=source),\
                    patch.object(audit,'map_precision_plan',return_value=(dict(mapped=True),{})):
                current,plan,_,process=audit.validate_sources(None,math_result,source['input_prepare'],observed_command=dict(argv=argv))
                self.assertEqual(current,source);self.assertEqual(plan,dict(mapped=True));self.assertIsNone(process)
                bad=list(argv);bad[bad.index('-expected')+1]=str(root/'foreign-reference.json')
                with self.assertRaises(ValueError):audit.validate_sources(None,math_result,source['input_prepare'],observed_command=dict(argv=bad))
                proof['current_program']=str(root/'old90/program.ptdsl');save(proof_path,proof)
                with self.assertRaises(ValueError):audit.validate_sources(None,math_result,source['input_prepare'],observed_command=dict(argv=argv))


if __name__=='__main__':unittest.main()
