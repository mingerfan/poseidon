"""Finite serial protocol fixtures; no real model, Provider or compiler runs."""
from argparse import Namespace
from contextlib import ExitStack
import copy
import errno
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_serial_continuation as driver


def save(path,record):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record)+'\n')


def fixture(root):
    results=root/'results';case=results/'retained';model=root/'model';inputs=root/'sst2'
    config=dict(model_type='bert',architectures=['BertForSequenceClassification'],hidden_size=4,num_hidden_layers=2,num_labels=2)
    save(model/'config.json',config);(model/'pytorch_model.bin').write_text('metadata-only source fixture')
    split=dict(kind='sst2_native',version='sst2-native-v1',seed=20261024,
        calibration=list(range(1000)),heldout=list(range(1000,2000)),confirm=list(range(2000,3000)),report=list(range(3000,3872)))
    save(case/'model/model_info.json',dict(source=dict(type='hf',local_path=str(model),inputs_path=str(inputs),config=config),
        task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5},split=split,
        inputs=[dict(name='inputs_embeds',kind='tokens',shape=[1,40,4])],sequence=dict(length=40)))
    save(case/'model/samples/source.json',dict(source=str(inputs),partition=split,native_test_read=False))
    prepared=dict(status='ok',phase='prepare',model=driver.MODEL,import_completed=True,model_dir=str(case/'model'),
        model_metadata=dict(unchanged_original_model=str(model)),reference_checks=[dict(passed=True) for _ in range(3)])
    prepared.update(dict.fromkeys(('native_test_reads','report_samples','confirm_samples','llm_calls','compiler_runs',
        'keygen_runs','real_ckks_attempts','real_ckks_completed'),0));save(case/'phases/001-prepare/worker_result.json',prepared)
    state=dict(driver.STATE_POLICY,phase='evaluate',iteration=2,calls=1,failed=None);save(case/'session/SESSION.json',state)
    template=root/'template.json';save(template,['{hecate_opt}','{input}','--pars',
        '--ckks-config={package}/config/compiler-config.json','--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
        '--runtime-plan-context-id=qwen-bootstrap48','--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
        '--waterline=43','--output-val=10','-o','{output_dir}/lowered.mlir'])
    return Namespace(source_case=case,out=results/'driver',checkpoint_out=results/'audit',compile_out=results/'compiled',
        intake_python=Path('/existing/intake/python'),data_python=Path('/existing/data/python'),
        ledger=results/'llm_calls.jsonl',allowance=results/'allowance.json',metadata_template=template,
        profile=driver.paths.PROFILES_DIR/'qwen48/compiler-config.json',frontend=driver.paths.DACAPO_ROOT/'python/hecate/hecate/expr.py',
        max_steps=25,overall_seconds=3600,launched=False,detach=False)


def frozen(case,*,selected=True,error=None):
    state=json.loads((case/'session/SESSION.json').read_text());state.update(phase='done',failed=error);save(case/'session/SESSION.json',state)
    confirmation=dict(selected=None,evals=[],confirm_rule='c2',applicable=True)
    if selected:
        program=case/'session/program.ptdsl';program.write_text('fn fixed_fixture() {}\n')
        confirmation.update(selected=dict(source=program.read_text(),iteration=2,origin='tuned'),
            evals=[dict(iteration=2,origin='tuned',total=1000,passed=True,diverged=False,n_diverged=0,acc_drop_pp=.2,
                max_u_calibrated=.8,max_u_analytic=1.,tail=dict(passed=True,total=3000,diverged=False,n_diverged=0,metrics=dict(acc_drop_pp=.3)))])
        save(case/'session/FROZEN.json',dict(program=str(program),origin='tuned',iteration=2,confirm=confirmation,
            **{k:driver.STATE_POLICY[k] for k in ('prompt_version','tune_rule','confirm_rule','split_version')}))
    save(case/'session/AGENT.json',dict(status='error' if error else 'feasible' if selected else 'confirm_failed',
        error=error,confirm=confirmation))


def checkpoint(args,source):
    from pt_agent.experiments.bert_checkpoint_audit import ZERO_FIELDS
    worker=dict(status='ok',source=dict(model=source['model'],model_info=source['model_info']),
        state_coverage=dict(passed=True,classifier=dict(both_original_checkpoint_keys=True)),**dict.fromkeys(ZERO_FIELDS,0))
    worker.update(dict.fromkeys(('checkpoint_load_attempts','checkpoint_load_completed','from_pretrained_attempts','from_pretrained_completed'),1))
    save(args.checkpoint_out/'result.json',dict(round='BERT-CHECKPOINT-AUDIT',model=driver.MODEL,status='ok',
        checkpoint_semantic_coverage_passed=True,worker=worker,process=observed()))


def report(args,acceptance):
    from pt_agent.experiments import bert_frozen_compile as compile
    path=args.source_case/'phases/020-report/worker_result.json';program=args.source_case/'session/program.ptdsl'
    worker=dict(status='ok',phase='report',model=driver.MODEL,frozen=True,report_completed=True,report_samples=872,
        task_policy={'acc_drop_pp':.5},task_policy_passed=acceptance,
        frozen_math=[dict(sample=i,checks=[dict(passed=True,criterion=compile.CRITERION) for _ in range(3)]) for i in range(3)])
    worker.update(dict.fromkeys(('native_test_reads','llm_calls','compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'),0))
    worker.update({k:None for k in ('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')})
    save(path,worker);save(args.source_case/'session/REPORT_ACCESS.json',dict(complete=True,program=str(program),report_samples=872,result=str(path)))
    save(args.source_case/'result.json',dict(phase='report',model=driver.MODEL,status='ok',worker=worker))


def observed(**changes):
    result=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],started_ns=1,ended_ns=2)
    result.update(changes);return result


class SerialBertContinuationChecks(unittest.TestCase):
    def test_retained_tuner_and_budget_are_required(self):
        for field,value in [('tuner',False),('tuner_budget_s',600)]:
            with self.subTest(field=field),TemporaryDirectory() as temp:
                args=fixture(Path(temp));path=args.source_case/'session/SESSION.json'
                state=json.loads(path.read_text());state[field]=value;save(path,state)
                with self.assertRaises(ValueError):driver.session_state(args.source_case)

    def test_main_confirmation_count_and_session_error_are_required(self):
        for field in ('count','failed'):
            with self.subTest(field=field),TemporaryDirectory() as temp:
                args=fixture(Path(temp));frozen(args.source_case)
                if field=='count':
                    path=args.source_case/'session/AGENT.json';agent=json.loads(path.read_text());agent['confirm']['evals'][0]['total']=999;save(path,agent)
                    path=args.source_case/'session/FROZEN.json';marker=json.loads(path.read_text());marker['confirm']=agent['confirm'];save(path,marker)
                else:
                    path=args.source_case/'session/SESSION.json';state=json.loads(path.read_text());state['failed']='retained failure';save(path,state)
                with self.assertRaises(ValueError):driver.frozen_state(args.source_case)

    def test_source_is_inferred_and_policy_and_partition_are_unchanged(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            with patch.object(driver.paths,'RESULTS_DIR',args.out.parent):source=driver.source_gate(args)
            self.assertEqual(source['model'],str(Path(temp)/'model'))
            self.assertEqual(source['initial_session']['calls'],1);self.assertFalse(source['report_feedback_permitted'])
            state=json.loads((args.source_case/'session/SESSION.json').read_text());state['confirm_rule']='c1';save(args.source_case/'session/SESSION.json',state)
            with patch.object(driver.paths,'RESULTS_DIR',args.out.parent),self.assertRaises(ValueError):driver.source_gate(args)

    def test_commands_are_exact_existing_entries_with_recorded_early_context(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            with patch.object(driver.paths,'RESULTS_DIR',args.out.parent):source=driver.source_gate(args)
            for phase in ('checkpoint','step','report'):
                command=driver.command(args,source,phase)
                self.assertEqual(command[3],driver.CHECKPOINT if phase=='checkpoint' else driver.RECORDED)
                self.assertNotIn('--phase run',command)
                if phase!='checkpoint':
                    self.assertEqual(command[command.index('--phase')+1],phase)
                    self.assertEqual(command[command.index('--metadata-template')+1],str(args.metadata_template))
            frozen(args.source_case);report(args,False)
            command=driver.command(args,source,'compile');self.assertEqual(command[3],driver.COMPILE)
            self.assertIn('--report-result',command);self.assertNotIn('--model-factory',command)

    def test_done_unselected_is_quality_rejected_but_agent_error_is_separate(self):
        for error in (None,'provider_error'):
            with TemporaryDirectory() as temp:
                args=fixture(Path(temp));frozen(args.source_case,selected=False,error=error)
                if error:
                    with self.assertRaises(RuntimeError):driver.frozen_state(args.source_case)
                else:self.assertEqual(driver.frozen_state(args.source_case),dict(outcome='quality_rejected',frozen=False))

    def test_report_gate_rejects_changed_selected_source_and_domain_failure(self):
        for kind in ('source','domain','tail'):
            with TemporaryDirectory() as temp:
                args=fixture(Path(temp));frozen(args.source_case)
                if kind=='source':(args.source_case/'session/program.ptdsl').write_text('changed')
                else:
                    path=args.source_case/'session/AGENT.json';agent=json.loads(path.read_text())
                    if kind=='domain':agent['confirm']['evals'][0]['max_u_calibrated']=.91
                    else:agent['confirm']['evals'][0]['tail']['passed']=False
                    save(path,agent);marker=args.source_case/'session/FROZEN.json';record=json.loads(marker.read_text());record['confirm']=agent['confirm'];save(marker,record)
                with self.subTest(kind=kind),self.assertRaises(ValueError):driver.frozen_state(args.source_case)

    def run_fixture(self,root,*,selected=True,fail=None,acceptance=True,partial_report=False,signal_number=None,reuse=False,remain=False):
        args=fixture(root)
        with patch.object(driver.paths,'RESULTS_DIR',args.out.parent):source=driver.source_gate(args)
        if reuse:checkpoint(args,source);frozen(args.source_case);report(args,acceptance)
        def child(argv,**kwargs):
            phase='checkpoint' if argv[3]==driver.CHECKPOINT else 'compile' if argv[3]==driver.COMPILE else argv[argv.index('--phase')+1]
            self.assertEqual(kwargs['mem_gib'],driver.PHASE_LIMITS[phase][1])
            if phase==fail:return observed(exit_code=1)
            if signal_number:return observed(exit_code=-signal_number,signal=signal_number)
            if phase=='checkpoint':checkpoint(args,source)
            elif phase=='step':
                if not remain:frozen(args.source_case,selected=selected)
                worker=dict(status='ok',phase='step',model=driver.MODEL,next_step='evaluate' if remain else 'done',report_samples=0,
                    **dict.fromkeys(('native_test_reads','compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'),0))
                worker.update({k:None for k in ('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')})
                save(args.source_case/'result.json',dict(status='ok',phase='step',model=driver.MODEL,worker=worker))
            elif phase=='report':
                if partial_report:save(args.source_case/'session/REPORT_ACCESS.json',dict(complete=False));return observed(exit_code=1)
                report(args,acceptance)
            else:
                provenance=driver.accepted_report(args)['source']
                save(args.compile_out/'result.json',dict(round='BERT-FROZEN-COMPILE',status='ok',source=provenance,
                    completed=dict(emit=1,trace=1,compile=1),compiled_bootstraps=8,predicted_latency_us=None))
            return observed()
        with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),patch.object(driver,'foreign_heavy_processes',return_value=[]),
              patch.object(driver,'run_process',side_effect=child) as calls,
              patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=20*driver.GIB)),
              patch('pt_agent.registry.git_rev',return_value='fixture')):result=driver.execute(args)
        return args,result,calls

    def test_complete_serial_chain_runs_once_and_retains_report_quality_rejection(self):
        with TemporaryDirectory() as temp:
            args,result,calls=self.run_fixture(Path(temp),acceptance=False)
            self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(calls.call_count,4)
            self.assertEqual(result['attempted'],dict(checkpoint=1,step=1,report=1,compile=1))
            self.assertFalse(result['task_report_acceptance']);self.assertEqual(result['compiled_bootstraps'],8)
            self.assertEqual([row['phase'] for row in result['processes']],['checkpoint','step','report','compile'])
            registry=[json.loads(line) for line in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(registry),1);self.assertEqual(set(registry[0]),driver.REGISTRY_FIELDS)
            completion=json.loads((args.out/'completion.json').read_text());self.assertTrue(completion['registry_persisted'])
            spans=result['timing']['spans'];self.assertEqual(sum(row['exclusive_ns'] for row in spans),
                sum(row['inclusive_ns'] for row in spans if row['parent_id'] is None))

    def test_quality_rejected_session_does_not_report_or_compile(self):
        with TemporaryDirectory() as temp:
            _,result,calls=self.run_fixture(Path(temp),selected=False)
        self.assertEqual(result['status'],'quality_rejected',result['error']);self.assertEqual(calls.call_count,2)
        self.assertEqual(result['attempted']['report'],0);self.assertEqual(result['attempted']['compile'],0)

    def test_first_error_or_term_stops_later_commands_and_preserves_partial(self):
        for kind in ('checkpoint','step','report','compile','signal'):
            with TemporaryDirectory() as temp:
                args,result,calls=self.run_fixture(Path(temp),fail=None if kind=='signal' else kind,
                    signal_number=signal.SIGTERM if kind=='signal' else None)
                self.assertEqual(result['status'],'error')
                self.assertTrue((args.out/'completion.json').is_file())
                self.assertEqual(calls.call_count,1 if kind in ('checkpoint','signal') else 2 if kind=='step' else 3 if kind=='report' else 4)
                if kind=='signal':self.assertEqual(result['termination'],dict(cause='incoming_signal',signal=signal.SIGTERM))

    def test_report_failure_retains_access_marker_and_no_second_report_or_compile(self):
        with TemporaryDirectory() as temp:
            args,result,calls=self.run_fixture(Path(temp),partial_report=True)
            self.assertEqual(result['status'],'error');self.assertEqual(calls.call_count,3)
            self.assertFalse(json.loads((args.source_case/'session/REPORT_ACCESS.json').read_text())['complete'])
            self.assertEqual(result['attempted']['compile'],0)

    def test_existing_semantic_audit_and_complete_report_are_reused_without_new_read(self):
        with TemporaryDirectory() as temp:
            _,result,calls=self.run_fixture(Path(temp),reuse=True)
        self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(calls.call_count,1)
        self.assertEqual(result['attempted'],dict(checkpoint=0,step=0,report=0,compile=1))

    def test_foreign_native_admission_stops_new_actions(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp));args.max_steps=1
            with patch.object(driver.paths,'RESULTS_DIR',args.out.parent):source=driver.source_gate(args)
            checkpoint(args,source)
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),patch.object(driver,'foreign_heavy_processes',return_value=[dict(pid=7,executable_name='r31-client')]),
                  patch.object(driver,'run_process',side_effect=AssertionError('No concurrent heavy launch')) as child,
                  patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=20*driver.GIB)),patch('pt_agent.registry.git_rev',return_value='fixture')):
                record=driver.execute(args)
            self.assertEqual(record['status'],'error');self.assertEqual(child.call_count,0)
            self.assertEqual(record['blocked_processes'][0]['pid'],7)

    def test_finite_25_step_allowance_stops_unchanged_inflight_checkpoint(self):
        with TemporaryDirectory() as temp:
            _,result,calls=self.run_fixture(Path(temp),remain=True)
        self.assertEqual(result['status'],'error');self.assertIn('step allowance exhausted',result['error'])
        self.assertEqual(result['attempted']['step'],25);self.assertEqual(calls.call_count,26)
        self.assertEqual(result['attempted']['report'],0);self.assertEqual(result['attempted']['compile'],0)

    def test_source_preflight_carries_saved_operator_identifier_without_new_digest(self):
        from pt_agent import toolchain
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),
                  patch.object(toolchain,'_file_sha256',side_effect=AssertionError('new digest forbidden')),
                  patch.object(toolchain.hashlib,'sha256',side_effect=AssertionError('new digest forbidden'))):
                source=driver.source_gate(args)
        self.assertEqual(source['initial_session']['confirm_rule'],'c2')

    def test_detach_uses_fixed_trusted_module_new_session_and_retains_launch_record(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp));args.detach=True
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),patch.object(driver,'foreign_heavy_processes',return_value=[]),
                  patch.object(driver.subprocess,'Popen',return_value=Namespace(pid=123)) as child):record=driver.start_detached(args)
            self.assertEqual(record['status'],'detached_started')
            argv=child.call_args.args[0];self.assertEqual(argv[:5],[driver.sys.executable,'-B','-m',driver.MODULE,'--launched'])
            self.assertTrue(child.call_args.kwargs['start_new_session']);self.assertTrue(child.call_args.kwargs['close_fds'])
            self.assertEqual(json.loads((args.out/'launch.json').read_text())['pid'],123)
            self.assertFalse((args.out/'result.json').exists())

    def test_nonblocking_serial_lease_and_small_budget_guard(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            with driver.serial_lease(root/'serial.lock'):
                with self.assertRaises(RuntimeError):
                    with driver.serial_lease(root/'serial.lock'):pass
            with patch.object(driver,'artifact_bytes',return_value=driver.CASE_BYTES+1),self.assertRaises(RuntimeError):driver.check_budget(root)

    def test_guard_cause_survives_child_cleanup_signal(self):
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp));args.out.mkdir();(args.out/'phases').mkdir()
            result=dict(termination=None,processes=[],attempted=dict.fromkeys(driver.PHASE_LIMITS,0),completed=dict.fromkeys(driver.PHASE_LIMITS,0))
            with patch.object(driver.paths,'RESULTS_DIR',args.out.parent):source=driver.source_gate(args)
            def stopped(argv,**kwargs):
                result['termination']=dict(cause='driver_guard',error='original finite wall allowance')
                return observed(exit_code=-15,signal=15)
            with (patch.object(driver,'foreign_heavy_processes',return_value=[]),patch.object(driver,'run_process',side_effect=stopped),
                  patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=20*driver.GIB)),self.assertRaises(RuntimeError)):
                driver._launch(args,source,'checkpoint',1,StageRecorder(args.out/'events.jsonl'),result,driver.time.monotonic()+10)
        self.assertEqual(result['termination']['cause'],'driver_guard')
        self.assertEqual(result['processes'][0]['cleanup_signal_after_recorded_termination'],15)

    def test_completion_reserve_is_charged_before_writer_and_actual_inventory_is_bounded(self):
        timing=dict(status='ok',attribution_ready=True,diagnostics=[],spans=[])
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'payload.json').write_text('x'*100)
            with patch.object(driver,'CASE_BYTES',4096),patch.object(driver,'COMPLETION_RESERVE_BYTES',2048):
                marker=driver.persist_terminal_completion(root,started_ns=driver.time.perf_counter_ns(),
                    result=dict(status='ok',timing=timing))
                self.assertEqual(marker['case_artifact_bytes'],driver.artifact_bytes(root))
                self.assertLessEqual(driver.artifact_bytes(root),4096)
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'payload.json').write_text('x'*2049)
            with (patch.object(driver,'CASE_BYTES',4096),patch.object(driver,'COMPLETION_RESERVE_BYTES',2048),
                  patch.object(driver,'persist_case_completion',side_effect=AssertionError('Reserve admission must precede marker')) as write,
                  self.assertRaises(RuntimeError)):
                driver.persist_terminal_completion(root,started_ns=1,result=dict(status='error',timing=timing))
            self.assertEqual(write.call_count,0)

    def test_oversized_completion_marker_is_rejected_without_claiming_final_budget_pass(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            def write_marker(out,**kwargs):
                (out/'completion.json').write_text('x'*256)
                return dict(case_artifact_bytes=driver.artifact_bytes(out))
            with (patch.object(driver,'CASE_BYTES',4096),patch.object(driver,'COMPLETION_RESERVE_BYTES',128),
                  patch.object(driver,'persist_case_completion',side_effect=write_marker),self.assertRaises(RuntimeError)):
                driver.persist_terminal_completion(root,started_ns=1,result=dict(status='ok',timing={}))

    def test_low_disk_guard_still_persists_failed_registry_and_completion_without_new_child(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),
                  patch.object(driver,'foreign_heavy_processes',return_value=[]),
                  patch.object(driver,'run_process',side_effect=AssertionError('Disk guard must stop before a child')) as child,
                  patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=9*driver.GIB)),
                  patch('pt_agent.registry.git_rev',return_value='fixture')):
                result=driver.execute(args)
            self.assertEqual(child.call_count,0);self.assertEqual(result['status'],'error')
            self.assertEqual(result['terminal_budget']['observed_disk_free_bytes'],9*driver.GIB)
            self.assertFalse(result['terminal_budget']['disk_reserve_passed'])
            rows=[json.loads(line) for line in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['metrics']['status'],'error')
            self.assertEqual(set(rows[0]),driver.REGISTRY_FIELDS)
            completion=json.loads((args.out/'completion.json').read_text())
            self.assertEqual(completion['status'],'error');self.assertTrue(completion['registry_persisted'])
            self.assertFalse(completion['resource_budget']['disk_reserve_passed'])
            self.assertEqual(completion['resource_budget']['observed_disk_free_bytes'],9*driver.GIB)
            self.assertEqual(completion['case_artifact_bytes'],driver.artifact_bytes(args.out))
            self.assertLessEqual((args.out/'completion.json').stat().st_size,driver.COMPLETION_RESERVE_BYTES)

    def test_case_guard_closes_failure_and_records_actual_overshoot_without_relaxing_success_cap(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp));case_cap=16384;reserved=8192
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),
                  patch.object(driver,'foreign_heavy_processes',return_value=[]),
                  patch.object(driver,'run_process',side_effect=AssertionError('Case guard must stop before a child')) as child,
                  patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=20*driver.GIB)),
                  patch.object(driver,'CASE_BYTES',case_cap),patch.object(driver,'COMPLETION_RESERVE_BYTES',reserved),
                  patch('pt_agent.registry.git_rev',return_value='fixture')):
                result=driver.execute(args)
            self.assertEqual(child.call_count,0);self.assertEqual(result['status'],'error')
            self.assertFalse(result['terminal_budget']['case_allowance_with_reserved_completion_passed'])
            self.assertGreater(result['terminal_budget']['observed_case_overshoot_bytes'],0)
            completion=json.loads((args.out/'completion.json').read_text())
            self.assertEqual(completion['status'],'error')
            self.assertEqual(completion['case_artifact_bytes'],driver.artifact_bytes(args.out))
            self.assertFalse(completion['resource_budget']['actual_case_within_cap'])
            self.assertEqual(completion['resource_budget']['actual_case_overshoot_bytes'],driver.artifact_bytes(args.out)-case_cap)
            self.assertEqual(completion['resource_budget']['case_limit_bytes'],case_cap)
            self.assertTrue(completion['resource_budget']['successful_case_cap_unchanged'])
            self.assertLessEqual((args.out/'completion.json').stat().st_size,reserved)
            rows=[json.loads(line) for line in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['metrics']['status'],'error')
            self.assertEqual(rows[0]['metrics']['terminal_budget'],result['terminal_budget'])

    def test_failed_resource_closure_requires_error_and_cannot_admit_an_oversized_success(self):
        timing=dict(status='ok',attribution_ready=True,diagnostics=[],spans=[])
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'payload').write_text('x'*2049)
            result=dict(status='ok',timing=timing,terminal_budget=dict(reasons=['case exhausted']))
            with (patch.object(driver,'CASE_BYTES',4096),patch.object(driver,'COMPLETION_RESERVE_BYTES',2048),
                  patch.object(driver,'persist_case_completion',side_effect=AssertionError('No success cap bypass')) as marker):
                with self.assertRaises(ValueError):
                    driver.persist_terminal_completion(root,started_ns=1,result=result,allow_resource_failure=True)
                with self.assertRaises(RuntimeError):driver.persist_terminal_completion(root,started_ns=1,result=result)
            self.assertEqual(marker.call_count,0)
        self.assertEqual(driver.CASE_BYTES,512*1024**2)

    def test_actual_enospc_propagates_and_never_fabricates_registry_or_completion(self):
        with TemporaryDirectory() as temp:
            args=fixture(Path(temp));real_write=driver.durable_json
            def disk_full(path,record):
                if path.name=='result.json':raise OSError(errno.ENOSPC,'synthetic real filesystem full response')
                return real_write(path,record)
            with (patch.object(driver.paths,'RESULTS_DIR',args.out.parent),
                  patch.object(driver,'foreign_heavy_processes',return_value=[]),
                  patch.object(driver,'run_process',side_effect=AssertionError('Failed durable launch cannot start a child')) as child,
                  patch.object(driver.shutil,'disk_usage',return_value=Namespace(free=20*driver.GIB)),
                  patch.object(driver,'durable_json',side_effect=disk_full),
                  patch('pt_agent.registry.git_rev',return_value='fixture')):
                with self.assertRaises(OSError) as stopped:driver.execute(args)
            self.assertEqual(stopped.exception.errno,errno.ENOSPC);self.assertEqual(child.call_count,0)
            self.assertTrue((args.out/'events.jsonl').is_file())
            self.assertFalse((args.out/'completion.json').exists())
            self.assertFalse((args.out.parent/'registry.jsonl').exists())


if __name__=='__main__':unittest.main()
