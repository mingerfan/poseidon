"""Small synthetic sources and finite post-freeze protocol only; no numerical IO."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r13_common_models
from pt_agent.experiments import r30_alex_postfreeze as entry
from pt_agent.experiments import r30_alex_continuation_report as report
from pt_agent.experiments import r30_common_model_e2e as common
from pt_agent.experiments import r30_frozen_cnn_report as original_report
from pt_agent.planners.transformer_session import save_state


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def clean():
    return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])


def argv_template():
    return ['{hecate_opt}','{input}','--pars','--waterline=43',
        '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
        '--runtime-plan-context-id=qwen-bootstrap48',
        '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
        '--ckks-config={package}/config/compiler-config.json','-o','{output_dir}/lowered.mlir']


def fixture(root):
    root.mkdir(parents=True,exist_ok=True);old=root/'original';case=root/'continued';agent=case/'work/agent'
    agent.mkdir(parents=True);(agent/'api').mkdir()
    normal=dict(mean=[.485,.456,.406],std=[.229,.224,.225])
    metadata={key:root/key for key in ('source','checkpoint','inference_preprocessing_source')}
    for path in metadata.values():path.write_text('synthetic original source\n')
    metadata['inference_preprocessing_source'].write_text('transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])\n')
    records=root/'work-results/agent-cifar-offline-records-r1804/cifar10-records';records.mkdir(parents=True)
    profile=root/'qwen48.json';save(profile,dict(native_original_profile=True))
    frontend=root/'expr.py';frontend.write_text('# original trusted frontend fake\n')
    save(old/'source/compiler-profile.json',entry.read(profile));save(old/'source/compile-metadata-template.json',argv_template())
    (old/'source/r30_common_model_e2e.py').write_text(Path(common.__file__).read_text())
    info=old/'work/intake/model_info/model_info.json';cal=old/'work/calibration/calibration.json';save(cal,dict(cached_stat=True))
    save(cal.with_name('phase_result.json'),dict(calibration_completed=True,calibration_samples=1000,calibration_batch=1))
    save(cal.with_name('calibration-source.json'),dict(split='train',count=1000,indices=list(range(1000)),normalization=normal))
    save(info,dict(source=dict(factory=entry.FACTORY,type='torch',original_source=str(metadata['source']),
        checkpoint=str(metadata['checkpoint']),inference_preprocessing_source=str(metadata['inference_preprocessing_source']),
        normalization=normal,preprocessing_provenance='official inference test'),task='classification',
        metric='classification_top1',threshold={'acc_drop_pp':.5},split=dict(kind='legacy_cifar',report=list(range(8000)))))
    proof=dict(passed=True,finite=True,dtype_ok=True,shape_ok=True,criterion='1e-12+1e-12*abs(reference)',max_error_ratio=.1)
    program='@fhe.program(slots=16384, refresh="agent", placement="program-plan")\ndef alexnet_silu_avgpool(x):\n    return x\n'
    (agent/'program.ptdsl').write_text(program)
    selected=dict(iteration=4,origin='tuned',source=program)
    confirmed=dict(selected=selected,evals=[dict(iteration=4,origin='tuned',passed=True,diverged=False,
        n_diverged=0,total=1000,acc_drop_pp=.5,max_u_calibrated=.9,max_u_analytic=1.)])
    state=dict(entry.continuation.OPTIONS,phase='done',failed=None,calls=6,numeric_lock=False,
        confirmation=confirmed,best_program=program)
    save_state(agent/'SESSION.json',state)
    worker=dict(model=entry.MODEL,status='ok',frozen=True,tuner_enabled=True,max_api_calls=6,
        recovered_api_calls=4,actual_api_calls=2,source_model_info=str(info),source_calibration=str(cal),
        report_samples=0,compiler_runs=0,real_ckks_runs=0,quality_acceptance=dict(passed=True),
        agent=dict(llm_calls=6,confirm=confirmed,best_program=program),selected_program=str(agent/'program.ptdsl'),
        strict_math_probes=[dict(sample_id='test:'+str(9000+i),exact=proof,approximate=proof)for i in range(3)])
    save(agent/'worker-result.json',worker)
    for i in (5,6):
        save(agent/'api'/('call'+str(i)+'.request.json'),{});save(agent/'api'/('call'+str(i)+'.response.json'),{})
    processes=[dict(phase=p,observation=clean(),argv=['python','--frontend',str(frontend)])for p in ('intake','calibration','agent')]
    processes[-1]['observation'].update(exit_code=-15,signal=15,signals_sent=['TERM'])
    save(old/'result.json',dict(round='R30',model=entry.MODEL,status='error',tuner_enabled=True,
        variant=common.VARIANT,attempted=dict(intake=1,calibration=1,agent=1),completed=dict(intake=1,calibration=1),
        processes=processes,source=dict(records=str(records),compiler_profile=str(profile)),
        diagnostic_elide_final_ir=False))
    marker=dict(status='error',registry_persisted=True,required_artifacts_persisted=True,complete_case_elapsed_seconds=7200.)
    save(old/'completion.json',marker)
    intake=dict(model=entry.MODEL,status='ok',graph_math=dict(passed=True,samples=[dict(passed=True,finite=True)for i in range(3)]))
    for key in ('import_completed','real_probes_passed','graph_math_passed','roundtrip_passed','static_coverage'):intake[key]=True
    save(old/'work/intake/phase_result.json',intake)
    (old/'work/intake/training_probes.npz').write_bytes(b'synthetic-only-not-npz')
    save(old/'work/intake/training_probe_source.json',dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2']))
    source=dict(original_case=str(old),model_info=str(info),calibration=str(cal),initial_calls=4,remaining_calls=2,
        calibration_samples=1000,metadata_template=str(old/'source/compile-metadata-template.json'))
    result=dict(round=entry.continuation.ROUND,mode=entry.continuation.MODE,model=entry.MODEL,status='ok',error=None,
        monitor_error=None,guard_trigger=None,incoming_signal=None,worker_attempts=1,worker_completed=1,process=clean(),
        cold_start=False,source=source,original_api_calls=4,actual_new_api_calls=2,worker=worker,
        provider_history=dict(inherited_successful_calls=4,actual_new_calls=2,complete_provider_conversation=True,
            old_ledger_records_recreated=False),compiler_runs=0,real_ckks_runs=0,keygen_runs=0,report_samples=0)
    save(case/'result.json',result);save(case/'completion.json',dict(marker,status='ok',complete_case_elapsed_seconds=180.))
    return case,metadata


class AlexPostfreezeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.case,self.metadata=fixture(self.root)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(paths,'WORK_RESULTS',self.root/'work-results'))
        self.stack.enter_context(patch.dict(r13_common_models.FACTORIES,{entry.FACTORY:self.metadata}))
        self.results=self.root/'results';self.results.mkdir()
        self.stack.enter_context(patch.object(paths,'RESULTS_DIR',self.results))
        self.stack.enter_context(patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=40*entry.GIB)))
        self.stack.enter_context(patch.object(entry,'_available_ram_bytes',return_value=16*entry.GIB))
        self.out=self.results/'compiled-continuation'

    def args(self):
        return Namespace(source_case=self.case,out=self.out,python=Path('/original/intake-python'),
            memory_gib=12,case_gib=32,seconds=entry.SECONDS,minimum_free_gib=10)

    def test_success_source_uses_current_continuation_not_fresh_recovery_count(self):
        gate=entry.frozen_source(self.case)
        self.assertEqual(gate['total_api_calls'],6);self.assertEqual(gate['continuation_api_calls'],2)
        self.assertEqual(gate['tuner_budget_s'],600);self.assertEqual(gate['confirm_rule'],'c1')
        self.assertFalse(gate['cold_start']);self.assertTrue(gate['formal_frozen'])
        self.assertEqual(Path(gate['program']).parent,self.case/'work/agent')
        with self.assertRaises(ValueError):original_report.source_gate(self.case,entry.MODEL)

    def test_live_or_rejected_continuation_refused(self):
        target=self.case/'result.json';original=entry.read(target)
        for mutate in (lambda r:r.update(status='running'),lambda r:r.update(status='quality_rejected'),
                       lambda r:r['process'].update(remaining_pids=[999]),lambda r:r.update(report_samples=1),
                       lambda r:r.update(actual_new_api_calls=3)):
            record=deepcopy(original);mutate(record);save(target,record)
            with self.assertRaises(ValueError):entry.frozen_source(self.case)

    def test_policy_original_intake_math_and_source_ast_retained(self):
        worker_path=self.case/'work/agent/worker-result.json';record=entry.read(self.case/'result.json');worker=entry.read(worker_path)
        for mutate in (lambda w:w.update(max_api_calls=7),lambda w:w['agent']['confirm']['evals'][0].update(max_u_calibrated=.90001),
                       lambda w:w['strict_math_probes'][0]['exact'].update(passed=False)):
            changed=deepcopy(worker);mutate(changed);save(worker_path,changed);save(self.case/'result.json',dict(record,worker=changed))
            with self.assertRaises(ValueError):entry.frozen_source(self.case)
        save(worker_path,worker);save(self.case/'result.json',record)
        (self.case/'work/agent/program.ptdsl').write_text('def changed(x):\n    return x + 1\n')
        with self.assertRaises(ValueError):entry.frozen_source(self.case)

    def test_profile_and_normalization_semantic_source_gates(self):
        gate=entry.frozen_source(self.case)
        save(Path(gate['profile']),dict(new_parameters=True))
        with self.assertRaises(ValueError):entry.frozen_source(self.case)
        save(Path(gate['profile']),dict(native_original_profile=True))
        info=entry.read(gate['model_info']);info['source']['normalization']['mean'][0]=0.
        save(Path(gate['model_info']),info)
        with self.assertRaises(ValueError):entry.frozen_source(self.case)

    def test_phase_commands_only_original_postfreeze_and_explicit_paths(self):
        args=self.args();gate=entry.frozen_source(self.case)
        for phase,bound in entry.BOUNDS.items():
            argv,seconds=entry.phase_command(args,gate,phase,self.out/'work'/phase,entry.SECONDS)
            self.assertEqual(seconds,bound+120);self.assertEqual(argv[argv.index('--seconds')+1],str(bound+60))
            self.assertIn(gate['program'],argv);self.assertIn(gate['model_info'],argv)
            self.assertNotIn('--ledger',argv);self.assertNotIn('--allowance',argv)
            self.assertNotIn('agent',argv)

    def fake_run(self,fail_phase=None,timeout_phase=None):
        def launch(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1])
            emit=self.out/'post-freeze/emit';value=dict(status='ok',phase=phase)
            if phase=='placement':
                value['boots_pred']=11;save(self.out/'post-freeze/refresh-plan.json',dict(boots=11))
            if phase=='emit':
                emit.mkdir();(emit/'constants.npz').write_bytes(b'synthetic generated constants\n'*2000)
                (emit/'candidate_trace.py').write_text('# synthetic trusted source\n');value['trace_source']=str(emit/'candidate_trace.py')
            if phase=='trace':
                (emit/'candidate_trace.mlir').write_text('module { /* synthetic complete */ }\n')
                value.update(process=clean(),mlir=dict(path=str(emit/'candidate_trace.mlir'),bytes=(emit/'candidate_trace.mlir').stat().st_size))
            if phase=='compile':
                self.assertFalse((emit/'constants.npz').exists());self.assertTrue((emit/'constants.npz.gz').is_file())
                value.update(process=clean(),compiled_bootstraps=11,predicted_latency_us=12345)
                save(work/'native-argv.json',['/native/compiler',str(emit/'candidate_trace.mlir')])
            if phase==fail_phase:value.update(status='error',error='synthetic native phase failed')
            if phase==timeout_phase:value.update(status='error',error='synthetic wall timeout',worker_wall_timeout_observed=True)
            save(work/'phase_result.json',value)
            observation=clean()
            if phase==fail_phase:observation['exit_code']=1
            if phase==timeout_phase:observation.update(exit_code=1,timed_out=True)
            return observation
        return launch

    def test_fake_complete_archive_before_compile_unique_registry_and_clocks(self):
        with patch.object(entry,'run_process',side_effect=self.fake_run())as launch:
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'ok');self.assertEqual(launch.call_count,4)
        self.assertEqual(list(result['completed']),list(entry.PHASES));self.assertEqual(result['compiled_bootstraps'],11)
        self.assertFalse(result['cold_start']);self.assertEqual(result['source']['original_complete_seconds'],7200.)
        rows=[json.loads(line)for line in (self.results/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertTrue(entry.read(self.out/'completion.json')['exclusive_attribution_ready'])
        with self.assertRaises(ValueError):entry._register(self.out,result)
        compiled=entry.compiled_source(self.out,self.case)
        self.assertEqual(compiled['source_case'],str(self.case));self.assertTrue(compiled['clean_native_compile'])

    def test_fake_native_first_error_stops_and_preserves_partial(self):
        with patch.object(entry,'run_process',side_effect=self.fake_run('trace'))as launch:
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'error');self.assertEqual(launch.call_count,3)
        self.assertEqual(result['completed'],dict(placement=1,emit=1));self.assertNotIn('compile',result['attempted'])
        self.assertEqual(result['phases']['trace']['status'],'error')
        self.assertTrue(entry.read(self.out/'completion.json')['registry_persisted'])

    def test_worker_delegates_unchanged_shared_implementation(self):
        args=Namespace(worker_phase='placement')
        with patch.object(common,'worker',return_value=0)as called:
            self.assertEqual(entry.worker(args),0)
        self.assertIs(called.call_args.args[0],args);self.assertEqual(args.model,entry.MODEL)
        self.assertTrue(args.tuner)

    def test_nan_diagnostics_preserved_without_changing_finite_selected_policy(self):
        worker_path=self.case/'work/agent/worker-result.json';record=entry.read(self.case/'result.json')
        worker=entry.read(worker_path);worker['agent']['history']=[dict(rejected_diagnostic=float('nan'))]
        worker['agent']['confirm']['evals'].append(dict(iteration=1,origin='llm',passed=False,max_u=float('nan')))
        state_path=self.case/'work/agent/SESSION.json'
        from pt_agent.planners.transformer_session import load_state
        state=load_state(state_path);state['confirmation']=worker['agent']['confirm'];save_state(state_path,state)
        save(worker_path,worker);save(self.case/'result.json',dict(record,worker=worker))
        gate=entry.frozen_source(self.case)
        self.assertEqual(gate['total_api_calls'],6)
        with patch.object(entry,'run_process',side_effect=self.fake_run()):result=entry.execute(self.args())
        self.assertEqual(result['status'],'ok');self.assertTrue(entry.compiled_source(self.out)['clean_native_compile'])

    def test_phase_timeout_is_distinct_from_external_signal_and_resource_failure(self):
        with patch.object(entry,'run_process',side_effect=self.fake_run(timeout_phase='placement')):
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'timeout');self.assertIsNone(result['incoming_signal'])
        self.assertIsNone(result['guard_trigger']);self.assertFalse(result['wall_timeout_observed'])
        self.assertEqual(result['phase_timeouts'],{'placement':['outer-R10','worker-wall']})
        self.assertTrue(result['processes'][0]['observation']['timed_out'])

    def test_explicit_disk_reserve_and_original_default_private_guards(self):
        for bad in (True,False,3,11,4.0):
            with self.assertRaises(ValueError):entry.minimum_free(Namespace(minimum_free_gib=bad))
        originals=(common.worker,common._post_freeze,common.archive_emit_constants_after_trace)
        namespace,proof=entry.trusted_namespace(4)
        self.assertEqual(proof['storage_guard_replacements'],{'_post_freeze':1,'archive_emit_constants_after_trace':1})
        self.assertIs(namespace['worker'].__globals__,namespace)
        self.assertEqual(originals,(common.worker,common._post_freeze,common.archive_emit_constants_after_trace))
        original,_=entry.trusted_namespace(10);self.assertIs(original['worker'],common.worker)
        self.assertIs(original['archive_emit_constants_after_trace'],common.archive_emit_constants_after_trace)
        emit=self.out/'post-freeze/emit';emit.mkdir(parents=True)
        source=emit/'constants.npz';source.write_bytes(b'synthetic compressed fixture\n'*2000)
        (emit/'candidate_trace.py').write_text('# trusted synthetic\n');mlir=emit/'candidate_trace.mlir';mlir.write_text('module {}\n')
        staged=dict(model=entry.MODEL,attempted=dict(emit=1,trace=1),completed=dict(emit=1,trace=1),
            phases=dict(emit=dict(status='ok',trace_source=str(emit/'candidate_trace.py')),
                trace=dict(status='ok',process=clean(),mlir=dict(path=str(mlir),bytes=mlir.stat().st_size))),
            processes=[dict(phase='trace',observation=clean())])
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=6*entry.GIB)):
            with self.assertRaises(RuntimeError):original['archive_emit_constants_after_trace'](staged,self.out)
            result=namespace['archive_emit_constants_after_trace'](staged,self.out)
        self.assertEqual(result['status'],'ok');self.assertFalse(source.exists())

    def test_disk_four_or_ten_guard_and_invalid_cli(self):
        args=self.args()
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=6*entry.GIB)),\
             patch.object(entry,'run_process')as launch:
            refused=entry.execute(args)
        self.assertEqual(refused['status'],'resource_stopped');launch.assert_not_called()
        self.out=self.results/'explicit-four';args=self.args();args.minimum_free_gib=4
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=6*entry.GIB)),\
             patch.object(entry,'run_process',side_effect=self.fake_run())as launch:
            accepted=entry.execute(args)
        self.assertEqual(accepted['status'],'ok');self.assertEqual(launch.call_count,4)
        self.assertEqual(accepted['limits']['minimum_disk_free_gib'],4)
        for value in ('3','11','True'):
            with patch.object(entry,'execute')as execute,self.assertRaises(SystemExit):
                entry.main(['--source-case',str(self.case),'--out',str(self.results/'bad'),'--minimum-free-gib',value])
            execute.assert_not_called()

    def test_report_private_namespace_and_explicit_compile_path(self):
        original_gate=original_report.source_gate;gate=entry.frozen_source(self.case)
        with patch.object(entry,'compiled_source',return_value=gate),patch.object(original_report,'run_process',return_value=clean())as launched:
            namespace=report.report_namespace(self.out)
            self.assertIs(original_report.source_gate,original_gate)
            self.assertIs(namespace['execute'].__globals__,namespace)
            self.assertEqual(namespace['__spec__'].name,report.__name__)
            self.assertEqual(namespace['source_gate'](self.case,entry.MODEL)['source_case'],str(self.case))
            output=self.results/'report';output.mkdir()
            namespace['run_process'](['python','--out',str(output)],timeout=1)
            self.assertEqual(launched.call_args.args[0][-2:],['--compile-case',str(self.out)])
            self.assertTrue((output/'source/r30_alex_continuation_report.py').is_file())
            with self.assertRaises(ValueError):namespace['source_gate'](self.case,'vgg16_silu_avgpool')

    def test_report_claim_uses_frozen_identity_across_compile_cases(self):
        gate=entry.frozen_source(self.case)
        with patch.object(entry,'compiled_source',return_value=gate):
            namespace=report.report_namespace(self.out)
            selected=namespace['source_gate'](self.case,entry.MODEL)
            marker=namespace['claim_access'](self.results/'one-report',selected)
            self.assertEqual(marker.name,self.case.name+'.json')
            with self.assertRaises(FileExistsError):namespace['claim_access'](self.results/'another-report',selected)

    def test_report_installs_metadata_context_before_trusted_worker(self):
        from contextlib import contextmanager
        from pt_agent.experiments import r29_local_compile_matrix
        gate=entry.frozen_source(self.case);order=[]
        @contextmanager
        def metadata(template):
            self.assertEqual(template,argv_template())
            order.append('context-enter')
            try:yield
            finally:order.append('context-exit')
        def trusted(args):
            self.assertEqual(order,['context-enter']);order.append('trusted-worker')
            self.assertTrue(all(os.environ[key]=='1'for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')))
            save(args.out/'worker-result.json',dict(status='ok',report_completed=True))
            return 0
        args=Namespace(source_case=self.case,model=entry.MODEL,out=self.results/'report-worker');args.out.mkdir()
        with patch.object(entry,'compiled_source',return_value=gate),patch.object(original_report,'worker',trusted),\
             patch.object(r29_local_compile_matrix,'native_metadata_commands',metadata),patch.dict(os.environ):
            self.assertEqual(report.report_namespace(self.out)['worker'](args),0)
        self.assertEqual(order,['context-enter','trusted-worker','context-exit'])
        self.assertTrue(entry.read(args.out/'worker-result.json')['continuation_report_adapter'])

    def test_report_actual_argv_list_context_skips_legacy_identifier_builder(self):
        from pt_agent import toolchain
        gate=entry.frozen_source(self.case)
        args=Namespace(source_case=self.case,model=entry.MODEL,out=self.results/'report-list');args.out.mkdir()
        def trusted(args):
            argv=toolchain.compile_argv('pars','qwen48',Path('/metadata-only'))
            self.assertEqual(argv[:2],['/hecate-opt','/out/candidate_trace.mlir'])
            save(args.out/'worker-result.json',dict(status='ok'));return 0
        with patch.object(entry,'compiled_source',return_value=gate),patch.object(original_report,'worker',trusted),\
             patch.object(toolchain,'compile_argv',side_effect=AssertionError('legacy identifier computation forbidden'))as forbidden,patch.dict(os.environ):
            self.assertEqual(report.report_namespace(self.out)['worker'](args),0)
            forbidden.assert_not_called()

    def test_finite_cli_and_fresh_output_reserve(self):
        with patch.object(entry,'execute',return_value={'status':'ok','error':None})as called:
            self.assertEqual(entry.main(['--source-case',str(self.case),'--out',str(self.out)]),0)
            self.assertEqual(called.call_args.args[0].memory_gib,12)
        self.out.mkdir()
        with self.assertRaises(ValueError):entry.execute(self.args())
        result=dict(status='ok');entry._terminal(self.out,result,32*entry.GIB)
        with self.assertRaises(RuntimeError):entry._terminal(self.out,result,entry.artifact_bytes(self.out)+entry.RESERVE-1)


if __name__=='__main__':unittest.main()
