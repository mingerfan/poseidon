"""Synthetic metadata/CLI tests; no models, arrays, compiler or report rows."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.experiments import r30_vgg_postfreeze as subject
from pt_agent.experiments import r30_alex_postfreeze as shared
from pt_agent.experiments import r30_frozen_cnn_report as report
from pt_agent.experiments import r13_common_models
from pt_agent.tests import test_vgg_continuation_terminal_audit as saved_fixture


class VGGPostfreezeChecks(unittest.TestCase):
    def setUp(self):
        fixture=saved_fixture.ReadbackChecks();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.f=fixture;self.root=fixture.root;self.case=fixture.case
        self.cold=Path(fixture.source['original_case']);self.audit=self.root/'audit';self.compiled=self.root/'compiled'
        self.sources=self.root/'official';self.sources.mkdir()
        self.official={key:self.sources/(key+'.py')for key in ('source','checkpoint','inference_preprocessing_source')}
        for path in self.official.values():path.write_text('retained synthetic source\n')
        self.norm=dict(mean=[.485,.456,.406],std=[.229,.224,.225])
        self.official['inference_preprocessing_source'].write_text(
            'transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])\n')
        self.factory_patch=patch.dict(r13_common_models.FACTORIES,{subject.FACTORY:self.official})
        self.factory_patch.start();self.addCleanup(self.factory_patch.stop)
        for name in ('WORK_RESULTS','RESULTS_DIR'):
            setter=patch.object(subject.paths,name,self.root);setter.start();self.addCleanup(setter.stop)
        self.info=self.cold/'work/intake/model_info/model_info.json'
        self.cal=self.cold/'work/calibration/calibration.json'
        self.profile=self.root/'compiler-profile.json';self.write(self.profile,{'synthetic':True})
        self.template=self.cold/'source/compile-metadata-template.json';self.write(self.template,{'synthetic':True})
        self.front=self.root/'expr.py';self.front.write_text('synthetic trusted frontend\n')
        self.write(self.cold/'source/compiler-profile.json',{'synthetic':True})
        self.write(self.info,dict(task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5},
            split=dict(kind='legacy_cifar',heldout=list(range(9000,10000)),confirm=list(range(8000,9000)),report=list(range(8000))),
            source=dict(factory=subject.FACTORY,type='torch',original_source=str(self.official['source']),
                checkpoint=str(self.official['checkpoint']),inference_preprocessing_source=str(self.official['inference_preprocessing_source']),
                normalization=self.norm,preprocessing_provenance='official inference test')))
        self.write(self.cal,{})
        self.write(self.cold/'work/calibration/phase_result.json',dict(calibration_completed=True,calibration_samples=1000,calibration_batch=1))
        self.write(self.cold/'work/calibration/calibration-source.json',dict(split='train',count=1000,indices=list(range(1000)),normalization=self.norm))
        proof=dict(passed=True,samples=[dict(passed=True,finite=True)for _ in range(3)])
        self.write(self.cold/'work/intake/phase_result.json',dict(model=subject.MODEL,status='ok',import_completed=True,
            real_probes_passed=True,graph_math_passed=True,roundtrip_passed=True,static_coverage=True,graph_math=proof))
        self.write(self.cold/'work/intake/training_probe_source.json',dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2']))
        (self.cold/'work/intake/training_probes.npz').write_text('unopened synthetic placeholder')
        self.write(self.cold/'result.json',dict(round='R30',model=subject.MODEL,status='error',
            source=dict(compiler_profile=str(self.profile),records=str(self.root/'agent-cifar-offline-records-r1804/cifar10-records')),
            processes=[dict(argv=['--frontend',str(self.front)])]))
        self.marker=dict(status='error',required_artifacts_persisted=True,registry_persisted=True,complete_case_elapsed_seconds=3575.)
        self.write(self.cold/'completion.json',self.marker)
        fixture.source.update(model_info=str(self.info),calibration=str(self.cal))
        fixture.worker.update(source_model_info=str(self.info),source_calibration=str(self.cal))
        fixture.record.update(source=fixture.source,worker=fixture.worker)
        self.write(self.case/'result.json',fixture.record);self.write(fixture.agent/'worker-result.json',fixture.worker)
        verified=dict(source_case=str(self.case),source_result=str(self.case/'result.json'),original_case=str(self.cold),
            worker_result=str(fixture.agent/'worker-result.json'),selected_program=str(fixture.program),
            model_info=str(self.info),calibration=str(self.cal),strict_math_checks=6,confirmation_runs_verified=1,
            original_api_calls=4,continued_api_calls=2,total_session_calls=6,
            selected_AST_verified=True,JSON_confirmation_semantics_verified=True)
        self.audit_record=dict(round=subject.terminal_audit.ROUND,mode=subject.terminal_audit.MODE,model=subject.MODEL,
            status='ok',error=None,source_case=str(self.case),formal_frozen=True,quality_acceptance_passed=True,
            strict_math_passed=True,source_case_modified=False,proof=verified,
            **dict.fromkeys(('model_runs','api_calls','new_confirmations','report_samples','compiler_runs','real_ckks_runs'),0))
        self.write(self.audit/'result.json',self.audit_record)
        self.write(self.audit/'completion.json',dict(self.marker,status='ok',complete_case_elapsed_seconds=.3))

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def gate(self):return subject.frozen_source(self.case,self.audit)

    def prepare_compiled(self):
        gate=self.gate();mlir=self.compiled/'post-freeze/emit/candidate_trace.mlir'
        mlir.parent.mkdir(parents=True);mlir.write_text('synthetic compiler input')
        self.write(self.compiled/'work/compile/native-argv.json',['compiler',str(mlir)])
        child=dict(status='ok',process=dict(self.f.record['process']))
        phases={phase:dict(child)for phase in shared.PHASES}
        phases['placement']['boots_pred']=211
        phases['compile'].update(compiled_bootstraps=211,runtime_plan=str(self.compiled/'post-freeze/emit/lowered.test.runtime-plan.json'))
        self.write(Path(phases['compile']['runtime_plan']),{'synthetic':True})
        record=dict(round=subject.ROUND,mode=subject.MODE,model=subject.MODEL,status='ok',error=None,
            guard_trigger=None,monitor_error=None,source=gate,compiled_bootstraps=211,
            attempted=dict.fromkeys(shared.PHASES,1),completed=dict.fromkeys(shared.PHASES,1),phases=phases,
            processes=[dict(phase=phase,observation=dict(self.f.record['process']))for phase in shared.BOUNDS],
            **dict.fromkeys(shared.ZERO,0))
        self.write(self.compiled/'result.json',record)
        self.write(self.compiled/'completion.json',dict(self.marker,status='ok',complete_case_elapsed_seconds=70.))
        return gate,record,mlir

    def test_frozen_gate_preserves_218_as_history_and_distinct_clocks(self):
        gate=self.gate()
        self.assertEqual(gate['model'],subject.MODEL);self.assertEqual(gate['historical_boots_pred'],218)
        self.assertEqual(gate['original_complete_seconds'],3575.)
        self.assertEqual(gate['continuation_complete_seconds'],4727.)
        self.assertEqual(gate['audit_complete_seconds'],.3)
        self.assertFalse(gate['confirmation_repeated']);self.assertEqual(gate['total_api_calls'],6)
        self.assertEqual(gate['program'],str(self.f.program));self.assertEqual(gate['probes'],str(self.cold/'work/intake/training_probes.npz'))

    def test_audit_must_be_accepted_closed_and_bound_to_selected_source(self):
        for field,value in [('formal_frozen',False),('status','error')]:
            with self.subTest(field=field):
                self.write(self.audit/'result.json',dict(self.audit_record,**{field:value}))
                with self.assertRaises(ValueError):self.gate()
        changed=deepcopy(self.audit_record);changed['proof']['selected_program']=str(self.root/'other.ptdsl')
        self.write(self.audit/'result.json',changed)
        with self.assertRaises(ValueError):self.gate()

    def test_existing_math_and_selected_AST_failures_still_reject(self):
        self.f.program.write_text('def model(x):\n    return y\n')
        with self.assertRaises(ValueError):self.gate()
        self.f.program.write_text(self.f.text)
        changed=deepcopy(self.f.worker);changed['strict_math_probes'][0]['approximate']['passed']=False
        self.write(self.f.agent/'worker-result.json',changed)
        with self.assertRaises(ValueError):self.gate()

    def test_changed_model_policy_or_training_partition_rejected(self):
        info=json.loads(self.info.read_text());bad=deepcopy(info);bad['threshold']={'acc_drop_pp':1.}
        self.write(self.info,bad)
        with self.assertRaises(ValueError):self.gate()
        self.write(self.info,info)
        self.write(self.cold/'work/calibration/calibration-source.json',dict(split='test',count=1000,indices=list(range(1000)),normalization=self.norm))
        with self.assertRaises(ValueError):self.gate()

    def test_private_delegation_keeps_old_modules_and_finite_phases(self):
        old=(shared.MODEL,shared.MODULE,shared.frozen_source,shared.worker,report.source_gate)
        namespace=subject.postfreeze_namespace(self.audit)
        self.assertEqual(namespace['MODEL'],subject.MODEL);self.assertEqual(namespace['MODULE'],subject.MODULE)
        self.assertEqual(namespace['PHASES'],('placement','emit','trace','archive_emit_constants','compile'))
        self.assertIs(namespace['worker'].__code__,shared.worker.__code__)
        self.assertEqual((shared.MODEL,shared.MODULE,shared.frozen_source,shared.worker,report.source_gate),old)

    def test_phase_argv_has_vgg_paths_limits_and_explicit_elision_only(self):
        namespace=subject.postfreeze_namespace(self.audit);gate=self.gate()
        args=type('Args',(),dict(python=Path('/existing/intake'),out=self.compiled,memory_gib=12,
            case_gib=32,minimum_free_gib=10,diagnostic_elide_final_ir=False))()
        for phase,limit in [('placement',1800),('trace',1800),('compile',7200)]:
            argv,outer=namespace['phase_command'](args,gate,phase,self.root/phase,13560)
            self.assertEqual(argv[argv.index('--worker-phase')+1],phase)
            self.assertEqual(argv[argv.index('--worker-seconds')+1],str(limit))
            self.assertEqual(outer,limit+120);self.assertIn(subject.MODULE,argv)
            self.assertEqual(argv[argv.index('--program')+1],str(self.f.program))
            self.assertEqual(argv[argv.index('--quality-audit-case')+1],str(self.audit))
            self.assertNotIn('--diagnostic-elide-final-ir',argv)
        args.diagnostic_elide_final_ir=True
        argv,_=namespace['phase_command'](args,gate,'compile',self.root/'compile',13560)
        self.assertIn('--diagnostic-elide-final-ir',argv)

    def test_current_compile_count_is_independent_of_historical_218(self):
        self.prepare_compiled();gate=subject.compiled_source(self.compiled,self.case,self.audit)
        self.assertTrue(gate['clean_native_compile']);self.assertEqual(gate['historical_boots_pred'],218)
        self.assertEqual(gate['post_freeze_complete_seconds'],70.)
        self.assertEqual(gate['compiled_bootstraps'],211)
        self.assertEqual(gate['runtime_plan'],str(self.compiled/'post-freeze/emit/lowered.test.runtime-plan.json'))

    def test_report_rejects_missing_or_foreign_runtime_plan(self):
        _,record,_=self.prepare_compiled()
        (self.compiled/'post-freeze/emit/lowered.test.runtime-plan.json').unlink()
        with self.assertRaises(ValueError):subject.compiled_source(self.compiled,self.case,self.audit)
        record['phases']['compile']['runtime_plan']=str(self.root/'foreign.runtime-plan.json')
        self.write(Path(record['phases']['compile']['runtime_plan']),{'synthetic':True})
        self.write(self.compiled/'result.json',record)
        with self.assertRaises(ValueError):subject.compiled_source(self.compiled,self.case,self.audit)

    def test_incomplete_compile_or_different_current_plan_count_rejected(self):
        _,record,_=self.prepare_compiled()
        for change in ('incomplete','count'):
            altered=deepcopy(record)
            if change=='incomplete':altered['completed']['compile']=0
            else:altered['phases']['placement']['boots_pred']=218
            self.write(self.compiled/'result.json',altered)
            with self.subTest(change=change),self.assertRaises(ValueError):subject.compiled_source(self.compiled,self.case,self.audit)

    def test_report_requires_compiled_source_and_claims_this_frozen_identity_once(self):
        with self.assertRaises(FileNotFoundError):subject.report_namespace(self.compiled,self.audit)['source_gate'](self.case,subject.MODEL)
        self.prepare_compiled();namespace=subject.report_namespace(self.compiled,self.audit)
        gate=namespace['source_gate'](self.case,subject.MODEL)
        self.assertFalse(gate['report_outcome_used_for_selection'])
        marker=namespace['claim_access'](self.root/'report',gate)
        self.assertEqual(marker.name,self.case.name+'.json')
        with self.assertRaises(FileExistsError):namespace['claim_access'](self.root/'second-report',gate)
        with self.assertRaises(ValueError):namespace['source_gate'](self.case,'alexnet_silu_avgpool')

    def test_report_launch_routes_both_explicit_paths_to_same_module(self):
        self.prepare_compiled();target=self.root/'report';target.mkdir()
        with patch.object(report,'run_process',return_value={'synthetic':True})as run:
            namespace=subject.report_namespace(self.compiled,self.audit)
            argv=['python','-m',subject.MODULE,'--worker','--model',subject.MODEL,'--out',str(target)]
            namespace['run_process'](argv,timeout=7320)
        sent=run.call_args.args[0]
        self.assertEqual(sent[sent.index('--phase')+1],'report')
        self.assertEqual(sent[sent.index('--compile-case')+1],str(self.compiled))
        self.assertEqual(sent[sent.index('--quality-audit-case')+1],str(self.audit))
        self.assertTrue((target/'source/r30_vgg_postfreeze.py').is_file())

    def test_full_fake_compile_chain_persists_current_plan_without_reconfirmation(self):
        namespace=subject.postfreeze_namespace(self.audit)
        args=type('Args',(),dict(source_case=self.case,out=self.compiled,python=Path('/existing/intake'),
            seconds=13560,memory_gib=12,case_gib=32,minimum_free_gib=10,diagnostic_elide_final_ir=True))()
        phases=[]
        def run(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];phases.append(phase)
            work=Path(argv[argv.index('--work-dir')+1]);row=dict(status='ok',process=dict(self.f.record['process']))
            if phase=='placement':row['boots_pred']=211
            if phase=='compile':row.update(compiled_bootstraps=211,predicted_latency_us=None,
                runtime_plan=str(self.compiled/'post-freeze/emit/lowered.test.runtime-plan.json'))
            self.write(work/'phase_result.json',row)
            return dict(self.f.record['process'])
        namespace.update(run_process=run,physical_memory_observation=lambda *a,**k:{'stop':False},
            trusted_namespace=lambda reserve:({'archive_emit_constants_after_trace':lambda *a:{'status':'ok'}},{'synthetic':True}))
        old=(self.case/'result.json').read_text();result=namespace['execute'](args)
        self.assertEqual(result['status'],'ok');self.assertEqual(phases,['placement','emit','trace','compile'])
        self.assertEqual(result['compiled_bootstraps'],211)
        self.assertEqual(result['phases']['compile']['runtime_plan'],str(self.compiled/'post-freeze/emit/lowered.test.runtime-plan.json'))
        for field in ('api_calls','agent_runs','tuner_runs','calibration_runs','confirm_samples','report_samples'):
            self.assertEqual(result[field],0)
        self.assertEqual((self.case/'result.json').read_text(),old)
        rows=[json.loads(x)for x in self.f.registry.read_text().splitlines()]
        self.assertEqual(set(rows[-1]),subject.terminal_audit.FIELDS)
        marker=json.loads((self.compiled/'completion.json').read_text())
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['required_artifacts_persisted'])

    def test_report_math_and_fixed_sample_loop_keep_original_implementation(self):
        namespace=subject.report_namespace(self.compiled,self.audit)
        for name in ('training_math','report_rows','read_report_sample','claim_access'):
            self.assertIs(namespace[name].__code__,getattr(report,name).__code__)
        self.assertEqual(namespace['REPORT_COUNT'],8000);self.assertEqual(namespace['CRITERION'],report.CRITERION)

    def test_report_worker_forwards_saved_argv_list_into_metadata_context(self):
        self.prepare_compiled();template=['compiler','--existing-operator-identifier=retained']
        self.write(self.template,template)
        args=type('Args',(),dict(source_case=self.case,model=subject.MODEL,out=self.root/'report'))()
        with patch.object(report,'worker',return_value=0)as worker,patch(
                'pt_agent.experiments.r29_local_compile_matrix.native_metadata_commands',return_value=nullcontext())as metadata:
            namespace=subject.report_namespace(self.compiled,self.audit)
            self.assertEqual(namespace['worker'](args),0)
        metadata.assert_called_once_with(template);worker.assert_called_once_with(args)

    def test_cli_dispatch_keeps_compile_and_report_separate(self):
        with patch.object(subject,'postfreeze_namespace',return_value={'main':lambda argv:argv})as build:
            result=subject.main(['--quality-audit-case',str(self.audit),'--source-case',str(self.case),'--out',str(self.compiled)])
        self.assertIn('--source-case',result);build.assert_called_once_with(self.audit)
        with patch.object(subject,'report_namespace',return_value={'main':lambda argv:argv})as reports:
            result=subject.main(['--phase','report','--quality-audit-case',str(self.audit),
                '--compile-case',str(self.compiled),'--model',subject.MODEL])
        reports.assert_called_once_with(self.compiled,self.audit);self.assertEqual(result,['--model',subject.MODEL])


if __name__=='__main__':unittest.main()
