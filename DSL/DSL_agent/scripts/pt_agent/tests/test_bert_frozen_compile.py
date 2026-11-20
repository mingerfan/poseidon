"""Frozen-source and finite phase fixtures; no real BERT or compiler execution."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_frozen_compile as compile


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def template():
    return ['{hecate_opt}','{input}','--pars','--ckks-config={package}/config/compiler-config.json',
        '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,'--runtime-plan-context-id=qwen-bootstrap48',
        '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5','--waterline=43','--output-val=10',
        '-o','{output_dir}/lowered.mlir']


def fixture(root,report_acceptance=True):
    case=root/'session-case';session=case/'session';session.mkdir(parents=True)
    program=session/'program.ptdsl';program.write_text('fn synthetically_frozen_only() {}\n')
    split=dict(kind='sst2_native',version='sst2-native-v1',seed=20261024,
        calibration=list(range(1000)),heldout=list(range(1000,2000)),confirm=list(range(2000,3000)),report=list(range(3000,3872)))
    model=root/'original-hf';inputs=root/'public-sst2'
    save(case/'model/model_info.json',dict(source=dict(type='hf',local_path=str(model),inputs_path=str(inputs),
        config=dict(model_type='bert',num_labels=2)),task='classification',metric='classification_top1',
        threshold={'acc_drop_pp':.5},split=split,sequence={'length':40},
        inputs=[dict(name='inputs_embeds',kind='tokens',shape=[1,40,128]),dict(name='attention_mask',kind='mask',shape=[1,40])]))
    save(case/'model/samples/source.json',dict(source=str(inputs),partition=split,native_test_read=False))
    selected=dict(source=program.read_text(),iteration=2,origin='tuned')
    confirmed=dict(iteration=2,origin='tuned',passed=True,diverged=False,n_diverged=0,
        acc_drop_pp=.2,max_u_calibrated=.8,max_u_analytic=1.,
        tail=dict(passed=True,total=3000,diverged=False,n_diverged=0,metrics=dict(acc_drop_pp=.3)))
    confirm=dict(selected=selected,evals=[confirmed],confirm_rule='c2',applicable=True)
    save(session/'AGENT.json',dict(confirm=confirm))
    save(session/'SESSION.json',dict(phase='done',prompt_version='r8-v11',tune_rule='v8-6.25',confirm_rule='c2',
        split_version='sst2-native-v1',numeric_lock=True,focus_retune=True,tune_policy='all',tune_search='sweep',engine_profile='tensor'))
    save(session/'FROZEN.json',dict(program=str(program),prompt_version='r8-v11',confirm_rule='c2',
        tune_rule='v8-6.25',split_version='sst2-native-v1',iteration=2,origin='tuned',confirm=confirm))
    report=case/'phases/010-report/worker_result.json'
    record=dict(model=compile.MODEL,phase='report',status='ok',frozen=True,report_completed=True,report_samples=872,
        task_policy={'acc_drop_pp':.5},task_policy_passed=report_acceptance,
        frozen_math=[dict(sample=i,checks=[dict(passed=True,criterion=compile.CRITERION) for _ in range(3)]) for i in range(3)])
    record.update(dict.fromkeys(('native_test_reads','llm_calls','compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'),0))
    save(report,record);save(session/'REPORT_ACCESS.json',dict(complete=True,report_samples=872,program=str(program),result=str(report)))
    checkpoint_audit=root/'checkpoint-audit/result.json'
    worker=dict(status='ok',state_coverage=dict(passed=True,classifier=dict(both_original_checkpoint_keys=True)),
        source=dict(model_info=str(case/'model/model_info.json'),model=str(model)),model_forwards=0,raw_sample_reads=0,api_calls=0,integrity_checks=0)
    save(checkpoint_audit,dict(round='BERT-CHECKPOINT-AUDIT',model=compile.MODEL,status='ok',checkpoint_semantic_coverage_passed=True,worker=worker))
    return case,checkpoint_audit,report


class FrozenBertCompileChecks(unittest.TestCase):
    def test_completed_frozen_same_source_math_is_admitted_and_report_rejection_retained(self):
        for accepted in (True,False):
            with TemporaryDirectory() as temp:
                case,audit,report=fixture(Path(temp),accepted);result=compile.source_gate(case,audit,report)
                self.assertIs(result['report_task_acceptance'],accepted)
                self.assertFalse(result['report_outcome_used_for_selection']);self.assertFalse(result['api_feedback_permitted'])
                self.assertEqual(result['inputs'][1]['shape'],[1,40])

    def test_source_or_unfrozen_or_confirmation_or_math_or_checkpoint_mismatch_is_rejected(self):
        for which in ('source','unfrozen','confirm','domain','tail','math','checkpoint','partition'):
            with TemporaryDirectory() as temp:
                case,audit,report=fixture(Path(temp))
                if which=='source':(case/'session/program.ptdsl').write_text('changed')
                elif which=='unfrozen':
                    path=case/'session/SESSION.json';record=json.loads(path.read_text());record['phase']='evaluate';save(path,record)
                elif which in ('confirm','domain','tail'):
                    path=case/'session/AGENT.json';record=json.loads(path.read_text())
                    if which=='confirm':record['confirm']['evals'][0]['passed']=False
                    elif which=='domain':record['confirm']['evals'][0]['max_u_calibrated']=.91
                    else:record['confirm']['evals'][0]['tail']['passed']=False
                    save(path,record);frozen=case/'session/FROZEN.json';r=json.loads(frozen.read_text());r['confirm']=record['confirm'];save(frozen,r)
                elif which=='math':
                    record=json.loads(report.read_text());record['frozen_math'][2]['checks'][1]['passed']=False;save(report,record)
                elif which=='checkpoint':
                    record=json.loads(audit.read_text());record['worker']['source']['model']='/foreign';save(audit,record)
                else:
                    path=case/'model/model_info.json';record=json.loads(path.read_text());record['split']['heldout'][0]=0;save(path,record)
                with self.subTest(which=which),self.assertRaises(ValueError):compile.source_gate(case,audit,report)

    def test_explicit_nix_command_preserves_inferred_model_signature_paths_and_7200_seconds(self):
        args=Namespace(intake_python=Path('/existing/intake/python'),**{k:Path('/original')/k for k in
            ('source_case','checkpoint_audit','report_result','metadata_template','profile','frontend','model_info','program')})
        command=compile.phase_command(args,'compile',Path('/isolated/work'))
        self.assertEqual(command[command.index('--seconds')+1],'7260')
        self.assertEqual(command[command.index('--python')+1],'/existing/intake/python')
        self.assertEqual(command[command.index('--worker-phase')+1],'compile')
        for key in ('model-info','program','report-result','checkpoint-audit'):self.assertIn('--'+key,command)

    def test_saved_metadata_command_scope_avoids_both_new_digest_paths_and_restores_builder(self):
        from pt_agent import toolchain
        with TemporaryDirectory() as temp:
            config=Path(temp);save(config/'compiler-config.json',dict(rescalingFactor=43))
            old=toolchain.compile_argv
            with (patch.object(toolchain,'BOOT_CONFIG_DIR',config),patch.object(toolchain,'boot_window_rule',return_value='dacapo-39c1fb2'),
                  patch.object(toolchain,'_file_sha256',side_effect=AssertionError('new digest forbidden')),
                  patch.object(toolchain.hashlib,'sha256',side_effect=AssertionError('new digest forbidden'))):
                with compile.native_metadata_commands(template()):
                    observed=toolchain.management_profile('dacapo-depth-dp','qwen48')
                    self.assertEqual(observed['waterline'],43);self.assertEqual(observed['output_val'],10)
            self.assertIs(toolchain.compile_argv,old)

    def execute_fixture(self,root,fail=None,report_acceptance=True):
        case,audit,report=fixture(root,report_acceptance);results=root/'results';results.mkdir()
        profiles=root/'profiles';profile=profiles/'qwen48/compiler-config.json';save(profile,dict(latencyTable={}))
        frontend=root/'dacapo/python/hecate/hecate/expr.py';frontend.parent.mkdir(parents=True);frontend.write_text('# trusted source fixture\n')
        metadata=root/'recorded-template.json';save(metadata,template())
        args=Namespace(out=results/'compiled',source_case=case,checkpoint_audit=audit,report_result=report,
            profile=profile,frontend=frontend,metadata_template=metadata,intake_python=Path('/existing/python'))
        def run(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];self.assertEqual(kwargs['mem_gib'],12)
            self.assertLessEqual(kwargs['timeout'],compile.PHASE_SECONDS[phase]+120)
            worker=dict(round=compile.ROUND,model=compile.MODEL,mode=compile.MODE,phase=phase,
                source=compile.source_gate(case,audit,report),status='error' if phase==fail else 'ok',
                **dict.fromkeys(compile.ZERO_FIELDS,0))
            if phase=='compile':worker.update(compiled_bootstraps=7,predicted_latency_us=None,runtime_plan='/new/plan',prediction={'missing':['relinearize']})
            save(args.out/'work'/(phase+'-result.json'),worker)
            return dict(exit_code=1 if phase==fail else 0,timed_out=False,signal=None,remaining_pids=[])
        with (patch.object(compile.paths,'RESULTS_DIR',results),patch.object(compile.paths,'PROFILES_DIR',profiles),
              patch.object(compile.paths,'DACAPO_ROOT',root/'dacapo'),patch.object(compile,'run_process',side_effect=run) as call,
              patch.object(compile,'_available_ram_bytes',return_value=20*compile.GIB),
              patch.object(compile.shutil,'disk_usage',return_value=Namespace(free=20*compile.GIB)),
              patch('pt_agent.registry.git_rev',return_value='synthetic')):result=compile.execute(args)
        return args,result,call

    def test_success_runs_three_once_and_keeps_report_rejection_without_candidate_change(self):
        with TemporaryDirectory() as temp:
            args,result,call=self.execute_fixture(Path(temp),report_acceptance=False)
            self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(call.call_count,3)
            self.assertEqual(result['completed'],dict(emit=1,trace=1,compile=1))
            self.assertFalse(result['task_report_acceptance']);self.assertIsNone(result['predicted_latency_us'])
            self.assertEqual(result['compiled_bootstraps'],7);self.assertTrue(all(result[k]==0 for k in compile.ZERO_FIELDS))
            rows=[json.loads(s) for s in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]),8)
            completion=json.loads((args.out/'completion.json').read_text());self.assertTrue(completion['registry_persisted'])
            spans=result['timing']['spans'];root=[r for r in spans if r['parent_id'] is None][0]
            self.assertEqual(sum(r['exclusive_ns'] for r in spans),root['inclusive_ns'])

    def test_first_phase_failure_stops_all_following_native_actions_and_retains_partial(self):
        with TemporaryDirectory() as temp:
            args,result,call=self.execute_fixture(Path(temp),fail='trace')
            self.assertEqual(result['status'],'error');self.assertEqual(call.call_count,2)
            self.assertEqual(result['attempted'],dict(emit=1,trace=1,compile=0))
            self.assertEqual(result['completed'],dict(emit=1,trace=0,compile=0))
            self.assertTrue((args.out/'completion.json').exists());self.assertIsNone(result['compiled_bootstraps'])

    def test_compiled_plan_observation_retains_positions_and_rejects_boot_count_mismatch(self):
        for planned in (1,2):
            with TemporaryDirectory() as temp:
                root=Path(temp);case,audit,report=fixture(root);work=root/'work';emit=work/'emit';emit.mkdir(parents=True)
                metadata=root/'template.json';save(metadata,template());profile=root/'profile.json'
                save(profile,dict(latencyTable=dict(bootstrap_single=[0,123.])))
                save(work/'refresh-plan.json',dict(boots=planned))
                plan=dict(values=[dict(id='in',level=1),dict(id='out',level=5)],execution=[
                    dict(kind='compute',op='boot',inputs=['in'],output='out')])
                args=Namespace(worker_phase='compile',work_dir=work,source_case=case,checkpoint_audit=audit,
                    report_result=report,program=case/'session/program.ptdsl',model_info=case/'model/model_info.json',
                    metadata_template=metadata,profile=profile,frontend=root/'frontend.py')
                def child(args,phase,argv,recorder,**kwargs):
                    self.assertEqual(phase,'compile');self.assertIn('--pars',argv)
                    self.assertNotIn('--enable-debug-printer',argv)
                    save(emit/'lowered.synthetic.runtime-plan.json',plan)
                    return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
                with patch.object(compile,'_child',side_effect=child),patch.dict(compile.os.environ,{'HECATE_PYTHON_LIBRARY_PATH':'/existing/library'}):
                    result=compile.worker(args)
                self.assertEqual(result['status'],'ok' if planned==1 else 'error')
                if planned==1:
                    self.assertEqual(result['compiled_bootstraps'],1)
                    self.assertEqual(result['bootstrap_positions'][0]['input_levels'],[1])
                    self.assertEqual(result['predicted_latency_us'],123.)


if __name__=='__main__':unittest.main()
