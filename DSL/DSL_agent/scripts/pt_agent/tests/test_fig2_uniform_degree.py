"""Pure fixtures only: no model, numerical backend, data or API execution."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import fig2_uniform_degree as subject
from pt_agent.experiments import fig2_uniform_degree_worker as worker
from pt_agent.experiments.r10_paper_metrics import StageRecorder


MODEL='resnet20_silu'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def base_program():
    body=['@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")',
          'def resnet20_silu(x: Cipher[1, 3, 32, 32]):']
    for index in range(19):
        body.append(f'    a{index} = act(x, fn="silu", method="cheb", deg=63, domain=(-1.5, 1.5))')
    body.append('    return a18')
    return '\n'.join(body)+'\n'


def metadata_template():
    return ['{hecate_opt}','{input}','--pars','--waterline=43',
        '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
        '--runtime-plan-context-id=qwen-bootstrap48',
        '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
        '--ckks-config={package}/config/compiler-config.json','-o','{output_dir}/lowered.mlir']


def fixtures(root):
    root.mkdir(parents=True,exist_ok=True)
    normal=dict(mean=[.485,.456,.406],std=[.229,.224,.225])
    records=root/'records';records.mkdir();(records/'test-09000.records').write_bytes(b'fake-only')
    source=dict(records=str(records),calibration_source={'records':str(records)},normalization=normal,factory='pt_agent.experiments.r13_common_models:'+MODEL,
                checkpoint=str(root/'original-checkpoint'))
    info=root/'r13/model_info/model_info.json';program=root/'r17/worker/program.ptdsl'
    program.parent.mkdir(parents=True);program.write_text(base_program())
    cache=program.with_name('heldout_outputs.npz');cache.write_bytes(b'not-opened-by-pure-source-tests')
    paths=dict(rule_result=root/'r17/result.json',r14_result=root/'r14/result.json',r13_result=root/'r13/result.json',
        model_info=info,base_program=program,cached_outputs=cache,records=records,
        calibration=root/'r14/calibration.json',calibration_source=root/'r14/calibration_source.json',
        metadata_template=root/'saved-command.json')
    save(info,dict(source=source,task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5}))
    save(info.parent/'graph/graph.json',dict(nodes=[['n'+str(i),'act',['f'+str(i)],['o'+str(i)],{'fn':'silu'},{}] for i in range(19)]))
    save(paths['calibration'],{'f'+str(i):dict(min=-1.,max=1.) for i in range(19)})
    save(paths['calibration_source'],dict(split='train',count=1000,indices=list(range(1000)),normalization=normal))
    save(paths['r14_result'],{});save(paths['r13_result'],{});save(paths['metadata_template'],metadata_template())
    w=dict(status='ok',model=source['factory'],model_source=source,rule_options=dict(deg=63,margin=1.5,slots=16384,
        placement='depth-dp',refresh='compiler',act_method='cheb'),completed_samples=1000,
        source_preparation={k:str(v) for k,v in paths.items() if k in ('r14_result','r13_result','calibration','calibration_source')},
        artifacts=dict(program=str(program),heldout_outputs=str(cache)),
        sample_source=dict(indices=list(range(9000,10000)),count=1000,normalization=normal),
        count_stats=dict(counts={'boot':0}),validation_metrics={'total':1000},max_u=.8,candidate_frozen=False)
    w['source_preparation']['model_info']=str(info.parent)
    for key in ('source_checked','calibration_reused','math_probes_passed','evaluation_completed',
                'task_passed','domain_passed','all_finite','heldout_acceptance'):w[key]=True
    save(paths['rule_result'],dict(round='R17',status='ok',model=source['factory'],worker=w))
    return paths


def completed_worker(reject31=False):
    result=worker.initial_result(MODEL);result.update(status='ok',source_checked=True,metadata_template_checked=True,
        model_restore_attempts=1,model_restore_completed=1,heldout_load_attempts=1,heldout_load_completed=1,
        reference_cache_load_attempts=1,reference_cache_load_completed=1)
    for degree in subject.DEGREES:
        row=worker.new_variant(degree);reject=reject31 and degree==31
        row.update(status='quality_rejected' if reject else 'accepted',completed_samples=1000,math_passed=True,
            all_finite=True,heldout_acceptance=not reject,task_passed=not reject,domain_passed=True,
            ast_check={'only_degree_changed':True},planned_bootstraps=7,plan_replay={'inner_boots':7},
            validation_metrics=dict(total=1000,acc_drop_pp=1. if reject else 0.),
            domain_rows=[dict(count=1000,max_u=.8) for _ in range(19)],domain_row_count=19,max_u=.8)
        for name,count in [('count',1),('plan',1),('replay',1),('tensor',1000),('mock',3),('math',3)]:
            row[name+'_attempts']=row[name+'_completed']=count
        result['variants'].append(row)
    return result


class SourceAndVariantChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=fixtures(self.root)

    def test_real_source_protocol_does_not_open_payloads(self):
        result=subject.source_gate(MODEL,self.source)
        self.assertEqual(result['calibration_samples'],1000)
        self.assertEqual(result['heldout_samples'],1000)
        self.assertFalse(result['original_training_images_reread'])

    def test_only_degree_mutation_roundtrip(self):
        original=self.source['base_program'].read_text()
        for degree in (31,63):
            changed=subject.degree_variant(original,degree)
            self.assertEqual(subject.check_variant(original,changed,degree)['activation_count'],19)
        with self.assertRaises(ValueError):subject.degree_variant(original,127)

    def test_non_approximation_or_domain_change_rejected(self):
        original=base_program();candidate=subject.degree_variant(original,31)
        for changed in [candidate.replace('(-1.5, 1.5)','(-2.0, 2.0)',1),
                        candidate.replace('refresh=\'compiler\'','refresh=\'agent\''),
                        candidate.replace('return a18','return a17')]:
            with self.assertRaises(ValueError):subject.check_variant(original,changed,31)

    def test_original_margin_and_planner_source_policy(self):
        self.source['base_program'].write_text(base_program().replace('depth-dp','program-plan'))
        with self.assertRaises(ValueError):subject.source_gate(MODEL,self.source)
        self.source['base_program'].write_text(base_program())
        save(self.source['calibration'],{'f'+str(i):dict(min=-1.,max=2.) for i in range(19)})
        with self.assertRaises(ValueError):subject.source_gate(MODEL,self.source)

    def test_reference_cache_identity_and_heldout_order(self):
        foreign=dict(self.source,cached_outputs=self.root/'foreign.npz')
        foreign['cached_outputs'].write_bytes(b'fake')
        with self.assertRaises(ValueError):subject.source_gate(MODEL,foreign)
        record=subject.read(self.source['rule_result']);record['worker']['sample_source']['indices'][0]=8000
        save(self.source['rule_result'],record)
        with self.assertRaises(ValueError):subject.source_gate(MODEL,self.source)


class VariantProtocolChecks(unittest.TestCase):
    def test_quality_rejection_is_complete_exit_zero_protocol(self):
        self.assertTrue(subject.validate_worker(completed_worker(reject31=True),MODEL))

    def test_counts_math_and_original_policy_are_required(self):
        for mutate in [lambda r:r['variants'][0].update(mock_completed=2),
                       lambda r:r.update(model_restore_attempts=2),
                       lambda r:r['variants'][0].update(math_passed=False),
                       lambda r:r['variants'][0]['validation_metrics'].update(acc_drop_pp=1.),
                       lambda r:r['variants'][0].update(max_u=.91),
                       lambda r:r.update(predicted_latency=12.)]:
            r=completed_worker();mutate(r)
            with self.assertRaises(ValueError):subject.validate_worker(r,MODEL)

    def test_quality_rejection_continues_but_math_error_stops(self):
        with tempfile.TemporaryDirectory() as name:
            work=Path(name)/'worker';work.mkdir();args=SimpleNamespace(work_dir=work)
            rec=StageRecorder(work/'events.jsonl');result=worker.initial_result(MODEL);calls=[]
            def reject(row):calls.append(row['degree']);row['status']='quality_rejected'
            worker.run_variants(args,result,rec,reject)
            self.assertEqual(calls,[31,63])
            result=worker.initial_result(MODEL);calls=[]
            def fail(row):
                calls.append(row['degree']);row['math_attempts']=1
                row['validation_metrics']={'must_clear':True};raise ValueError('strict math fake failure')
            with self.assertRaises(ValueError):worker.run_variants(args,result,rec,fail)
            self.assertEqual(calls,[31]);self.assertEqual(result['variants'][0]['math_attempts'],1)
            self.assertIsNone(result['variants'][0]['validation_metrics'])

    def test_observer_accepts31_and_rejects_wrong_cfg_or_missing_call(self):
        class Array:
            size=2
            def min(self):return -.2
            def max(self):return .3
        np=SimpleNamespace(ndarray=Array,isfinite=lambda data:SimpleNamespace(all=lambda:True))
        op=SimpleNamespace(kind='act',id=1,inputs=('x',),attrs={'deg':31,'domain':(-1.,1.)})
        rows=[dict(op_id=1,count=0,obs_min=None,obs_max=None,max_u=0.,u_sample_id=None)]
        obs,finish=worker.domain_observer(SimpleNamespace(ops=[op]),'test:9000',rows,np,lambda *a,**k:.3)
        with self.assertRaises(ValueError):finish()
        with self.assertRaises(ValueError):obs('x',Array(),('cheb',63,-1.,1.))
        obs('x',Array(),('cheb',31,-1.,1.));finish()
        self.assertEqual(rows[0]['count'],1);self.assertEqual(rows[0]['max_u'],.3)

    def test_metadata_context_precedes_worker_numeric_route(self):
        from pt_agent.experiments import r29_local_compile_matrix
        entered=[]
        @contextmanager
        def context(template):
            entered.append('enter')
            try:yield
            finally:entered.append('exit')
        def execute(args,recorder,result):
            self.assertEqual(entered,['enter']);result['status']='ok'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);metadata=p/'metadata.json';save(metadata,metadata_template())
            argv=['--model',MODEL,'--work-dir',str(p/'worker')]
            for key in ('rule-result','r14-result','r13-result','model-info','calibration','calibration-source',
                        'base-program','cached-outputs','records','metadata-template'):
                argv+=['--'+key,str(metadata)]
            with patch.object(r29_local_compile_matrix,'native_metadata_commands',context),patch.object(worker,'run_model',execute),patch.dict(os.environ):
                self.assertEqual(worker.main(argv),0)
        self.assertEqual(entered,['enter','exit'])

    def test_actual_list_schema_reaches_real_context_without_legacy_digest_builder(self):
        from pt_agent import toolchain
        calls=[]
        def run(args,recorder,result):
            argv=toolchain.compile_argv('pars','qwen48',Path('/metadata-only'))
            self.assertEqual(argv[:2],['/hecate-opt','/out/candidate_trace.mlir'])
            self.assertIn('--pars',argv);calls.append(1);result['status']='ok'
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);metadata=p/'metadata.json';save(metadata,metadata_template())
            argv=['--model',MODEL,'--work-dir',str(p/'worker')]
            for key in ('rule-result','r14-result','r13-result','model-info','calibration','calibration-source',
                        'base-program','cached-outputs','records','metadata-template'):
                argv+=['--'+key,str(metadata)]
            with patch.object(worker,'run_model',run),patch.object(toolchain,'compile_argv',side_effect=AssertionError('legacy identifier computation forbidden'))as forbidden,patch.dict(os.environ):
                self.assertEqual(worker.main(argv),0);forbidden.assert_not_called()
            result=subject.read(p/'worker/worker_result.json')
            self.assertTrue(result['metadata_template_checked']);self.assertEqual(calls,[1])

    def test_object_recipe_rejected_before_model_or_source_execution(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);metadata=p/'metadata.json';save(metadata,dict(invalid_object_recipe=True))
            argv=['--model',MODEL,'--work-dir',str(p/'worker')]
            for key in ('rule-result','r14-result','r13-result','model-info','calibration','calibration-source',
                        'base-program','cached-outputs','records','metadata-template'):
                argv+=['--'+key,str(metadata)]
            with patch.object(worker,'run_model')as run,patch.dict(os.environ):
                self.assertEqual(worker.main(argv),1);run.assert_not_called()
            result=subject.read(p/'worker/worker_result.json')
            self.assertFalse(result['metadata_template_checked']);self.assertEqual(result['model_restore_attempts'],0)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=fixtures(self.root/'sources')
        self.results=self.root/'results';self.results.mkdir()
        self.out=self.results/'fake-fig2-comparison'

    def launch(self,record=None,exit_code=0):
        record=completed_worker(reject31=True) if record is None else record
        def fake(argv,**kwargs):
            self.assertEqual(kwargs['mem_gib'],12);self.assertEqual(kwargs['timeout'],1860)
            self.assertIn('1830',argv)
            save(self.out/'worker/worker_result.json',record)
            return dict(exit_code=exit_code,timed_out=False,signal=None,started_ns=1,ended_ns=2,
                termination_observed_ns=2,peak_rss_kib_by_pid={},outer_peak_rss_kib=0,limits={},
                signals_sent=[],remaining_pids=[])
        with patch.object(subject.paths,'RESULTS_DIR',self.results),patch.object(subject,'selected_sources',return_value=self.source),\
             patch.object(subject,'_available_ram_bytes',return_value=16*subject.GIB),patch.object(subject,'run_process',side_effect=fake) as run:
            result=subject.execute(SimpleNamespace(model=MODEL,out=self.out))
        return result,run

    def test_complete_quality_reject_case_has_unique_eight_field_registry(self):
        result,run=self.launch();self.assertEqual(result['status'],'ok');self.assertEqual(run.call_count,1)
        rows=[json.loads(line) for line in (self.results/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        marker=subject.read(self.out/'completion.json')
        self.assertEqual(marker['case_artifact_bytes'],subject.artifact_bytes(self.out))
        self.assertTrue(marker['exclusive_attribution_ready'])
        with self.assertRaises(ValueError):subject.register(self.out,result,self.results/'registry.jsonl')

    def test_failed_worker_preserves_actual_partial_counts(self):
        record=worker.initial_result(MODEL);record.update(status='error',source_checked=True)
        row=worker.new_variant(31);row.update(status='error',tensor_attempts=3,tensor_completed=2)
        record['variants']=[row]
        result,run=self.launch(record,exit_code=1)
        self.assertEqual(result['status'],'error');self.assertEqual(run.call_count,1)
        self.assertEqual(result['worker']['variants'][0]['tensor_completed'],2)
        self.assertEqual(len(result['worker']['variants']),1)

    def test_fresh_directory_and_terminal_reserve_enforced(self):
        self.out.mkdir()
        with patch.object(subject.paths,'RESULTS_DIR',self.results):
            with self.assertRaises(ValueError):subject.execute(SimpleNamespace(model=MODEL,out=self.out))
        value={'model':MODEL,'status':'ok'}
        subject._terminal(self.out,value)
        self.assertEqual(value['artifact_bytes'],subject.artifact_bytes(self.out))
        with patch.object(subject,'CASE_BYTES',value['artifact_bytes']+subject.RESERVE_BYTES-1):
            with self.assertRaises(RuntimeError):subject._terminal(self.out,value)

    def test_disk_preflight_stops_before_any_child(self):
        with patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            result,run=self.launch()
        self.assertEqual(run.call_count,0);self.assertEqual(result['status'],'resource_stopped')
        self.assertIsNotNone(result['guard_trigger'])


if __name__=='__main__':unittest.main()
