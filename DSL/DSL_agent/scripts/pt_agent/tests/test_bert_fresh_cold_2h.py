"""Small, payload-free fixtures for the fresh BERT two-hour orchestration."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pt_agent.experiments import bert_fresh_cold_2h as entry
from pt_agent.experiments import bert_compile_first_deferred_report as deferred
from pt_agent.experiments import bert_frozen_compile as compiler


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def state():
    return dict(entry.serial.STATE_POLICY,phase='evaluate',iteration=2,calls=1,failed=None,pending=None,
                pool=[dict(iteration=1)],records={'typed-key':{'retained':True}},history=[{'iteration':1}],
                messages=[{'role':'system','content':'tiny'}],tune_cache={},best_iteration=1)


def args(root):
    return SimpleNamespace(model=root/'original',inputs=root/'inputs',out=root/'fresh',
        checkpoint_out=root/'checkpoint',intake_python=Path('/existing/intake'),data_python=Path('/existing/data'),
        ledger=root/'ledger.jsonl',allowance=root/'allowance.json',metadata_template=root/'template.json',
        profile=root/'profile.json',frontend=root/'frontend.py',paper_runs=None)


class FreshColdTests(unittest.TestCase):
    def test_budget_stop_preserves_entire_original_state(self):
        with TemporaryDirectory()as folder:
            case=Path(folder);before=state();write(case/'session/SESSION.json',before)
            observed=entry.stop_search(case,elapsed_seconds=123)
            self.assertEqual(entry.read(case/'session/SESSION.json'),dict(before,phase='confirm'))
            self.assertEqual(entry.read(case/'budget-stop-before-session.json'),before)
            self.assertEqual(observed['abandoned_partial_steps'],0)
            self.assertEqual(before['max_iters'],10)

    def test_pending_failed_empty_and_policy_change_rejected(self):
        for update in ({'pending':{'partial':True}},{'failed':'bad'},{'pool':[]},{'max_iters':1},{'phase':'tune'}):
            with self.subTest(update=update),TemporaryDirectory()as folder:
                case=Path(folder);write(case/'session/SESSION.json',dict(state(),**update))
                with self.assertRaises(ValueError):entry.stop_search(case,elapsed_seconds=100)
                self.assertFalse((case/'budget-stop.json').exists())

    def test_commands_never_expose_report_and_preserve_pure_worker_paths(self):
        a=args(Path('/fixture'))
        for phase in ('prepare','step','confirm','math','emit','trace','compile','checkpoint'):
            command=entry.command(a,phase,Path('/fixture/work'),400)
            self.assertNotIn('report',command)
            if phase!='checkpoint':
                self.assertEqual(command[command.index('--python')+1],str(a.intake_python))
                inner=command[command.index('--')+1:]
                self.assertEqual(inner[:2],['-B','-m'])
                self.assertNotIn(str(a.ledger),inner if phase not in ('prepare','step','confirm')else [])
            if phase in ('emit','trace','compile'):
                self.assertIn(str(a.out/'session/program.ptdsl'),command)

    def test_worker_resource_clone_and_original_functions_preserved(self):
        a=SimpleNamespace(worker_phase='emit',remaining_seconds=500,math_result=Path('/math'),metadata_template=Path('/template'))
        original_case=compiler.CASE_BYTES; captured={}
        def factory(seconds):
            namespace={'CASE_BYTES':1}
            def worker(value):captured.update(namespace);return {'status':'ok'}
            namespace['worker']=worker;return namespace
        @contextmanager
        def commands(template):yield
        with patch.object(deferred,'compile_namespace',side_effect=factory),\
             patch.object(entry,'read',return_value={'compiler_inputs':{'same':True}}),\
             patch.object(deferred,'compiler_metadata',return_value={'same':True}),\
             patch.object(compiler,'native_metadata_commands',side_effect=commands):
            self.assertEqual(entry.worker(a),{'status':'ok'})
        self.assertEqual(captured['CASE_BYTES'],64*1024**3)
        self.assertEqual(compiler.CASE_BYTES,original_case)

    def test_compile_context_and_metadata_identity_precede_original_dispatch(self):
        a=SimpleNamespace(worker_phase='compile',remaining_seconds=300,math_result=Path('/math'),metadata_template=Path('/template'))
        events=[]
        @contextmanager
        def context(template):
            events.append('enter')
            try:yield
            finally:events.append('exit')
        def factory(seconds):
            self.assertEqual(events,['enter'])
            return {'worker':lambda value:events.append('original-worker')or{'status':'ok'}}
        with patch.object(entry,'read',return_value={'compiler_inputs':{'same':True}}),\
             patch.object(deferred,'compiler_metadata',return_value={'same':True}),\
             patch.object(compiler,'native_metadata_commands',side_effect=context),\
             patch.object(deferred,'compile_namespace',side_effect=factory):
            self.assertEqual(entry.worker(a),{'status':'ok'})
        self.assertEqual(events,['enter','original-worker','exit'])
        with patch.object(entry,'read',return_value={'compiler_inputs':{}}),\
             patch.object(deferred,'compiler_metadata',return_value={'same':True}),\
             patch.object(deferred,'compile_namespace')as factory:
            with self.assertRaises(ValueError):entry.worker(a)
            factory.assert_not_called()

    def test_paper_current_c2_exclusive_steps_native_clock_and_failure_class(self):
        with TemporaryDirectory()as folder:
            a=args(Path(folder));a.paper_runs=Path(folder)/'runs.jsonl';captured=[]
            native=dict(exit_code=-11,started_ns=100,ended_ns=200)
            resources=Path(folder)/'resources.jsonl'
            resources.write_text(json.dumps({'processes':[{'vmrss_kib':10},{'vmrss_kib':20}]})+'\n')
            result=dict(status='error',compile_accepted=False,processes=[dict(phase='compile',worker_result={'process':native},
                observation={'timed_out':False},resources=str(resources))],phases={},prediction=None,termination=None,
                confirmation_quality={'top1':91.2,'exact_top1':91.4,'passed':True,'total':1000},
                compiled_bootstraps=None,timing={'attribution_ready':True,'spans':[
                    dict(stage_id='step',exclusive_ns=10**9),dict(stage_id='step',exclusive_ns=2*10**9),
                    dict(stage_id='fresh_cold_complete',exclusive_ns=10**9)]},
                api_calls=1,error='native failed',wall_policy={'search_stop':'first boundary'})
            record=dict(persisted_pipeline_within_two_hours=False,elapsed_seconds=20,
                completion_elapsed_seconds=20,usable_definition='actual',status='error')
            from pt_agent.experiments import a_group_runs
            with patch.object(a_group_runs,'append_record',side_effect=lambda path,row:captured.append(row)):
                entry.paper_record(a,result,record)
            row=captured[0];self.assertEqual(row['model'],'bert_tiny');self.assertEqual(row['status'],'compiler_crash')
            self.assertEqual(row['compile_s'],1e-7);self.assertEqual(row['stages_s']['step'],3)
            self.assertEqual(sum(row['stages_s'].values()),20);self.assertEqual(row['quality']['value'],91.2)
            self.assertIsNone(row['generation_e2e_s'])
            self.assertEqual(row['peak_rss_gib'],30*1024/entry.GIB)

    def test_failed_native_saved_process_and_late_record_scope(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a=args(root)
            write(root/'compile-result.json',{'status':'error','process':None})
            write(root/'compile-process.json',{'exit_code':-11,'timed_out':False,'started_ns':1,'ended_ns':2})
            recovered=entry._read_worker(a,'compile',root)
            self.assertEqual(recovered['process']['exit_code'],-11)
            self.assertEqual(entry.read(root/'compile-result.json')['process'],None)
            record=entry.two_hour_observation({'compile_accepted':True},
                {'required_artifacts_persisted':True,'registry_persisted':True,'ended_ns':7201*10**9,
                 'complete_case_elapsed_seconds':7201},0,7202*10**9)
            self.assertFalse(record['persisted_pipeline_within_two_hours'])
            self.assertEqual(record['status'],'error')

    def test_native_typed_plan_and_bundle_metadata_with_no_payload_read(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);plan=root/'lowered.test.runtime-plan.json';bundle=root/'lowered.test.bundle'
            value=lambda name,kind,level:dict(id=name,kind=kind,context='qwen-bootstrap48',ntt=True,
                place={'kind':'host','rank':0},level=level,scale_log2=43,components=1 if kind=='plaintext' else 2)
            document=dict(format_version=1,finalization=[],external_inputs=['x'],final_outputs=['y'],
                values=[value('x','ciphertext',8),value('p','plaintext',5),value('y','ciphertext',5)],
                initialization=[dict(kind='encode',output='p',payload={'kind':'bundle','content':'tiny:0'})],
                execution=[dict(op='boot',inputs=['x'],output='y')],plaintext_bundle={'id':'tiny'})
            write(plan,document);write(bundle/'manifest.json',dict(bundle_format_version=1,version=1,bundle_id='tiny',
                blobs=[dict(content='tiny:0',byte_length=8)]))
            (bundle/'data').mkdir();(bundle/'data/0.bin').write_bytes(b'12345678')
            accepted=entry.native_artifacts(dict(runtime_plan=str(plan),compiled_bootstraps=1))
            self.assertEqual(accepted['payload_bytes'],8);self.assertEqual(accepted['blob_payload_reads'],0)
            document['values'][-1]['level']=8;write(plan,document)
            with self.assertRaises(ValueError):entry.native_artifacts(dict(runtime_plan=str(plan),compiled_bootstraps=1))

    def test_exact_inclusive_deadline_and_late_persistence_rejected(self):
        result=dict(compile_accepted=True,native_completed_ns=200)
        completion=dict(required_artifacts_persisted=True,registry_persisted=True,ended_ns=300,
                        complete_case_elapsed_seconds=0.0000003)
        limit=7200*10**9
        self.assertTrue(entry.two_hour_observation(result,completion,0,limit)['persisted_pipeline_within_two_hours'])
        self.assertFalse(entry.two_hour_observation(result,completion,0,limit+1)['persisted_pipeline_within_two_hours'])
        self.assertFalse(entry.two_hour_observation(result,dict(completion,ended_ns=limit+1),0,400)['persisted_pipeline_within_two_hours'])
        self.assertFalse(entry.two_hour_observation(dict(result,compile_accepted=False),completion,0,400)['persisted_pipeline_within_two_hours'])

    def test_real_outer_fake_children_fresh_prepare_c2_math_compile_and_unique_closure(self):
        self._outer_fixture(False)

    def test_partial_step_failure_never_confirms_or_compiles(self):
        self._outer_fixture(True)

    def _outer_fixture(self,fail):
        with TemporaryDirectory()as folder:
            root=Path(folder);a=args(root);a.ledger.touch();a.metadata_template.write_text('[]')
            registry=root/'registry.jsonl';registry.touch();phases=[]
            def fake_run(command,**kwargs):
                name=entry.read(a.out/'result.json')['active_phase'];phases.append(name)
                self.assertEqual(Path(kwargs['env']['TMPDIR']),a.out/'tmp')
                self.assertTrue(Path(kwargs['env']['TMPDIR']).is_dir())
                if name=='math':self.assertFalse((a.out/'math').exists())
                process=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],
                    started_ns=100,ended_ns=200,cleanup_unknown=False)
                if name in ('prepare','step','confirm'):
                    work=Path(command[command.index('--work-dir')+1])
                    value=dict(status='ok',phase='prepare'if name=='prepare'else'step')
                    if name=='step':
                        write(a.out/'session/SESSION.json',dict(state(),phase='tune',pending={'partial':True})if fail else state())
                        if fail:process['exit_code']=1
                    if name=='confirm':
                        latest=entry.read(a.out/'session/SESSION.json');self.assertEqual(latest['phase'],'confirm')
                        write(a.out/'session/SESSION.json',dict(latest,phase='done'))
                        write(a.out/'session/AGENT.json',{'confirm':{'selected':{'iteration':1,'origin':'tiny'},
                            'evals':[{'iteration':1,'origin':'tiny','passed':True,'top1':91.,'exact_top1':91.,'total':1000}]}})
                    write(work/'worker_result.json',value)
                elif name=='checkpoint':
                    value=dict(status='ok',checkpoint_semantic_coverage_passed=True)
                    write(a.checkpoint_out/'result.json',value)
                else:
                    work=Path(command[command.index('--work-dir')+1])
                    value=dict(status='ok',math_passed=True)
                    if name=='compile':
                        plan=a.out/'compile-work/emit/plan.json';write(plan,{})
                        value.update(runtime_plan=str(plan),compiled_bootstraps=3,prediction=None)
                    write(work/(name+'-result.json'),value)
                return process
            with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'source_gate',return_value={'fresh':True}),\
                 patch.object(entry,'run_process',side_effect=fake_run),patch.object(entry,'_budget',return_value=0),\
                 patch.object(entry.original,'check_worker'),patch.object(entry.serial,'frozen_state',return_value={'frozen':True}),\
                 patch.object(entry,'native_artifacts',return_value={'accepted':True}),\
                 patch.object(compiler,'_available_ram_bytes',return_value=8*1024**3):
                result=entry.execute(a)
            self.assertEqual(result['status'],'error'if fail else'ok')
            self.assertEqual(phases,['prepare','checkpoint','step']if fail else
                             ['prepare','checkpoint','step','confirm','math','emit','trace','compile'])
            self.assertEqual(result['attempted']['confirm'],0 if fail else 1)
            self.assertEqual(result['report_samples'],0)
            rows=[json.loads(line)for line in registry.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),entry.REGISTRY_FIELDS)
            completion=entry.read(a.out/'completion.json');self.assertTrue(completion['registry_persisted'])
            record=entry.read(a.out/'two-hour-record.json')
            self.assertEqual(record['persisted_pipeline_within_two_hours'],not fail)
            self.assertTrue(result['monitor_stopped'])


if __name__=='__main__':unittest.main()
