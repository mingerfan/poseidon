import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.experiments import autonomous_serial_batch as m

class SerialBatch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.batch=self.root/'batch';self.batch.mkdir();self.watch=self.root/'watched';self.watch.mkdir()
    def closed(self,status='ok'):
        for name in ('result.json','completion.json'):(self.watch/name).write_text(json.dumps({'status':status,'required_artifacts_persisted':True,'registry_persisted':True}))
    def job(self,name,**extra):
        out=self.root/name
        row={'id':name,'out':str(out),'seconds':10,'argv':['python3','-B','-m',m.PREFIX+'orbit_common_model_frontend','--out',str(out)]};row.update(extra);return row
    def manifest(self,jobs):return {'schema_version':1,'watch':{'case':str(self.watch),'pid':1,'starttime':10,'seconds':10},'jobs':jobs}
    def test_commands_reject_detachment_shell_and_output_mismatch(self):
        good=self.job('a');m.check_command(good['argv'],good['out'])
        for argv in (['bash','-c','true'],good['argv']+['--detach'],good['argv'][:3]+['other.module']+good['argv'][4:]):
            with self.assertRaises(ValueError):m.check_command(argv,good['out'])
        with self.assertRaises(ValueError):m.check_command(good['argv'],self.root/'different')
    def test_finite_unique_dependencies_and_results_scope(self):
        m.validate(self.manifest([self.job('a'),self.job('b',requires=['a'])]),self.root)
        for jobs in ([self.job('a',requires=['b'])],[self.job('a'),self.job('a')],[self.job('a',seconds=0)]):
            with self.assertRaises(ValueError):m.validate(self.manifest(jobs),self.root)
    def test_explicit_finite_disk_reserve_keeps_default(self):
        m.validate(self.manifest([self.job('a',minimum_free_gib=4)]),self.root)
        for value in (3,11,4.5,5.0,float('nan'),float('inf'),True,'4'):
            with self.assertRaises(ValueError):m.validate(self.manifest([self.job('a',minimum_free_gib=value)]),self.root)
        rows=m.execute_jobs([self.job('a',minimum_free_gib=4)],out=self.batch,results_root=self.root,runner=self.fake(),free=lambda _:9*m.GIB)
        self.assertEqual(rows[0]['status'],'ok');self.assertEqual(self.started,['a'])
    def test_only_bounded_trusted_native_continuation_and_partial_archive_added(self):
        for name in ('dacapo_compile_continuation','failed_trace_archive',
                     'r30_alex_postfreeze','r30_alex_continuation_report'):
            job=self.job(name);job['argv'][3]=m.PREFIX+name
            self.assertEqual(m.check_command(job['argv'],job['out']),job['argv'])
    def test_phase_conditions_limit_to_recorded_attempt_counts(self):
        source=str(self.watch)
        m.validate(self.manifest([self.job('a',requires_case_fields={source:{'trace_completed':1,'compile_completed':0}})]),self.root)
        for fields in ({},{'accuracy':1},{'trace_completed':True},{'compile_completed':2}):
            with self.assertRaises(ValueError):m.validate(self.manifest([self.job('a',requires_case_fields={source:fields})]),self.root)
    def test_failed_compile_uses_saved_trace_once_and_skips_partial_trace_archive(self):
        self.closed('error');f=self.watch/'result.json';d=json.loads(f.read_text());d.update(trace_completed=1,compile_completed=0);f.write_text(json.dumps(d))
        source=str(self.watch)
        jobs=[self.job('continue',requires_case_fields={source:{'trace_completed':1,'compile_completed':0}}),
              self.job('partial',requires_case_fields={source:{'trace_completed':0}})]
        rows=m.execute_jobs(jobs,out=self.batch,results_root=self.root,runner=self.fake(),free=lambda _:20*m.GIB)
        self.assertEqual([x['status'] for x in rows],['ok','dependency_skipped']);self.assertEqual(self.started,['continue'])
    def test_phase_conditions_require_actual_integer_count(self):
        self.closed('error');f=self.watch/'result.json'
        for value in (True,False,1.0,'1',None):
            f.write_text(json.dumps({'trace_completed':value}))
            self.assertFalse(m.case_phase_matches(self.watch,{'trace_completed':1}))
        f.write_text(json.dumps({'trace_completed':1}))
        self.assertTrue(m.case_phase_matches(self.watch,{'trace_completed':1}))
    def test_prepared_status_only_for_explicit_qwen_prompt(self):
        argv=['python3','-B','-m',m.PREFIX+'qwen_bridge_agent','--through','prompt']
        self.assertTrue(m.accepted_status(argv,'prepared'))
        self.assertFalse(m.accepted_status(argv[:-1]+['agent'],'prepared'))
        self.assertFalse(m.accepted_status(argv,'quality_rejected'))
        self.assertFalse(m.accepted_status(self.job('a')['argv'],'prepared'))
    def test_pid_reuse_closed_case_never_restarts(self):
        self.closed('error');w=self.manifest([])['watch']
        d=m.wait_closed(w,probe=lambda p:{'starttime':11,'state':'S'})
        self.assertEqual(d['status'],'error');self.assertTrue(d['same_original_process_not_restarted'])
    def test_external_alive_timeout(self):
        times=iter([0,0,11]);w=self.manifest([])['watch']
        with self.assertRaises(TimeoutError):m.wait_closed(w,probe=lambda p:{'starttime':10,'state':'S'},sleep=lambda _:None,clock=lambda:next(times))
    def test_gone_unclosed_case_refuses_next_job(self):
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_missing_terminal_persistence_refuses_admission(self):
        self.closed();(self.watch/'completion.json').write_text(json.dumps({'status':'ok'}))
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_external_case_residual_descendants_refuse_admission(self):
        self.closed('error');f=self.watch/'result.json';d=json.loads(f.read_text());d['processes']=[{'observation':{'remaining_pids':[4]}}];f.write_text(json.dumps(d))
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_external_unknown_cleanup_refuses_admission(self):
        self.closed('error');f=self.watch/'result.json';d=json.loads(f.read_text());d['jobs']=[{'cleanup_unknown':True}];f.write_text(json.dumps(d))
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_completed_batch_with_observations_is_closed(self):
        self.closed('completed_with_observations')
        self.assertEqual(m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)['status'],'completed_with_observations')
    def test_batch_active_process_missing_from_terminal_refuses_admission(self):
        self.closed('error');f=self.watch/'result.json';d=json.loads(f.read_text());d['mode']='finite-trusted-serial-experiments';f.write_text(json.dumps(d));(self.watch/'active.json').write_text(json.dumps({'id':'lost'}))
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_batch_job_cancellation_signals_refuse_successor(self):
        self.closed('error');f=self.watch/'result.json'
        for job in ({'signal':15},{'process':{'signal':2}}):
            d=json.loads(f.read_text());d['jobs']=[job];f.write_text(json.dumps(d))
            with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_batch_empty_active_observation_refuses_successor(self):
        self.closed('error');f=self.watch/'result.json';d=json.loads(f.read_text());d.update(mode='finite-trusted-serial-experiments',jobs=[{'id':'a','process':{}}]);f.write_text(json.dumps(d));(self.watch/'active.json').write_text(json.dumps({'id':'a'}))
        with self.assertRaises(RuntimeError):m.wait_closed(self.manifest([])['watch'],probe=lambda p:None)
    def test_conditional_recovery_skips_successful_original(self):
        self.closed();runner=self.fake()
        rows=m.execute_jobs([self.job('a',requires_case_statuses={str(self.watch):'error'})],out=self.batch,results_root=self.root,runner=runner,free=lambda _:20*m.GIB)
        self.assertEqual(rows[0]['status'],'dependency_skipped');self.assertEqual(self.started,[])
    def test_lease_wait_is_finite(self):
        times=iter([0,0,2])
        with patch.object(m.fcntl,'flock',side_effect=BlockingIOError):
            with self.assertRaises(TimeoutError):m.acquire_lease(1,1,clock=lambda:next(times),sleep=lambda _:None)
    def test_lease_wait_retries_before_starting(self):
        with patch.object(m.fcntl,'flock',side_effect=[BlockingIOError,None]) as lock:
            m.acquire_lease(1,2,clock=lambda:0,sleep=lambda _:None)
        self.assertEqual(lock.call_count,2)
    def fake(self,fail=(),signal_job=None):
        self.started=[]
        def run(argv,**kw):
            out=Path(argv[argv.index('--out')+1]);self.started.append(out.name);out.mkdir();status='error' if out.name in fail else 'ok';(out/'result.json').write_text(json.dumps({'status':status}))
            return {'exit_code':0 if status=='ok' else 1,'timed_out':False,'signal':15 if out.name==signal_job else None,'remaining_pids':[],'signals_sent':[]}
        return run
    def test_failed_dependency_skips_only_its_dependents(self):
        rows=m.execute_jobs([self.job('a'),self.job('b',requires=['a']),self.job('c')],out=self.batch,results_root=self.root,runner=self.fake(fail=['a']),free=lambda _:20*m.GIB)
        self.assertEqual([x['status'] for x in rows],['failed','dependency_skipped','ok']);self.assertEqual(self.started,['a','c'])
    def test_low_disk_and_existing_outputs_never_launch(self):
        runner=self.fake();rows=m.execute_jobs([self.job('a')],out=self.batch,results_root=self.root,runner=runner,free=lambda _:9*m.GIB)
        self.assertEqual(rows[0]['status'],'resource_deferred');self.assertEqual(self.started,[])
        (self.root/'a').mkdir();rows=m.execute_jobs([self.job('a')],out=self.batch,results_root=self.root,runner=runner,free=lambda _:20*m.GIB)
        self.assertEqual(rows[0]['status'],'preserved_existing_case');self.assertEqual(self.started,[])
    def test_interrupt_stops_independent_later_jobs(self):
        rows=m.execute_jobs([self.job('a'),self.job('b')],out=self.batch,results_root=self.root,runner=self.fake(signal_job='a'),free=lambda _:20*m.GIB)
        self.assertEqual(len(rows),1);self.assertEqual(self.started,['a'])
    def test_residual_processes_stop_independent_later_jobs(self):
        base=self.fake()
        def run(argv,**kw):
            row=base(argv,**kw);row['remaining_pids']=[23];return row
        rows=m.execute_jobs([self.job('a'),self.job('b')],out=self.batch,results_root=self.root,runner=run,free=lambda _:20*m.GIB)
        self.assertEqual(len(rows),1);self.assertEqual(self.started,['a']);self.assertEqual(rows[0]['status'],'failed')
    def test_unknown_runner_cleanup_stops_independent_later_jobs(self):
        def run(*args,**kwargs):raise OSError('supervisor failed')
        rows=m.execute_jobs([self.job('a'),self.job('b')],out=self.batch,results_root=self.root,runner=run,free=lambda _:20*m.GIB)
        self.assertEqual(len(rows),1);self.assertTrue(rows[0]['cleanup_unknown'])
    def test_signal_during_prelaunch_metadata_stops_later_jobs(self):
        with patch.object(m,'check_command',side_effect=m.BatchInterrupted(2)):
            rows=m.execute_jobs([self.job('a'),self.job('b')],out=self.batch,results_root=self.root,runner=self.fake(),free=lambda _:20*m.GIB)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['signal'],2)
    def test_signal_between_children_persists_terminal_metadata(self):
        f=self.root/'manifest.json';f.write_text(json.dumps(self.manifest([])));out=self.root/'actualbatch'
        with patch.object(m.paths,'RESULTS_DIR',self.root),patch.object(m,'wait_closed',side_effect=m.BatchInterrupted(15)),patch('pt_agent.registry.git_rev',return_value='test'):
            code=m.run(SimpleNamespace(manifest=f,out=out))
        self.assertEqual(code,1);self.assertEqual(json.loads((out/'result.json').read_text())['signal'],15)
        self.assertTrue(json.loads((out/'completion.json').read_text())['registry_persisted'])
    def test_archiver_manifest_uses_single_file_metadata(self):
        single=self.root/'traced.mlir';single.write_text('constant')
        closed={'eligible_files':{'compiler_ir_input':single},'eligible_bundle':self.root/'absent'}
        job=self.job('archive',kind='archive',source_case=str(self.watch),minimum_free_gib=6);f=self.batch/'selected.json'
        with patch('pt_agent.experiments.completed_compile_archive.validate_completed_case',return_value=closed):
            argv=m.archive_argv(job,f,self.root)
        d=json.loads(f.read_text());self.assertEqual(d['files'][0]['inventory'][0]['kind'],'file');self.assertEqual(d['files'][0]['inventory'][0]['bytes'],8);self.assertEqual(argv[-2:],['--minimum-free-gib','6'])

if __name__=='__main__':unittest.main()
