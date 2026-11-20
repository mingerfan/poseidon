"""Tiny metadata fixtures and a fake archive primitive; no actual compression."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import resource
import signal
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import closed_dacapo_trace_archive as entry


class QuietThread:
    """Only the real main-thread admission/terminal checks run in these fixtures."""
    def __init__(self, *args, **kwargs): self.started = False
    def start(self): self.started = True
    def join(self, timeout=None): pass
    def is_alive(self): return False


class ClosedTraceArchiveChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.case = self.root/entry.CASE
        self.trace = self.case/'work/trace/traced/AlexNet.mlir'
        self.trace.parent.mkdir(parents=True); self.trace.write_text('tiny trusted fixture\n')
        self.plan = self.case/'work/compile/lowered._hecate_AlexNet.runtime-plan.json'
        self.plan.parent.mkdir(parents=True); self.plan.write_text('{}')
        self.bundle = self.plan.with_name('lowered._hecate_AlexNet.bundle'); self.bundle.mkdir()
        self.owner = {'pid':900, 'starttime':9}
        self.table = {
            os.getpid(): {'pid':os.getpid(), 'ppid':900, 'state':'S', 'starttime':10},
            900: {'pid':900, 'ppid':1, 'state':'S', 'starttime':9}}
        observation = dict(exit_code=0, timed_out=False, signal=None,
            signals_sent=[], remaining_pids=[], peak_rss_kib_by_pid={'101':10})
        self.result = dict(round='DACAPO-ORIGINAL', mode='original-source-native-profile-compile',
            model=entry.MODEL, status='ok', source_checked=True, original_source_variant=True,
            r29_cheb63_variant=False, cold_trace_compile_completed=True,
            original_trace_reused=False, prior_times_included=False,
            trace_worker={'status':'ok', 'source_unmodified':True, 'mlir':{'path':str(self.trace),
                'size_bytes':self.trace.stat().st_size,'mtime_ns':self.trace.stat().st_mtime_ns}},
            compile_worker={'status':'ok', 'process':copy.deepcopy(observation)},
            trace_process=copy.deepcopy(observation), compile_process=copy.deepcopy(observation),
            plan={'path':str(self.plan)}, complete_saved_scientific_seconds=624.9,
            **{key:1 for key in ('trace_attempts','trace_completed','compile_attempts',
                'compile_completed','native_compile_attempts','native_compile_completed')})
        self.completion = dict(round='DACAPO-ORIGINAL',status='ok',required_artifacts_persisted=True,
            registry_persisted=True,case_artifact_bytes=10000)
        self.write_source()
        self.queue = self.root/'archive-owner-queue'; self.queue.mkdir()
        self.out = self.root/'closed-dacapo-alex-trace-archive-fixture'
        self.args = SimpleNamespace(source_case=self.case, queue_case=[self.queue],
            out=self.out, seconds=30)
        self.manifest = {'jobs':[
            {'id':'completed-source', 'out':str(self.case), 'argv':['python','-m','past',str(self.case)]},
            {'id':'future-independent', 'out':str(self.root/'qwen'), 'argv':['python','-m','qwen']},
            {'id':'archive-self', 'out':str(self.out), 'argv':['python','-B','-m',entry.MODULE,
                '--source-case',str(self.case),'--queue-case',str(self.queue),'--out',str(self.out)]}]}
        self.queue_state = {'jobs':[{'id':'completed-source','status':'ok','process':{'remaining_pids':[]}},
            {'id':'archive-self','status':'running'}]}
        self.write_queue()
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(paths,'RESULTS_DIR',self.root))
        stack.enter_context(patch.object(entry.lease_tools,'_scan',side_effect=lambda:copy.deepcopy(self.table)))
        stack.enter_context(patch.object(entry.lease_tools,'_lease_owners',return_value=[900]))
        stack.enter_context(patch.object(entry.lease_tools,'_process_name',return_value='python3'))
        stack.enter_context(patch.object(entry.serial,'available_ram',return_value=16*entry.GIB))
        stack.enter_context(patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)))

    def dump(self,path,value): path.write_text(json.dumps(value))
    def write_source(self):
        self.dump(self.case/'result.json',self.result); self.dump(self.case/'completion.json',self.completion)
    def write_queue(self):
        self.dump(self.queue/'manifest.json',self.manifest)
        self.dump(self.queue/'progress.json',self.queue_state)
        self.dump(self.queue/'supervisor.json',{'pid':900,'process':{'starttime':9}})
    def source(self): return entry.source_gate(self.case)
    def kept(self,*args):
        return dict(source=str(self.trace),status='kept_uncompressed',raw_bytes=self.trace.stat().st_size,
            compressed_bytes=None,released_bytes=0,elapsed_ns=1)
    def fake_run(self,primitive=None):
        with patch.object(entry,'archive_one',side_effect=primitive or self.kept) as archive, \
                patch.object(entry.threading,'Thread',QuietThread):
            result=entry.execute(self.args)
        return result,archive
    def assert_closed(self,status='error'):
        result=json.loads((self.out/'result.json').read_text())
        completion=json.loads((self.out/'completion.json').read_text())
        rows=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual((result['status'],completion['status'],rows[0]['metrics']['status']),(status,)*3)
        self.assertTrue(completion['required_artifacts_persisted']); self.assertTrue(completion['registry_persisted'])
        self.assertEqual(rows[0]['metrics']['archive_accepted'],result['archive_accepted'])
        return result

    def test_original_stream_primitive_is_unmodified_and_scope_is_raw_trace(self):
        self.assertIs(entry.archive_one,entry.original.archive_one)
        source=self.source()
        self.assertEqual(source['source'],str(self.trace)); self.assertFalse(source['raw_content_read_for_admission'])
        self.assertEqual(source['RuntimePlan'],str(self.plan)); self.assertEqual(source['bundle'],str(self.bundle))
        self.assertEqual(source['source_case_bytes'],10000)

    def test_wrong_model_round_variant_or_incomplete_source_rejects(self):
        changes=[('model','vgg16'),('round','R34'),('mode','saved-trace'),('status','running'),
            ('source_checked',False),('original_source_variant',False),('r29_cheb63_variant',True),
            ('cold_trace_compile_completed',False),('original_trace_reused',True),('prior_times_included',True),
            ('trace_attempts',0),('native_compile_completed',0)]
        for key,value in changes:
            with self.subTest(key=key):
                original=self.result[key]; self.result[key]=value; self.write_source()
                with self.assertRaises(ValueError): self.source()
                self.result[key]=original
        self.write_source()

    def test_source_completion_clean_process_and_active_original_identity(self):
        for key in ('status','required_artifacts_persisted','registry_persisted'):
            original=self.completion[key]; self.completion[key]=False; self.write_source()
            with self.subTest(key=key),self.assertRaises(ValueError):self.source()
            self.completion[key]=original
        self.write_source();self.result['compile_process']['remaining_pids']=[101];self.write_source()
        with self.assertRaisesRegex(ValueError,'clean process'): self.source()
        self.result['compile_process']['remaining_pids']=[];self.write_source()
        self.table[101]={'pid':101,'ppid':1,'state':'S','starttime':15}
        with self.assertRaisesRegex(ValueError,'still active'): self.source()

    def test_wrong_source_case_or_protected_payload_path_rejects(self):
        with self.assertRaises(ValueError):entry.source_gate(self.root/'other')
        for path in (self.plan,self.bundle,self.case/'weights.pt',self.case/'samples.npz'):
            self.result['trace_worker']['mlir']['path']=str(path);self.write_source()
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'raw Alex trace'):self.source()

    def test_symlink_trace_and_missing_plan_or_bundle_reject(self):
        self.trace.unlink();target=self.root/'fixture.mlir';target.write_text('fixture')
        self.trace.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'raw Alex trace'):self.source()
        self.trace.unlink();self.trace.write_text('fixture')
        self.result['trace_worker']['mlir'].update(size_bytes=self.trace.stat().st_size,mtime_ns=self.trace.stat().st_mtime_ns)
        self.write_source()
        self.plan.unlink()
        with self.assertRaisesRegex(ValueError,'RuntimePlan'):self.source()
        self.plan.write_text('{}');self.bundle.rmdir()
        with self.assertRaisesRegex(ValueError,'RuntimePlan'):self.source()

    def test_existing_final_or_partial_archive_is_preserved_and_rejected(self):
        for suffix in ('.gz','.gz.partial'):
            target=Path(str(self.trace)+suffix);target.write_text('previous attempt')
            with self.subTest(suffix=suffix),self.assertRaisesRegex(ValueError,'existing archive'):self.source()
            self.assertTrue(target.is_file());target.unlink()

    def test_pending_consumer_and_unmatched_owner_queue_reject(self):
        source=self.source()
        self.assertEqual(len(entry.queue_consumers(self.args,source,self.owner)),3)
        self.manifest['jobs'][1]['argv'].append(str(self.trace));self.write_queue()
        with self.assertRaisesRegex(ValueError,'pending or running'):entry.queue_consumers(self.args,source,self.owner)
        self.manifest['jobs'][1]['argv'].pop();self.write_queue()
        with self.assertRaisesRegex(ValueError,'owner queue'):entry.queue_consumers(self.args,source,{'pid':901,'starttime':9})

    def test_dependency_consumer_and_spoofed_self_exemption_reject(self):
        self.manifest['jobs'][1]['requires_case_statuses']={str(self.case):'ok'};self.write_queue()
        with self.assertRaisesRegex(ValueError,'pending or running'):entry.queue_consumers(self.args,self.source(),self.owner)
        self.manifest['jobs'][1].pop('requires_case_statuses')
        self.manifest['jobs'][2]['argv'][3]='other.module';self.write_queue()
        with self.assertRaisesRegex(ValueError,'pending or running'):entry.queue_consumers(self.args,self.source(),self.owner)

    def test_queue_symlink_and_duplicate_job_reject(self):
        link=self.root/'queue-alias';link.symlink_to(self.queue,target_is_directory=True)
        changed=SimpleNamespace(**dict(vars(self.args),queue_case=[link]))
        with self.assertRaisesRegex(ValueError,'direct original'):entry.queue_consumers(changed,self.source(),self.owner)
        self.manifest['jobs'].append(copy.deepcopy(self.manifest['jobs'][0]));self.write_queue()
        with self.assertRaisesRegex(ValueError,'duplicate'):entry.queue_consumers(self.args,self.source(),self.owner)

    def test_global_lease_requires_live_strict_ancestor_and_same_identity(self):
        self.assertEqual(entry.lease_observation()['owner'],self.owner)
        for owners in ([],[900,901],[os.getpid()]):
            with self.subTest(owners=owners),patch.object(entry.lease_tools,'_lease_owners',return_value=owners),self.assertRaises(ValueError):
                entry.lease_observation()
        with self.assertRaises(ValueError):entry.lease_observation({'pid':900,'starttime':8})
        self.table[900]['state']='Z'
        with self.assertRaises(ValueError):entry.lease_observation()

    def test_foreign_compiler_rejects_while_same_case_descendant_is_allowed(self):
        self.table[901]={'pid':901,'ppid':1,'state':'S','starttime':12}
        with patch.object(entry.lease_tools,'_process_name',side_effect=lambda pid:'hecate-opt' if pid==901 else 'python3'):
            with self.assertRaisesRegex(ValueError,'Another native compiler'):entry.lease_observation()
            self.table[901]['ppid']=os.getpid()
            self.assertEqual(entry.lease_observation()['foreign_compilers'],[])

    def test_resource_reserves_deadline_and_temporary_case_budget(self):
        self.out.mkdir();source=self.source();future=time.monotonic()+60
        self.assertFalse(entry.resources(self.args,source,future,self.owner,admission=True)['stop'])
        for available,free in ((entry.GIB,100*entry.GIB),(16*entry.GIB,9*entry.GIB)):
            with self.subTest(available=available,free=free),patch.object(entry.serial,'available_ram',return_value=available), \
                    patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
                self.assertTrue(entry.resources(self.args,source,future,self.owner)['stop'])
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=25*entry.GIB)):
            self.assertTrue(entry.resources(self.args,source,future,self.owner,admission=True)['stop'])
        self.assertTrue(entry.resources(self.args,source,time.monotonic()-1,self.owner)['stop'])
        source['other_source_case_bytes']=entry.SOURCE_CASE_BYTES
        self.assertTrue(entry.resources(self.args,source,future,self.owner)['stop'])
        source=self.source()
        with patch.object(entry,'artifact_bytes',return_value=entry.META_CASE_BYTES):
            self.assertTrue(entry.resources(self.args,source,future,self.owner)['stop'])

    def test_limits_are_halfGiB_and16GiB_without_setting_test_process_limits(self):
        with patch.object(entry.resource,'getrlimit',return_value=(-1,resource.RLIM_INFINITY)), \
                patch.object(entry.resource,'setrlimit') as setter:
            entry.process_limits()
        self.assertEqual(setter.call_args_list[0].args,(resource.RLIMIT_AS,(entry.GIB//2,)*2))
        self.assertEqual(setter.call_args_list[1].args,(resource.RLIMIT_FSIZE,(16*entry.GIB,)*2))

    def test_fake_primitive_only_after_gate_preserves_scientific_clocks(self):
        before=(copy.deepcopy(self.result),copy.deepcopy(self.completion))
        result,archive=self.fake_run()
        self.assertEqual(archive.call_count,1);self.assertEqual((result['attempted'],result['completed']),(1,1))
        self.assertTrue(result['archive_accepted']);self.assertTrue(result['monitor_stopped'])
        self.assertEqual(self.assert_closed('ok')['released_bytes'],0)
        self.assertEqual((json.loads((self.case/'result.json').read_text()),json.loads((self.case/'completion.json').read_text())),before)
        self.assertTrue(self.trace.is_file());self.assertTrue(self.plan.is_file());self.assertTrue(self.bundle.is_dir())

    def test_rejected_source_never_calls_primitive_but_closes_own_result(self):
        self.result['model']='wrong';self.write_source();result,archive=self.fake_run()
        archive.assert_not_called();self.assertEqual(result['attempted'],0)
        self.assertFalse(self.assert_closed()['archive_accepted']);self.assertTrue(self.trace.is_file())

    def test_primitive_failure_keeps_raw_and_partial_and_records_attempt_once(self):
        def failed(source,target,temporary):
            temporary.write_text('tiny partial fixture');raise OSError('fake disk error')
        result,archive=self.fake_run(failed)
        self.assertEqual(archive.call_count,1);self.assertEqual((result['attempted'],result['completed']),(1,0))
        self.assertTrue(result['final_files']['source']['exists']);self.assertTrue(result['final_files']['temporary']['exists'])
        self.assertFalse(result['final_files']['archive']['exists']);self.assertIn('fake disk error',self.assert_closed()['error'])

    def test_inconsistent_primitive_state_retains_completed_fact_but_refuses_acceptance(self):
        def malformed(*args):return dict(status='compressed',released_bytes=5)
        result,_=self.fake_run(malformed)
        self.assertEqual(result['completed'],1);self.assertFalse(result['archive_accepted'])
        self.assertTrue(self.trace.is_file());self.assertIn('inconsistent file state',self.assert_closed()['error'])

    def test_terminal_resource_failure_revokes_acceptance_and_retains_completed_fact(self):
        def kept_then_low_memory(*args):
            row=self.kept(*args)
            self.low_memory=patch.object(entry.serial,'available_ram',return_value=entry.GIB)
            self.low_memory.start();self.addCleanup(self.low_memory.stop)
            return row
        result,_=self.fake_run(kept_then_low_memory)
        self.assertEqual(result['completed'],1);self.assertTrue(result['guard']['stop'])
        self.assertFalse(self.assert_closed()['archive_accepted'])

    def test_signal_failure_preserves_raw_restores_handlers_and_cleans_monitor(self):
        before={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM,signal.SIGXFSZ)}
        def interrupted(*args):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        result,_=self.fake_run(interrupted)
        self.assertEqual(result['incoming_signal'],signal.SIGTERM);self.assertTrue(result['monitor_stopped'])
        self.assertEqual({sig:signal.getsignal(sig) for sig in before},before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0));self.assertTrue(self.trace.is_file())
        self.assertFalse(self.assert_closed()['archive_accepted'])

    def test_existing_output_and_nonfinite_allowance_reject_before_primitive(self):
        for seconds in (0,1801):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                entry.execute(SimpleNamespace(**dict(vars(self.args),seconds=seconds)))
        self.out.mkdir()
        with self.assertRaises(ValueError):entry.execute(self.args)

    def test_completed_fake_final_archive_is_recorded_without_readback(self):
        def finalized(source,target,temporary):
            # Tiny fixture file operations stand in for the original primitive.
            # No compression, archived-content read or comparison is performed.
            target.write_text('fixture');source.unlink()
            return dict(status='compressed',released_bytes=12,restore_argv=['gzip','-dk','--',str(target)])
        result,archive=self.fake_run(finalized)
        self.assertEqual(archive.call_count,1);self.assertTrue(self.assert_closed('ok')['archive_accepted'])
        self.assertEqual(result['released_bytes'],12)
        self.assertFalse(result['final_files']['source']['exists']);self.assertTrue(result['final_files']['archive']['exists'])
        self.assertEqual(result['CRC_or_hash_or_readback_checks'],0)

    def test_finalize_failure_preserves_final_and_raw_without_retry(self):
        def finalized_then_failed(source,target,temporary):
            target.write_text('fixture');raise OSError('fake directory persistence error')
        result,archive=self.fake_run(finalized_then_failed)
        self.assertEqual(archive.call_count,1);self.assertEqual(result['completed'],0)
        self.assertTrue(result['final_files']['source']['exists']);self.assertTrue(result['final_files']['archive']['exists'])
        self.assertIn('fake directory persistence error',self.assert_closed()['error'])

    def test_pending_scalar_case_reference_also_requires_explicit_closure(self):
        self.manifest['jobs'][1]['requires_case_fields']={str(self.case):{'compile_completed':1}}
        self.write_queue()
        with self.assertRaisesRegex(ValueError,'pending or running'):
            entry.queue_consumers(self.args,self.source(),self.owner)

    def test_owner_first_job_without_progress_keeps_other_jobs_pending(self):
        self.manifest['jobs']=[self.manifest['jobs'][2],self.manifest['jobs'][1]]
        self.write_queue();(self.queue/'progress.json').unlink()
        rows=entry.queue_consumers(self.args,self.source(),self.owner)
        self.assertTrue(rows[0]['archive_self']);self.assertFalse(rows[0]['closed'])
        self.assertFalse(rows[1]['closed'])
        self.manifest['jobs'][1]['argv'].append(str(self.trace));self.write_queue()
        (self.queue/'progress.json').unlink()
        with self.assertRaisesRegex(ValueError,'pending or running'):
            entry.queue_consumers(self.args,self.source(),self.owner)

    def test_missing_or_mismatched_saved_trace_stat_provenance_rejects(self):
        source=self.source()
        self.assertEqual(source['source_binding_scope'],
            'saved and current stat metadata only; original content not revalidated')
        for key in ('size_bytes','mtime_ns'):
            original=self.result['trace_worker']['mlir'][key]
            self.result['trace_worker']['mlir'].pop(key);self.write_source()
            with self.subTest(key=key,case='missing'),self.assertRaisesRegex(ValueError,'stat size/mtime'):
                self.source()
            self.result['trace_worker']['mlir'][key]=original+1;self.write_source()
            with self.subTest(key=key,case='mismatch'),self.assertRaisesRegex(ValueError,'stat size/mtime'):
                self.source()
            self.result['trace_worker']['mlir'][key]=original;self.write_source()


if __name__=='__main__':unittest.main()
