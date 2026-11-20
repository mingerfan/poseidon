"""Small metadata and original gzip-stream fixtures; no actual BERT access."""
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_closed_trace_archive as entry


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(root):
    cases={role:root/name for role,name in entry.CASES.items()};records={};summary={};births=[];rows=[]
    process=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
    for index,(role,case) in enumerate(cases.items()):
        case.mkdir();phases=['emit','trace','compile']if role=='native'else['math','compile']
        result=dict(round='fixture-'+role,model=entry.MODEL,mode='original-fixture',status='ok',metrics=dict(status='ok'),report_samples=872 if role=='report'else 0)
        if role=='report':
            result.update(attempted_report=1,completed_report=1,report_accepted=True,process=process,
                compile_case=str(cases['master']),source=dict(native_case=str(cases['native'])))
        else:
            result.update(attempted={p:1 for p in phases},completed={p:1 for p in phases},compiled_bootstraps=9180,
                processes=[dict(phase=p,observation=process)for p in phases])
            if role=='master':result['compile_case']=str(cases['native'])
        write(case/'result.json',result)
        write(case/'completion.json',dict(round=result['round'],status='ok',registry_persisted=True,required_artifacts_persisted=True))
        row=dict(run_id=case.name,round=result['round'],model=result['model'],mode=result['mode'],metrics=result['metrics'],git_rev='fixture',artifacts_dir=str(case),ts='fixture')
        rows.append(row);records[role]=result
        birth=dict(pid=900000+index,starttime=100+index);births.append(birth)
        write(case/'fixture-resources.jsonl',dict(processes=[birth],outer_process=None))
        summary[role]=dict(case=str(case),status='ok',attempted=result.get('attempted'),completed=result.get('completed'),
            compiled_bootstraps=result.get('compiled_bootstraps'),report_samples=result['report_samples'])
    (root/'registry.jsonl').write_text('\n'.join(json.dumps(row)for row in rows)+'\n')
    emit=cases['native']/'work/emit';emit.mkdir(parents=True)
    raw=emit/'candidate_trace.mlir';raw.write_text('public finite compiler-text fixture\n'*10000)
    preserved=[]
    for name in ('constants.npz','candidate_trace.py','lowered.mlir'):
        path=emit/name;path.write_text('opaque fixture artifact\n');preserved.append(entry.metadata(path))
    plan=emit/'lowered._hecate_golden.runtime-plan.json';plan.write_text('opaque runtime plan body is never opened')
    bundle=emit/'lowered._hecate_golden.bundle';(bundle/'data').mkdir(parents=True)
    blobs=[];total=0
    for i in range(2):
        path=bundle/'data'/('fixture'+str(i)+'.bin');path.write_text('opaque constant\n');size=path.stat().st_size;total+=size
        blobs.append(dict(content='opaque:fixture'+str(i),byte_length=size))
    manifest=bundle/'manifest.json';write(manifest,dict(bundle_format_version=1,version=1,blobs=blobs))
    audit=dict(must_fix=[],checks=[dict(passed=True)],case_summaries=summary,recorded_births=births,
        raw=entry.metadata(raw),runtime_plan=entry.metadata(plan),preserved_emit_artifacts=preserved,
        bundle=dict(manifest=entry.metadata(manifest),blob_count=2,referenced_blob_count=2,blob_bytes=total))
    auditpath=root/'audit.json';write(auditpath,audit)
    return auditpath,raw,audit


class BertArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.audit,self.raw,self.saved=fixture(self.root)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry.paths,'RESULTS_DIR',self.root))
        self.stack.enter_context(patch.object(entry,'_process',return_value=None))
        self.stack.enter_context(patch.object(entry,'_available_ram_bytes',return_value=4*entry.GIB))
        self.stack.enter_context(patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=64*entry.GIB)))

    def args(self,suffix='fixture'):
        return SimpleNamespace(audit=self.audit,out=self.root/('bert-closed-trace-archive-'+suffix),seconds=30)

    def test_metadata_gate_never_opens_raw_plan_or_constant_payload(self):
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path in (self.raw,Path(self.saved['runtime_plan']['path'])) or path.suffix in ('.bin','.npz','.py','.mlir'):
                raise AssertionError('gate opened scientific payload')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):proof=entry.source_gate(self.audit)
        self.assertEqual(proof['recorded_birth_count'],3);self.assertEqual(proof['bundle_blobs'],2)
        self.assertEqual(proof['payload_reads_for_admission'],0)

    def test_raw_stat_active_birth_and_existing_broken_target_rejected(self):
        with patch.object(entry,'_process',return_value=dict(starttime=100,state='S')):
            with self.assertRaises(ValueError):entry.source_gate(self.audit)
        target=self.raw.with_name(self.raw.name+'.gz');target.symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):entry.source_gate(self.audit)
        target.unlink();self.raw.write_text('changed stat fixture')
        with self.assertRaises(ValueError):entry.source_gate(self.audit)

    def test_duplicate_registry_or_report_incomplete_rejected(self):
        registry=self.root/'registry.jsonl';original=registry.read_text();registry.write_text(original+original.splitlines()[0]+'\n')
        with self.assertRaises(ValueError):entry.source_gate(self.audit)
        registry.write_text(original)
        report=self.root/entry.CASES['report']/'result.json';value=entry.read(report);value['completed_report']=0;write(report,value)
        with self.assertRaises(ValueError):entry.source_gate(self.audit)

    def test_real_original_archive_once_preserves_executables_and_registers_completion(self):
        result=entry.execute(self.args())
        self.assertEqual(result['status'],'ok');self.assertTrue(result['archive_accepted'])
        self.assertEqual((result['attempted'],result['completed']),(1,1));self.assertFalse(self.raw.exists())
        self.assertTrue(Path(str(self.raw)+'.gz').is_file());self.assertFalse(Path(str(self.raw)+'.gz.partial').exists())
        self.assertGreater(result['released_bytes'],0)
        out=self.args().out;done=entry.read(out/'completion.json');self.assertEqual(done['status'],'ok')
        rows=[json.loads(line)for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['run_id']==out.name for row in rows),1)
        self.assertTrue(all(entry.metadata(Path(row['path']))==row for row in result['source']['preserved']))
        self.assertEqual(result['hash_CRC_readback_checks'],0)

    def test_original_stream_interruption_retains_raw_partial_and_error_closure(self):
        stopped=InterruptedError('finite fixture stream stop')
        with patch.object(entry.original.shutil,'copyfileobj',side_effect=stopped):
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'error');self.assertFalse(result['archive_accepted'])
        self.assertEqual((result['attempted'],result['completed']),(1,0));self.assertTrue(self.raw.is_file())
        self.assertTrue(Path(str(self.raw)+'.gz.partial').is_file());self.assertFalse(Path(str(self.raw)+'.gz').exists())
        self.assertEqual(entry.read(self.args().out/'completion.json')['status'],'error')

    def test_existing_archive_refused_before_original_primitive(self):
        Path(str(self.raw)+'.gz').write_text('retained prior archive')
        with patch.object(entry,'archive_one',side_effect=AssertionError('should not archive')):
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'error');self.assertEqual(result['attempted'],0)
        self.assertTrue(self.raw.exists())

    def test_late_resource_stop_revokes_acceptance_without_removing_archive(self):
        original=entry.resources;calls=[]
        def resources(*args,**kwargs):
            observed=original(*args,**kwargs);calls.append(observed)
            if len(calls)>=3:observed.update(stop=True,reasons=['late fixture reserve stop'])
            return observed
        with patch.object(entry,'resources',resources):result=entry.execute(self.args())
        self.assertEqual(result['completed'],1);self.assertEqual(result['status'],'error');self.assertFalse(result['archive_accepted'])
        self.assertTrue(Path(str(self.raw)+'.gz').is_file())
        self.assertEqual(entry.read(self.args().out/'completion.json')['status'],'error')

    def test_resource_worst_temporary_reserve_and_physical_RAM_limits(self):
        args=self.args();args.out.mkdir();source=entry.source_gate(self.audit)
        reserve=source['source']['bytes']+(source['source']['bytes']+99)//100
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB+reserve-1)):
            self.assertTrue(entry.resources(args,source,__import__('time').monotonic()+30,admission=True)['stop'])
        with patch.object(entry,'_available_ram_bytes',return_value=entry.GIB):
            self.assertTrue(entry.resources(args,source,__import__('time').monotonic()+30)['stop'])

    def test_independent_single_lock_and_exact_process_limits(self):
        with entry.local_lock(self.root):
            with self.assertRaises(BlockingIOError):
                with entry.local_lock(self.root):pass
        set_limits=[]
        with patch.object(entry.resource,'getrlimit',return_value=(resource.RLIM_INFINITY,resource.RLIM_INFINITY)), \
                patch.object(entry.resource,'setrlimit',side_effect=lambda *args:set_limits.append(args)), \
                patch.object(entry.os,'getpriority',return_value=5),patch.object(entry.os,'nice')as nice:
            entry.process_limits()
        self.assertEqual(set_limits,[(resource.RLIMIT_AS,(entry.GIB//2,entry.GIB//2)),(resource.RLIMIT_FSIZE,(32*entry.GIB,32*entry.GIB))])
        nice.assert_called_once_with(14)
        self.assertFalse((self.root/'autonomous-heavy-experiments.lock').exists())

    def test_execute_join_signal_regression_persists_terminal_and_restores_closed_monitor(self):
        import signal
        signals=(signal.SIGINT,signal.SIGTERM,signal.SIGALRM,signal.SIGXFSZ)
        saved={sig:signal.getsignal(sig)for sig in signals};timer=signal.getitimer(signal.ITIMER_REAL)
        observations=[]
        class JoinedWatcher:
            def __init__(self,*args,**kwargs):pass
            def start(self):pass
            def join(watcher,timeout):
                self.assertEqual(timeout,3)
                self.assertTrue(all(signal.getsignal(sig)==signal.SIG_IGN for sig in signals))
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
                observations.append('signals ignored while joining')
            def is_alive(self):return False
        try:
            with patch.object(entry.threading,'Thread',JoinedWatcher):result=entry.execute(self.args())
            self.assertEqual(observations,['signals ignored while joining'])
            self.assertEqual(result['status'],'ok');self.assertTrue(result['monitor_stopped'])
            self.assertFalse(result['late_monitor_signals_ignored_until_exit'])
            self.assertEqual(entry.read(self.args().out/'completion.json')['status'],'ok')
            self.assertEqual({sig:signal.getsignal(sig)for sig in signals},saved)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in saved.items():signal.signal(sig,handler)
            if timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*timer)

    def test_execute_unclosed_monitor_keeps_signals_ignored_and_records_error_terminal(self):
        import signal
        signals=(signal.SIGINT,signal.SIGTERM,signal.SIGALRM,signal.SIGXFSZ)
        saved={sig:signal.getsignal(sig)for sig in signals};timer=signal.getitimer(signal.ITIMER_REAL)
        class UnclosedWatcher:
            def __init__(self,*args,**kwargs):pass
            def start(self):pass
            def join(watcher,timeout):
                self.assertEqual(timeout,3)
                self.assertTrue(all(signal.getsignal(sig)==signal.SIG_IGN for sig in signals))
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
            def is_alive(self):return True
        try:
            signal.setitimer(signal.ITIMER_REAL,30)
            with patch.object(entry.threading,'Thread',UnclosedWatcher):result=entry.execute(self.args())
            self.assertEqual(result['status'],'error');self.assertFalse(result['archive_accepted'])
            self.assertFalse(result['monitor_stopped']);self.assertTrue(result['late_monitor_signals_ignored_until_exit'])
            self.assertTrue(all(signal.getsignal(sig)==signal.SIG_IGN for sig in signals))
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
            signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
            self.assertEqual(entry.read(self.args().out/'completion.json')['status'],'error')
            rows=[json.loads(line)for line in (self.root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['metrics']['status'],'error')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in saved.items():signal.signal(sig,handler)
            if timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*timer)


if __name__=='__main__':unittest.main()
