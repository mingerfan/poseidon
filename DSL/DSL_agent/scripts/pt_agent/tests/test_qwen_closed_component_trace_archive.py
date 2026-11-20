"""Small metadata fixtures; archive primitives and process launches are fake."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent.experiments import qwen_closed_component_trace_archive as entry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.args = argparse.Namespace(results_root=self.root, out=self.root/'observer',
            watch_pid=entry.WATCH_PID, watch_starttime=entry.WATCH_START,
            seconds=180, poll_seconds=1., consumer_manifest=[], archive_request=None,
            archive_response=None, run=True,
            **{n+'_case':self.root/s for n,s in entry.CASE_NAMES.items()})
        self.args.out.mkdir()
        self.case = self.args.remaining_case/'layer1'; self.emit = self.case/'codegen/emit'
        self.emit.mkdir(parents=True)
        self.plan = self.emit/'lowered.original.runtime-plan.json'; write(self.plan, {})
        for name in entry.TRACE_NAMES:
            (self.emit/name).write_text('tiny trusted original trace\n')
        self.marker = dict(raw_trace=str(self.emit/entry.TRACE_NAMES[0]),
            marked_trace=str(self.emit/entry.TRACE_NAMES[1]), arithmetic_rewritten=False,
            raw_trace_retained=True)
        self.record = dict(round='COMPONENT', model=entry.MODEL, mode='one-component', status='ok',
            phase_results=dict(trace=dict(trace_marker=self.marker)))
        write(self.case/'result.json', self.record)
        write(self.case/'completion.json', dict(status='ok', registry_persisted=True,
            required_artifacts_persisted=True))
        self.acceptance = dict(status='ok', runtime_plan=str(self.plan), native=dict(native_ABI_verified=True))
        row = dict(run_id='layer1', round='COMPONENT', model=entry.MODEL, mode='one-component',
            git_rev=None, metrics=dict(status='ok'), artifacts_dir=str(self.case), ts='fixture')
        self.registry_row = row
        (self.root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        for phase in ('trace', 'compile'):
            (self.case/(phase+'-resources.jsonl')).write_text(json.dumps(dict(
                processes=[dict(pid=901, starttime=55)],outer_process=dict(pid=900,starttime=54)))+'\n')
        self.gate = patch.object(entry, 'original_acceptance', return_value=self.acceptance)
        self.live = patch.object(entry, 'same_live', return_value=False)
        self.gate_mock = self.gate.start(); self.addCleanup(self.gate.stop)
        self.live_mock = self.live.start(); self.addCleanup(self.live.stop)

    def component(self):
        return entry.accepted_component(self.args, 'layer1')

    def request(self):
        accepted = self.component(); proof = entry.file_proof(accepted, entry.TRACE_NAMES[0])
        stem = self.args.out/('layer1-'+entry.TRACE_NAMES[0])
        request = stem.with_name(stem.name+'.request.json')
        response = stem.with_name(stem.name+'.response.json')
        write(request, dict(args=entry.args_record(self.args), component='layer1', accepted=accepted, file=proof))
        return request, response, proof

    def fake_manifest(self):
        names = ['qwen_retained_component_replay','qwen_layer_native_probe',
            'qwen_generation_kernel_compile','qwen_remaining_prefill_compile','qwen_component_link_validation']
        cases = [self.args.source_case,self.args.probe_case,self.args.kernel_case,self.args.remaining_case,self.root/'link']
        jobs=[]
        for i,(name,case)in enumerate(zip(names,cases)):
            argv=['python','-m','pt_agent.experiments.'+name,'--out',str(case)]
            if i:argv.extend(['--source-case',str(self.args.source_case)])
            jobs.append(dict(id=str(i),out=str(case),argv=argv))
        return dict(deadline_utc=entry.DEADLINE.isoformat(),jobs=jobs)

    def test_closed_child_is_eligible_while_parent_running_without_parent_registry(self):
        write(self.args.remaining_case/'result.json',dict(status='running'))
        value=self.component()
        self.assertEqual(value['runtime_plan'],str(self.plan));self.assertEqual(value['registry']['field_count'],8)
        self.assertEqual(value['births']['same_birth_live'],0)
        self.gate_mock.assert_called_once_with(self.args,'layer1')

    def test_pending_or_failed_child_never_reaches_native_gate(self):
        (self.case/'completion.json').unlink()
        with self.assertRaises(entry.Pending):self.component()
        self.gate_mock.assert_not_called()
        write(self.case/'completion.json',dict(status='error'))
        with self.assertRaises(ValueError):self.component()
        self.gate_mock.assert_not_called()

    def test_original_source_gate_and_foreign_native_plan_rejections_preserved(self):
        self.gate_mock.side_effect=ValueError('original source mismatch')
        with self.assertRaisesRegex(ValueError,'original source mismatch'):self.component()
        self.gate_mock.side_effect=None;self.gate_mock.return_value=dict(self.acceptance,runtime_plan=str(self.root/'foreign.json'))
        with self.assertRaisesRegex(ValueError,'RuntimePlan'):self.component()

    def test_actual_registry_is_unique_exact_eight_fields(self):
        path=self.root/'registry.jsonl';path.write_text(json.dumps(self.registry_row)+'\n'+json.dumps(self.registry_row)+'\n')
        with self.assertRaisesRegex(ValueError,'One original'):self.component()
        path.write_text(json.dumps(dict(self.registry_row,extra=1))+'\n')
        with self.assertRaisesRegex(ValueError,'eight-field'):self.component()

    def test_trace_and_native_birth_closure_required_and_reused_pid_permitted(self):
        self.live_mock.side_effect=lambda pid,start:pid==901 and start==55
        with self.assertRaises(entry.Pending):self.component()
        self.live_mock.side_effect=None;self.live_mock.return_value=False
        self.assertEqual(len(self.component()['births']['identities']),2)
        (self.case/'compile-resources.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'Both original'):self.component()

    def test_original_gate_dispatch_uses_probe_remaining_and_kernel_interfaces(self):
        self.gate.stop()
        remaining=ModuleType('pt_agent.experiments.qwen_remaining_prefill_compile')
        checked=dict(first_layer_runtime_plan='actual-plan',first_layer_native=dict(probe_accepted=True),
            bindings=dict(layers=[dict(original=0)]))
        remaining.admission=Mock(return_value=checked);remaining.accept_component=Mock(return_value=dict(actual='layer'))
        kernel=ModuleType('pt_agent.experiments.qwen_generation_kernel_compile')
        kernel.admission=Mock(return_value=dict(original='C2'));kernel.accept_component=Mock(return_value=dict(actual='kernel'))
        with patch.dict('sys.modules',{remaining.__name__:remaining,kernel.__name__:kernel}):
            with patch('pt_agent.experiments.qwen_remaining_prefill_compile',remaining,create=True), \
                    patch('pt_agent.experiments.qwen_generation_kernel_compile',kernel,create=True):
                self.assertEqual(entry.original_acceptance(self.args,'layer0')['runtime_plan'],'actual-plan')
                self.assertEqual(entry.original_acceptance(self.args,'head'),dict(actual='layer'))
                self.assertEqual(entry.original_acceptance(self.args,'first'),dict(actual='kernel'))
        remaining.accept_component.assert_called_once_with(self.args.remaining_case/'head','head',checked)
        kernel.accept_component.assert_called_once_with(self.args.kernel_case/'first','first',dict(original='C2'))

    def test_only_two_exact_mlir_names_and_clean_file_states_allowed(self):
        accepted=self.component()
        for name in ('constants.data','manifest.json','lowered.original.runtime-plan.json'):
            with self.assertRaises(ValueError):entry.file_proof(accepted,name)
        proof=entry.file_proof(accepted,entry.TRACE_NAMES[0])
        Path(proof['archive']).write_text('preexisting')
        with self.assertRaises(ValueError):entry.file_proof(accepted,entry.TRACE_NAMES[0])
        Path(proof['archive']).unlink();Path(proof['temporary']).write_text('partial')
        with self.assertRaises(ValueError):entry.file_proof(accepted,entry.TRACE_NAMES[0])
        Path(proof['temporary']).unlink();Path(proof['source']).unlink()
        Path(proof['source']).symlink_to(self.plan)
        with self.assertRaises(ValueError):entry.file_proof(accepted,entry.TRACE_NAMES[0])

    def test_consumer_manifest_retains_five_original_modules_and_rejects_future_trace(self):
        value=self.fake_manifest();write(self.args.watch_case/'manifest.json',value)
        self.assertEqual(len(entry.consumer_proof(self.args)),5)
        value['jobs'][-1]['argv'].extend(['--input',str(self.emit/entry.TRACE_NAMES[0])])
        write(self.args.watch_case/'manifest.json',value)
        with self.assertRaisesRegex(ValueError,'future raw'):entry.consumer_proof(self.args)

    def test_pinned_identity_and_single_live_heavy_owner(self):
        write(self.args.watch_case/'supervisor.json',dict(pid=entry.WATCH_PID,process=dict(starttime=entry.WATCH_START)))
        self.live_mock.return_value=True
        with patch.object(entry,'lease_owners',return_value=[entry.WATCH_PID]):
            self.assertTrue(entry.watch_proof(self.args)['live'])
        with patch.object(entry,'lease_owners',return_value=[123]):
            with self.assertRaisesRegex(ValueError,'unique heavy'):entry.watch_proof(self.args)
        bad=copy.copy(self.args);bad.watch_pid+=1
        with self.assertRaises(ValueError):entry.validate_args(bad)

    def test_disk_and_temporary_reserve_preserve_raw_source(self):
        proof=entry.file_proof(self.component(),entry.TRACE_NAMES[0])
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB)), \
                patch.object(entry,'available_ram',return_value=2*entry.GIB):
            with self.assertRaises(entry.Pending):entry.disk_guard(self.args,'layer1',proof,admission=True)
        self.assertTrue(Path(proof['source']).is_file())
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)), \
                patch.object(entry,'available_ram',return_value=entry.GIB):
            with self.assertRaises(entry.Pending):entry.disk_guard(self.args,'layer1',proof)

    def test_worker_fake_archive_preserves_constants_native_bundle_and_clock_scope(self):
        request,response,proof=self.request()
        constants=self.emit/'constant-store/constants.data';constants.parent.mkdir();constants.write_text('constant fixture')
        bundle=self.emit/'lowered.original.bundle';bundle.mkdir();write(bundle/'manifest.json',{})
        before={str(p):p.stat().st_size for p in (constants,self.plan,bundle/'manifest.json')}
        def fake(source,target,temporary):
            self.assertEqual(source,Path(proof['source']));target.write_text('mock archive');source.unlink()
            return dict(status='compressed',released_bytes=3,elapsed_ns=9)
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof'), \
                patch.object(entry,'watch_proof'),patch.object(entry,'disk_guard'), \
                patch.object(entry,'archive_one',side_effect=fake):
            self.assertEqual(entry.archive_worker(request,response),0)
        result=entry.read(response)
        self.assertEqual(result['status'],'ok');self.assertTrue(result['final_files']['archive']['exists'])
        self.assertFalse(result['final_files']['source']['exists'])
        self.assertEqual(before,{str(p):p.stat().st_size for p in (constants,self.plan,bundle/'manifest.json')})

    def test_worker_compression_failure_keeps_raw_and_reports_partial_without_reopen(self):
        request,response,proof=self.request()
        def fail(source,target,temporary):
            temporary.write_text('mock partial');raise InterruptedError('fixture interrupt')
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof'), \
                patch.object(entry,'watch_proof'),patch.object(entry,'disk_guard'), \
                patch.object(entry,'archive_one',side_effect=fail):
            self.assertEqual(entry.archive_worker(request,response),1)
        result=entry.read(response)
        self.assertEqual(result['status'],'error');self.assertTrue(result['final_files']['source']['exists'])
        self.assertTrue(result['final_files']['temporary']['exists']);self.assertFalse(result['final_files']['archive']['exists'])

    def test_worker_rechecks_stat_and_rejects_foreign_response_path(self):
        request,response,proof=self.request();Path(proof['source']).write_text('changed metadata and size')
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof'), \
                patch.object(entry,'watch_proof'),patch.object(entry,'archive_one')as primitive:
            self.assertEqual(entry.archive_worker(request,response),1)
        primitive.assert_not_called()
        with self.assertRaisesRegex(ValueError,'request/response'):entry.archive_worker(request,self.root/'foreign.json')

    def test_async_parent_error_is_same_object_and_only_owned_child_is_stopped(self):
        error=InterruptedError('fixture async');child=Mock(pid=999,returncode=None)
        child.poll.side_effect=error
        accepted=self.component()
        def stop(process,birth):
            self.assertIs(process,child);self.assertEqual(birth,77);child.returncode=-15
            return [dict(pid=999,starttime=77,signal=15)]
        with patch.object(entry,'disk_guard'),patch.object(entry.subprocess,'Popen',return_value=child), \
                patch.object(entry,'process',return_value=dict(starttime=77)),patch.object(entry,'stop_owned',side_effect=stop):
            with self.assertRaises(InterruptedError)as caught:
                entry.run_archive(self.args,accepted,entry.TRACE_NAMES[0],entry.time.monotonic()+20)
        self.assertIs(caught.exception,error)
        saved=entry.read(self.args.out/('layer1-'+entry.TRACE_NAMES[0]+'.observation.json'))
        self.assertEqual(saved['status'],'error');self.assertTrue(saved['final_files']['source']['exists'])
        self.assertEqual([row['pid']for row in saved['signals_sent']],[999])

    def test_stop_owned_never_signals_changed_birth_or_controller(self):
        child=Mock(pid=999);child.poll.return_value=None;child.wait.return_value=0
        with patch.object(entry,'same_live',return_value=False),patch.object(entry.os,'kill')as kill:
            self.assertEqual(entry.stop_owned(child,77),[])
        kill.assert_not_called()
        child.wait.side_effect=[subprocess.TimeoutExpired('own',3),0]
        with patch.object(entry,'same_live',return_value=True),patch.object(entry.os,'kill')as kill:
            rows=entry.stop_owned(child,77)
        self.assertEqual([call.args for call in kill.call_args_list],[(999,signal.SIGTERM),(999,signal.SIGKILL)])
        self.assertTrue(all(row['pid']not in (entry.WATCH_PID,192085)for row in rows))

    def test_argument_json_roundtrip_preserves_exact_paths_and_deadline(self):
        restored=entry.restore_args(json.loads(json.dumps(entry.args_record(self.args))))
        self.assertEqual(vars(restored),vars(self.args));self.assertEqual(entry.validate_args(restored),self.root)
        self.assertEqual(entry.DEADLINE,datetime(2026,11,17,10,tzinfo=timezone.utc))

    def test_postcommit_persistence_error_reports_actual_archive_state(self):
        request,response,proof=self.request()
        def committed_then_error(source,target,temporary):
            target.write_text('mock committed archive');source.unlink();raise OSError('directory fsync fixture')
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof'), \
                patch.object(entry,'watch_proof'),patch.object(entry,'disk_guard'), \
                patch.object(entry,'archive_one',side_effect=committed_then_error), \
                patch.object(entry.signal,'setitimer')as timer:
            self.assertEqual(entry.archive_worker(request,response),1)
        row=entry.read(response)
        self.assertEqual(row['status'],'error');self.assertTrue(row['final_files']['archive']['exists'])
        self.assertFalse(row['final_files']['source']['exists']);self.assertFalse(row['final_files']['temporary']['exists'])
        self.assertEqual(timer.call_args.args,(signal.ITIMER_REAL,0))

    def test_expired_deadline_creates_no_observer_case(self):
        args=copy.copy(self.args);args.out=self.root/'too-late'
        fake=Mock();fake.now.return_value=entry.DEADLINE
        with patch.object(entry,'datetime',fake):
            with self.assertRaisesRegex(ValueError,'Leave30'):entry.execute(args)
        self.assertFalse(args.out.exists())

    def test_observer_fake_archive_has_independent_registry_and_clean_completion(self):
        accepted=self.component();args=copy.copy(self.args);args.out=self.root/'observer-complete'
        original_record=entry.read(self.case/'result.json')
        def fake_archive(args,component,filename,deadline):
            row=dict(status='ok',pid=999,elapsed_seconds=.01,signals_sent=[],final_files={})
            write(args.out/(filename+'.observation.json'),row)
            return row
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof',return_value=[]), \
                patch.object(entry,'watch_proof',return_value=dict(closed=True)), \
                patch.object(entry,'accepted_component',return_value=accepted), \
                patch.object(entry,'run_archive',side_effect=fake_archive),patch.object(entry,'NAMES',('layer1',)), \
                patch.object(entry.signal,'setitimer')as timer:
            result=entry.execute(args)
        self.assertEqual(result['status'],'completed_with_observations')
        self.assertEqual((result['attempted_archives'],result['completed_archives']),(2,2))
        self.assertEqual(entry.read(args.out/'completion.json')['status'],result['status'])
        rows=[json.loads(x)for x in (self.root/'registry.jsonl').read_text().splitlines()]
        owned=[r for r in rows if r['run_id']==args.out.name]
        self.assertEqual(len(owned),1);self.assertEqual(set(owned[0]),entry.REGISTRY_FIELDS)
        self.assertEqual(entry.read(self.case/'result.json'),original_record)
        self.assertIn('Independent observer',result['timing_scope'])
        self.assertEqual(timer.call_args.args,(signal.ITIMER_REAL,0))

    def test_unavailable_archive_admission_has_zero_launched_attempts(self):
        accepted=self.component();args=copy.copy(self.args);args.out=self.root/'observer-no-reserve'
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof',return_value=[]), \
                patch.object(entry,'watch_proof',return_value=dict(closed=True)), \
                patch.object(entry,'accepted_component',return_value=accepted), \
                patch.object(entry,'disk_guard',side_effect=entry.Pending('reserve unavailable')), \
                patch.object(entry.subprocess,'Popen')as launch,patch.object(entry,'NAMES',('layer1',)), \
                patch.object(entry.signal,'setitimer'):
            result=entry.execute(args)
        launch.assert_not_called()
        self.assertEqual((result['archive_admissions'],result['attempted_archives'],result['completed_archives']),(1,0,0))
        self.assertTrue((self.emit/entry.TRACE_NAMES[0]).is_file())

    def test_interrupted_observer_recovers_the_actual_launched_attempt(self):
        accepted=self.component();args=copy.copy(self.args);args.out=self.root/'observer-interrupted'
        def interrupted(args,component,filename,deadline):
            write(args.out/(filename+'.observation.json'),dict(status='error',pid=999,signals_sent=[dict(pid=999,signal=15)]))
            raise InterruptedError('fixture async')
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof',return_value=[]), \
                patch.object(entry,'watch_proof',return_value=dict(closed=True)), \
                patch.object(entry,'accepted_component',return_value=accepted), \
                patch.object(entry,'run_archive',side_effect=interrupted),patch.object(entry,'NAMES',('layer1',)), \
                patch.object(entry.signal,'setitimer'):
            result=entry.execute(args)
        self.assertEqual(result['status'],'error')
        self.assertEqual((result['attempted_archives'],result['completed_archives']),(1,0))
        self.assertEqual(result['archives'][0]['pid'],999)

    def test_new_actual_producer_family_is_explicit_and_old_identity_rejected(self):
        self.assertEqual((entry.WATCH_PID,entry.WATCH_START),(199735,4311840))
        self.assertEqual(self.args.source_case.name,'qwen-retained13-component-expanded-wall-recovery-20261117')
        self.assertEqual(self.args.watch_case.name,'deadline-qwen-retained-expanded-wall-serial-batch-20261117')
        self.assertTrue(all('component-expanded-wall' in getattr(self.args,n+'_case').name
            for n in ('source','probe','remaining','kernel')))
        self.assertEqual(entry.validate_args(self.args),self.root)
        old=copy.copy(self.args);old.watch_pid=191281;old.watch_starttime=4143079
        with self.assertRaisesRegex(ValueError,'pinned'):entry.validate_args(old)

    def test_expanded_wall_binding_rejects_previous_family_and_keeps_consumer_policy(self):
        prior=copy.copy(self.args);prior.watch_pid=196938;prior.watch_starttime=4255701
        with self.assertRaisesRegex(ValueError,'pinned'):entry.validate_args(prior)
        prior=copy.copy(self.args)
        prior.source_case=self.root/'qwen-retained13-component-dead-count-recovery-20261117'
        with self.assertRaisesRegex(ValueError,'pinned component family'):entry.validate_args(prior)
        value=self.fake_manifest();write(self.args.watch_case/'manifest.json',value)
        self.assertEqual(len(entry.consumer_proof(self.args)),5)
        self.assertEqual(entry.TRACE_NAMES,('candidate_trace.mlir','candidate_trace.segment.earth.mlir'))
        self.assertEqual(entry.DEADLINE,datetime(2026,11,17,10,tzinfo=timezone.utc))

    def test_late_cleanup_signals_still_reap_and_preserve_original_interruption(self):
        original_error=InterruptedError('original cancellation')
        late_error=InterruptedError('late cleanup cancellation')
        child=Mock(pid=999,returncode=None);child.poll.side_effect=[original_error,None]
        accepted=self.component();signals=(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)
        def incoming(*args):raise late_error
        handlers={sig:incoming for sig in signals}
        def set_handler(sig,handler):
            previous=handlers[sig];handlers[sig]=handler;return previous
        observed_cleanup=[]
        def wait(timeout):
            self.assertEqual(timeout,3)
            for sig in signals:
                observed_cleanup.append(sig)
                handler=handlers[sig]
                if callable(handler):handler(sig,None)
                self.assertEqual(handler,signal.SIG_IGN)
            child.returncode=-15;return -15
        child.wait.side_effect=wait
        saved_durable=entry.durable
        def persist(path,value):
            if Path(path).name.endswith('.observation.json'):
                self.assertTrue(all(handlers[sig]==signal.SIG_IGN for sig in signals))
            return saved_durable(path,value)
        with patch.object(entry,'disk_guard'),patch.object(entry.subprocess,'Popen',return_value=child), \
                patch.object(entry,'process',return_value=dict(starttime=77)), \
                patch.object(entry,'same_live',return_value=True),patch.object(entry.os,'kill')as kill, \
                patch.object(entry.signal,'getsignal',side_effect=lambda sig:handlers[sig]), \
                patch.object(entry.signal,'signal',side_effect=set_handler),patch.object(entry,'durable',side_effect=persist):
            with self.assertRaises(InterruptedError)as caught:
                entry.run_archive(self.args,accepted,entry.TRACE_NAMES[0],entry.time.monotonic()+20)
        self.assertIs(caught.exception,original_error)
        self.assertEqual(observed_cleanup,list(signals));child.wait.assert_called_once_with(timeout=3)
        kill.assert_called_once_with(999,signal.SIGTERM)
        self.assertTrue(all(handlers[sig]is incoming for sig in signals))
        row=entry.read(self.args.out/('layer1-'+entry.TRACE_NAMES[0]+'.observation.json'))
        self.assertEqual(row['status'],'error');self.assertEqual(row['exit_code'],-15)
        self.assertEqual(row['error'],'InterruptedError: original cancellation')

    def test_successful_first_file_survives_pending_second_and_each_primitive_runs_once(self):
        args=copy.copy(self.args);args.out=self.root/'observer-file-progress'
        state=dict(closed=False,blocked=True);calls=[];primitives=[];pending=[]
        constants=self.emit/'constants.data';constants.write_text('own constant fixture')
        bundle=self.emit/'lowered.original.bundle';bundle.mkdir();write(bundle/'manifest.json',{})
        retained={str(p):p.stat().st_size for p in (constants,self.plan,bundle/'manifest.json')}
        def disk(args,name,proof,*,admission=False):
            if admission and Path(proof['source']).name==entry.TRACE_NAMES[1]and state['blocked']:
                raise entry.Pending('temporary reserve unavailable')
            return dict(free_bytes=100*entry.GIB)
        def primitive(source,target,temporary):
            primitives.append(source.name);target.write_text('own fake archive');source.unlink()
            return dict(status='compressed',released_bytes=1,elapsed_ns=1)
        def archive(args,accepted,filename,deadline):
            calls.append(filename);proof=entry.file_proof(accepted,filename)
            disk(args,'layer1',proof,admission=True)
            stem=args.out/('layer1-'+filename)
            request=stem.with_name(stem.name+'.request.json');response=stem.with_name(stem.name+'.response.json')
            write(request,dict(args=entry.args_record(args),component='layer1',accepted=accepted,file=proof))
            self.assertEqual(entry.archive_worker(request,response),0)
            row=dict(status='ok',pid=999,elapsed_seconds=.01,signals_sent=[],
                file=proof,worker=entry.read(response))
            write(stem.with_name(stem.name+'.observation.json'),row)
            return row
        def next_poll(seconds):
            value=entry.read(args.out/'result.json')['components']['layer1'];pending.append(value)
            self.assertEqual(value['status'],'pending');self.assertFalse(value['terminal'])
            self.assertEqual(list(value['file_progress']),[entry.TRACE_NAMES[0]])
            self.assertEqual(len(value['files']),1)
            state.update(closed=True,blocked=False)
        with patch.object(entry,'apply_limits'),patch.object(entry,'consumer_proof',return_value=[]), \
                patch.object(entry,'watch_proof',side_effect=lambda args:dict(closed=state['closed'])), \
                patch.object(entry,'run_archive',side_effect=archive),patch.object(entry,'disk_guard',side_effect=disk), \
                patch.object(entry,'archive_one',side_effect=primitive),patch.object(entry,'NAMES',('layer1',)), \
                patch.object(entry.time,'sleep',side_effect=next_poll),patch.object(entry.signal,'setitimer'):
            result=entry.execute(args)
        self.assertEqual(result['status'],'completed_with_observations')
        self.assertEqual(calls,[entry.TRACE_NAMES[0],entry.TRACE_NAMES[1],entry.TRACE_NAMES[1]])
        self.assertEqual(primitives,list(entry.TRACE_NAMES));self.assertEqual(len(pending),1)
        self.assertEqual((result['attempted_archives'],result['completed_archives']),(2,2))
        component=result['components']['layer1'];self.assertEqual(component['status'],'archived')
        self.assertEqual(list(component['file_progress']),list(entry.TRACE_NAMES));self.assertEqual(len(component['files']),2)
        self.assertEqual(retained,{str(p):p.stat().st_size for p in (constants,self.plan,bundle/'manifest.json')})
        self.assertTrue(all(not(self.emit/name).exists()and(self.emit/(name+'.gz')).is_file()for name in entry.TRACE_NAMES))


if __name__=='__main__':
    unittest.main()
