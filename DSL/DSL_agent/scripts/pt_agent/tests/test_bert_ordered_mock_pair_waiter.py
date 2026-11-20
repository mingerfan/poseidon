"""Finite no-payload wait and exact trusted-exec admission fixtures."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_ordered_mock_pair_waiter as entry
from test_bert_ordered_mock_pair import fixture as outer_fixture


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    results=root/'results';results.mkdir()
    outer,_=outer_fixture(results)
    deadline=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    argv=[sys.executable,'-B','-m',entry.pair.MODULE]
    for name in('source_case','compile_case','first_case','prior_case','out','metadata_template',
                'intake_python','data_python','ledger','allowance'):
        argv+=['--'+name.replace('_','-'),str(getattr(outer,name))]
    for flag,value in(('--prior-pid',outer.prior_pid),('--prior-starttime',outer.prior_starttime),
                      ('--host-pid-namespace',outer.host_pid_namespace),('--deadline-utc',deadline)):
        argv += [flag,str(value)]
    manifest=dict(schema=entry.SCHEMA,automatic_restart=False,argv=argv,deadline_utc=deadline,
        watch=dict(case=str(outer.prior_case),pid=outer.prior_pid,starttime=outer.prior_starttime,seconds=90000))
    readiness=dict(schema='bert-ordered-mock-pair-launch-readiness-v1',source_stopped=True,must_fix=[],
        **dict.fromkeys(('outer_tests_passed','worker_tests_passed','waiter_tests_passed',
            'old_source_preflight_passed','old_producer_preflight_passed'),True))
    args=SimpleNamespace(manifest=root/'manifest.json',readiness=root/'readiness.json',metadata_out=root/'metadata-wait')
    write(args.manifest,manifest);write(args.readiness,readiness)
    return args,outer,manifest,readiness


def probe(pid):
    if pid==os.getpid():return entry.birth_probe(pid)
    return None


class WaiterTests(unittest.TestCase):
    def test_exact_manifest_and_watched_birth(self):
        with TemporaryDirectory()as folder:
            args,outer,value,_=fixture(Path(folder))
            with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent):selected,deadline=entry.validate(value)
            self.assertEqual(selected.out,outer.out)
            self.assertEqual(selected.prior_pid,outer.prior_pid)
            self.assertGreater(deadline,0)
            for update in({'pid':1},{'case':'/wrong-owner'},{'starttime':1}):
                changed=dict(value,watch=dict(value['watch'],**update))
                with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent),self.assertRaises(ValueError):entry.validate(changed)

    def test_unreviewed_source_and_scientific_root_metadata_output_rejected(self):
        for fault in('readiness','location'):
            with self.subTest(fault=fault),TemporaryDirectory()as folder:
                args,outer,_,ready=fixture(Path(folder))
                if fault=='readiness':ready['source_stopped']=False;write(args.readiness,ready)
                else:args.metadata_out=outer.out.parent/'wrong-metadata'
                with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent),self.assertRaises(ValueError):entry.run(args)
                self.assertFalse(args.metadata_out.exists());self.assertFalse(outer.out.exists())

    def test_wait_precedes_exec_and_exec_failure_has_no_scientific_case(self):
        with TemporaryDirectory()as folder:
            args,outer,manifest,_=fixture(Path(folder));calls=[];limit_calls=[]
            def wait(value):calls.append('wait');return{'status':'completed_with_observations'}
            def execute(binary,argv):
                calls.append('exec');self.assertEqual(argv,manifest['argv']);raise OSError('tiny controlled exec failure')
            with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent):
                value=entry.run(args,wait=wait,probe=probe,executor=execute,set_limits=limit_calls.append)
            self.assertEqual(calls,['wait','exec']);self.assertEqual(limit_calls,[True,False])
            self.assertEqual(value['status'],'error');self.assertTrue(value['original_process_declared_terminal'])
            self.assertFalse(outer.out.exists());self.assertFalse(value['actual_heavy_lease_acquired'])

    def test_observation_timeout_keeps_original_owner_unproven(self):
        with TemporaryDirectory()as folder:
            args,outer,_,_=fixture(Path(folder));calls=[]
            def wait(value):raise TimeoutError('same original handle remains pending')
            with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent):
                value=entry.run(args,wait=wait,probe=probe,executor=lambda *x:calls.append(x),set_limits=lambda x:None)
            self.assertEqual(value['status'],'wait_expired_owner_unproven')
            self.assertFalse(value['original_process_declared_terminal']);self.assertEqual(calls,[])
            self.assertFalse(outer.out.exists())

    def test_full_pair_deadline_reserve_prevents_late_scientific_entry(self):
        with TemporaryDirectory()as folder:
            args,outer,manifest,_=fixture(Path(folder));calls=[]
            deadline=entry.batch.deadline_timestamp(manifest['deadline_utc'])
            with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent):
                value=entry.run(args,wait=lambda x:calls.append('wait'),probe=probe,
                    executor=lambda *x:calls.append('exec'),set_limits=lambda x:None,
                    now=lambda:deadline-entry.pair.SECONDS-20)
            self.assertEqual(value['status'],'skipped_deadline');self.assertEqual(calls,[])
            self.assertFalse(outer.out.exists())

    def test_changed_mainline_plan_returns_for_automatic_reconciliation(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,outer,manifest,_=fixture(root);plan=root/'CODEX_TASKS.md'
            plan.write_text('Only a plan update detector fixture\n')
            manifest['plan_observation']=dict(path=str(plan),mtime_ns=plan.stat().st_mtime_ns-1)
            write(args.manifest,manifest);calls=[]
            with patch.object(entry.pair.paths,'RESULTS_DIR',outer.out.parent):
                result=entry.run(args,wait=lambda x:{'status':'completed_with_observations'},probe=probe,
                    executor=lambda *x:calls.append(x),set_limits=lambda x:None)
            self.assertEqual(result['status'],'plan_changed_reconcile_automatically')
            self.assertEqual(calls,[]);self.assertFalse(outer.out.exists())

    def test_strict_missing_or_reused_birth_and_zombie_wait(self):
        self.assertTrue(entry.gone(7,9,probe=lambda p:None)['original_birth_gone'])
        self.assertTrue(entry.gone(7,9,probe=lambda p:{'starttime':10})['pid_reused'])
        clock=iter([0,31])
        with self.assertRaises(TimeoutError):entry.gone(7,9,probe=lambda p:{'starttime':9,'state':'Z'},
            sleep=lambda x:None,clock=lambda:next(clock))

    def test_private_controlled_exec_keeps_pid_without_any_numeric_entry(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,outer,manifest,_=fixture(root);marker=root/'exec-observation.json'
            script=r'''
from pathlib import Path
from types import SimpleNamespace
import os,sys,json
from pt_agent.experiments import bert_ordered_mock_pair_waiter as entry
entry.pair.paths.RESULTS_DIR=Path(sys.argv[1])
args=SimpleNamespace(manifest=Path(sys.argv[2]),readiness=Path(sys.argv[3]),metadata_out=Path(sys.argv[4]))
marker=sys.argv[5]
before=os.getpid()
def probe(pid):return entry.birth_probe(pid)if pid==os.getpid()else None
def execute(binary,argv):
    code="import os,json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'prior_pid':int(sys.argv[2]),'requested_argv':json.loads(sys.argv[3])}))"
    os.execv(sys.executable,[sys.executable,'-B','-c',code,marker,str(before),json.dumps(argv)])
entry.run(args,wait=lambda x:{'status':'completed_with_observations'},probe=probe,
          executor=execute,set_limits=lambda x:None)
raise RuntimeError('Controlled exec should replace this image')
'''
            process=subprocess.run([sys.executable,'-B','-c',script,str(outer.out.parent),str(args.manifest),
                str(args.readiness),str(args.metadata_out),str(marker)],capture_output=True,text=True,timeout=10)
            self.assertEqual(process.returncode,0,process.stderr)
            value=json.loads(marker.read_text());self.assertEqual(value['pid'],value['prior_pid'])
            self.assertEqual(value['requested_argv'],manifest['argv'])
            self.assertFalse(outer.out.exists());self.assertTrue((args.metadata_out/'launch-intent.json').is_file())


if __name__=='__main__':unittest.main()
