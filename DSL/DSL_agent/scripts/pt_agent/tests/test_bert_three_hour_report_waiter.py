"""Conditional frozen-report scheduling; all scientific work is a fixture."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_three_hour_report_waiter as entry


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(root,*,accepted=False):
    results=root/'results';results.mkdir()
    source=results/'fresh';report=results/'report';serial=results/'report-serial';owner=results/'source-owner'
    argv=['python3','-B','-m',entry.report.MODULE,'--source-case',str(source),
          '--supervisor-case',str(owner),'--out',str(report),'--host-pid-namespace','fixture',
          '--metadata-template',str(root/'template.json')]
    job=dict(id='report',out=str(report),argv=argv,seconds=1920,minimum_free_gib=10,
        outer_address_space_gib=4,outer_file_gib=8,requires_case_statuses={str(source):'ok'})
    value=dict(schema=entry.SCHEMA,automatic_retry=False,serial_case=str(serial),source_case=str(source),
        report_case=str(report),batch=dict(schema_version=1,deadline_utc='2030-01-01T00:00:00+00:00',
            watch=dict(case=str(owner),pid=11,starttime=110,seconds=90000),jobs=[job]))
    args=argparse.Namespace(manifest=root/'manifest.json',readiness=root/'readiness.json',metadata_out=root/'metadata')
    write(args.manifest,value);write(args.readiness,dict(status='ok',must_fix=[],source_stopped=True))
    status='ok'if accepted else'error';end=8000*10**9
    write(source/'result.json',dict(round=entry.fresh.ROUND,mode=entry.fresh.MODE,model=entry.fresh.MODEL,
        status=status,compile_accepted=accepted))
    write(source/'completion.json',dict(round=entry.fresh.ROUND,status=status,started_ns=100,
        ended_ns=end,required_artifacts_persisted=True,registry_persisted=True))
    write(source/'three-hour-record.json',dict(round=entry.fresh.ROUND,mode=entry.fresh.MODE,
        budget_seconds=10800,status=status,compile_accepted=accepted,persisted_pipeline_within_three_hours=accepted,
        persisted_pipeline_within_two_hours=False,started_ns=100,required_artifacts_and_registry_completed_ns=end,
        observed_ns=end+100,elapsed_seconds=8000.))
    return args,value,results


def probe(pid):return dict(pid=pid,starttime=999)if pid==os.getpid()else None


class ThreeHourReportWaiterTests(unittest.TestCase):
    def test_private_parent_limits_only_apply_to_the_exact_single_report(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder));modules=set(entry.batch.MODULES)
            with patch.object(entry.paths,'RESULTS_DIR',root):scope=entry.private_batch(value)
            self.assertEqual(scope['parent_limits'](value['batch']['jobs'][0]),(4,8))
            changed=copy.deepcopy(value['batch']['jobs'][0]);changed['out']=str(root/'different')
            with self.assertRaises(ValueError):scope['parent_limits'](changed)
            self.assertEqual(entry.batch.MODULES,modules)

    def test_false_source_acceptance_skips_without_a_scientific_case(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder));limits=[]
            with patch.object(entry.paths,'RESULTS_DIR',root):
                actual=entry.run(args,wait=lambda watch:{'closed_fixture':True},probe=probe,set_limits=limits.append)
            self.assertEqual(actual['status'],'skipped_unaccepted_source')
            self.assertEqual(actual['report_dispatches'],0);self.assertEqual(limits,[True])
            self.assertFalse(Path(value['report_case']).exists())
            self.assertFalse(Path(value['serial_case']).exists())

    def test_positive_source_launches_one_original_batch_after_wait(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder),accepted=True);calls=[];limits=[]
            def run_batch(a):
                calls.append(a)
                write(a.out/'result.json',dict(status='ok',jobs=[dict(process={'fixture_only':True})]))
                Path(value['report_case']).mkdir()
                return 0
            with patch.object(entry.paths,'RESULTS_DIR',root):
                scope=entry.private_batch(value);scope['run']=run_batch
                with patch.object(entry,'private_batch',return_value=scope):
                    actual=entry.run(args,wait=lambda watch:{'closed_fixture':True},probe=probe,set_limits=limits.append)
            self.assertEqual(actual['status'],'completed');self.assertEqual(actual['report_dispatches'],1)
            self.assertEqual(limits,[True,False]);self.assertEqual(len(calls),1)
            self.assertEqual(calls[0].out,Path(value['serial_case']))
            self.assertFalse(actual['source_admission']['usable_within_two_hours'])
            self.assertTrue(actual['source_admission']['usable_within_three_hours'])

    def test_missing_source_record_with_dependency_skip_does_not_create_report(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder));source=Path(value['source_case'])
            (source/'three-hour-record.json').unlink()
            write(Path(value['batch']['watch']['case'])/'result.json',
                  dict(jobs=[dict(out=str(source),status='dependency_skipped')]))
            with patch.object(entry.paths,'RESULTS_DIR',root):
                actual=entry.run(args,wait=lambda watch:{},probe=probe,set_limits=lambda value:None)
            self.assertEqual(actual['status'],'skipped_unaccepted_source')
            self.assertEqual(actual['report_dispatches'],0)

    def test_late_positive_record_is_rejected_and_no_report_runs(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder),accepted=True)
            record=entry.read(Path(value['source_case'])/'three-hour-record.json')
            record['observed_ns']=10801*10**9;write(Path(value['source_case'])/'three-hour-record.json',record)
            with patch.object(entry.paths,'RESULTS_DIR',root):
                actual=entry.run(args,wait=lambda watch:{},probe=probe,set_limits=lambda value:None)
            self.assertEqual(actual['status'],'error');self.assertEqual(actual['report_dispatches'],0)
            self.assertFalse(Path(value['report_case']).exists())

    def test_missing_actual_waiter_identity_is_rejected_before_metadata_or_report(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder))
            with patch.object(entry.paths,'RESULTS_DIR',root):
                with self.assertRaisesRegex(ValueError,'owner birth'):
                    entry.run(args,wait=lambda watch:{},probe=lambda pid:None,set_limits=lambda value:None)
            self.assertFalse(args.metadata_out.exists())

    def test_cancellation_finalization_masks_signals_and_restores_original_handlers(self):
        with TemporaryDirectory()as folder:
            args,value,root=fixture(Path(folder));original_save=entry.save
            handlers={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT)}
            def wait(watch):raise InterruptedError('controlled metadata cancellation')
            def save(path,value):
                if Path(path).name in('result.json','completion.json'):
                    for sig in handlers:
                        self.assertEqual(signal.getsignal(sig),signal.SIG_IGN)
                        signal.raise_signal(sig)
                original_save(path,value)
            with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'save',save):
                actual=entry.run(args,wait=wait,probe=probe,set_limits=lambda value:None)
            self.assertEqual(actual['status'],'error');self.assertEqual(actual['report_dispatches'],0)
            for sig,handler in handlers.items():self.assertIs(signal.getsignal(sig),handler)


if __name__=='__main__':unittest.main()
