"""Recovered lineage and automatic report/publication; no dataset execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_recovery_frozen_report as report
from pt_agent.experiments import bert_recovery_report_waiter as waiter
from pt_agent.experiments import bert_recovery_terminal_metadata as metadata
from pt_agent.experiments import bert_recovery_metadata_observer as observer
from pt_agent.tests.test_bert_fresh_frozen_report_3h import owner_fixture, write
from pt_agent.tests.test_bert_three_hour_report_waiter import fixture as waiter_fixture, probe
from pt_agent.tests.test_bert_three_hour_terminal_metadata import fixture as metadata_fixture


class RecoveryFollowupTests(unittest.TestCase):
    def test_old_interrupted_source_cannot_gain_recovered_report_admission(self):
        from argparse import Namespace
        args=Namespace(source_case=Path('/fixture/old'),supervisor_case=Path('/fixture/old-owner'))
        with patch.object(report.base,'closed_case')as reader:
            with self.assertRaisesRegex(ValueError,'new independent'):report.admission(args)
            reader.assert_not_called()

    def test_recovered_owner_registry_uses_recovery_validation(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,result,_=owner_fixture(root);called=[]
            with patch.object(report.paths,'RESULTS_DIR',root),patch.object(report.recovery,'namespace',
                    return_value={'validate':lambda value,path:called.append((value,path))}):
                row=report.queue_registry_row(args.supervisor_case,result)
            self.assertEqual(row['metrics']['jobs'],result['jobs'])
            self.assertEqual(len(called),1);self.assertEqual(called[0][1],root)

    def test_recovered_producer_still_rejects_live_controller_or_source_birth(self):
        for live,birth in((11,110),(12,120)):
            with TemporaryDirectory()as folder:
                root=Path(folder);args,_,_=owner_fixture(root)
                scope=report.admission_namespace()
                with patch.object(report.paths,'RESULTS_DIR',root),patch.object(report.recovery,'namespace',
                        return_value={'validate':lambda value,path:value}):
                    with self.assertRaisesRegex(ValueError,'birth remains'):
                        scope['producer_closure'](args.source_case,args.supervisor_case,args.host_pid_namespace,
                            process_reader=lambda pid:dict(starttime=birth)if pid==live else None)

    def test_report_keeps_original_worker_shell_and1800second_limit(self):
        scope=report.namespace()
        self.assertEqual(scope['SECONDS'],1800);self.assertEqual(scope['MODULE'],report.MODULE)
        self.assertIs(scope['admission'],report.admission)
        for name in('command','check_report','resources','claim','registry_row'):
            self.assertIs(scope[name].__code__,getattr(report.base,name).__code__)

    def test_false_recovered_source_creates_no_report(self):
        with TemporaryDirectory()as folder:
            args,value,root=waiter_fixture(Path(folder));value['batch']['jobs'][0]['argv'][3]=report.MODULE
            write(args.manifest,value);scope=waiter.namespace()
            with patch.object(report.paths,'RESULTS_DIR',root):
                result=scope['run'](args,wait=lambda watch:{},probe=probe,set_limits=lambda flag:None)
            self.assertEqual(result['status'],'skipped_unaccepted_source')
            self.assertEqual(result['report_dispatches'],0)
            self.assertFalse(Path(value['report_case']).exists())

    def test_accepted_recovered_source_dispatches_once_with_unchanged_4_8_limits(self):
        with TemporaryDirectory()as folder:
            args,value,root=waiter_fixture(Path(folder),accepted=True)
            value['batch']['jobs'][0]['argv'][3]=report.MODULE;write(args.manifest,value)
            scope=waiter.namespace();calls=[]
            with patch.object(report.paths,'RESULTS_DIR',root):
                batch=scope['private_batch'](value)
                self.assertEqual(batch['parent_limits'](value['batch']['jobs'][0]),(4,8))
                def run_case(a):
                    calls.append(a);write(a.out/'result.json',dict(status='ok',jobs=[dict(process={})]))
                    Path(value['report_case']).mkdir();return 0
                batch['run']=run_case;scope['private_batch']=lambda value:batch
                result=scope['run'](args,wait=lambda watch:{},probe=probe,set_limits=lambda flag:None)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(calls),1)
            self.assertEqual(result['report_dispatches'],1)
            self.assertFalse(result['source_admission']['usable_within_two_hours'])

    def test_recovered_metadata_source_retains_actual3h_and2h_flags(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);manifest,result,done,record,row=metadata_fixture(root)
            old=Path(manifest['source_case']);new=root/metadata.FRESH;old.rename(new)
            manifest['source_case']=str(new)
            row.update(run_id=new.name,artifacts_dir=str(new))
            (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
            (root/'runs.jsonl').write_text(json.dumps(dict(run_id=new.name,source=str(new/'result.json'),status='ok',budget_s=10800))+'\n')
            proof=metadata.namespace().source_proof(manifest,'agent',process_reader=lambda pid:None)
            self.assertTrue(proof['usable_within_three_hours']);self.assertFalse(proof['usable_within_two_hours'])

    def test_new_publications_preserve_interrupted_comparison_and_quality_files(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);dest=[root/'a',root/'b'];[p.mkdir()for p in dest]
            scope=observer.namespace()
            for directory in dest:
                (directory/'bert_3h_comparison_terminal.json').write_text('old interrupted comparison')
                (directory/'bert_fresh_3h_final_quality.json').write_text('old unrelated quality')
            manifest=dict(destinations=list(map(str,dest)))
            scope['publish'](manifest,'bert_3h_comparison_terminal.json',dict(recovered=True))
            scope['publish'](manifest,'bert_fresh_3h_final_quality.json',dict(recovered=True))
            for directory in dest:
                self.assertEqual((directory/'bert_3h_comparison_terminal.json').read_text(),'old interrupted comparison')
                self.assertEqual((directory/'bert_fresh_3h_final_quality.json').read_text(),'old unrelated quality')
                self.assertTrue(json.loads((directory/'bert_3h_recovery_comparison_terminal.json').read_text())['recovered'])
            self.assertEqual(scope['evidence'].FRESH,metadata.FRESH)
            self.assertIs(scope['evidence'].report_entry,report)


if __name__=='__main__':unittest.main()
