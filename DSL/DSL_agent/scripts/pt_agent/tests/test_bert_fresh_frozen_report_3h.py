"""Three-hour source and real-schema queue fixtures; no report data reads."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_fresh_frozen_report_3h as entry
from pt_agent.tests import test_bert_fresh_frozen_report as oldfixtures
from pt_agent.tests.test_bert_three_hour_comparison_queue import fixture as queue_fixture

arguments,clean,write=oldfixtures.arguments,oldfixtures.clean,oldfixtures.write


def admission_fixture(root):
    # Reuse an existing synthetic helper, without running or changing old tests.
    args,result,expected=oldfixtures.IsolatedReportTests()._admission_fixture(root)
    result.update(round=entry.fresh.ROUND,mode=entry.fresh.MODE)
    end=8000*10**9
    write(args.source_case/'completion.json',dict(started_ns=0,ended_ns=end))
    write(args.source_case/'three-hour-record.json',dict(round=entry.fresh.ROUND,mode=entry.fresh.MODE,
        status='ok',compile_accepted=True,persisted_pipeline_within_three_hours=True,
        persisted_pipeline_within_two_hours=False,started_ns=0,
        required_artifacts_and_registry_completed_ns=end,observed_ns=end+10**9))
    return args,result,expected


def owner_fixture(root):
    manifest=queue_fixture(root);args=arguments(root)
    args.source_case=Path(manifest['jobs'][1]['out']);args.compile_case=args.source_case
    owner=args.supervisor_case;source=args.source_case;source.mkdir()
    jobs=[]
    for index,declared in enumerate(manifest['jobs']):
        job=dict(declared,status='failed'if index==2 else'ok',
            experiment_status='error'if index==2 else'ok',cleanup_unknown=False,
            process=dict(clean(),exit_code=1 if index==2 else 0))
        jobs.append(job)
    result=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',
                status='completed_with_observations',jobs=jobs)
    write(owner/'result.json',result)
    write(owner/'completion.json',dict(round=result['round'],status=result['status'],
        required_artifacts_persisted=True,registry_persisted=True))
    write(owner/'manifest.json',manifest)
    write(owner/'agent-command.json',jobs[1]['argv'])
    write(owner/'supervisor.json',dict(pid=11,process=dict(pid=11,starttime=110)))
    sample=dict(ts_ns=150,outer_process=dict(pid=11,starttime=110),processes=[dict(pid=12,starttime=120)])
    (owner/'agent-resources.jsonl').write_text(json.dumps(sample)+'\n')
    registry=dict(run_id=owner.name,round=result['round'],model='original-model-experiment-queue',
        mode=result['mode'],git_rev=None,metrics=dict(status=result['status'],jobs=jobs),
        artifacts_dir=str(owner),ts='fixture')
    (root/'registry.jsonl').write_text(json.dumps(registry)+'\n')
    return args,result,registry


class ThreeHourFrozenReportTests(unittest.TestCase):
    def test_accepted_three_hour_source_can_report_even_when_completion_exceeded_two_hours(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,result,expected=admission_fixture(root)
            with patch.object(entry.paths,'RESULTS_DIR',root), \
                    patch.object(entry.base,'closed_case',return_value=(result,{'unique':True})), \
                    patch.object(entry.deferred,'compile_source',return_value=expected), \
                    patch.object(entry.fresh,'native_artifacts',return_value={'accepted':True}), \
                    patch.object(entry,'producer_closure',return_value={'saved_births_checked':3}):
                proof=entry.admission(args)
            self.assertTrue(proof['fresh_three_hour_accepted'])
            self.assertEqual(proof['report_samples'],872)
            self.assertFalse(proof['report_feedback_permitted'])
            self.assertIn('10800',proof['report_clock_scope'])

    def test_late_three_hour_record_is_rejected_before_math_or_report(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,result,_=admission_fixture(root)
            record=entry.base.read(args.source_case/'three-hour-record.json')
            record['observed_ns']=10801*10**9;write(args.source_case/'three-hour-record.json',record)
            with patch.object(entry.paths,'RESULTS_DIR',root), \
                    patch.object(entry.base,'closed_case',return_value=(result,{})), \
                    patch.object(entry.deferred,'compile_source')as math:
                with self.assertRaisesRegex(ValueError,'three-hour'):entry.admission(args)
                math.assert_not_called()

    def test_complete_actual_queue_schema_without_top_level_model_or_metrics_is_accepted(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,result,_=owner_fixture(root)
            self.assertNotIn('model',result);self.assertNotIn('metrics',result)
            with patch.object(entry.paths,'RESULTS_DIR',root):
                proof=entry.producer_closure(args.source_case,args.supervisor_case,
                    args.host_pid_namespace,process_reader=lambda pid:None)
            self.assertEqual(proof['saved_births_checked'],2)
            self.assertTrue(proof['all_serial_jobs_terminal'])
            self.assertTrue(proof['unrelated_serial_failures_preserved'])

    def test_source_and_controller_births_must_all_be_gone(self):
        for live,birth in[(11,110),(12,120)]:
            with TemporaryDirectory()as folder:
                root=Path(folder);args,_,_=owner_fixture(root)
                with patch.object(entry.paths,'RESULTS_DIR',root):
                    with self.assertRaisesRegex(ValueError,'birth remains'):
                        entry.producer_closure(args.source_case,args.supervisor_case,args.host_pid_namespace,
                            process_reader=lambda pid:dict(starttime=birth)if pid==live else None)

    def test_registry_model_or_jobs_change_is_rejected(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,result,row=owner_fixture(root)
            row['model']='wrong-queue-model';(root/'registry.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(entry.paths,'RESULTS_DIR',root):
                with self.assertRaisesRegex(ValueError,'registry'):entry.queue_registry_row(args.supervisor_case,result)

    def test_original_report_shell_worker_checks_and_safe_finalization_remain_delegated(self):
        values=entry.namespace()
        for name in('shell_bindings','command','check_report','resources','claim','registry_row'):
            self.assertIs(values[name].__code__,getattr(entry.base,name).__code__)
        self.assertEqual(values['ROUND'],entry.ROUND)
        self.assertEqual(values['SECONDS'],1800)
        args=arguments(Path('/fixture'));argv=values['command'](args,{'model':'/model','inputs':'/inputs'},200)
        self.assertEqual(entry._option(argv,'--phase'),'report')
        self.assertEqual(entry._option(argv,'--session-dir'),str(args.out/'session'))
        self.assertNotIn(str(args.source_case/'session'),argv)


if __name__=='__main__':unittest.main()
