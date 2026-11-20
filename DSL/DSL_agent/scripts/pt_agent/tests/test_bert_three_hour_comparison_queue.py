"""Finite private scheduler checks, with no real wait, model or compiler."""
import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments import bert_three_hour_comparison_queue as entry


def fixture(root):
    out=root/'queue';watch=root/'old-two-hour-controller'
    paper=root/'runs.jsonl';archive=root/'raw-archive';agent=root/'fresh-agent';native=root/'native-search'
    commands=[
        [sys.executable,'-B','-m',entry.archive.MODULE,'--out',str(archive),
         '--host-pid-namespace','fixture','--seconds','1740','--run','--no-future-restored-input-consumers'],
        [sys.executable,'-B','-m',entry.agent.MODULE,'--out',str(agent),'--paper-runs',str(paper)],
        [sys.executable,'-B','-m',entry.native.MODULE,'--out',str(native),'--paper-runs',str(paper),
         '--seconds','13200','--native-seconds','10800']]
    jobs=[]
    for index,(argv,case,seconds) in enumerate(zip(commands,[archive,agent,native],[1860,10920,13320])):
        jobs.append(dict(id=['archive','agent','native'][index],argv=argv,out=str(case),seconds=seconds,
                         requires=[]if index==0 else['archive']))
    return dict(schema_version=1,watch=dict(case=str(watch),pid=12345,starttime=987,seconds=90000),
                jobs=jobs,direct_human_budget_seconds=10800,original_two_hour_batch_untouched=True)


class ThreeHourQueueTests(unittest.TestCase):
    def test_private_admission_does_not_change_original_scheduler(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);manifest=fixture(root);old=set(entry.original.MODULES)
            values=entry.namespace()
            self.assertEqual(values['validate'](manifest,root),manifest)
            self.assertEqual(entry.original.MODULES,old)
            self.assertNotIn(entry.agent.MODULE.removeprefix(entry.original.PREFIX),old)
            self.assertIs(values['run'].__code__,entry.original.run.__code__)
            self.assertIs(values['run'].__globals__,values)

    def test_no_unrelated_module_or_shortened_native_allocation(self):
        changes=[lambda m:m['jobs'][2]['argv'].__setitem__(3,'pt_agent.experiments.qwen_bridge_agent'),
                 lambda m:m['jobs'][2].update(seconds=10920),
                 lambda m:m['jobs'][2]['argv'].__setitem__(-1,'7200'),
                 lambda m:m['jobs'][2].update(requires=['agent']),
                 lambda m:m.update(direct_human_budget_seconds=7200)]
        for update in changes:
            with TemporaryDirectory()as folder:
                root=Path(folder);value=fixture(root);update(value)
                with self.assertRaises(ValueError):entry.namespace()['validate'](value,root)

    def test_agent_failure_does_not_skip_independent_native_trial(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);value=fixture(root);out=root/'queue';out.mkdir()
            calls=[]
            def runner(argv,**kwargs):
                from pt_agent.experiments.r10_paper_metrics import write_json
                target=Path(argv[argv.index('--out')+1]);target.mkdir()
                is_agent=argv[3]==entry.agent.MODULE
                write_json(target/'result.json',dict(status='error'if is_agent else'ok'))
                calls.append(argv[3])
                return dict(exit_code=1 if is_agent else 0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
            rows=entry.namespace()['execute_jobs'](value['jobs'],out=out,results_root=root,
                runner=runner,free=lambda p:100*1024**3)
            self.assertEqual(calls,[entry.archive.MODULE,entry.agent.MODULE,entry.native.MODULE])
            self.assertEqual([row['status']for row in rows],['ok','failed','ok'])

    def test_archive_failure_skips_both_scientific_runs(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);value=fixture(root);out=root/'queue';out.mkdir();calls=[]
            def runner(argv,**kwargs):
                calls.append(argv[3])
                return dict(exit_code=1,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
            rows=entry.namespace()['execute_jobs'](value['jobs'],out=out,results_root=root,
                runner=runner,free=lambda p:100*1024**3)
            self.assertEqual(calls,[entry.archive.MODULE])
            self.assertEqual([row['status']for row in rows],['failed','dependency_skipped','dependency_skipped'])


if __name__=='__main__':unittest.main()
