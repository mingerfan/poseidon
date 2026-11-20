"""Synthetic terminal evidence and publication checks; no model or report reads."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_three_hour_terminal_metadata as entry
from pt_agent.experiments import bert_three_hour_metadata_observer as observer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    source, native, owner, report, report_queue = [root/name for name in
        (entry.FRESH, entry.SEARCH, entry.QUEUE, entry.REPORT, entry.REPORT_QUEUE)]
    result = dict(round=entry.fresh.ROUND, model=entry.fresh.MODEL, mode=entry.fresh.MODE,
                  status='ok', monitor_stopped=True, compile_accepted=True, frozen=True,
                  completed=dict.fromkeys(('prepare', 'checkpoint', 'step', 'confirm', 'math', 'emit', 'trace', 'compile'), 1))
    result['attempted'] = dict(result['completed']); result['metrics'] = dict(status='ok')
    start, end = 100, 8000*10**9+100
    done = dict(round=result['round'], status='ok', started_ns=start, ended_ns=end,
                registry_persisted=True, required_artifacts_persisted=True,
                complete_case_elapsed_ns=end-start, complete_case_elapsed_seconds=8000.)
    record = dict(round=result['round'], model=result['model'], mode=result['mode'], budget_seconds=10800,
                  status='ok', compile_accepted=True, started_ns=start, observed_ns=end+10**9,
                  required_artifacts_and_registry_completed_ns=end, elapsed_seconds=8001.,
                  completion_elapsed_seconds=8000., persisted_pipeline_within_three_hours=True,
                  persisted_pipeline_within_two_hours=False)
    write(source/'result.json', result); write(source/'completion.json', done)
    write(source/'three-hour-record.json', record)
    (source/'phase-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=12, starttime=120)]))+'\n')
    row = dict(run_id=source.name, round=result['round'], model=result['model'], mode=result['mode'],
               metrics=result['metrics'], git_rev=None, artifacts_dir=str(source), ts='fixture')
    (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
    (root/'runs.jsonl').write_text(json.dumps(dict(run_id=source.name, source=str(source/'result.json'),
                                               status='ok', budget_s=10800))+'\n')
    helpers = root/'helper.py'; helpers.write_text('# fixture metadata helper\n')
    destinations = [root/'delivery', root/'paper']; [p.mkdir() for p in destinations]
    manifest = dict(schema='bert-three-hour-terminal-metadata-manifest-v1', source_case=str(source),
        native_case=str(native), source_queue=str(owner), report_case=str(report), report_queue=str(report_queue),
        report_waiter_case=str(root/'report-waiter'), registry=str(root/'registry.jsonl'), runs=str(root/'runs.jsonl'),
        comparison_watch=dict(pid=20, starttime=200), report_watch=dict(pid=30, starttime=300),
        attribution_helper=str(helpers), quality_helper=str(helpers), destinations=list(map(str, destinations)),
        deadline_utc='2026-11-18T12:00:00+00:00')
    return manifest, result, done, record, row


class ThreeHourTerminalMetadataTests(unittest.TestCase):
    def test_real_schema_source_and_both_time_flags(self):
        with TemporaryDirectory() as folder:
            manifest, _, _, _, _ = fixture(Path(folder))
            proof = entry.source_proof(manifest, 'agent', process_reader=lambda pid: None)
            self.assertTrue(proof['usable_within_three_hours'])
            self.assertFalse(proof['usable_within_two_hours'])
            self.assertEqual(proof['registry_fields'], 8)
            self.assertEqual(proof['producer_closure']['saved_births'], 1)

    def test_live_zombie_and_permission_error_are_never_closed(self):
        with TemporaryDirectory() as folder:
            manifest, _, _, _, _ = fixture(Path(folder))
            for state in ('R', 'Z'):
                with self.assertRaises(entry.Pending):
                    entry.source_proof(manifest, 'agent', process_reader=lambda pid: dict(starttime=120, state=state))
            def denied(pid):
                raise PermissionError('fixture host namespace inaccessible')
            with self.assertRaises(PermissionError): entry.source_proof(manifest, 'agent', process_reader=denied)

    def test_duplicate_or_changed_registry_and_paper_are_rejected(self):
        for what in ('duplicate', 'metrics', 'paper'):
            with TemporaryDirectory() as folder:
                root=Path(folder); manifest, _, _, _, row=fixture(root)
                if what=='duplicate':
                    with (root/'registry.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                elif what=='metrics':
                    row['metrics']['status']='error'; (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
                else:
                    (root/'runs.jsonl').write_text(json.dumps(dict(run_id=entry.FRESH, source='/wrong', budget_s=10800))+'\n')
                with self.assertRaises(ValueError): entry.source_proof(manifest,'agent',process_reader=lambda pid:None)

    def test_late_positive_incomplete_pipeline_and_false_first7200_claim_rejected(self):
        with TemporaryDirectory() as folder:
            _, result, done, record, _=fixture(Path(folder))
            later=copy.deepcopy(record); later.update(observed_ns=later['started_ns']+10801*10**9, elapsed_seconds=10801.)
            with self.assertRaisesRegex(ValueError,'every accepted'):entry.clock_binding(result,done,later)
            wrong=copy.deepcopy(record);wrong['persisted_pipeline_within_two_hours']=True
            with self.assertRaisesRegex(ValueError,'first7200'):entry.clock_binding(result,done,wrong)
            incomplete=copy.deepcopy(result);incomplete['completed']['math']=0
            with self.assertRaisesRegex(ValueError,'every accepted'):entry.clock_binding(incomplete,done,record)

    def test_positive_clock_with_failed_original_paper_row_cannot_be_published(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);manifest,_,_,_,_=fixture(root)
            (root/'runs.jsonl').write_text(json.dumps(dict(run_id=entry.FRESH,
                source=str(Path(manifest['source_case'])/'result.json'),status='driver_error',budget_s=10800))+'\n')
            with self.assertRaisesRegex(ValueError,'successful paper row'):
                entry.source_proof(manifest,'agent',process_reader=lambda pid:None)

    def test_actual_queue_result_without_model_or_metrics_binds_registry(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);case=root/'queue';case.mkdir()
            result=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',
                        status='completed_with_observations',jobs=[dict(id='native',status='failed')])
            done=dict(round=result['round'],status=result['status'],registry_persisted=True,required_artifacts_persisted=True)
            write(case/'result.json',result);write(case/'completion.json',done)
            row=dict(round=result['round'],mode=result['mode'],model='original-model-experiment-queue',
                     metrics=dict(status=result['status'],jobs=result['jobs']))
            actual,_=entry.closed(case,{case.name:row})
            self.assertEqual(actual,result)
            row['metrics']['jobs']=[]
            with self.assertRaisesRegex(ValueError,'schema and jobs'):entry.closed(case,{case.name:row})

    def test_native_timeout_is_preserved_with_its_own_compile_clock(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);manifest,_,done,_,_=fixture(root);case=Path(manifest['native_case'])
            result=dict(round=entry.native.ROUND,model=entry.native.MODEL,mode=entry.native.MODE,
                status='error',monitor_stopped=True,phases=dict(compile=dict(native_process=dict(
                timed_out=True,exit_code=-15,signal=None,remaining_pids=[],signals_sent=[dict(pid=70,signal=15)]))))
            write(case/'result.json',result)
            write(case/'completion.json',dict(done,round=result['round'],status='error'))
            (case/'native-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=70,starttime=700)]))+'\n')
            row=dict(run_id=case.name,round=result['round'],model=result['model'],mode=result['mode'],git_rev=None,
                     metrics=dict(status='error'),artifacts_dir=str(case),ts='fixture')
            with (root/'registry.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            with (root/'runs.jsonl').open('a') as stream:stream.write(json.dumps(dict(run_id=case.name,
                     source=str(case/'result.json'),status='timeout',budget_s=10800,compile_s=10800.4))+'\n')
            proof=entry.source_proof(manifest,'native',process_reader=lambda pid:None)
            self.assertTrue(proof['native_timed_out'])
            self.assertEqual(proof['paper_record_original']['compile_s'],10800.4)
            self.assertEqual(proof['outer_restore_compile_budget_s'],13200)
            self.assertTrue(proof['original_math_quality_unmeasured'])

    def test_complete_three_job_queue_with_independent_native_failure_is_accepted(self):
        from pt_agent.tests.test_bert_three_hour_comparison_queue import fixture as queue_fixture
        with TemporaryDirectory() as folder:
            root=Path(folder);manifest,_,done,_,_=fixture(root);owner=Path(manifest['source_queue'])
            declared=queue_fixture(root)
            for index,key in ((1,'source_case'),(2,'native_case')):
                job=declared['jobs'][index];previous=job['out'];job['out']=manifest[key]
                job['argv']=[manifest[key] if x==previous else x for x in job['argv']]
            jobs=[dict(j,status='failed'if i==2 else'ok',experiment_status='error'if i==2 else'ok')
                  for i,j in enumerate(declared['jobs'])]
            result=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',
                        status='completed_with_observations',jobs=jobs)
            write(owner/'result.json',result);write(owner/'manifest.json',declared)
            write(owner/'completion.json',dict(done,round=result['round'],status=result['status']))
            write(owner/'supervisor.json',dict(pid=20,process=dict(pid=20,starttime=200)))
            for job in jobs:write(owner/(job['id']+'-command.json'),job['argv'])
            (owner/'agent-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=21,starttime=210)]))+'\n')
            row=dict(run_id=owner.name,round=result['round'],model='original-model-experiment-queue',mode=result['mode'],
                metrics=dict(status=result['status'],jobs=jobs),artifacts_dir=str(owner),ts='fixture',git_rev=None)
            with (root/'registry.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            proof=entry.queue_proof(manifest,process_reader=lambda pid:None)
            self.assertTrue(proof['unrelated_failures_preserved'])
            self.assertEqual(proof['original_job_outcomes'][-1]['status'],'failed')

    def test_attribution_names_and_clock_are_separate_from_original_two_hour_files(self):
        with TemporaryDirectory() as folder:
            manifest,_,_,_,_=fixture(Path(folder))
            data=dict(inflight=False,source_two_hour_record='/old',stage_rows=[],iteration_observations=[],
                      missing=dict(search_Count_only_s=None),original_cold_case_s=8000.)
            with patch.object(entry,'load_helper',return_value=SimpleNamespace(build=lambda case:data)):
                blobs=entry.attribution(manifest,dict(usable_within_three_hours=True))
            self.assertEqual(set(blobs),set(entry.ATTRIBUTION_OUTPUTS))
            value=json.loads(blobs[entry.ATTRIBUTION_OUTPUTS[0]])
            self.assertNotIn('source_two_hour_record',value)
            self.assertTrue(value['source_three_hour_record'].endswith('three-hour-record.json'))
            self.assertIsNone(value['missing']['search_Count_only_s'])

    def test_independent_quality_preserves_three_hour_acceptance_false_two_hour_and_quality_rejection(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);manifest,fresh_result,done,_,fresh_row=fixture(root)
            source,owner,report,queue=[Path(manifest[k])for k in('source_case','source_queue','report_case','report_queue')]
            admission=dict(fresh_three_hour_accepted=True,fresh_three_hour_record=str(source/'three-hour-record.json'),
                           source_case=str(source),compile_case=str(source),report_feedback_permitted=False,
                           source_producer_closure=dict(supervisor_case=str(owner),all_serial_jobs_terminal=True))
            run=dict(round=entry.report_entry.ROUND,model=entry.fresh.MODEL,mode=entry.report_entry.MODE,
                     status='ok',monitor_stopped=True,source_case=str(source),compile_case=str(source),
                     admission_passed=True,source=admission,report_accepted=True,attempted_report=1,
                     shell=dict(frozen_program_path_rebased_only=True,original_case_writes=0),
                     shell_recheck=dict(frozen_program_path_rebased_only=True,original_case_writes=0),metrics=dict(status='ok'))
            write(report/'admission.json',admission)
            worker=dict(phase='report',model=entry.fresh.MODEL,**dict.fromkeys(('llm_calls','native_test_reads',
                         'compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'),0))
            write(report/'worker/worker_result.json',worker)
            write(report/'session/REPORT_ACCESS.json',dict(program=str(report/'session/program.ptdsl'),
                  complete=True,result=str(report/'worker/worker_result.json')))
            values={report:run}
            for p in(owner,queue):
                values[p]=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',
                               status='completed_with_observations'if p==owner else'ok',jobs=[])
                write(p/'supervisor.json',dict(pid=50,process=dict(pid=50,starttime=500)))
            rows=[fresh_row]
            for p,value in values.items():
                write(p/'result.json',value)
                write(p/'completion.json',dict(done,round=value['round'],status=value['status']))
                (p/'job-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=60,starttime=600)]))+'\n')
                rows.append(dict(run_id=p.name,round=value['round'],model=value.get('model','original-model-experiment-queue'),
                  mode=value['mode'],metrics=value.get('metrics',dict(status=value['status'],jobs=[])),
                  git_rev=None,artifacts_dir=str(p),ts='fixture'))
            (root/'registry.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
            def queue_binding(p,*args,**kwargs):return dict(job_resources=str(p/'job-resources.jsonl'))
            numeric=dict(complete_scalar_quality=True,task_policy_passed=False)
            helper=SimpleNamespace(queue_binding=queue_binding,quality=lambda *args:numeric,
               quality_admission=lambda n,r:dict(eligible_for_final_quality=True,quality_policy_accepted=False))
            with patch.object(entry,'load_helper',return_value=helper):
                quality=entry.final_quality(manifest,process_reader=lambda pid:None)
                self.assertTrue(quality['fresh_usable_within_three_hours'])
                self.assertFalse(quality['fresh_usable_within_two_hours'])
                self.assertTrue(quality['eligible_for_final_quality'])
                self.assertFalse(quality['quality_policy_accepted'])
                self.assertFalse(quality['report_feedback_to_Agent'])
                admission['fresh_three_hour_record']='/wrong';write(report/'admission.json',admission)
                with self.assertRaisesRegex(ValueError,'source admission'):entry.final_quality(manifest,process_reader=lambda pid:None)

    def test_quality_waits_for_all_four_completion_markers_before_registry_lookup(self):
        with TemporaryDirectory()as folder:
            manifest,_,_,_,_=fixture(Path(folder))
            write(Path(manifest['report_case'])/'result.json',dict(status='running'))
            with patch.object(entry.closure,'registry_rows')as registry:
                with self.assertRaises(entry.Pending):entry.final_quality(manifest)
                registry.assert_not_called()

    def test_observer_publishes_closed_failure_attribution_and_comparison_without_fake_quality(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);manifest,_,_,_,_=fixture(root)
            waiter=Path(manifest['report_waiter_case'])
            write(waiter/'result.json',dict(status='skipped',reason='fresh_compile_not_accepted'))
            write(waiter/'completion.json',dict(status='skipped',metadata_persisted=True))
            proofs={kind:dict(scientific_status='error',usable_within_three_hours=False)for kind in('agent','native')}
            with patch.object(entry,'source_proof',side_effect=lambda m,k,**kw:proofs[k]), \
                    patch.object(entry,'queue_proof',return_value=dict(status='completed_with_observations')), \
                    patch.object(entry,'attribution',return_value={name:'{}\n' for name in entry.ATTRIBUTION_OUTPUTS}), \
                    patch.object(entry,'final_quality')as quality:
                value=observer.observe(manifest,root/'observer',process_reader=lambda pid:None,
                    now=lambda:datetime(2026,11,17,tzinfo=timezone.utc),sleep=lambda seconds:self.fail('No wait needed'))
            self.assertEqual(value['status'],'ok');quality.assert_not_called()
            self.assertEqual(value['quality_publications'],0)
            self.assertEqual(value['comparison_publications'],2)
            self.assertEqual(value['attribution_publications'],6)
            self.assertFalse(any(p.name=='bert_fresh_3h_final_quality.json'for p in root.rglob('*')))

    def test_observer_masks_terminal_signals_through_persistence_and_restores_handlers(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);manifest,_,_,_,_=fixture(root)
            before={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
            saved=entry.save;calls=[]
            def persist(path,data):
                if Path(path).name in('result.json','completion.json'):
                    self.assertTrue(all(signal.getsignal(s)==signal.SIG_IGN for s in before))
                    os.kill(os.getpid(),signal.SIGTERM);os.kill(os.getpid(),signal.SIGINT);calls.append(Path(path).name)
                return saved(path,data)
            with patch.object(entry,'source_proof',side_effect=entry.Pending('fixture pending')), \
                    patch.object(entry,'save',side_effect=persist):
                value=observer.observe(manifest,root/'observer',process_reader=lambda pid:dict(starttime=300),
                    now=lambda:datetime(2026,11,19,tzinfo=timezone.utc),sleep=lambda seconds:None)
            self.assertEqual(value['status'],'deadline_without_all_terminal_sources')
            self.assertEqual(calls,['result.json','completion.json'])
            self.assertEqual({s:signal.getsignal(s)for s in before},before)


if __name__=='__main__':unittest.main()
