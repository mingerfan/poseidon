import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import experiment_delivery_refresh as subject


class MetadataRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.results=self.root/'results';self.results.mkdir()
        self.case=self.results/'case';self.case.mkdir()
        self.out=self.root/'publication'
        self.manifest=self.root/'manifest.json'
        self.row=dict(id='case',case=str(self.case),model='VGG16',system='Agent',variant_role='primary-method')
        self.write(self.manifest,dict(schema='explicit-experiment-metadata-cases-v1',cases=[self.row]))
        self.start=datetime(2026,11,16,0,0,tzinfo=timezone.utc).timestamp()

    def write(self,path,value):
        path.write_text(json.dumps(value))

    def args(self,**updates):
        values=dict(manifest=self.manifest,publish_dir=self.out,results_root=self.results,
                    poll_seconds=60,until='2026-11-16T00:02:00Z',once=False)
        values.update(updates)
        return argparse.Namespace(**values)

    def test_saved_terminal_and_phase_clocks_publish_without_payloads(self):
        self.write(self.case/'result.json',dict(status='ok',compiled_bootstraps=211,api_delta=0,
             messages=['FORBIDDEN_PROMPT'],worker={'source':'FORBIDDEN_DSL'},weights='FORBIDDEN_WEIGHT',
             attempted={'compile':1,'FORBIDDEN '+('x'*100):'payload'},completed={'compile':1},
             processes=[dict(stage='compile',argv=['FORBIDDEN_ARGV'],observation=dict(exit_code=0,
                 started_ns=10,ended_ns=30,peak_rss_kib_by_pid={'17':100},remaining_pids=[]))]))
        self.write(self.case/'completion.json',dict(status='ok',complete_case_elapsed_seconds=2.,
             registry_persisted=True,required_artifacts_persisted=True,
             phase_spans=[dict(stage_id='compile',parent_id=None,inclusive_ns=20,exclusive_ns=20,status='ok',metadata='FORBIDDEN_EXTRA')]))
        with patch('subprocess.Popen',side_effect=AssertionError('No child execution')):
            self.assertEqual(subject.execute(self.args(once=True),now=lambda:self.start),1)
        text=(self.out/'LIVE_CASE_METADATA.json').read_text()
        self.assertNotIn('FORBIDDEN',text)
        saved=json.loads(text)['cases'][0]
        self.assertTrue(saved['terminal_observed'])
        self.assertEqual(saved['result']['compiled_bootstraps'],211)
        self.assertEqual(saved['result']['processes'][0]['single_PID_HWM_KiB'],100)
        self.assertEqual(saved['stage_clocks'][0]['inclusive_ns'],20)
        self.assertTrue((self.out/'LIVE_CASE_METADATA.csv').is_file())

    def test_missing_cases_are_unobserved_not_zero_or_success(self):
        row=subject.snapshot([self.row],self.start)['cases'][0]
        self.assertIsNone(row['result']);self.assertIsNone(row['completion'])
        self.assertFalse(row['terminal_observed'])

    def test_partial_json_is_a_read_error_not_a_terminal(self):
        (self.case/'result.json').write_text('{"status":')
        row=subject.snapshot([self.row],self.start)['cases'][0]
        self.assertEqual(row['metadata_read_error_class'],'JSONDecodeError')
        self.assertFalse(row['terminal_observed'])

    def test_unknown_nested_keys_and_nonfinite_counts_are_not_published(self):
        record=subject.summarize_record(dict(status='running',messages=['payload'],
             compiled_bootstraps=float('nan'),api_delta=float('inf'),error='ValueError: payload',
             attempted={'compile':1,'tune_cache':'notnumeric','valid_bool':False},
             limits={'memory_gib':12,'argv':'payload'}))
        self.assertNotIn('messages',record);self.assertNotIn('compiled_bootstraps',record)
        self.assertNotIn('api_delta',record);self.assertEqual(record['error_class'],'ValueError')
        self.assertEqual(record['attempted'],{'compile':1,'valid_bool':False})
        self.assertEqual(record['limits'],{'memory_gib':12})

    def test_manifest_refuses_external_paths_or_payload_fields(self):
        value=json.loads(self.manifest.read_text())
        value['cases'][0]['case']=str(self.root)
        with self.assertRaises(ValueError):subject.validate_manifest(value,self.results)
        value['cases'][0]=dict(self.row,messages='payload')
        with self.assertRaises(ValueError):subject.validate_manifest(value,self.results)

    def test_publication_canonical_path_cannot_overlap_raw_evidence(self):
        alias=self.root/'alias';alias.symlink_to(self.case,target_is_directory=True)
        with self.assertRaises(ValueError):subject.execute(self.args(publish_dir=alias,once=True),now=lambda:self.start)
        self.assertFalse((self.case/'LIVE_CASE_METADATA.json').exists())

    def test_final_symlink_refused_and_destination_preserved(self):
        target=self.root/'retained';target.write_text('keep')
        self.out.mkdir();link=self.out/'LIVE_CASE_METADATA.json';link.symlink_to(target)
        with self.assertRaises(ValueError):subject.execute(self.args(once=True),now=lambda:self.start)
        self.assertEqual(target.read_text(),'keep')

    def test_fake_clock_stops_at_finite_deadline_without_children(self):
        clock=[self.start];delays=[]
        def sleep(value):delays.append(value);clock[0]+=value
        with patch('subprocess.Popen',side_effect=AssertionError('No child execution')):
            polls=subject.execute(self.args(),now=lambda:clock[0],sleep=sleep)
        self.assertEqual(polls,2);self.assertEqual(delays,[60,60])
        self.assertEqual(clock[0],self.start+120)

    def test_deadline_and_poll_validation_precede_publication(self):
        for kwargs in [dict(poll_seconds=True),dict(poll_seconds=1),dict(until='2026-11-18T00:00:00Z'),
                       dict(until='2026-11-16T00:02:00'),dict(until='2026-11-15T00:00:00Z')]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                subject.execute(self.args(**kwargs),now=lambda:self.start)
        self.assertFalse(self.out.exists())

    def test_metadata_symlinks_are_read_errors(self):
        target=self.root/'different.json';self.write(target,dict(status='ok'))
        (self.case/'result.json').symlink_to(target)
        row=subject.snapshot([self.row],self.start)['cases'][0]
        self.assertEqual(row['metadata_read_error_class'],'ValueError')
        self.assertFalse(row['terminal_observed'])


if __name__=='__main__':unittest.main()
