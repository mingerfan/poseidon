"""Synthetic metadata and fake-clock checks; no model or experiment is started."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import live_execution_observer as observer


class ObserverChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.batch, self.watch, self.active = [self.root/name for name in ('batch','watched','active')]
        self.publish = self.root/'publication'
        self.ledger, self.allowance = self.root/'ledger.jsonl', self.root/'allowance.json'
        self.args = SimpleNamespace(batch_case=self.batch, watch_case=self.watch,
            ledger=self.ledger, allowance=self.allowance, publish_dir=self.publish,
            poll_seconds=30, seconds=100)
        self.write(self.batch/'manifest.json', dict(jobs=[dict(id='one', out=str(self.active),
            seconds=1800, argv=['python3','-B','-m','trusted','--memory-gib','12','--case-gib','32'])]))
        self.write(self.allowance, dict(version=1, ledger=str(self.ledger.resolve()),
            ceiling=600, observed_attempts=574, increment=120, authorization='DO_NOT_PUBLISH'))
        self.ledger.write_text('{"status":200,"purpose":"DO_NOT_PUBLISH"}\n{"status":null}\n')
        self.write(self.watch/'progress.json', dict(status='running', attempted={'agent':1}, completed={}))
        self.write(self.watch/'work/agent/SESSION.json', dict(phase='tune', iteration=6,
            calls=5, max_iters=6, tuner=True, messages=[dict(content='PRIVATE_PROMPT_MARKER')],
            pending=dict(source='PRIVATE_PROGRAM_MARKER'), report='PRIVATE_REPORT_MARKER'))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def close(self, case, status='ok'):
        self.write(case/'result.json', dict(status=status))
        self.write(case/'completion.json', dict(status=status, required_artifacts_persisted=True,
            registry_persisted=True, complete_case_elapsed_seconds=10.5))

    def test_selector_skips_ignored_conversations_without_decoding(self):
        original = json.loads
        decoded = []
        def loads(value, **kwargs):
            decoded.append(value)
            return original(value, **kwargs)
        with patch.object(observer.json, 'loads', side_effect=loads):
            result = observer.projection(self.watch/'work/agent/SESSION.json', observer.SESSION_FIELDS)
        self.assertEqual(result, dict(phase='tune',iteration=6,calls=5,max_iters=6,tuner=True))
        self.assertTrue(all('PRIVATE_' not in value for value in decoded))

    def test_unknown_long_string_is_not_accumulated(self):
        for raw in ('{"messages": ["' + ('x'*20000) + '"], "calls": 4}',
                    '{"tune_cache": {"' + ('x'*20000) + '":[]}, "calls":4}'):
            self.assertEqual(observer._Selector(io.StringIO(raw), 30000).project({'calls':True}), {'calls':4})

    def test_summary_filters_payloads_and_keeps_unknowns(self):
        value = observer.collect(self.args)
        encoded = json.dumps(value)
        self.assertNotIn('PRIVATE_', encoded)
        self.assertNotIn('DO_NOT_PUBLISH', encoded)
        self.assertIsNone(value['active_case'])
        self.assertEqual(value['watched_case']['session']['iteration'], 6)
        self.assertEqual(value['watched_case']['phase'], 'agent')
        self.assertEqual(value['queue']['jobs'][0]['recorded_resource_limits'],
                         dict(address_space_gib=12.0, artifact_gib=32.0, outer_wall_seconds=1800))
        self.assertEqual(value['api']['complete_accounting_rows'],2)
        self.assertEqual(value['api']['http_200_rows'],1)
        self.assertEqual(value['api']['remaining_recorded_allowance'],598)
        self.assertFalse(value['queue']['durable_completion'])
        self.assertEqual(value['display_timezone'],'Asia/Taipei')
        self.assertTrue(value['observed_at_local'].endswith('+08:00'))

    def test_ledger_partial_and_malformed_are_explicit(self):
        self.ledger.write_text('{"status":200}\nnot-json\n{"status":200}')
        missing=[]; value=observer.ledger_summary(self.ledger,self.allowance,missing)
        self.assertEqual(value['complete_accounting_rows'],1)
        self.assertEqual(value['malformed_rows'],1)
        self.assertTrue(value['partial_tail'])
        self.assertIsNone(value['remaining_recorded_allowance'])

    def test_allowance_requires_the_same_version_and_explicit_ledger(self):
        for version, ledger in ((2,str(self.ledger)),(1,str(self.root/'other-ledger'))):
            with self.subTest(version=version,ledger=ledger):
                self.write(self.allowance,dict(version=version,ledger=ledger,ceiling=600))
                missing=[]; result=observer.ledger_summary(self.ledger,self.allowance,missing)
                self.assertIsNone(result['absolute_allowance'])
                self.assertIsNone(result['remaining_recorded_allowance'])
                self.assertIn('api.allowance_ledger_binding',missing)

    def test_active_case_must_be_declared_and_within_the_results_parent(self):
        self.write(self.batch/'active.json',dict(id='one',status='running'))
        self.write(self.active/'progress.json',dict(status='running',phase='trace'))
        result=observer.collect(self.args)
        self.assertEqual(result['active_case']['phase'],'trace')
        self.write(self.batch/'manifest.json',dict(jobs=[dict(id='one',out='/unrelated-case')]))
        result=observer.collect(self.args)
        self.assertIsNone(result['active_case'])
        self.assertIn('active.case_path_outside_results',result['missing'])

    def test_terminal_queue_requires_both_persisted_flags(self):
        self.close(self.batch)
        value=observer.collect(self.args)
        self.assertTrue(value['queue']['durable_completion'])
        self.write(self.batch/'completion.json',dict(status='ok',required_artifacts_persisted=True))
        self.assertFalse(observer.collect(self.args)['queue']['durable_completion'])

    def test_stale_active_and_resource_terminal_statuses(self):
        self.write(self.batch/'active.json',dict(id='one',status='running'))
        self.write(self.batch/'progress.json',dict(status='running',jobs=[dict(id='one',status='failed')]))
        self.close(self.watch,'resource_stopped')
        value=observer.collect(self.args)
        self.assertIsNone(value['queue']['active_job_id'])
        self.assertEqual(value['watched_case']['status'],'resource_stopped')
        self.assertTrue(value['watched_case']['durable_completion'])

    def test_missing_incomplete_and_malformed_metadata_are_unavailable(self):
        self.ledger.unlink()
        (self.watch/'work/agent/SESSION.json').write_text('{"messages":[')
        self.write(self.batch/'manifest.json',dict(jobs='wrong'))
        self.write(self.watch/'progress.json',dict(status='running',attempted=['wrong']))
        value=observer.collect(self.args)
        self.assertIsNone(value['api']['complete_accounting_rows'])
        self.assertIn('watch.session',value['missing'])
        self.assertEqual(value['queue']['jobs'],[])
        self.assertIsNone(value['watched_case']['session']['calls'])

    def test_bounded_inputs_and_finite_numeric_fields(self):
        path=self.root/'large.json';path.write_text('{"calls":123}')
        with self.assertRaises(observer.Unavailable):
            observer.projection(path,{'calls':True},maximum=4)
        self.assertIsNone(observer._number(float('nan')))
        self.assertIsNone(observer._number(True))
        self.assertIsNone(observer._number(1.5,integral=True))

    def test_html_escapes_and_publication_never_updates_inputs(self):
        value=observer.collect(self.args)
        value['queue']['jobs'][0]['id']='<script>alert(1)</script>'
        before=self.ledger.read_text()
        observer.publish(self.publish,value,30)
        page=(self.publish/observer.PUBLIC_HTML).read_text()
        self.assertIn('http-equiv="refresh" content="30"',page)
        self.assertNotIn('<script>',page)
        self.assertIn('&lt;script&gt;',page)
        self.assertEqual(self.ledger.read_text(),before)
        self.assertEqual({path.name for path in self.publish.iterdir()},
                         {observer.PUBLIC_JSON,observer.PUBLIC_HTML})

    def test_publication_refuses_symlink_and_never_follows_old_tmp(self):
        self.publish.mkdir()
        original=self.root/'original';original.write_text('retained')
        (self.publish/(observer.PUBLIC_JSON+'.tmp')).symlink_to(original)
        observer._atomic(self.publish/observer.PUBLIC_JSON,'new publication')
        self.assertEqual(original.read_text(),'retained')
        (self.publish/observer.PUBLIC_JSON).unlink()
        (self.publish/observer.PUBLIC_JSON).symlink_to(original)
        with self.assertRaises(ValueError):
            observer._atomic(self.publish/observer.PUBLIC_JSON,'forbidden')
        self.assertEqual(original.read_text(),'retained')

    def test_cli_poll_duration_and_canonical_publication_separation(self):
        argv=['--batch-case',str(self.batch),'--watch-case',str(self.watch),
              '--ledger',str(self.ledger),'--allowance',str(self.allowance),
              '--publish-dir',str(self.publish)]
        args=observer.parse_args(argv)
        self.assertEqual((args.poll_seconds,args.seconds),(30,7*86400))
        for extra in (['--poll-seconds','9'],['--seconds','604801'],['--seconds','nan']):
            with self.subTest(extra=extra),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
                observer.parse_args(argv+extra)
        alias=self.root/'alias';alias.symlink_to(self.watch,target_is_directory=True)
        with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
            observer.parse_args(argv+['--publish-dir',str(alias/'subdirectory')])

    def test_fake_clock_stops_at_queue_completion_without_experiment_actions(self):
        value=observer.collect(self.args);now=[0.];saved=[];calls=[]
        def collect(args):
            result=copy.deepcopy(value);calls.append(1)
            result['queue']['durable_completion']=len(calls)==3
            return result
        def sleep(seconds):now[0]+=seconds
        result=observer.observe(self.args,clock=lambda:now[0],sleep=sleep,collector=collect,
            publisher=lambda directory,data,poll:saved.append(data))
        self.assertEqual(now[0],60)
        self.assertEqual(result['observer_status'],'queue_complete')
        self.assertEqual(result['observations'],3)
        self.assertTrue(all(row['experiment_actions']==0 for row in saved))

    def test_fake_clock_limit_and_signal_stop_observation_only(self):
        for signalled in (False,True):
            with self.subTest(signalled=signalled):
                now=[0.];saved=[]
                result=observer.observe(self.args,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n),
                    collector=lambda args:observer.collect(args),
                    publisher=lambda directory,data,poll:saved.append(data),stopped=lambda:signalled)
                self.assertEqual(result['observer_status'],'observer_signal' if signalled else 'observation_limit_reached')
                self.assertEqual(now[0],0 if signalled else 100)
                self.assertEqual(result['watched_case']['status'],'running')


if __name__=='__main__':
    unittest.main()
