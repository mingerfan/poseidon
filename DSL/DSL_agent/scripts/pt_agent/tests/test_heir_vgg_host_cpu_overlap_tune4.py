"""Exact tune4 private guard delta, using tiny metadata/fake observations only."""
import json
from unittest.mock import patch
import unittest

from pt_agent.experiments import heir_vgg_host_cpu_overlap as entry
from pt_agent.tests import test_heir_vgg_host_cpu_overlap as fixtures

put=fixtures.put


class Tune4Checks(unittest.TestCase):
    def setUp(self):
        # Reuse the existing small fixture builder; no old test body is run.
        fixtures.HostOverlapChecks.setUp(self)
        self.args.bert_pid,self.args.bert_starttime=entry.IDENTIFIED_BERT
        row=self.table.pop(64170);row.update(pid=self.args.bert_pid,starttime=self.args.bert_starttime)
        self.table[self.args.bert_pid]=row
        put(self.args.bert_case/'session/SESSION.json',dict(phase='tune',iteration=4,calls=4,failed=None))
        path=self.args.bert_run_case/'result.json';run=json.loads(path.read_text())
        run['attempted']['step']=3;run['completed']['step']=2;put(path,run)

    def fake(self,**options):return fixtures.HostOverlapChecks.fake(self,**options)

    def test_exact_current_raw_metadata_and_original_guard_accept(self):
        files=[self.args.bert_case/'session/SESSION.json',self.args.bert_run_case/'result.json',
            self.args.batch_case/'active.json',self.args.batch_case/'supervisor.json']
        before=[p.read_text()for p in files]
        accepted=entry.metadata_admission(self.args)
        self.assertEqual((accepted['phase'],accepted['iteration'],accepted['calls']),('tune',4,4))
        self.args.out.mkdir();guard=entry._namespace(self.args)['physical_observation']
        observed=guard(None,12*entry.GIB,50*entry.GIB)
        self.assertFalse(observed['stop'],observed['reasons'])
        self.assertEqual(observed['heavy_lease_owners'],[36762])
        self.assertEqual([p.read_text()for p in files],before)

    def test_old_next_mixed_failed_or_other_phase_rejected(self):
        path=self.args.bert_case/'session/SESSION.json'
        original=json.loads(path.read_text())
        for delta in (dict(iteration=3,calls=3),dict(iteration=5,calls=5),dict(iteration=4,calls=3),
                dict(phase='evaluate'),dict(phase='confirm'),dict(failed='observed failure')):
            with self.subTest(delta=delta):
                put(path,dict(original,**delta))
                with self.assertRaises(ValueError):entry.metadata_admission(self.args)

    def test_wrong_step_counts_status_or_source_rejected(self):
        path=self.args.bert_run_case/'result.json';original=json.loads(path.read_text())
        for attempted,completed in ((1,0),(4,3),(3,3),(2,2)):
            with self.subTest(counts=(attempted,completed)):
                row=dict(original,attempted=dict(step=attempted),completed=dict(step=completed));put(path,row)
                with self.assertRaises(ValueError):entry.metadata_admission(self.args)
        for delta in (dict(status='ok'),dict(termination=dict(signal=15)),dict(source=dict(source_case='/foreign'))):
            with self.subTest(delta=delta):
                put(path,dict(original,**delta))
                with self.assertRaises(ValueError):entry.metadata_admission(self.args)

    def test_exact_worker_batch_and_live_ancestor_owner_checks_preserved(self):
        original=self.args.bert_pid;self.args.bert_pid=64170
        with self.assertRaises(ValueError):entry.metadata_admission(self.args)
        self.args.bert_pid=original;self.args.out.mkdir();guard=entry._namespace(self.args)['physical_observation']
        self.table[original]['starttime']+=1;self.assertTrue(guard(None,12*entry.GIB,50*entry.GIB)['stop'])
        self.table[original]['starttime']-=1;self.table[original]['ppid']=1
        self.assertTrue(guard(None,12*entry.GIB,50*entry.GIB)['stop'])
        self.table[original]['ppid']=40000
        with patch.object(entry.overlap,'_lease_owners',return_value=[999]):
            # The cloned original guard uses its captured ordinary helper, so
            # build the namespace inside this scoped fixture patch.
            altered=entry._namespace(self.args)['physical_observation']
            self.assertTrue(altered(None,12*entry.GIB,50*entry.GIB)['stop'])

    def test_wrong_queue_terminal_batch_or_foreign_native_rejected(self):
        path=self.args.batch_case/'active.json';active=json.loads(path.read_text())
        put(path,dict(active,status='completed'))
        with self.assertRaises(ValueError):entry.metadata_admission(self.args)
        put(path,active);put(self.args.batch_case/'result.json',dict(status='ok'))
        with self.assertRaises(ValueError):entry.metadata_admission(self.args)
        (self.args.batch_case/'result.json').unlink();self.args.out.mkdir()
        self.table[65432]=dict(pid=65432,ppid=1,starttime=11,state='R')
        with patch.object(entry.overlap,'_process_name',side_effect=lambda pid:'heir-opt'if pid==65432 else'python3'):
            guard=entry._namespace(self.args)['physical_observation']
            self.assertTrue(guard(None,12*entry.GIB,50*entry.GIB)['stop'])

    def test_original_worker_math_native_sequence_and_runtime_scope_after_delta(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error'])
        self.assertEqual(len(self.calls),2);self.assertTrue(result['strict_math_passed'])
        self.assertTrue(result['model_compiled_accepted']);self.assertIn('tune4',result['mode'])
        self.assertEqual(result['identified_bert']['worker_pid'],93984)
        self.assertEqual(self.calls[0][0][:4],[str(entry.host_python()),'-B','-m',entry.original.MODULE])
        self.assertEqual(self.calls[1][0][0],str(self.args.heir_opt))
        self.assertEqual(result['limits']['subtree_RSS_gib'],10);self.assertEqual(result['limits']['minimum_available_gib'],2)

    def test_late_guard_revokes_acceptance_and_retains_actual_native_completion(self):
        result=self.fake(late=True);self.assertNotEqual(result['status'],'ok')
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_original_contextmanager_numeric_and_native_functions_stay_original(self):
        namespace=entry._namespace(self.args)
        for name in ('worker','_runtime','run_probes','check_worker','native_command','check_native','serial_lock'):
            self.assertIs(namespace[name],getattr(entry.original,name))
        self.assertEqual(entry.overlap._metadata_admission.__globals__['ROUND'],entry.overlap.ROUND)


if __name__=='__main__':unittest.main()
