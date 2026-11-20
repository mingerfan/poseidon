"""Public process/lease metadata and fake-stage serial Alex fixtures only."""
import json
import os
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_alexnet_host_serial_json as entry
from pt_agent.tests import test_heir_alexnet_host_cpu_overlap as fixtures


class SerialJSONChecks(unittest.TestCase):
    def setUp(self):
        fixtures.AlexHostChecks.setUp(self);self.args.out=self.root/(entry.OUT_PREFIX+'fixture')
        # This new supervisor is an actual descendant of the unique lease-owner
        # fixture, without a BERT argument or session file.
        self.table[os.getpid()]['ppid']=40000
        for name in('bert_case','bert_run_case','batch_case'):
            shutil.rmtree(getattr(self.args,name));delattr(self.args,name)
        for name in('bert_pid','bert_starttime','batch_pid','batch_starttime'):delattr(self.args,name)

    def fake(self,**options):
        return entry.shared._clone_plain(fixtures.AlexHostChecks.fake,dict(vars(fixtures),entry=entry))(self,**options)

    def test_original_reviewed_JSON_worker_source_native_and_serial_resource_contract(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.calls[0][0][:4],[str(entry.host.host_python()),'-B','-m',entry.numeric.MODULE])
        self.assertEqual(self.source_call.args,(entry.alex.MODEL,));self.assertEqual(len(self.math_calls),1)
        self.assertEqual(self.calls[1][0][0],str(self.args.heir_opt));self.assertTrue(result['strict_math_passed'])
        self.assertTrue(result['model_compiled_accepted']);self.assertEqual(result['limits']['total_seconds'],10800)
        for key,value in dict(AS_gib=24,subtree_RSS_gib=14,minimum_available_gib=1.25,file_gib=16,case_gib=32,minimum_free_disk_gib=10).items():
            self.assertEqual(result['limits'][key],value)
        self.assertEqual(result['ancestor_heavy_lease_owner'],dict(pid=36762,starttime=961036))
        self.assertEqual(result['runtime_variant'],'existing-host-runtime-serial');self.assertFalse(result['BERT_session_dependency'])

    def test_unique_real_lease_owner_must_be_strict_original_ancestor(self):
        observed=entry.lease_observation();self.assertEqual(observed['owner'],dict(pid=36762,starttime=961036))
        for owners in([], [999], [36762,40000], [os.getpid()]):
            with self.subTest(owners=owners),patch.object(entry.shared,'_lease_owners',return_value=owners):
                with self.assertRaises(ValueError):entry.lease_observation()
        with self.assertRaises(ValueError):entry.lease_observation(dict(pid=36762,starttime=961037))

    def test_live_namespace_owner_identity_and_no_foreign_native_required(self):
        own=self.table.pop(os.getpid())
        with self.assertRaises(ValueError):entry.lease_observation()
        self.table[os.getpid()]=own;self.table[36762]['state']='Z'
        with self.assertRaises(ValueError):entry.lease_observation()
        self.table[36762]['state']='S'
        with patch.object(entry.shared,'_process_name',return_value='heir-opt'):
            with self.assertRaises(ValueError):entry.lease_observation()

    def test_metadata_guard_pins_owner_and_preserves_RSS14_available1p25_free10(self):
        guard=entry._namespace(self.args)['physical_observation']
        self.assertFalse(guard(dict(subtree_rss_bytes=13*entry.original.GIB),2*entry.original.GIB,50*entry.original.GIB)['stop'])
        self.assertTrue(guard(dict(subtree_rss_bytes=15*entry.original.GIB),2*entry.original.GIB,50*entry.original.GIB)['stop'])
        self.assertTrue(guard(None,entry.original.GIB,50*entry.original.GIB)['stop'])
        self.assertTrue(guard(None,2*entry.original.GIB,9*entry.original.GIB)['stop'])
        self.table[36762]['starttime']+=1;self.assertTrue(guard(None,2*entry.original.GIB,50*entry.original.GIB)['stop'])

    def test_no_BERT_metadata_or_tune_guard_calls_and_initializer_is_original(self):
        namespace=entry._namespace(self.args);self.assertIs(namespace['serial_lock'],entry.original.serial_lock)
        self.assertIs(namespace['check_worker'],entry.alex.check_worker)
        self.assertIs(namespace['native_command'].__code__,entry.original.native_command.__code__)
        with patch.object(entry.shared,'_metadata_admission',side_effect=AssertionError('No BERT metadata access')):
            result=self.fake()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['existing_initializer_lock_retained'])

    def test_original_source_and_math_failures_never_start_native(self):
        result=self.fake(math_bad=True);self.assertEqual(result['status'],'error')
        self.assertEqual(len(self.calls),1);self.assertEqual(result['attempted']['native_compile'],0)
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_terminal_resource_failure_revokes_acceptance_and_preserves_native_fact(self):
        result=self.fake(late=True);self.assertEqual(result['status'],'error')
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))
        self.assertTrue(result['terminal_serial_guard']['stop'])
        row=json.loads(self.args.registry.read_text());self.assertEqual(len(row),8)
        self.assertFalse(row['metrics']['model_compiled_accepted']);self.assertEqual(row['metrics']['status'],'error')

    def test_host_environment_profile_snapshot_registry_and_new_one_attempt_mode(self):
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'foreign','NIX_PATH':'foreign','PYTHONHOME':'foreign'}):result=self.fake()
        env=self.calls[0][1]['env']
        for key in('LD_LIBRARY_PATH','NIX_PATH','PYTHONHOME'):self.assertNotIn(key,env)
        self.assertEqual(env['OMP_NUM_THREADS'],'1');self.assertTrue(result['host_runtime']['original_worker_code_unchanged'])
        self.assertEqual(entry.original.read(self.args.out/'native/native-parameters.json'),self.checked['native_parameters'])
        self.assertTrue((self.args.out/'source/heir-alexnet-host-serial-json.py').is_file())
        self.args.out=self.root/(entry.OUT_PREFIX+'second')
        result=self.fake();self.assertEqual(result['status'],'error');self.assertEqual(result['attempted'],dict(export_math=0,native_compile=0))

    def test_parser_has_no_BERT_arguments_and_rejects_wrong_runtime_or_result_prefix(self):
        args=entry.parser().parse_args(['--out',str(self.args.out)])
        self.assertFalse(any(name.startswith('bert')for name in vars(args)))
        self.args.out=self.root/'foreign-case'
        with self.assertRaises(ValueError):entry.execute(self.args)
        self.args.out=self.root/(entry.OUT_PREFIX+'fixture');self.args.python=self.root/'foreign-python'
        with self.assertRaises(ValueError):entry.execute(self.args)
        self.assertEqual(self.calls,[])

    def test_absolute_original_case_deadline_at_closure_revokes_accepted_native_fact(self):
        with patch.object(entry,'time',SimpleNamespace(monotonic=__import__('unittest.mock',fromlist=['Mock']).Mock(side_effect=[0.,10801.]))):
            result=self.fake()
        self.assertEqual(result['status'],'error');self.assertTrue(result['model_compiled'])
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue(result['terminal_deadline_exhausted'])


if __name__=='__main__':unittest.main()
