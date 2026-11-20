"""Tiny Alex-source metadata and fake bounded-stage supervision fixtures."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_alexnet_host_cpu_overlap as entry
from pt_agent.experiments import heir_common_model_pipeline as pipeline
from pt_agent.tests import test_heir_vgg_host_cpu_overlap_tune4 as fixtures
from pt_agent.tests.test_heir_vgg_host_cpu_overlap import put,clean


class AlexHostChecks(unittest.TestCase):
    def setUp(self):
        fixtures.Tune4Checks.setUp(self)
        self.args.out=self.root/(entry.OUT_PREFIX+'fixture');self.math_calls=[]

    def fake(self,*,worker_bad=False,math_bad=False,late=False,source_bad=False):
        def process(command,**options):
            self.calls.append((command,options));put(Path(options['log']),'synthetic bounded response only\n')
            put(Path(options['resources']),'')
            if '--worker'in command:
                out=Path(command[command.index('--out')+1]);put(out/'result.json',dict(status='ok',lowered_adapter_contract=dict(role='fixture')))
                put(out/'model.mlir','fixture original source metadata only\n')
            return clean(exit_code=1 if worker_bad else 0)
        def check(record,checked):
            self.math_calls.append((record,checked))
            if math_bad:raise ValueError('strict original Alex check rejected')
            return dict(passed=True,training_samples=3,activation_pairs=21,logit_elements=30,emitted_constants=31)
        def native(*args):
            if late:self.available=entry.host.GIB
            return dict(static_bootstrap_operations=2)
        def snapshot(out):(out/'source').mkdir();return[]
        with(patch.object(pipeline,'sources',return_value=self.source)as source,
                patch.object(entry.alex,'source_gate',side_effect=ValueError('original source rejected')if source_bad else None,return_value=self.checked),
                patch.object(entry.alex,'check_worker',side_effect=check),patch.object(entry.alex,'_snapshot',side_effect=snapshot),
                patch.object(entry.host.original,'run_process',side_effect=process),
                patch.object(entry.host.original,'check_native',side_effect=native)):
            result=entry.execute(self.args)
        self.source_call=source.call_args;return result

    def test_direct_original_Alex_worker_full_source_flags_and_10800_contract(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.source_call.args,(entry.alex.MODEL,));self.assertEqual(result['model'],entry.alex.MODEL)
        self.assertEqual(result['variant'],entry.alex.VARIANT);self.assertEqual(result['round'],entry.ROUND)
        self.assertEqual(result['original_worker_module'],entry.alex.MODULE)
        argv,options=self.calls[0];self.assertEqual(argv[:4],[str(entry.host.host_python()),'-B','-m',entry.alex.MODULE])
        self.assertIn('--worker',argv);self.assertNotIn('pt_agent.tools.nixrun',argv)
        for name,path in self.source.items():self.assertEqual(argv[argv.index('--'+name.replace('_','-'))+1],str(path))
        self.assertEqual(argv[argv.index('--memory-gib')+1],'24');self.assertEqual(options['mem_gib'],24)
        self.assertEqual(result['limits']['total_seconds'],10800);self.assertEqual(result['limits']['subtree_RSS_gib'],10)
        self.assertEqual(result['limits']['minimum_available_gib'],2);self.assertEqual(result['limits']['case_gib'],32)
        self.assertEqual(result['limits']['file_gib'],16);self.assertEqual(result['limits']['minimum_free_disk_gib'],10)

    def test_original_math_worker_source_native_and_contextmanager_objects_retained(self):
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['worker'],entry.alex.worker);self.assertIs(namespace['check_worker'],entry.alex.check_worker)
        self.assertIs(namespace['source_gate'].__closure__[0].cell_contents,self.args)
        for name in('native_command','check_native'):
            self.assertIs(namespace[name].__code__,getattr(entry.host.original,name).__code__)
        self.assertIs(namespace['serial_lock'],entry.host.original.serial_lock)
        result=self.fake();self.assertTrue(result['strict_math_passed']);self.assertEqual(len(self.math_calls),1)
        self.assertEqual(self.math_calls[0][1],self.checked)

    def test_native_argv_preserves_requested_parameters_and_does_not_use_host_worker_environment(self):
        result=self.fake();argv,options=self.calls[1]
        self.assertEqual(argv[0],str(self.args.heir_opt));self.assertIn('--mlir-disable-threading',argv)
        self.assertEqual(entry.host.original.read(self.args.out/'native/native-parameters.json'),self.checked['native_parameters'])
        self.assertTrue(any(x.startswith('--pass-pipeline=')for x in argv))
        self.assertEqual(options['mem_gib'],24);self.assertEqual(options['fsize_gib'],16)
        self.assertTrue(result['model_compiled_accepted']);self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))

    def test_host_versions_environment_scope_registry_and_snapshots_are_explicit(self):
        with patch.dict('os.environ',{'LD_LIBRARY_PATH':'/foreign','NIX_PATH':'/foreign','PYTHONHOME':'/foreign','CONDA_PREFIX':'/foreign'}):
            result=self.fake()
        env=self.calls[0][1]['env']
        for name in('LD_LIBRARY_PATH','NIX_PATH','PYTHONHOME','CONDA_PREFIX'):self.assertNotIn(name,env)
        self.assertEqual(env['OMP_NUM_THREADS'],'1')
        self.assertEqual(result['host_runtime']['last_observed_import_versions'],entry.host.OBSERVED_HOST_VERSIONS)
        self.assertFalse(result['host_runtime']['original_Nix_runtime_equivalence_claimed'])
        self.assertIsNone(result['limits']['nix_seconds']);self.assertIn('Alex',result['scope'])
        self.assertTrue((self.args.out/'source/heir-alexnet-host-cpu-overlap.py').is_file())
        row=json.loads(self.args.registry.read_text());self.assertEqual(len(row),8)
        self.assertEqual(row['model'],entry.alex.MODEL);self.assertEqual(row['mode'],entry.MODE)
        self.assertTrue(row['metrics']['model_compiled_accepted']);self.assertEqual(row['metrics']['bert_signal_actions'],0)

    def test_math_rejection_stops_before_native_and_closes_without_acceptance(self):
        result=self.fake(math_bad=True);self.assertNotEqual(result['status'],'ok');self.assertEqual(len(self.calls),1)
        self.assertEqual(result['attempted']['native_compile'],0);self.assertFalse(result['model_compiled_accepted'])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_original_source_rejection_closes_with_no_process(self):
        result=self.fake(source_bad=True);self.assertNotEqual(result['status'],'ok');self.assertEqual(self.calls,[])
        self.assertEqual(result['attempted'],dict(export_math=0,native_compile=0));self.assertTrue((self.args.out/'completion.json').is_file())

    def test_native_fact_retained_after_terminal_resource_rejection(self):
        result=self.fake(late=True);self.assertNotEqual(result['status'],'ok');self.assertTrue(result['model_compiled'])
        self.assertFalse(result['model_compiled_accepted']);self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))

    def test_exact_original_tune4_identity_topology_owner_and_other_compiler_guard(self):
        self.args.out.mkdir();guard=entry._namespace(self.args)['physical_observation']
        self.assertFalse(guard(None,12*entry.host.GIB,50*entry.host.GIB)['stop'])
        self.assertTrue(guard(dict(subtree_rss_bytes=11*entry.host.GIB),12*entry.host.GIB,50*entry.host.GIB)['stop'])
        self.table[93984]['starttime']+=1;self.assertTrue(guard(None,12*entry.host.GIB,50*entry.host.GIB)['stop'])
        self.table[93984]['starttime']-=1
        with patch.object(entry.host.overlap,'_process_name',return_value='heir-opt'):
            foreign=entry._namespace(self.args)['physical_observation']
            self.assertTrue(foreign(None,12*entry.host.GIB,50*entry.host.GIB)['stop'])

    def test_Alex_variant_has_own_one_attempt_budget_and_requires_new_Alex_result_prefix(self):
        self.args.out=self.root/(entry.host.OUT_PREFIX+'foreign-VGG')
        with self.assertRaises(ValueError):entry.execute(self.args)
        self.args.out=self.root/(entry.OUT_PREFIX+'fixture')
        put(self.args.registry,dict(round=entry.ROUND,mode=entry.MODE,metrics=dict(attempted=dict(export_math=1))))
        result=self.fake();self.assertNotEqual(result['status'],'ok');self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
