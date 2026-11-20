"""RSS-only recovery tested through the original fake trace/native pipeline."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import dacapo_fresh_cold as base
from pt_agent.experiments import dacapo_original_compile as original
from pt_agent.experiments import dacapo_vgg_fresh_cold_rss16 as entry
from pt_agent.tests import test_dacapo_fresh_cold as fixtures


class VGGFreshRSS16Checks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FreshOriginalColdChecks(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.old_policies = copy.deepcopy(base.POLICIES)
        self.old_run = base.run
        self.old_main = base.main

    def tearDown(self):
        self.assertEqual(base.POLICIES, self.old_policies)
        self.assertIs(base.run, self.old_run)
        self.assertIs(base.main, self.old_main)

    def arguments(self):
        return self.fixture.arguments('vgg16')

    def policy(self, args):
        with patch.object(paths, 'RESULTS_DIR', self.fixture.root):
            return entry.resource_policy(args)

    def run_fake(self, *, failure=None):
        # Reuse the old tiny fixture as an ordinary bound function. No old test
        # method is executed and neither module's entry functions are patched.
        facade = SimpleNamespace(**dict(vars(base), run=entry.run, main=entry.main,
                                        POLICIES=entry.POLICIES))
        function = fixtures.FreshOriginalColdChecks.run_fake
        private = dict(function.__globals__, entry=facade)
        clone = FunctionType(function.__code__, private, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        return MethodType(clone, self.fixture)('vgg16', failure=failure, via_main=True)

    def test_only_RSS_changes_and_original_policy_remains14(self):
        args = self.arguments()
        with patch.object(paths, 'RESULTS_DIR', self.fixture.root):
            old = base.resource_policy(args)
            new = entry.resource_policy(args)
        self.assertEqual({k: new[k] for k in old}, dict(old, RSS_gib=16))
        self.assertEqual(base.POLICIES['vgg16'], {'native_AS_gib':24, 'RSS_gib':14})
        self.assertEqual(base.POLICIES['alexnet'], {'native_AS_gib':32, 'RSS_gib':16})
        self.assertTrue(new['resource_only_recovery'])
        self.assertEqual(new['original_adapter_module'], base.MODULE)
        self.assertEqual(new['original_adapter_policy'], base.POLICIES['vgg16'])

    def test_Alex_and_other_models_reject_before_output(self):
        args = self.arguments()
        for model in ('alexnet', 'resnet20', None):
            changed = SimpleNamespace(**dict(vars(args), model=model))
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'only original VGG16'):
                self.policy(changed)
        self.assertFalse(args.out.exists())

    def test_RSS14to16_accepted_and_existing_reserves_preserved(self):
        policy = self.policy(self.arguments())
        for rss in (14.5*base.GIB,16*base.GIB):
            sample = {'subtree_plus_supervisor_rss_bytes':rss,'sample_ts_ns':1}
            self.assertFalse(base.physical_observation(policy,sample,1.25*base.GIB,10*base.GIB)['stop'])
        sample = {'subtree_plus_supervisor_rss_bytes':16*base.GIB+1,'sample_ts_ns':1}
        self.assertTrue(base.physical_observation(policy,sample,20*base.GIB,10*base.GIB)['stop'])
        self.assertTrue(base.physical_observation(policy,None,1.25*base.GIB-1,10*base.GIB)['stop'])
        self.assertTrue(base.physical_observation(policy,None,20*base.GIB,10*base.GIB-1)['stop'])

    def test_both_worker_argv_identical_to_original_adapter(self):
        args=self.arguments();policy=self.policy(args)
        (args.out/'source').mkdir(parents=True)
        for phase in ('trace','compile'):
            options={'command_file':self.fixture.root/'argv.json'} if phase=='compile' else {}
            expected=base.phase_worker(args,phase,self.fixture.root/phase,policy=policy,**options)
            actual=entry.phase_worker(args,phase,self.fixture.root/phase,policy=policy,**options)
            self.assertEqual(actual,expected)
            self.assertEqual(base._one_argument(actual,'--memory-gib'),'12' if phase=='trace' else '24')
            self.assertEqual(base._one_argument(actual,'--file-gib'),'16')
            self.assertIn('pt_agent.experiments.dacapo_original_compile',actual)
        self.assertEqual(args.memory_gib,12)

    def test_original_full_fresh_fake_has_same_compiler_argv_math_scope_and_new_snapshots(self):
        args,code,calls,result,closed=self.run_fake()
        self.assertEqual(code,0);self.assertEqual(result['status'],'ok')
        self.assertEqual(result['round'],original.ROUND);self.assertEqual(result['mode'],original.MODE)
        self.assertTrue(result['fresh_compile_admission']['passed'])
        self.assertTrue(result['cold_trace_compile_completed'])
        self.assertFalse(result['original_trace_reused']);self.assertFalse(result['prior_times_included'])
        adapter=result['resource_adapter'];self.assertEqual(adapter['variant'],entry.VARIANT)
        self.assertEqual([x['AS_gib'] for x in adapter['phases']],[12,24])
        self.assertEqual([x['RSS_gib'] for x in adapter['phases']],[16,16])
        self.assertEqual(result['limits']['subtree_plus_supervisor_RSS_gib'],16)
        self.assertEqual(result['limits']['minimum_available_RAM_gib'],1.25)
        for name in ('dacapo_original_compile.py','dacapo_fresh_cold.py','dacapo_vgg_fresh_cold_rss16.py'):
            self.assertTrue((args.out/'source'/name).is_file())
        self.assertEqual(adapter['source_adapter_snapshot']['path'],str(args.out/'source'/Path(entry.__file__).name))
        self.assertEqual(adapter['policy']['original_adapter_source'],str(Path(base.__file__).resolve()))
        native=[x for x in calls if x['command'][0]==str(args.compiler.resolve())]
        expected=original.compiler_argv(json.loads(args.metadata_template.read_text()),binary=args.compiler,
            source=args.out/'work/trace'/original.model_paths(args.upstream,'vgg16')['trace_relative'],
            profile=args.out/'compiler-config.json',output=args.out/'work/compile/lowered.mlir',
            waterline=40,diagnostic_elide_final_ir=True)
        self.assertEqual(len(native),1);self.assertEqual(native[0]['command'],expected)
        self.assertEqual(native[0]['options']['mem_gib'],24)
        self.assertEqual(native[0]['options']['timeout'],7200)
        self.assertEqual(result['api_calls'],0);self.assertEqual(result['real_ckks_attempts'],0)
        for field in original.FORMAL_FIELDS:self.assertIsNone(result[field])
        registry=json.loads(args.registry.read_text())
        self.assertEqual(set(registry),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(registry['metrics']['status'],'ok')
        self.assertTrue(closed['registry_persisted']);self.assertEqual(closed['status'],'ok')
        self.assertEqual(closed['case_artifact_bytes'],original.artifact_bytes(args.out))

    def test_resource_stop_after_native_keeps_real1_1_and_failure_closure(self):
        args,code,_,result,closed=self.run_fake(failure='native_guard_completed')
        self.assertEqual(code,1);self.assertEqual(result['status'],'resource_stopped')
        self.assertEqual(result['native_compile_attempts'],1)
        self.assertEqual(result['native_compile_completed'],1)
        self.assertFalse(result['cold_trace_compile_completed'])
        self.assertEqual(result['resource_adapter']['variant'],entry.VARIANT)
        self.assertEqual(result['limits']['subtree_plus_supervisor_RSS_gib'],16)
        self.assertEqual(json.loads(args.registry.read_text())['metrics']['status'],'resource_stopped')
        self.assertEqual(closed['status'],'resource_stopped')

    def test_private_plain_entry_clones_keep_original_contextmanagers(self):
        private=entry._namespace()
        self.assertIs(private['original'],original)
        self.assertIs(private['supervision'],base.supervision)
        self.assertIs(private['supervision'].serial_lease,base.supervision.serial_lease)
        self.assertIs(private['original']._budget_guard,original._budget_guard)
        self.assertIs(private['guarded_process'],base.guarded_process)
        self.assertEqual(private['__file__'],entry.__file__)
        for flag in ('--worker-trace','--worker-compile'):
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,'complete original outer'):
                entry.main([flag])


if __name__=='__main__':
    unittest.main()
