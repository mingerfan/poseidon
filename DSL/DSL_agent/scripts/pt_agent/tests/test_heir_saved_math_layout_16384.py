"""Small saved-metadata fixtures for the official one-layout-pass candidate."""
import ast
import copy
import inspect
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_saved_math_layout_16384 as entry
from pt_agent.tests import test_heir_saved_math_no_linear_fusion as fixtures


class OriginalFusionLayoutCases(unittest.TestCase):
    # Reuse only metadata setup/helpers; no previous test method is inherited.
    configure = fixtures.SavedMathCases.configure
    persist = fixtures.SavedMathCases.persist
    lease = fixtures.SavedMathCases.lease
    native_text = fixtures.SavedMathCases.native_text
    fake_run = fixtures.SavedMathCases.fake_run

    def setUp(self):
        fixtures.SavedMathCases.setUp(self)

    def run_entry(self, runner=None):
        base = entry.base
        with patch.object(base.paths, 'RESULTS_DIR', self.root), \
                patch.object(base.serial, 'lease_observation', side_effect=self.lease), \
                patch.object(base, '_available_ram_bytes', side_effect=lambda:self.available), \
                patch.object(base.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)), \
                patch.object(base, 'run_process', side_effect=runner or self.fake_run):
            return entry.execute(self.args)

    def test_one_pass_insert_preserves_original_fusion_and_all_other_characters(self):
        old = 'builtin.module(\r\n  linalg-fuse-linear-ops,\r\n  '+entry.PROPAGATION_PASS+',\r\n  cse,\r\n  end\r\n)'
        inserted = '  '+entry.LAYOUT_PASS+',\r\n'
        new = entry.add_layout_optimization(old)
        self.assertEqual(new,old.replace('  cse,',inserted+'  cse,'))
        self.assertEqual(new.replace(inserted,''),old)
        self.assertEqual(new.count('linalg-fuse-linear-ops'),1)

    def test_pass_rejects_ambiguous_wrong_slots_existing_or_missing_fusion(self):
        old = self.source['pipeline']
        for bad in [old+old,old.replace('min-slot-count=16384','min-slot-count=8192'),
                old.replace('linalg-fuse-linear-ops,',''),old.replace('linalg-fuse-linear-ops,','linalg-fuse-linear-ops{foreign=true},'),
                old.replace(entry.PROPAGATION_PASS,entry.PROPAGATION_PASS+','+entry.LAYOUT_PASS),
                old.replace('layout-propagation{min-slot-count=16384},','layout-propagation{min-slot-count=16384}')]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):entry.add_layout_optimization(bad)

    def test_both_original_host_sources_strict_math_and_failure_proof_retained(self):
        for variant,pairs in [('alex-serial-json',21),('vgg-rss12-json-overlap',45)]:
            self.configure(variant);self.pipeline_mock.return_value=self.pipeline
            checked=entry.source_gate(self.args)
            self.assertEqual(checked['inherited_strict_math']['activation_pairs'],pairs)
            self.assertEqual(checked['inherited_strict_math']['logit_elements'],30)
            self.assertEqual(checked['original_source'],self.source)
            self.assertEqual(checked['original_native_argv'],self.old_argv)
            self.assertEqual(checked['new_math_runs'],0)
            self.assertTrue(checked['candidate_configuration']['original_linear_fusion_retained'])
            self.assertFalse(checked['original_MLIR_payload_read'])

    def test_original_source_gate_function_body_and_rejection_remain_exact(self):
        function=entry._clone(entry.base.source_gate,dict(vars(entry.base),remove_linear_fusion=entry.add_layout_optimization))
        self.assertIs(function.__code__,entry.base.source_gate.__code__)
        self.record['strict_math_passed']=False;self.persist()
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_exact_original_pipeline_and_params_with_only_one_added_pass(self):
        checked=entry.source_gate(self.args);self.out.mkdir()
        original_text=Path.read_text
        def small_only(path,*args,**kwargs):
            if path==self.case/'worker/model.mlir':raise AssertionError('original IR payload read')
            return original_text(path,*args,**kwargs)
        with patch.object(Path,'read_text',small_only):command=entry.native_command(self.args,self.out,checked)
        pipeline=(self.out/'native/explicit.pipeline').read_text().strip()
        rebased=entry.base._pipeline_at(self.source,self.out/'native')
        self.assertEqual(pipeline,entry.add_layout_optimization(rebased))
        self.assertEqual(command[:-4],entry.base._compiler_argv(self.tool,self.case/'worker/model.mlir',self.out/'native',pipeline))
        self.assertEqual(tuple(command[-4:]),entry.FAILURE_FLAGS)
        self.assertEqual(json.loads((self.out/'native/native-parameters.json').read_text()),self.parameters)
        self.assertEqual((self.case/'native/explicit.pipeline').read_text(),self.old_pipeline+'\n')

    def test_trusted_supervisor_AST_changes_only_two_recording_sites(self):
        tree=entry._execute_tree()
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant)and node.value=='saved-math-original-fusion-layout-16384-continuation':
                node.value='saved-math-no-linear-fusion-continuation'
            if isinstance(node,ast.keyword)and node.arg=='pass_added':
                node.arg='pass_removed';node.value.value='linalg-fuse-linear-ops'
            if isinstance(node,ast.keyword)and node.arg=='added_count':node.arg='removed_count'
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(inspect.getsource(entry.base.execute)),include_attributes=False))
        namespace=entry._namespace()
        self.assertIs(namespace['serial_lock'],entry.base.serial_lock)
        self.assertIs(namespace['run_process'],entry.base.run_process)
        self.assertIs(namespace['physical_observation'],entry.base.physical_observation)
        self.assertIs(namespace['original'].check_native,entry.base.original.check_native)
        self.assertIs(namespace['original'].clean_process,entry.base.original.clean_process)
        self.assertIs(namespace['original'].durable_json,entry.durable_json)
        for name,value in vars(entry.base.original).items():
            if name!='durable_json':self.assertIs(getattr(namespace['original'],name),value)
        self.assertEqual(namespace['_registry'].__code__,entry.base._registry.__code__)
        with self.assertRaises(TypeError):entry._clone(entry.base.serial_lock,{})

    def test_fake_complete_native_scope_identity_resource_and_registry(self):
        result=self.run_entry()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['model_compiled_accepted'])
        self.assertEqual(result['round'],entry.ROUND);self.assertEqual(result['mode'],entry.MODE)
        self.assertEqual(result['variant'],entry.VARIANT)
        self.assertEqual(result['attempted'],{'native_compile':1});self.assertEqual(result['completed'],{'native_compile':1})
        self.assertEqual(result['native_request']['pass_added'],entry.LAYOUT_PASS)
        self.assertEqual(result['native_request']['added_count'],1)
        self.assertNotIn('pass_removed',result['native_request'])
        self.assertTrue(result['original_linear_fusion_retained'])
        self.assertTrue(result['prior_default_layout_crash_risk_retained'])
        self.assertEqual(result['limits'],dict(total_seconds=7440,native_seconds=7200,AS_gib=24,
            subtree_RSS_gib=16,minimum_available_gib=1.25,minimum_free_disk_gib=10,
            file_gib=16,case_gib=32,concurrency=1,completion_reserve_bytes=1024**2))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][1]['mem_gib'],24);self.assertEqual(self.calls[0][1]['fsize_gib'],16)
        self.assertLessEqual(self.calls[0][1]['timeout'],7200)
        self.assertTrue(result['math_inherited']);self.assertFalse(result['new_math_measured'])
        self.assertTrue(all(result[key]==0 for key in entry.ZERO_FIELDS))
        self.assertTrue(all(result[key]is None for key in entry.base.original.FORMAL))
        snapshots={row['original']for row in result['source_snapshots']}
        self.assertIn(str(Path(entry.__file__).resolve()),snapshots)
        self.assertIn(str(Path(entry.base.__file__).resolve()),snapshots)
        rows=[json.loads(line)for line in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertEqual(set(rows[-1]),entry.base.host_sources.REGISTRY_FIELDS)
        self.assertEqual(rows[-1]['round'],entry.ROUND);self.assertEqual(rows[-1]['mode'],entry.MODE)
        self.assertTrue(rows[-1]['metrics']['model_compiled_accepted'])
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'ok')

    def test_fake_failure_keeps_original_case_and_unknown_native_inventory(self):
        def failed(command,**options):
            fixtures.put(options['log'],'original fake failure');fixtures.put(options['resources'],'{}\n')
            return fixtures.observation(1)
        result=self.run_entry(failed)
        self.assertEqual(result['status'],'error');self.assertFalse(result['model_compiled_accepted'])
        self.assertFalse(result['model_compiled']);self.assertIsNone(result['static_native_bootstraps'])
        self.assertEqual(result['completed'],{'native_compile':0})
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'error')

    def test_terminal_physical_guard_revokes_acceptance_keeps_completed_fact(self):
        check=entry.base.original.check_native
        def completed(*args):
            artifact=check(*args);self.available=entry.GIB;return artifact
        with patch.object(entry.base.original,'check_native',side_effect=completed):result=self.run_entry()
        self.assertEqual(result['status'],'error');self.assertEqual(result['original_termination_status'],'resource_stopped')
        self.assertTrue(result['model_compiled']);self.assertEqual(result['completed'],{'native_compile':1})
        self.assertFalse(result['model_compiled_accepted']);self.assertIsNotNone(result['guard_trigger'])
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'error')
        self.assertFalse(json.loads(self.registry.read_text().splitlines()[-1])['metrics']['model_compiled_accepted'])

    def test_monitor_failure_revokes_acceptance_and_stopped_contextmanager_reused(self):
        class Stuck:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def is_alive(self):return True
            def join(self,timeout):self.timeout=timeout
        with patch.object(entry.base.threading,'Thread',Stuck):result=self.run_entry()
        self.assertEqual(result['status'],'error');self.assertFalse(result['monitor_stopped'])
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        with entry.base.serial_lock(self.root):pass

    def test_prepare_and_parser_never_invoke_tool(self):
        self.args.run=False;result=self.run_entry()
        self.assertEqual(result['status'],'prepared');self.assertEqual(self.calls,[])
        self.assertFalse(result['model_compiled_accepted'])
        parsed=entry.parser().parse_args(['--source-case',str(self.case),'--source-variant','alex-serial-json',
            '--out',str(self.out),'--heir-opt',str(self.tool),'--run'])
        self.assertEqual(parsed.source_variant,'alex-serial-json')
        with self.assertRaises(ValueError):self.run_entry()


if __name__=='__main__':unittest.main()
