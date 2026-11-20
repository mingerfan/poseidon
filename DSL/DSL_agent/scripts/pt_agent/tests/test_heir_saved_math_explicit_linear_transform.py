"""New bounded metadata fixtures for the explicit target capability variant."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_saved_math_explicit_linear_transform as entry
from pt_agent.tests import test_heir_saved_math_no_linear_fusion as fixtures


class ExplicitTargetCases(unittest.TestCase):
    # Reuse metadata constructors only; no old test class is inherited.
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

    def test_insert_is_unique_after_annotation_and_inverse_preserves_pipeline(self):
        old = self.source['pipeline'].replace('\n','\r\n')
        added = '  '+entry.OVERRIDE_PASS+',\r\n'
        new = entry.add_explicit_linear_transform(old)
        anchor = '  '+entry.ANNOTATION_PASS+',\r\n'
        self.assertEqual(new,old.replace(anchor,anchor+added))
        self.assertEqual(new.replace(added,''),old)
        self.assertEqual(new.count(entry.OVERRIDE_PASS),1)
        self.assertEqual(new.count('linalg-fuse-linear-ops'),1)
        self.assertNotIn('layout-optimization',new)

    def test_wrong_backend_scheme_annotation_or_fusion_rejected(self):
        old=self.source['pipeline']
        for bad in [old+old,old.replace('backend=lattigo','backend=openfhe'),
                old.replace('scheme=ckks','scheme=bgv'),
                old.replace(entry.ANNOTATION_PASS,entry.ANNOTATION_PASS+','+entry.ANNOTATION_PASS),
                old.replace('linalg-fuse-linear-ops,',''),
                old.replace('linalg-fuse-linear-ops,','linalg-fuse-linear-ops{foreign=true},'),
                old.replace(entry.ANNOTATION_PASS+',',entry.ANNOTATION_PASS)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):entry.add_explicit_linear_transform(bad)

    def test_duplicate_or_other_existing_overrides_are_rejected(self):
        old=self.source['pipeline']
        for inserted in [entry.OVERRIDE_PASS,'apply-config-override{config=bootstrapLevelsConsumed=1}',
                'backend.config_override','hasKernelLinearTransform=0']:
            bad=old.replace('  linalg-fuse-linear-ops,','  '+inserted+',\n  linalg-fuse-linear-ops,')
            with self.subTest(inserted=inserted),self.assertRaises(ValueError):entry.add_explicit_linear_transform(bad)
        with self.assertRaises(ValueError):entry.add_explicit_linear_transform(entry.add_explicit_linear_transform(old))

    def test_closed_host_source_and_original_math_are_delegated_for_both_models(self):
        for variant,pairs,constants in [('alex-serial-json',21,31),('vgg-rss12-json-overlap',45,61)]:
            self.configure(variant);self.pipeline_mock.return_value=self.pipeline
            checked=entry.source_gate(self.args)
            self.assertEqual(checked['inherited_strict_math'],dict(passed=True,training_samples=3,
                activation_pairs=pairs,logit_elements=30,emitted_constants=constants))
            self.assertEqual(checked['original_source'],self.source)
            self.assertEqual(checked['original_native_argv'],self.old_argv)
            self.assertEqual(checked['new_math_runs'],0)
            self.assertFalse(checked['original_MLIR_payload_read'])
            self.assertEqual(checked['candidate_configuration']['pass_added'],entry.OVERRIDE_PASS)
            self.assertFalse(checked['candidate_configuration']['installed_default_or_model_branch_verified'])

    def test_original_source_identity_and_math_rejections_are_retained(self):
        original=copy.deepcopy(self.record)
        for field,value in [('round',entry.ROUND),('mode',entry.MODE),('strict_math_passed',False),
                ('model_compiled_accepted',True)]:
            self.record=copy.deepcopy(original);self.record[field]=value;self.persist()
            with self.subTest(field=field),self.assertRaises(ValueError):entry.source_gate(self.args)
        namespace=dict(vars(entry.base),remove_linear_fusion=entry.add_explicit_linear_transform)
        self.assertIs(entry._clone(entry.base.source_gate,namespace).__code__,entry.base.source_gate.__code__)

    def test_actual_command_builder_changes_only_pipeline_and_fresh_storage(self):
        checked=entry.source_gate(self.args);self.out.mkdir()
        original_read=Path.read_text
        def small_only(path,*args,**kwargs):
            if path==self.case/'worker/model.mlir':raise AssertionError('Original model IR payload read')
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',small_only):command=entry.native_command(self.args,self.out,checked)
        pipeline=(self.out/'native/explicit.pipeline').read_text().strip()
        before=entry.base._pipeline_at(self.source,self.out/'native')
        original_command=entry.base._compiler_argv(self.tool,self.case/'worker/model.mlir',self.out/'native',before)
        self.assertEqual(pipeline,entry.add_explicit_linear_transform(before))
        self.assertEqual(command[:2],original_command[:2]);self.assertEqual(command[3:-4],original_command[3:])
        self.assertEqual(command[2],'--pass-pipeline='+pipeline)
        self.assertEqual(tuple(command[-4:]),entry.FAILURE_FLAGS)
        self.assertEqual(json.loads((self.out/'native/native-parameters.json').read_text()),self.parameters)
        self.assertEqual((self.case/'native/explicit.pipeline').read_text(),self.old_pipeline+'\n')

    def test_private_namespace_and_inverse_AST_keep_original_supervision(self):
        original_names={k:getattr(entry.base,k) for k in ('MODULE','ROUND','MODE','VARIANT','source_gate','native_command','durable_json')}
        tree=entry._execute_tree()
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant)and node.value=='saved-math-explicit-linear-transform-continuation':
                node.value='saved-math-no-linear-fusion-continuation'
            if isinstance(node,ast.keyword)and node.arg=='pass_added':
                node.arg='pass_removed';node.value.value='linalg-fuse-linear-ops'
            if isinstance(node,ast.keyword)and node.arg=='added_count':node.arg='removed_count'
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(inspect.getsource(entry.base.execute)),include_attributes=False))
        namespace=entry._namespace()
        for key in ('serial_lock','run_process','physical_observation','observe_child','TOTAL_SECONDS','NATIVE_SECONDS','CASE_BYTES','RESERVE_BYTES'):
            self.assertIs(namespace[key],getattr(entry.base,key))
        self.assertIs(namespace['source_gate'],entry.source_gate);self.assertIs(namespace['native_command'],entry.native_command)
        self.assertIs(namespace['original'].durable_json,entry.durable_json)
        for key,value in vars(entry.base.original).items():
            if key!='durable_json':self.assertIs(getattr(namespace['original'],key),value)
        self.assertIs(namespace['_registry'].__code__,entry.base._registry.__code__)
        for key,value in original_names.items():self.assertIs(getattr(entry.base,key),value)
        with self.assertRaises(TypeError):entry._clone(entry.base.serial_lock,{})

    def test_fake_supervisor_records_separate_identity_and_complete_source_snapshot(self):
        result=self.run_entry()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['model_compiled_accepted'])
        self.assertEqual((result['round'],result['mode'],result['variant']),(entry.ROUND,entry.MODE,entry.VARIANT))
        self.assertEqual(result['native_request']['pass_added'],entry.OVERRIDE_PASS)
        self.assertEqual(result['native_request']['added_count'],1);self.assertNotIn('pass_removed',result['native_request'])
        self.assertTrue(result['original_linear_fusion_retained'])
        self.assertFalse(result['model_conversion_branch_verified']);self.assertFalse(result['model_specific_native_codegen_verified'])
        self.assertEqual(result['attempted'],dict(native_compile=1));self.assertEqual(result['completed'],dict(native_compile=1))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][1]['mem_gib'],24);self.assertEqual(self.calls[0][1]['fsize_gib'],16)
        self.assertLessEqual(self.calls[0][1]['timeout'],7200)
        self.assertEqual(result['limits'],dict(total_seconds=7440,native_seconds=7200,AS_gib=24,
            subtree_RSS_gib=16,minimum_available_gib=1.25,minimum_free_disk_gib=10,
            file_gib=16,case_gib=32,concurrency=1,completion_reserve_bytes=1024**2))
        self.assertTrue(all(result[k]==0 for k in entry.ZERO_FIELDS))
        self.assertTrue(all(result[k]is None for k in entry.base.original.FORMAL))
        snapshots={row['original']for row in result['source_snapshots']}
        self.assertIn(str(Path(entry.__file__).resolve()),snapshots);self.assertIn(str(Path(entry.base.__file__).resolve()),snapshots)
        rows=[json.loads(line)for line in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertEqual(set(rows[-1]),entry.base.host_sources.REGISTRY_FIELDS)
        self.assertEqual(rows[-1]['round'],entry.ROUND);self.assertEqual(rows[-1]['mode'],entry.MODE)
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'ok')

    def test_fake_compiler_failure_preserves_original_failure_and_unknown_inventory(self):
        def failed(command,**options):
            fixtures.put(options['log'],'New fake native failure');fixtures.put(options['resources'],'{}\n')
            return fixtures.observation(1)
        result=self.run_entry(failed)
        self.assertEqual(result['status'],'error');self.assertFalse(result['model_compiled_accepted'])
        self.assertFalse(result['model_compiled']);self.assertIsNone(result['static_native_bootstraps'])
        self.assertEqual(result['completed'],dict(native_compile=0))
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        self.assertFalse(json.loads(self.registry.read_text().splitlines()[-1])['metrics']['model_compiled_accepted'])


if __name__=='__main__':unittest.main()
