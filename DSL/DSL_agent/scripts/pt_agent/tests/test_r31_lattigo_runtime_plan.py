"""Stdlib checks for the independent plan adapter; no crypto or model loading."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import r31_lattigo_runtime_plan as adapter


class PlanInventoryChecks(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.work=Path(temporary.name)
        plan,bundle,self.input,self.expected=adapter.make_regression(self.work)
        self.plan=json.loads(plan.read_text());self.manifest=json.loads((bundle/'manifest.json').read_text())

    def test_full_fixture_symbols_liveness_and_original_parameter_variant(self):
        result=adapter.plan_inventory(self.plan,self.manifest)
        self.assertEqual(result['execution_count'],11)
        self.assertEqual(result['encode_count'],2)
        self.assertEqual(result['bootstrap_count'],0)
        self.assertEqual(result['rotation_steps'],[1,16383])
        self.assertEqual(result['terminal_ciphertext_ids'],['y'])
        self.assertLess(result['last_use_ciphertext_peak_count'],len(self.plan['values']))
        self.assertEqual(result['op_counts']['rescale'],2)
        self.assertEqual(result['logical_slots'],14)
        self.assertIsNone(result['security_acceptance'])
        self.assertFalse(result['concrete_prime_binding_verified'])

    def test_scale_components_level_and_topology_failures_are_explicit(self):
        for field,value in (('components',2),('scale_log2',43),('level',4)):
            changed=copy.deepcopy(self.plan)
            square=next(v for v in changed['values'] if v['id']=='square');square[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                adapter.plan_inventory(changed,self.manifest)
        changed=copy.deepcopy(self.plan);changed['execution'][0]['inputs']=['y']
        with self.assertRaises(ValueError):adapter.plan_inventory(changed,self.manifest)

    def test_bundle_strings_are_locators_and_traversal_is_rejected(self):
        self.assertEqual(adapter.content_filename('fixture:quarter'),'quarter.bin')
        literal='a'*64
        self.assertEqual(adapter.content_filename('sha256:'+literal),literal+'.bin')
        for name in ('sha256:../secret','sha256:/absolute','unqualified','sha256:'):
            with self.subTest(name=name),self.assertRaises(ValueError):adapter.content_filename(name)
        changed=copy.deepcopy(self.manifest);changed['blobs'][0]['byte_length']-=8
        with self.assertRaises(ValueError):adapter.plan_inventory(self.plan,changed)

    def test_original_native_boot_target_is_preserved_in_static_plan(self):
        changed=copy.deepcopy(self.plan)
        source=changed['external_inputs'][0]
        old_output=changed['final_outputs'][0]
        new_value=copy.deepcopy(next(v for v in changed['values'] if v['id']==old_output))
        new_value.update(id='refreshed',level=5)
        changed['values'].append(new_value)
        changed['execution'].append(dict(kind='compute',ordinal=13,op='boot',inputs=[old_output],
            output='refreshed',place=dict(kind='host',rank=0),attrs=dict(implementation='native',
            operator_profile='original-metadata',target_level=5,target_scale_log2=43,target_components=2)))
        changed['final_outputs']=['refreshed']
        result=adapter.plan_inventory(changed,self.manifest)
        self.assertEqual(result['bootstrap_count'],1)
        self.assertEqual(changed['execution'][-1]['attrs']['operator_profile'],'original-metadata')
        self.assertEqual(source,'x')

    def test_inventory_opens_only_metadata_and_never_bundle_payloads(self):
        with patch.object(adapter,'run_process',side_effect=AssertionError('no subprocess')):
            result=adapter.plan_inventory(self.plan,self.manifest)
        self.assertEqual(result['unique_blob_count'],2)
        self.assertEqual(result['extra_integrity_checks'],0)

    def test_native_client_retains_scale_and_only_final_decryption(self):
        source=adapter.GO_SOURCE.read_text()
        compact=''.join(source.split())
        self.assertEqual(source.count('rlwe.NewDecryptor('),1)
        self.assertIn('final.decrypt_decode_compare',source)
        self.assertIn('pt.LogDimensions.Cols=logSlots',compact)
        self.assertIn('eval.DropLevel(out,out.Level()-v.Level)',compact)
        self.assertIn('ct.Scale.Cmp(predicted)!=0',compact)
        self.assertNotIn('sha256.New',source)
        self.assertNotIn('SecretKeyBootstrapper',source)
        self.assertNotIn('DecryptNew(a)',source)


class NativeBootstrapPreparationChecks(unittest.TestCase):
    def test_physical_guard_separates_virtual_allowance_rss_and_host_reserve(self):
        sample=dict(subtree_rss_bytes=13*adapter.GIB,sample_ts_ns=123)
        observed=adapter.physical_memory_observation(sample,2*adapter.GIB)
        self.assertFalse(observed['stop'])
        self.assertNotIn('address_space_bytes',observed)
        self.assertTrue(adapter.physical_memory_observation(
            dict(subtree_rss_bytes=15*adapter.GIB),2*adapter.GIB)['stop'])
        self.assertTrue(adapter.physical_memory_observation(None,adapter.GIB//2)['stop'])
        self.assertEqual(observed['rss_allowance_bytes'],14*adapter.GIB)

    def test_bootstrap_probe_preserves_level1_to5_and_native_post_operations(self):
        from pt_agent.experiments.r31_lattigo_input_worker import make_bootstrap_plan
        target=dict(target_id='poseidon-ckks-cpu',adapter_probe=True)
        plan=make_bootstrap_plan(target)
        manifest=dict(bundle_format_version=1,version=1,bundle_id=plan['plaintext_bundle']['id'],
            blobs=[dict(content='fixture:offset',byte_length=16384*8)])
        inventory=adapter.plan_inventory(plan,manifest)
        self.assertEqual(inventory['op_counts'],{'add_cp':1,'boot':1,'rotate':1})
        self.assertEqual(plan['values'][0]['level'],1)
        self.assertEqual(plan['execution'][0]['attrs']['target_level'],5)
        self.assertEqual(plan['target'],target)
        self.assertEqual(inventory['terminal_ciphertext_ids'],['y'])

    def test_source_gate_is_metadata_only_and_preserves_frozen_training_paths(self):
        from pt_agent.experiments import r31_lattigo_input_worker as worker
        from types import SimpleNamespace
        with TemporaryDirectory() as temporary:
            root=Path(temporary);program=root/'program.ptdsl';program.write_text('def fixture(x):\n return x\n')
            probes=root/'never-read-probes.npz';probes.write_text('metadata-only fixture')
            probe_source=root/'probe-source.json';probe_source.write_text(json.dumps(dict(split='train',
                indices=[0,1,2],sample_ids=['train:0','train:1','train:2'],normalization={'preserved':True})))
            r13=root/'r13.json';model_info=root/'model_info'
            r13.write_text(json.dumps(dict(round='R13',status='ok',model=worker.MODEL,artifacts=dict(
                training_probes=str(probes),training_probe_source=str(probe_source),model_info=str(model_info)))))
            original=root/'agent-result.json'
            original.write_text(json.dumps(dict(status='ok',model='resnet20_silu',worker=dict(
                model='resnet20_silu',source_model_info=str(model_info/'model_info.json'),
                frozen=True,quality_acceptance={'passed':True}))))
            audit=root/'post-run-quality-audit.json';audit.write_text(json.dumps(dict(status='quality_accepted',
                formal_frozen=True,strict_math_passed=True,quality_acceptance_passed=True,
                model='resnet20_silu',source_result=str(original),selected_program=str(program))))
            compilation=root/'compile-result.json';compilation.write_text(json.dumps(dict(round='R29',status='ok',
                system='agent-pars',source_model_info=str(model_info/'model_info.json'),source=dict(
                independent_audit_passed=True,program=str(program),source_result=str(original)))))
            args=SimpleNamespace(r13_result=r13,compilation_result=compilation,probes=probes,
                                 probe_source=probe_source,program=program)
            source=worker.check_sources(args)
            self.assertEqual(source['sample_id'],'train:0')
            self.assertFalse(source['normalization_reapplied'])
            changed=json.loads(audit.read_text());changed['formal_frozen']=False;audit.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):worker.check_sources(args)


if __name__=='__main__':unittest.main()
