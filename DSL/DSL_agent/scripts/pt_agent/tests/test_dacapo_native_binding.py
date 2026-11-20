"""Key-free public fixtures for the separate original-DaCapo binding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import dacapo_native_binding as binding


class DaCapoNativeBindingChecks(unittest.TestCase):
    def test_original_n17_literal_has_no_precision_mapping(self):
        p=binding.native_parameters()
        self.assertEqual((p['log_n'],p['logical_log_slots'],p['max_level'],p['log_default_scale']),(17,16,16,51))
        self.assertEqual(p['log_q'],[55]+[51]*16)
        self.assertEqual(p['log_p'],[61]*3)
        self.assertFalse(p['precision_mapping_applied'])
        self.assertIsNone(p['actual_q'])
        self.assertIsNone(p['security_acceptance'])

    def test_public_high_scale_fixture_and_independent_formula(self):
        plan,manifest,raw,vectors,expected=binding.make_public_fixture()
        observed=binding.plan_inventory(plan,manifest,public_fixture=True)
        self.assertEqual(observed['execution_count'],11)
        self.assertEqual(observed['bootstrap_count'],0)
        self.assertIn(87,observed['value_scales'])
        self.assertIn(127,observed['value_scales'])
        self.assertEqual(observed['inline_scalar_count'],1)
        actual=binding.clear_replay(plan,raw,vectors)
        for ref,got in zip(expected,actual):
            self.assertLessEqual(abs(ref-got),1e-12+1e-12*abs(ref))

    def test_public_bootstrap_fixture_uses_original_branch_after_boot(self):
        plan,manifest,raw,vectors,expected=binding.make_public_fixture(bootstrap=True)
        observed=binding.plan_inventory(plan,manifest,public_fixture=True)
        self.assertEqual(observed['bootstrap_count'],1)
        self.assertEqual(observed['execution_count'],11)
        self.assertEqual(plan['execution'][8]['inputs'],['c'])
        self.assertEqual(observed['terminal_ciphertext_ids'],['final'])
        actual=binding.clear_replay(plan,raw,vectors)
        for ref,got in zip(expected,actual):
            self.assertLessEqual(abs(ref-got),1e-12+1e-12*abs(ref))
        self.assertFalse(observed['source_gates']['original_training_math_verified'])

    def test_original_model_stays_unsupported_without_site_and_math_evidence(self):
        plan,manifest,*_=binding.make_public_fixture()
        observed=binding.plan_inventory(plan,manifest)
        self.assertFalse(observed['execution_scope_ready'])
        self.assertEqual(len(observed['source_gates']['unsupported_reasons']),2)
        with self.assertRaisesRegex(binding.UnsupportedPlan,'scalar versus one-element'):
            binding.require_execution_ready(observed)

    def test_source_proof_requires_every_original_site_and_three_fixed_train_ids(self):
        plan,manifest,*_=binding.make_public_fixture()
        ids=[step['output'] for step in plan['initialization'] if step['payload']['kind']=='inline']
        plan_path=Path(__file__).resolve()
        sites=dict(status='scalar_sites_verified',source_factory=binding.SOURCE_FACTORY,source_plan=str(plan_path),
            original_dsl_scalar_broadcast_interpretation=True,original_plan_modified=False,
            sites=[dict(value_id=name,interpretation='full-slot-scalar',source_path='/trusted/source.py',source_line=72) for name in ids])
        math=dict(status='strict_math_passed',source_factory=binding.SOURCE_FACTORY,source_plan=str(plan_path),
            sample_ids=['train:0','train:1','train:2'],original_poly_preserved=True,
            original_pack_bb32_preserved=True,original_plan_modified=False,
            absolute_tolerance=1e-12,relative_tolerance=1e-12,all_three_passed=True)
        observed=binding.plan_inventory(plan,manifest,source_proof=sites,source_math=math,plan_path=plan_path)
        self.assertTrue(observed['execution_scope_ready'])
        del sites['original_plan_modified']
        observed=binding.plan_inventory(plan,manifest,source_proof=sites,source_math=math,plan_path=plan_path)
        self.assertFalse(observed['execution_scope_ready'])
        math['sample_ids']=['test:0','train:1','train:2']
        observed=binding.plan_inventory(plan,manifest,source_proof=sites,source_math=math,plan_path=plan_path)
        self.assertFalse(observed['source_gates']['original_training_math_verified'])

    def test_source_proofs_bind_both_paths_to_the_current_plan(self):
        plan,manifest,*_=binding.make_public_fixture()
        ids=[row['output'] for row in plan['initialization'] if row['payload']['kind']=='inline']
        with tempfile.TemporaryDirectory() as directory:
            actual=Path(directory)/'plan.json'
            foreign=Path(directory)/'foreign.json'
            actual.write_text(json.dumps(plan))
            foreign.write_text(json.dumps(plan))
            sites=dict(status='scalar_sites_verified',source_factory=binding.SOURCE_FACTORY,
                source_plan=str(actual),original_dsl_scalar_broadcast_interpretation=True,
                original_plan_modified=False,sites=[dict(value_id=name,interpretation='full-slot-scalar',
                    source_path='/trusted/source.py',source_line=72) for name in ids])
            mathematics=dict(status='strict_math_passed',source_factory=binding.SOURCE_FACTORY,
                source_plan=str(actual),sample_ids=['train:0','train:1','train:2'],
                original_poly_preserved=True,original_pack_bb32_preserved=True,original_plan_modified=False,
                absolute_tolerance=1e-12,relative_tolerance=1e-12,all_three_passed=True)
            inventory=lambda:binding.plan_inventory(plan,manifest,source_proof=sites,
                source_math=mathematics,plan_path=actual)
            self.assertTrue(inventory()['execution_scope_ready'])
            for proof in (sites,mathematics):
                proof['source_plan']=str(foreign)
                observed=inventory()
                self.assertFalse(observed['execution_scope_ready'])
                with self.assertRaises(binding.UnsupportedPlan):
                    binding.require_execution_ready(observed)
                proof['source_plan']=str(actual)
            self.assertFalse(binding.plan_inventory(plan,manifest,source_proof=sites,
                source_math=mathematics)['execution_scope_ready'])

    def test_source_path_resolves_alias_and_rejects_relative_missing_or_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            actual=Path(directory)/'plan.json'
            actual.write_text('{}')
            alias=Path(directory)/'alias.json'
            alias.symlink_to(actual)
            self.assertTrue(binding._same_source_plan(str(alias),actual))
            self.assertTrue(binding._same_source_plan(str(actual),alias))
            for proof in ('plan.json','',None,str(Path(directory)/'missing.json'),str(Path(directory))):
                self.assertFalse(binding._same_source_plan(proof,actual))
            self.assertFalse(binding._same_source_plan(str(actual),None))
            self.assertFalse(binding._same_source_plan(str(actual),Path(directory)))

    def test_public_fixture_path_bypass_keeps_original_model_math_unverified(self):
        plan,manifest,*_=binding.make_public_fixture()
        observed=binding.plan_inventory(plan,manifest,source_proof={'source_plan':'foreign'},
            source_math={'source_plan':'foreign'},public_fixture=True)
        self.assertTrue(observed['execution_scope_ready'])
        self.assertFalse(observed['source_gates']['original_training_math_verified'])

    def test_unknown_operation_is_rejected_before_execution(self):
        plan,manifest,*_=binding.make_public_fixture()
        plan['execution'][0]['op']='plaintext_simulation'
        with self.assertRaisesRegex(binding.UnsupportedPlan,'Unknown operation'):
            binding.plan_inventory(plan,manifest,public_fixture=True)

    def test_rescale_uses51_and_keeps_original_equations(self):
        plan,manifest,*_=binding.make_public_fixture()
        plan['execution'][2]['attrs']['target_scale_log2']=77
        with self.assertRaisesRegex(binding.UnsupportedPlan,'one51-bit'):
            binding.plan_inventory(plan,manifest,public_fixture=True)

    def test_bootstrap_contract_remains3_51_to16_51(self):
        plan,manifest,*_=binding.make_public_fixture(bootstrap=True)
        plan['execution'][2]['attrs']['target_level']=5
        with self.assertRaisesRegex(binding.UnsupportedPlan,'bootstrap contract'):
            binding.plan_inventory(plan,manifest,public_fixture=True)

    def test_large_level_wrong_context_and_unsafe_locator_rejected(self):
        plan,manifest,*_=binding.make_public_fixture()
        plan['values'][0]['level']=17
        with self.assertRaisesRegex(binding.UnsupportedPlan,'value metadata'):
            binding.plan_inventory(plan,manifest,public_fixture=True)
        plan,manifest,*_=binding.make_public_fixture()
        plan['values'][0]['context']='qwen-bootstrap48'
        with self.assertRaisesRegex(binding.UnsupportedPlan,'value metadata'):
            binding.plan_inventory(plan,manifest,public_fixture=True)
        with self.assertRaisesRegex(binding.UnsupportedPlan,'Unsafe'):
            binding.content_filename('opaque:../credential')

    def test_inline_vector_and_nonfinite_scalar_fail_explicitly(self):
        for payload in ([1.,2.],[float('inf')]):
            plan,manifest,*_=binding.make_public_fixture()
            plan['initialization'][1]['payload']['values']=payload
            with self.assertRaisesRegex(binding.UnsupportedPlan,'finite singleton'):
                binding.plan_inventory(plan,manifest,public_fixture=True)

    def test_public_fixture_override_rejects_a_full_length_program(self):
        plan,manifest,*_=binding.make_public_fixture()
        # Use a valid extended SSA chain so the fixture limit is the reason.
        output=plan['values'][-1]
        for index in range(2):
            name='extra'+str(index)
            plan['values'].append(dict(output,id=name))
            plan['execution'].append(dict(kind='compute',op='negate',inputs=[plan['final_outputs'][0]],
                output=name,place=dict(kind='host',rank=0),ordinal=len(plan['initialization'])+len(plan['execution'])))
            plan['final_outputs']=[name]
        with self.assertRaisesRegex(binding.UnsupportedPlan,'twelve operations'):
            binding.plan_inventory(plan,manifest,public_fixture=True)

    def test_clear_oracle_preserves_source_and_scalar_is_full_slot(self):
        plan,manifest,raw,vectors,_=binding.make_public_fixture()
        before=copy.deepcopy((plan,raw,vectors))
        actual=binding.clear_replay(plan,raw,vectors)
        self.assertEqual((plan,raw,vectors),before)
        self.assertTrue(all(v < -.125 for v in actual))
        self.assertEqual(binding.plan_inventory(plan,manifest,public_fixture=True)['last_use_ciphertext_peak_count'],2)

    def test_public_fixture_preparation_writes_only_public_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)/'public'
            paths=binding.write_public_fixture(work)
            self.assertEqual(Path(paths['input']).stat().st_size,65536*8)
            self.assertEqual((work/'bundle/data/half.bin').stat().st_size,65536*8)
            self.assertEqual(paths['keygen_runs'],0)
            self.assertEqual(paths['ckks_runs'],0)
            expected=json.loads(Path(paths['expected']).read_text())
            self.assertEqual(len(expected['values']),8)
            self.assertIn('no original-model acceptance',expected['scope'])


if __name__=='__main__':
    unittest.main()
