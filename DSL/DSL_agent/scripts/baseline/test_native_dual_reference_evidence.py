"""Fixture controls only; patched proofs never count as native certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from delivery_component import bound,canonical,sha,generator_manifest
from first_output_assessment import evaluation_policy
from generation_protocol import Config
import lossless_host_bound_delivery as host_entry
import native_dual_reference_evidence as native
import dual_reference_first_output_assessment as assessment


class DualReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=b'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return x\n'
        self.key='a'*64;self.request=dict(request_id=self.key,layout=dict(output_selectors=[[0,0],[0,1]]))

    def write_native(self,core=None,actual=None,semantic=None):
        core=[[1.,2.],[3.,4.]]if core is None else core
        actual=copy.deepcopy(core)if actual is None else actual
        semantic=[[1.+1e-14,2.],[3.,4.]]if semantic is None else semantic
        root=self.root/'native';output=root/'attempt-00/output';output.mkdir(parents=True)
        inputs=[[[.25,.5]],[[.5,.75]]]
        records={root/'report.json':dict(attempts=[dict(comparison=dict(reference=core,actual=actual))]),
            output/'execution.json':dict(outputs=[[list(row)+[99.]]for row in actual]),output/'inputs.json':inputs}
        for p,data in records.items():p.write_bytes(canonical(data))
        rows=[dict(input_sha256=sha(canonical(sample)),source_sha256=sha(self.source),CKKS_real=list(a),CKKS_imag=None,
            declared_reference=list(d),native_core_reference=list(r))for sample,a,d,r in zip(inputs,actual,semantic,core)]
        witness=dict(source_sha256=sha(self.source),compiler_source_sha256=sha(self.source),compiled_validated=True,real_compiler=True,
            encrypted_execution=True,real_CKKS=True,test_only=False,inputs=rows,intermediate_decryptions=0,reference_or_tolerance_changed=False)
        proof=dict(evidence=str(root),real_compilation_observed=True,real_encrypted_execution_observed=True,unit_fixture=True)
        return witness,proof,records

    def native_check(self,witness,proof):
        with patch.object(native,'verify_backend_witness',return_value=proof):return native.verify_native_rows(witness,self.request,self.source)

    def test_small_reference_difference_is_preserved_and_checked(self):
        witness,proof,records=self.write_native();before=copy.deepcopy(witness);pins={p:sha(p.read_bytes())for p in records}
        report=self.native_check(witness,proof)
        self.assertTrue(report['passed']);self.assertEqual(report['input_groups'],2);self.assertEqual(report['compared_values'],4)
        self.assertGreater(report['inputs'][0]['reference_alignment']['max_absolute_error'],0.)
        self.assertEqual(witness,before);self.assertEqual(pins,{p:sha(p.read_bytes())for p in records})

    def test_changed_core_reference_is_an_evidence_error_even_inside_tolerance(self):
        witness,proof,_=self.write_native();witness['inputs'][0]['native_core_reference'][0]+=1e-14
        with self.assertRaisesRegex(ValueError,'core reference changed'):self.native_check(witness,proof)

    def test_changed_native_values_or_source_cannot_be_substituted(self):
        witness,proof,_=self.write_native();a=copy.deepcopy(witness);a['inputs'][0]['CKKS_real'][0]+=1e-14
        with self.assertRaisesRegex(ValueError,'original decoded output'):self.native_check(a,proof)
        for field in ['source_sha256','compiler_source_sha256']:
            a=copy.deepcopy(witness);a[field]='f'*64
            with self.assertRaisesRegex(ValueError,'Original source required'):self.native_check(a,proof)

    def test_large_reference_gap_is_reported_as_a_numeric_failure(self):
        witness,proof,_=self.write_native(semantic=[[1.1,2.],[3.,4.]])
        report=self.native_check(witness,proof);self.assertFalse(report['passed'])
        self.assertGreater(report['inputs'][0]['reference_alignment']['failed_values'],0)
        self.assertEqual(report['inputs'][0]['CKKS_against_core']['failed_values'],0)
        self.assertGreater(report['inputs'][0]['CKKS_against_declared']['failed_values'],0)

    def test_core_CKKS_failure_is_not_hidden_by_semantic_CKKS_success(self):
        witness,proof,_=self.write_native(core=[[-1e-12,2.],[3.,4.]],actual=[[1e-5,2.],[3.,4.]],semantic=[[0.,2.],[3.,4.]])
        report=self.native_check(witness,proof);row=report['inputs'][0]
        self.assertEqual(row['reference_alignment']['failed_values'],0);self.assertEqual(row['CKKS_against_declared']['failed_values'],0)
        self.assertEqual(row['CKKS_against_core']['failed_values'],1);self.assertFalse(report['passed'])

    def test_missing_duplicate_complex_or_nonfinal_input_evidence_is_rejected(self):
        witness,proof,_=self.write_native();cases=[]
        a=copy.deepcopy(witness);a['inputs']=a['inputs'][:1];cases.append(a)
        a=copy.deepcopy(witness);a['inputs']=[a['inputs'][0],a['inputs'][0]];cases.append(a)
        a=copy.deepcopy(witness);a['inputs'][0]['CKKS_imag']=[0.,0.];cases.append(a)
        a=copy.deepcopy(witness);a['intermediate_decryptions']=1;cases.append(a)
        a=copy.deepcopy(witness);a['test_only']=True;cases.append(a)
        for case in cases:
            with self.assertRaises(ValueError):self.native_check(case,proof)

    def external_fixture(self,source):
        output=self.root/'assessment';input_hash='c'*64
        policy=evaluation_policy([input_hash],domain_sha256='d'*64,approximation_requirements=[dict(id='max_absolute_error',comparison='<=',limit=0.)],application_requirements=[])
        plan=bound(dict(format='poseidon-first-output-delivery-plan-v1',slots={self.key:dict(request_id=self.key,request_sha256='b'*64,
            delivery_path=str(self.root/'delivery'),policy=policy,split='development',unseen=False)},independent_seal_verified=False))
        manifest=generator_manifest('external-fixture',list(assessment.source_manifest()),baseline_binding='0'*64,capabilities_sha256='0'*64,
            evaluation_policy_sha256=sha(canonical(plan)),provider_config=Config(max_calls=1,max_tokens=128,reasoning_effort='none'))
        manifest=bound(dict({k:v for k,v in manifest.items()if k!='binding'},host_bound_contract=host_entry.VERSION,
            host_request_bindings={self.key:dict(public_canonical_sha256='b'*64,binding='e'*64)},host_pre_generation_plans={self.key:{}},
            host_assessment_paths={self.key:str(output)},host_backend_identity=dict(unit_fixture=True)))
        receipt=dict(request_id=self.key,source_sha256=None if source is None else sha(source),status='unsupported'if source is None else'delivered',
            origin='pre_generation_admission'if source is None else'provider_first_output',first_output_eligible=False)
        return output,plan,manifest,receipt,policy,input_hash

    def test_external_mode_must_be_frozen_and_absent_source_never_calls_evaluator(self):
        output,plan,manifest,receipt,_,_=self.external_fixture(None);evaluator=Mock(side_effect=AssertionError('No source to assess'))
        with patch.object(assessment,'verify_host_delivery',return_value=(receipt,self.request,None,manifest)):
            with self.assertRaisesRegex(ValueError,'Pre-generation external mode'):assessment.assess_once(self.root/'delivery','f'*64,plan,output,evaluator)
        self.assertFalse(output.exists());selected=assessment.bind_assessment(manifest)
        with patch.object(assessment,'verify_host_delivery',return_value=(receipt,self.request,None,selected)):
            report=assessment.assess_once(self.root/'delivery','f'*64,plan,output,evaluator)
        self.assertEqual(report['status'],'unsupported');self.assertFalse(report['first_joint_passed']);evaluator.assert_not_called()
        self.assertEqual(json.loads((output/'attempt.json').read_bytes())['feedback_to_Agent'],0)

    def test_external_numeric_failure_keeps_valid_native_groups_and_compile_status(self):
        output,plan,manifest,receipt,_,input_hash=self.external_fixture(self.source);selected=assessment.bind_assessment(manifest)
        witness=dict(source_sha256=sha(self.source),compiler_source_sha256=sha(self.source),compiled_validated=True,real_compiler=True,
            encrypted_execution=True,real_CKKS=True,test_only=False,intermediate_decryptions=0,reference_or_tolerance_changed=False,
            inputs=[dict(input_sha256=input_hash,source_sha256=sha(self.source),original_reference=[1.,2.],declared_reference=[1.,2.],
                dsl_mathematical_output=[1.,2.],CKKS_real=[1.,2.],CKKS_imag=None,native_core_reference=[1.,2.],
                inside_frozen_domain=True,domain_sha256='d'*64,application_metrics={},whole_model_output=True)])
        pairs=bound(dict(format=native.VERSION,passed=False,unit_fixture=True))
        evaluator=Mock(return_value=witness)
        with patch.object(assessment,'verify_host_delivery',return_value=(receipt,self.request,self.source,selected)),\
            patch('first_output_numerical_evidence.verify_backend_witness',return_value=dict(unit_fixture=True)),\
            patch('first_output_numerical_evidence.verify_mathematical_rows',return_value=dict(unit_fixture=True)),\
            patch.object(native,'verify_native_rows',return_value=pairs):
            report=assessment.assess_once(self.root/'delivery','f'*64,plan,output,evaluator)
        self.assertEqual(report['status'],'numeric_failed');self.assertTrue(report['raw_compile_passed']);self.assertFalse(report['raw_numeric_passed'])
        self.assertTrue(report['native_values_binding_verified']);self.assertTrue(report['independent_math_binding_verified'])
        self.assertEqual(report['input_groups_evaluated'],1);self.assertEqual(report['compared_values'],2);evaluator.assert_called_once()
        self.assertFalse(report['first_joint_passed'])


if __name__=='__main__':unittest.main()
