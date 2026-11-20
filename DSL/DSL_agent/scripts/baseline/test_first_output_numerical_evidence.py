"""Inert protocol controls: never real Agent/compiler/native acceptance.

The fabricated artifacts below only exercise provenance rejection and exact
matching. They do not certify a mathematical producer or a backend capability.
"""
import copy
import json
from pathlib import Path
import unittest

from delivery_component import (bound, canonical, capture_response,
    generator_manifest, sha)
from first_output_assessment import assess_once, evaluation_policy
from first_output_backend import FrozenSourceBackendEvaluator
from first_output_numerical_evidence import (verify_backend_witness,
    verify_native_rows, verify_mathematical_rows)
import test_first_output_backend as fixtures


class NumericalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BackendSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        f = self.fixture
        self.root, self.request, self.source = f.root, f.request, f.source
        # Shape is sample -> ciphertext -> real slots in this recorded profile.
        f.write(f.output/'execution.json', dict(backend='poseidon_cpu',
            encrypted_execution=True, executor_secret_key=False, security='tc128',
            outputs=[[[1., 4.]]]))
        f.report['attempts'][0]['comparison'].update(reference=[[1., 4.]], actual=[[1., 4.]])
        f.refresh()
        self.proof = f.bind()
        sources = {}
        for index in range(2):
            path = self.root/('inert_producer_'+str(index)+'.py')
            path.write_text('# Protocol fixture only; role '+str(index)+'\n')
            sources[str(path)] = sha(path.read_bytes())
        self.policy = evaluation_policy([sha(canonical([1., 2.]))],
            domain_sha256='d'*64, approximation_requirements=[
                dict(id='max_absolute_error', comparison='<=', limit=.01)],
            application_requirements=[dict(id='accuracy', comparison='>=', limit=.9)],
            whole_model_required=True, independent_math_sources=sources,
            original_model_sha256=sha(canonical(self.request['model'])))
        row = dict(input_sha256=self.policy['inputs'][0], source_sha256=sha(self.source),
            original_reference=[1., 4.], declared_reference=[1., 4.],
            dsl_mathematical_output=[1., 4.], CKKS_real=[1., 4.], CKKS_imag=None,
            inside_frozen_domain=True, domain_sha256='d'*64,
            application_metrics=dict(accuracy=.95), whole_model_output=True)
        self.witness = dict(source_sha256=sha(self.source), compiler_source_sha256=sha(self.source),
            compiled_validated=True, real_compiler=True, encrypted_execution=True,
            real_CKKS=True, test_only=False, backend_binding=self.proof,
            intermediate_decryptions=0, reference_or_tolerance_changed=False,
            inputs=[row])
        self.math = dict(format='poseidon-three-layer-mathematical-evidence-v1',
            request_id=self.request['request_id'], source_sha256=sha(self.source),
            policy_sha256=sha(canonical(self.policy)),
            original_model_sha256=self.policy['original_model_sha256'],
            declared_model_sha256=sha(canonical(self.request['model'])),
            producer_sources=sources, candidate_python_executed=False,
            CKKS_outputs_used_as_reference=False, test_only=False,
            inputs=[{key: value for key, value in row.items() if key not in ('CKKS_real', 'CKKS_imag')}])
        self.save_math()

    def save_math(self):
        self.math_path = self.root/'inert-mathematical-record.json'
        self.math_path.write_bytes(canonical(bound(self.math)))
        self.witness['mathematical_evidence'] = dict(path=str(self.math_path), sha256=sha(self.math_path.read_bytes()))

    def assess(self, witness=None, policy=None):
        policy = self.policy if policy is None else policy
        mechanism = generator_manifest('inert-protocol-fixture-only-v1', [],
            baseline_binding='0'*64, capabilities_sha256='1'*64,
            evaluation_policy_sha256=sha(canonical(policy)))
        answer = json.dumps(dict(schema=1, request_id=self.request['request_id'], hecate_source=self.source.decode()))
        receipt = capture_response(self.request, answer, self.root/'retrospective', mechanism=mechanism)
        result = assess_once(self.root/'retrospective', receipt['binding'], policy,
            self.root/'assessment', lambda *_: self.witness if witness is None else witness)
        self.assertFalse(result['first_output_eligible'])
        self.assertFalse(result['first_joint_passed'])
        return result

    def test_exact_native_and_math_cells_match_protocol_without_becoming_acceptance(self):
        verify_native_rows(self.witness, self.request, self.source)
        verify_mathematical_rows(self.witness, self.policy, self.request, self.source)
        result = self.assess()
        self.assertTrue(result['backend_binding_verified'])
        self.assertTrue(result['native_values_binding_verified'])
        self.assertTrue(result['independent_math_binding_verified'])
        self.assertTrue(result['raw_joint_passed'])
        self.assertEqual(result['input_groups_evaluated'], 1)

    def test_boolean_success_without_original_source_evidence_is_blocked(self):
        self.witness.pop('backend_binding')
        result = self.assess()
        self.assertEqual(result['status'], 'backend_blocked')
        self.assertFalse(result['raw_compile_passed'])
        self.assertFalse(result['raw_numeric_passed'])
        self.assertEqual(result['input_groups_evaluated'], 0)
        self.assertEqual(result['unverified_numeric_claims']['compared_values'], 2)

    def test_compilation_survives_missing_math_but_numeric_success_does_not(self):
        self.witness.pop('mathematical_evidence')
        result = self.assess()
        self.assertTrue(result['raw_compile_passed'])
        self.assertFalse(result['raw_numeric_passed'])
        self.assertEqual(result['status'], 'backend_blocked')
        self.assertEqual(result['input_groups_evaluated'], 0)

    def test_ckks_cells_cannot_be_replaced_by_reference_or_other_values(self):
        self.witness['inputs'][0]['CKKS_real'] = [1.00000001, 4.]
        result = self.assess()
        self.assertFalse(result['raw_numeric_passed'])
        self.assertFalse(result['native_values_binding_verified'])

    def test_frozen_native_reference_and_input_identity_cannot_change(self):
        for field, replacement in [('declared_reference', [1., 4.000001]), ('input_sha256', '0'*64)]:
            witness = copy.deepcopy(self.witness)
            witness['inputs'][0][field] = replacement
            with self.assertRaises(ValueError): verify_native_rows(witness, self.request, self.source)

    def test_actual_native_decoder_and_unsupported_imaginary_cells_are_enforced(self):
        self.witness['inputs'][0]['CKKS_imag'] = [0., 0.]
        with self.assertRaises(ValueError): verify_native_rows(self.witness, self.request, self.source)
        self.witness['inputs'][0]['CKKS_imag'] = None
        f = self.fixture
        f.report['attempts'][0]['comparison']['actual'] = [[9., 4.]]
        f.refresh(); self.witness['backend_binding'] = f.bind()
        with self.assertRaises(ValueError): verify_native_rows(self.witness, self.request, self.source)

    def test_numeric_failure_still_requires_provenance_instead_of_only_checking_passes(self):
        self.witness['inputs'][0]['CKKS_real'] = [9., 4.]
        result = self.assess()
        self.assertEqual(result['status'], 'backend_blocked')
        self.assertEqual(result['input_groups_evaluated'], 0)

    def test_bound_actual_numerical_failure_retains_concrete_error(self):
        f = self.fixture
        f.write(f.output/'execution.json', dict(backend='poseidon_cpu', encrypted_execution=True,
            executor_secret_key=False, security='tc128', outputs=[[[9., 4.]]]))
        f.report['attempts'][0]['comparison'].update(passed=False, actual=[[9., 4.]])
        f.report['attempts'][0].update(status='failed', failure_layer='numerical_comparison')
        f.report['status'] = 'failed'; f.refresh()
        self.witness['backend_binding'] = f.bind()
        self.witness['inputs'][0]['CKKS_real'] = [9., 4.]
        result = self.assess()
        self.assertEqual(result['status'], 'numeric_failed')
        self.assertTrue(result['raw_compile_passed'])
        self.assertEqual(result['input_groups_evaluated'], 1)
        self.assertEqual(result['inputs'][0]['CKKS']['max_absolute_error'], 8.)

    def test_mathematical_record_and_producer_cannot_change_after_freeze(self):
        self.math_path.write_bytes(self.math_path.read_bytes()+b' ')
        with self.assertRaises(ValueError): verify_mathematical_rows(self.witness, self.policy, self.request, self.source)
        self.save_math()
        path = Path(next(iter(self.policy['independent_math_sources'])))
        path.write_text('# changed fixture producer\n')
        with self.assertRaises(ValueError): verify_mathematical_rows(self.witness, self.policy, self.request, self.source)

    def test_original_model_source_and_frozen_policy_are_required(self):
        for key in ('source_sha256', 'original_model_sha256', 'declared_model_sha256', 'policy_sha256'):
            original = self.math[key]; self.math[key] = '0'*64; self.save_math()
            with self.assertRaises(ValueError): verify_mathematical_rows(self.witness, self.policy, self.request, self.source)
            self.math[key] = original
        policy = copy.deepcopy(self.policy); policy['independent_math_sources'] = {}
        with self.assertRaises(ValueError): verify_mathematical_rows(self.witness, policy, self.request, self.source)

    def test_domain_application_and_original_mathematical_cells_are_bound(self):
        for field, value in [('inside_frozen_domain', False), ('domain_sha256', '0'*64),
                ('application_metrics', dict(accuracy=.9)), ('original_reference', [1., 4.1]),
                ('dsl_mathematical_output', [1., 4.0000001]), ('whole_model_output', False)]:
            witness = copy.deepcopy(self.witness); witness['inputs'][0][field] = value
            with self.assertRaises(ValueError): verify_mathematical_rows(witness, self.policy, self.request, self.source)

    def test_plaintext_candidate_or_ckks_reference_or_test_record_cannot_be_promoted(self):
        for key, value in [('candidate_python_executed', True), ('CKKS_outputs_used_as_reference', True), ('test_only', True)]:
            original = self.math[key]; self.math[key] = value; self.save_math()
            with self.assertRaises(ValueError): verify_mathematical_rows(self.witness, self.policy, self.request, self.source)
            self.math[key] = original

    def test_execution_flags_must_match_actual_bound_report(self):
        for key in ('compiled_validated', 'real_compiler', 'encrypted_execution', 'real_CKKS'):
            witness = copy.deepcopy(self.witness); witness[key] = False
            with self.assertRaises(ValueError): verify_backend_witness(witness, self.request, self.source)

    def test_external_reader_can_attach_math_descriptor_without_overwriting_source_proof(self):
        extra = {key: self.witness[key] for key in ('inputs', 'intermediate_decryptions',
            'reference_or_tolerance_changed', 'mathematical_evidence')}
        evaluator = FrozenSourceBackendEvaluator(self.fixture.backend(), batch=None,
            checkpoint=lambda: None, frozen_identity=dict(v=1), output=self.root/'reader-call',
            numerical_reader=lambda *_: extra, test_only=True)
        witness = evaluator(self.request, self.source)
        self.assertEqual(witness['mathematical_evidence'], extra['mathematical_evidence'])
        self.assertTrue(witness['test_only'])
        verify_native_rows(witness, self.request, self.source)


if __name__ == '__main__': unittest.main()
