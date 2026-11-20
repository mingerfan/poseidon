"""Twelve new fixture controls; patched compiler proofs are never certification."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from candidate_contract import request_input_names
from delivery_component import bound, canonical, sha
import delivery_component as delivery
import first_output_planning as planning
import lossless_host_bound_delivery as host
import prepared_first_delivery as prepared
import bounded_generation_context as context
import completed_delivery_startup as startup
from deepseek_provider import DeepSeekProvider
from generation_protocol import Config
import retained_math_producer as producer
import retained_dual_reference_rows as references
import retained_first_output_mode as mode
import retained_external_assessment as assessment
import test_retained_math_producer as mathematical_fixtures
import test_host_bound_delivery as host_fixtures
import test_bounded_prepared_delivery as wire_fixtures


class RetainedExternalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = producer.implementation_bundle()
        cls.authority = planning.source_authority()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        original = copy.deepcopy(self.bundle)
        def bundle(extra_sources=()):
            value = copy.deepcopy(original)
            value['sources'].update({str(Path(p)): sha(Path(p).read_bytes()) for p in extra_sources})
            return value
        cache = patch.object(producer, 'implementation_bundle', side_effect=bundle); cache.start(); self.addCleanup(cache.stop)
        cache = patch.object(planning, 'source_authority', return_value=copy.deepcopy(self.authority)); cache.start(); self.addCleanup(cache.stop)
        self.c = mathematical_fixtures.Case(self.root/'external')
        self.request, self.planning = planning.prepare_plan(self.c.model, host_fixtures.constraints(),
            options={}, candidate_limit=1, test_only=True)
        self.c.request = self.request
        self.c.target['request_id'] = self.request['request_id']
        s = self.c.selection
        self.c.selection = references.freeze(s['reader_contract'], s['retained_index'], s['decoder_source'], self.c.target)
        names = request_input_names(self.request)
        self.c.source = ('@hc.func("'+','.join('c' for _ in names)+'")\r\ndef golden('+', '.join(names)+'):\r\n    return '+names[0]+'*'+names[0]+'\r\n\r\n').encode()
        self.policy = self.c.policy(extra_sources=[Path(mode.__file__), Path(assessment.__file__)])
        self.record, self.descriptor = self.c.record(self.policy)
        self.backend = host_fixtures.Backend(); self.identity = self.backend.identity_manifest()
        self.config = Config(max_calls=1, max_tokens=128)
        self.delivery = self.root/'delivery'; self.external = self.root/'assessment'
        self.scope = mode.native_mode(mode.REAL, external_batch_name='fixture-external-after-delivery', native_evidence_root=self.c.root)
        self.witness = self.native_fixture(self.record, self.descriptor)
        self.selected = mode._record(self.scope, self.policy, self.policy['retained_mathematical_producer'], dict(host_backend_identity=self.identity))
        self.first_output_manifest = None

    def native_fixture(self, record, descriptor):
        c = self.c; native = c.root/'native'; output = native/'attempt-00/output'; output.mkdir(parents=True, exist_ok=True)
        physical = producer._normalized(c.request, c.samples)[1]
        selectors = c.request['layout']['output_selectors']; outputs = []
        for row in record['inputs']:
            matrix = [[0.]*(max(slot for ct, slot in selectors if ct == i)+1)
                      for i in range(max(ct for ct, slot in selectors)+1)]
            for (ct, slot), value in zip(selectors, row['dsl_mathematical_output']): matrix[ct][slot] = value
            outputs.append(matrix)
        cores = [row['declared_reference'] for row in record['inputs']]
        actual = [row['dsl_mathematical_output'] for row in record['inputs']]
        (native/'report.json').write_bytes(canonical(dict(attempts=[dict(comparison=dict(reference=cores, actual=actual))])))
        (output/'execution.json').write_bytes(canonical(dict(outputs=outputs)))
        (output/'inputs.json').write_bytes(canonical(physical))
        witness = c.witness(record, descriptor)
        witness.update(compiler_source_sha256=sha(c.source), compiled_validated=True, real_compiler=True,
            encrypted_execution=True, real_CKKS=True, test_only=False, intermediate_decryptions=0, reference_or_tolerance_changed=False)
        for row, core, cells in zip(witness['inputs'], cores, actual):
            row.update(native_core_reference=copy.deepcopy(core), CKKS_real=copy.deepcopy(cells), CKKS_imag=None)
        self.proof = dict(evidence=str(native), real_compilation_observed=True, real_encrypted_execution_observed=True,
                          backend_identity_sha256=sha(canonical(self.identity)))
        witness['backend_binding'] = copy.deepcopy(self.proof)
        return witness

    def check(self, witness=None, policy=None, selected=None, **kwargs):
        args = dict(mechanism_binding='a'*64, test_only=True); args.update(kwargs)
        with patch.object(assessment.backend_evidence, 'verify_backend_witness', return_value=self.proof), \
             patch.object(assessment.native_evidence, 'verify_backend_witness', return_value=self.proof):
            return assessment.check_witness(self.witness if witness is None else witness,
                self.policy if policy is None else policy, self.request, self.c.source,
                self.selected if selected is None else selected, **args)

    def freeze(self, abi=mode.REAL):
        path = self.root/'host.json'; path.write_bytes(canonical(self.request))
        binding = host.freeze_host_request(path)
        self.plan = delivery.delivery_plan([dict(request=self.request, policy=self.policy,
            delivery_path=str(self.delivery), split='development')])
        paths = set(mode.source_manifest()) | set(self.policy['independent_math_sources'])
        base = host.generator_manifest('retained-external-offline-controls-v1', sorted(paths),
            host_bindings=[binding], plans=[self.planning], assessment_paths={self.request['request_id']: str(self.external)},
            backend_identity=self.identity, baseline_binding='0'*64, evaluation_policy_sha256=sha(canonical(self.plan)), provider_config=self.config)
        base = startup.external.bind_assessment(base); base = startup.bind_startup(base)
        self.context = context.freeze(host.public_request(self.request))
        self.base = prepared.bind(base, {self.request['request_id']: self.context})
        scope = mode.native_mode(abi, external_batch_name='fixture-external-after-delivery', native_evidence_root=self.c.root)
        self.first_output_manifest = mode.bind(self.base, self.plan, {self.request['request_id']: scope})
        return self.first_output_manifest

    def generate(self, abi=mode.REAL):
        if self.first_output_manifest is None: self.freeze(abi)
        answer = json.dumps(dict(schema=1, request_id=self.request['request_id'], hecate_source=self.c.source.decode()))
        self.wire = wire_fixtures.OfflineWire(self.context, host.public_request(self.request), self.config, answer)
        self.provider = DeepSeekProvider(self.config, transport=self.wire)
        return mode.generate_once(self.request, self.provider, self.delivery, mechanism=self.first_output_manifest,
            plan=self.plan, backend=self.backend, backend_identity=self.identity, test_only=True)

    def evaluate(self, receipt, callback=None):
        with patch.object(assessment.backend_evidence, 'verify_backend_witness', return_value=self.proof), \
             patch.object(assessment.native_evidence, 'verify_backend_witness', return_value=self.proof):
            return assessment.assess_once(self.delivery, receipt['binding'], self.plan, self.external,
                (lambda request, source: copy.deepcopy(self.witness)) if callback is None else callback, test_only=True)

    def test_01_mode_and_all_math_sources_frozen_before_new_delivery(self):
        mechanism = self.freeze(); mode.verify_selection(mechanism, self.plan)
        self.assertNotEqual(mechanism['binding'], self.base['binding'])
        self.assertEqual(mechanism['retained_first_output_slots'][self.request['request_id']]['generation_status'], 'prepared')
        self.assertTrue(set(self.policy['independent_math_sources']) <= set(mechanism['source_sha256']))
        with self.assertRaises(ValueError): mode.verify_selection(self.base, self.plan)
        with self.assertRaises(ValueError): mode.bind(mechanism, self.plan, {self.request['request_id']: self.scope})

    def test_02_no_tolerance_relaxation_or_real_slot_subsampling(self):
        self.freeze(); altered = copy.deepcopy(self.plan); numeric = altered['slots'][self.request['request_id']]['policy']
        numeric['CKKS']['atol'] = .01
        with self.assertRaises(ValueError): mode.verify_selection(self.first_output_manifest, altered)
        policy = copy.deepcopy(self.policy); policy['inputs'] = [sha(str(i).encode()) for i in range(44)]
        blocked = mode._record(self.scope, policy, self.policy['retained_mathematical_producer'], dict(host_backend_identity=self.identity))
        self.assertEqual(blocked['retained_input_groups'], 44); self.assertEqual(blocked['generation_status'], 'resource_blocked')
        self.assertEqual(len(policy['inputs']), 44)

    def test_03_independent_retained_math_and_original_real_slot_readers_both_dispatch(self):
        result = self.check()
        self.assertTrue(result['raw_joint_passed']); self.assertTrue(result['test_only'])
        self.assertTrue(result['backend_binding_verified']); self.assertTrue(result['independent_math_binding_verified'])
        self.assertTrue(result['native_values_binding_verified']); self.assertEqual(result['compared_values'], 4)
        self.assertGreater(result['original_function_to_declared_approximation'][0]['max_absolute_error'], 0.)
        self.assertEqual(sum(r['mathematical']['failed_values'] for r in result['inputs']), 0)

    def test_04_missing_partial_and_changed_math_records_never_become_numeric_passes(self):
        missing = copy.deepcopy(self.witness); missing.pop('mathematical_evidence')
        result = self.check(missing); self.assertTrue(result['raw_compile_passed']); self.assertFalse(result['raw_numeric_passed'])
        self.assertEqual(result['input_groups_evaluated'], 0)
        partial, descriptor = self.c.record(self.policy, indices=[0])
        witness = copy.deepcopy(self.witness); witness['mathematical_evidence'] = descriptor; witness['inputs'] = witness['inputs'][:1]
        self.assertFalse(self.check(witness)['raw_numeric_passed'])
        changed = copy.deepcopy(self.witness); changed['inputs'][0]['dsl_mathematical_output'][0] += 1.
        self.assertFalse(self.check(changed)['independent_math_binding_verified'])

    def test_05_reported_native_cells_imaginary_and_core_reference_cannot_be_substituted(self):
        for field in ('CKKS_real', 'native_core_reference'):
            changed = copy.deepcopy(self.witness); changed['inputs'][0][field][0] += 1.
            with self.subTest(field=field):
                result = self.check(changed); self.assertFalse(result['raw_numeric_passed']); self.assertFalse(result['native_values_binding_verified'])
        changed = copy.deepcopy(self.witness); changed['inputs'][0]['CKKS_imag'] = [0., 0.]
        self.assertFalse(self.check(changed)['native_values_binding_verified'])

    def test_06_verified_outside_domain_values_remain_numerical_failures(self):
        policy = self.c.policy(domain=dict(input0=[-.001, .001]))
        record, descriptor = self.c.record(policy); witness = self.native_fixture(record, descriptor)
        selected = mode._record(self.scope, policy, policy['retained_mathematical_producer'], dict(host_backend_identity=self.identity))
        result = self.check(witness, policy, selected)
        self.assertTrue(result['independent_math_binding_verified']); self.assertFalse(result['raw_numeric_passed'])
        self.assertFalse(result['domain_verified']); self.assertEqual(result['input_groups_evaluated'], 2)

    def test_07_missing_original_full_application_evidence_is_not_inferred_from_cases(self):
        path, digest = next(iter(self.policy['independent_math_sources'].items()))
        policy = self.c.policy(whole_model_required=True, domain_contract=self.c.domain_contract,
            application_requirements=[dict(id='top1_accuracy_difference', comparison='<=', limit=.01)],
            application_contract=dict(samples=10000, dataset_sha256='e'*64, producer_sources={path: digest}), required_application_samples=10000)
        domain = self.c.domain(policy)
        record, descriptor = self.c.record(policy, domain_evidence=domain); witness = self.native_fixture(record, descriptor)
        selected = mode._record(self.scope, policy, policy['retained_mathematical_producer'], dict(host_backend_identity=self.identity))
        result = self.check(witness, policy, selected)
        self.assertTrue(result['independent_math_binding_verified']); self.assertFalse(result['raw_numeric_passed'])
        self.assertEqual(policy['retained_mathematical_producer']['required_application_samples'], 10000)
        self.assertTrue(all(not r['application_passed'] for r in result['inputs']))

    def test_08_original_terminal_source_scope_attempts_inode_and_freshness_are_required(self):
        root = self.c.root; st = root.stat(); started = time.time()-1
        plan = dict(source_binding='a'*64, stages={k: int(k == 'qualify') for k in assessment.shared_execution.KINDS})
        row = dict(plan=plan, plan_binding=sha(canonical(plan)), status='succeeded', active=None,
            used=copy.deepcopy(plan['stages']), started=started, finished=time.time(), artifacts=[dict(path=str(root), device=st.st_dev, inode=st.st_ino)],
            uncertain_stage=None, last_stage=dict(kind='qualify', number=1, outcome='completed', execution_uncertain=False))
        check = lambda r, minimum=int((started-.1)*1e9): assessment._resource_state(r, self.scope, self.proof, 'a'*64, minimum)
        self.assertTrue(check(row)['native_terminal_certain'])
        for change in ('binding', 'allowance', 'root', 'freshness'):
            wrong = copy.deepcopy(row)
            if change == 'binding': wrong['plan_binding'] = 'b'*64
            if change == 'allowance': wrong['used']['qualify'] = 0
            if change == 'root': wrong['artifacts'][0]['inode'] += 1
            if change == 'freshness': wrong['started'] = started-10
            with self.subTest(change=change), self.assertRaises(ValueError): check(wrong)
        uncertain = copy.deepcopy(row); uncertain['last_stage']['execution_uncertain'] = True
        self.assertFalse(check(uncertain)['native_terminal_certain'])

    def test_09_fixture_policy_and_complex_json_flags_cannot_claim_actual_native_eligibility(self):
        with self.assertRaises(ValueError): self.check(test_only=False)
        complex_scope = mode.native_mode(mode.COMPLEX, external_batch_name='fixture-complex', native_evidence_root=self.c.root)
        selected = mode._record(complex_scope, self.policy, self.policy['retained_mathematical_producer'], dict(host_backend_identity=self.identity))
        result = self.check(selected=selected)
        self.assertFalse(result['raw_compile_passed']); self.assertFalse(result['raw_numeric_passed'])
        self.assertFalse(result['complex_decoder_called']); self.assertFalse(result['unverified_CKKS_values_consumed'])
        wrong = dict(self.proof, backend_identity_sha256='b'*64)
        with patch.object(assessment.backend_evidence, 'verify_backend_witness', return_value=wrong):
            result = assessment.check_witness(self.witness, self.policy, self.request, self.c.source, self.selected, mechanism_binding='a'*64, test_only=True)
        self.assertFalse(result['raw_compile_passed'])

    def test_10_delivery_and_external_modes_integrate_without_first_output_promotion(self):
        receipt = self.generate(); seen = []
        def reader(request, source):
            seen.append((request, source)); return copy.deepcopy(self.witness)
        result = self.evaluate(receipt, reader)
        self.assertEqual(seen, [(self.request, self.c.source)])
        self.assertTrue(result['raw_joint_passed']); self.assertTrue(result['test_only'])
        self.assertFalse(result['first_compile_passed']); self.assertFalse(result['first_numeric_passed']); self.assertFalse(result['first_joint_passed'])
        self.assertEqual(self.provider.request_attempts, 1)
        self.assertEqual((self.delivery/'candidate.py').read_bytes(), self.c.source)
        with self.assertRaises(FileExistsError): self.evaluate(receipt, reader)
        self.assertEqual(len(seen), 1)

    def test_11_external_source_tampering_is_preserved_as_integrity_rejection(self):
        receipt = self.generate()
        def reader(request, source):
            path = self.delivery/'candidate.py'; path.chmod(0o600); path.write_bytes(b'replacement')
            return copy.deepcopy(self.witness)
        result = self.evaluate(receipt, reader)
        self.assertEqual(result['status'], 'integrity_rejected'); self.assertFalse(result['source_unchanged'])
        self.assertFalse(result['first_joint_passed']); self.assertEqual(result['compared_values'], 0)

    def test_12_unavailable_complex_route_keeps_request_denominator_without_any_provider_call(self):
        receipt = self.generate(mode.COMPLEX)
        self.assertEqual(receipt['status'], 'backend_blocked'); self.assertEqual(self.provider.request_attempts, 0)
        result = self.evaluate(receipt, lambda *_: self.fail('Blocked request called an evaluator'))
        self.assertEqual(result['status'], 'backend_blocked'); self.assertFalse(result['first_joint_passed'])
        self.assertEqual(result['request_id'], self.request['request_id']); self.assertEqual(result['compared_values'], 0)


if __name__ == '__main__': unittest.main()
