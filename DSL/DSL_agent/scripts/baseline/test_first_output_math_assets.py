"""Post-record dependency controls using inert provenance fixtures only."""
import copy
import unittest

from delivery_component import bound, canonical, sha
from first_output_numerical_evidence import verify_mathematical_rows
import test_first_output_numerical_evidence as fixtures


class MathematicalAssetTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.NumericalEvidenceTests()
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        f = self.f
        self.asset = f.root/'frozen-mathematical-data.json'
        self.asset.write_bytes(b'{"fixture":true}')
        self.domain = dict(input0=[-1., 1.])
        domain_hash = sha(canonical(self.domain))
        f.policy['frozen_domain_sha256'] = domain_hash
        f.witness['inputs'][0]['domain_sha256'] = domain_hash
        f.math['inputs'][0]['domain_sha256'] = domain_hash
        self.spec = dict(format='poseidon-frozen-mathematical-producer-v1',
            producer_sources=f.policy['independent_math_sources'],
            producer_assets={str(self.asset): sha(self.asset.read_bytes())},
            original_model_sha256=f.policy['original_model_sha256'],
            declared_model_sha256=sha(canonical(f.request['model'])),
            input_domain=self.domain, reference_kind='json_float64')
        self.save()

    def save(self):
        f = self.f
        spec = bound(copy.deepcopy(self.spec))
        policy = {k: v for k, v in f.policy.items() if k != 'binding'}
        policy['mathematical_producer'] = spec
        f.policy = bound(policy)
        f.math.update(policy_sha256=sha(canonical(f.policy)),
            producer_spec_binding=spec['binding'], reference_origin=spec['reference_kind'])
        f.save_math()

    def verify(self):
        f = self.f
        return verify_mathematical_rows(f.witness, f.policy, f.request, f.source)

    def test_exact_frozen_non_code_dependency_is_accepted_as_protocol_evidence(self):
        self.verify()

    def test_asset_change_after_record_is_rejected(self):
        self.asset.write_bytes(b'{"fixture":"changed"}')
        with self.assertRaises(ValueError): self.verify()

    def test_record_cannot_claim_a_different_producer_selection(self):
        self.f.math['producer_spec_binding'] = '0'*64; self.f.save_math()
        with self.assertRaises(ValueError): self.verify()

    def test_original_declared_and_domain_identities_are_enforced(self):
        for key, value in [('original_model_sha256', '0'*64), ('declared_model_sha256', '0'*64),
                ('input_domain', dict(input0=[-2., 2.]))]:
            original = self.spec[key]; self.spec[key] = value; self.save()
            with self.assertRaises(ValueError): self.verify()
            self.spec[key] = original

    def test_reference_kind_cannot_change_after_record(self):
        self.f.math['reference_origin'] = 'native_source'; self.f.save_math()
        with self.assertRaises(ValueError): self.verify()

    def test_indirect_assets_and_invalid_spec_binding_are_rejected(self):
        self.asset.unlink(); self.asset.symlink_to(self.f.root/'outside.json')
        (self.f.root/'outside.json').write_bytes(b'{"fixture":true}')
        with self.assertRaises(ValueError): self.verify()
        self.asset.unlink(); self.asset.write_bytes(b'{"fixture":true}')
        self.f.policy['mathematical_producer']['binding'] = '0'*64
        with self.assertRaises(ValueError): self.verify()


if __name__ == '__main__': unittest.main()
