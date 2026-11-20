import subprocess
import unittest
from pathlib import Path

from pt_agent import paths
from pt_agent.experiments import a4_prepared_disk_client as own
from pt_agent.tests._artifacts import require

FORMATTER = paths.WORKSPACE / 'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/gofmt'


class PreparedDiskClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(own.__file__).with_name('r31_lattigo_runtime_plan.go').read_text()
        cls.generated = own.build_source(cls.base)

    def test_dedup_uses_level_scale_and_existing_payload_identity(self):
        values = [dict(id=str(i), kind='plaintext', level=level, scale_log2=scale)
                  for i, (level, scale) in enumerate(((2, 50), (2, 50), (3, 50), (2, 100)))]
        plan = dict(values=values, initialization=[dict(kind='encode', output=str(i),
                    payload=dict(content='opaque-existing-locator')) for i in range(4)])
        inventory = own.storage_inventory(plan, dict(n=16, galois_count=2, one_key_bytes=100),
                                          dict(galois_count=3, one_key_bytes=200))
        self.assertEqual(inventory['unique_encoded_plaintexts'], 3)
        self.assertEqual(inventory['plaintext_coefficient_bytes'], 16 * (3 + 4 + 3) * 8)
        self.assertEqual(inventory['galois_coefficient_bytes'], 800)

    def test_disk_reserve_and_explicit_resource_rejection(self):
        self.assertTrue(own.disk_admission(31, 20, reserve_bytes=10)['accepted'])
        self.assertFalse(own.disk_admission(29, 20, reserve_bytes=10)['accepted'])
        with self.assertRaises(ValueError):
            own.disk_admission(-1, 0)

    def test_current_original_parameter_constructor_and_comparison_unchanged(self):
        for first, following in (('func parameters()', 'type lazyKeys struct'),
                                 ('func compare(', 'func execute('),
                                 ('func validateCipher(', 'func compare(')):
            a = self.base[self.base.index(first):self.base.index(following, self.base.index(first))]
            b = self.generated[self.generated.index(first):self.generated.index(following, self.generated.index(first))]
            self.assertEqual(a, b)

    def test_inference_cache_miss_has_no_generation_or_encoding(self):
        for first, following in (('func (l *lazyKeys) get(', 'func (l *lazyKeys) summary('),
                                 ('func (c *plaintextCache) get(', 'func (c *plaintextCache) summary(')):
            body = self.generated[self.generated.index(first):self.generated.index(following)]
            self.assertNotIn('GenGaloisKeyNew', body)
            self.assertNotIn('.Encode(', body)
        self.assertIn('l.kg=nil;l.sk=nil', self.generated)
        self.assertIn('key generation forbidden during prepared inference', self.generated)
        self.assertIn('unprepared plaintext value', self.generated)

    def test_clock_is_after_input_encoding_before_encrypt_and_after_final_decode(self):
        body = self.generated[self.generated.index('func execute('):self.generated.index('func main()')]
        self.assertLess(body.index('encoder.Encode(x, pt)'), body.index('preparedStartInference()'))
        self.assertLess(body.index('preparedStartInference()'), body.index('EncryptNew(pt)'))
        self.assertLess(body.index('DecryptNew(out)'), body.index('preparedStopInference(true)'))
        self.assertLess(body.index('encoder.Decode(pt, decoded)'), body.index('preparedStopInference(true)'))
        self.assertLess(body.index('preparedStopInference(true)'), body.index('compare(reference.Values, got)'))
        self.assertIn('if preparedTimed {preparedStopInference(false)}', body)

    def test_no_per_operation_rows_and_no_secret_serialization(self):
        self.assertNotIn('record["bootstrap_observations"] = append', self.generated)
        self.assertIn('per-operation observations must be disabled', self.generated)
        self.assertIn('if preparedTimed { return }', self.generated)
        self.assertNotIn('sk.WriteTo(', self.generated)
        self.assertNotIn('sk.MarshalBinary(', self.generated)
        self.assertIn('unknown or unsealed prepared key', self.generated)

    def test_source_anchor_drift_is_rejected_before_artifact_generation(self):
        with self.assertRaises(ValueError):
            own.build_source(self.base.replace('func save(force ...bool) {', 'func save_changed() {'))

    def test_terminal_acceptance_retains_strict_error_and_quiet_contract(self):
        plan = dict(execution=[dict(op='boot')])
        client = dict(status='ok', encrypted_plan_completed=True, ckks_inference_completed=True,
            no_keygen_or_constant_encoding_during_inference=True, per_operation_recording_enabled=False,
            input_ssa_check_enabled=False, public_key_request_trace_enabled=False,
            input_ssa_observations=[], bootstrap_observations=[], intermediate_decryptions=0,
            compute_attempts=1, compute_completed=1, op_attempts_by_kind={'boot': 1},
            op_completed_by_kind={'boot': 1}, bootstrap_attempts=1, bootstrap_completed=1,
            native_bootstrap_api_attempts=1, native_bootstrap_api_completed=1,
            encryption_attempts=1, encryption_completed=1, final_decryption_attempts=1,
            final_decryption_completed=1, ckks_inference_s=2, ckks_inference_observed_ns=2000000000,
            final_selected_values=[1.00001], comparison=dict(passed=True, absolute_tolerance=1e-5,
                                                           relative_tolerance=1e-4))
        self.assertTrue(own.check_client(client, plan, dict(values=[1]))['passed'])
        client['final_selected_values'] = [1.01]
        with self.assertRaises(ValueError):
            own.check_client(client, plan, dict(values=[1]))
        client['final_selected_values'] = [1]
        client['no_keygen_or_constant_encoding_during_inference'] = False
        with self.assertRaises(ValueError):
            own.check_client(client, plan, dict(values=[1]))


@require(FORMATTER)
class PreparedDiskSyntaxTests(unittest.TestCase):
    def test_generated_go_syntax_with_existing_formatter(self):
        # No Go build, native parameters, key generation or ciphertext execution.
        proc = subprocess.run([str(FORMATTER)], input=own.build_source(), text=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == '__main__':
    unittest.main()
