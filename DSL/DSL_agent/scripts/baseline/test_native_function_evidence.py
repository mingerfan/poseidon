"""Audit saved real compiler/SEAL evidence without running a provider or native VM."""
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from probe_native_calls import CASES


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_CPU'), 'requires actual native-call encrypted evidence')
class NativeCallEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_CURRENT'), 'requires current compile probe')
    def test_current_compile_reproduces_executed_bytes(self):
        from probe_native_calls import SOURCE
        current_root = Path(os.environ['POSEIDON_NATIVE_CALLS_CURRENT'])
        current = json.loads((current_root/'report.json').read_text())
        cpu = json.loads((Path(os.environ['POSEIDON_NATIVE_CALLS_CPU'])/'report.json').read_text())
        previous = json.loads((Path(cpu['probe'])/'report.json').read_text())
        self.assertEqual(current['frontend_source_sha256'], digest(SOURCE))
        self.assertTrue(current['all_compiled'])
        self.assertEqual(current['status'], 'compiled_nonidentical')
        self.assertEqual(current['frontend_library_sha256'], previous['frontend_library_sha256'])
        self.assertEqual([c['case'] for c in current['cases']], list(CASES))
        for new, old in zip(current['cases'], previous['cases']):
            for style in ('direct','call'):
                for name, key in [('lowered._hecate_golden.hevm','hevm_sha256'), ('_hecate_golden.cst','cst_sha256')]:
                    self.assertEqual(new[style][key], old[style][key])
                    self.assertEqual(digest(current_root/(new['case']+'-'+style)/name), new[style][key])

    def test_every_artifact_hash_plain_reference_and_decrypted_result(self):
        import numpy as np
        from run_native_call_goldens import reference
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        root = Path(os.environ['POSEIDON_NATIVE_CALLS_CPU'])
        report = json.loads((root/'report.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['agent_calls'], 0)
        self.assertFalse(report['poseidon_gpu_validated'])
        self.assertFalse(report['bootstrap_executed'])
        self.assertEqual(report['parameters']['security_check'], 'tc128')
        self.assertEqual(report['parameters']['polynomial_degree'], 32768)
        self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
        probe = Path(report['probe'])
        self.assertEqual(digest(probe/'report.json'), report['probe_sha256'])
        compiled = json.loads((probe/'report.json').read_text())
        self.assertEqual([c['case'] for c in compiled['cases']], list(CASES))
        differences = [c['case'] for c in compiled['cases'] if not c['identical_artifacts']]
        self.assertEqual(differences, ['array'])
        expected_rows = {(case, style, group) for case in CASES for style in ('direct','call')
                         for group in range(4 if case in ('array','pair') else 1)}
        self.assertEqual(len(report['cases']), len(expected_rows))
        self.assertEqual({(c['case'],c['style'],c['group']) for c in report['cases']}, expected_rows)
        values = 0; comparisons = []
        for row in report['cases']:
            folder = root/(row['case']+'-'+row['style']+'-'+str(row['group']))
            for name, expected in row['frozen_hashes'].items(): self.assertEqual(digest(folder/name), expected)
            original = next(c[row['style']] for c in compiled['cases'] if c['case'] == row['case'])
            self.assertEqual(digest(folder/'lowered._hecate_golden.hevm'), original['hevm_sha256'])
            self.assertEqual(digest(folder/'_hecate_golden.cst'), original['cst_sha256'])
            self.assertEqual((original['trace_exit'], original['compile_exit']), (0,0))
            binding = json.loads((folder/'binding.json').read_text())
            self.assertEqual(binding, row['binding'])
            inputs, independent = reference(row['case'])
            expected = np.stack([independent[:, i,j] for i,j in binding['selectors']], axis=1)
            with np.load(folder/'arrays.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['inputs'], inputs)
                np.testing.assert_array_equal(arrays['reference'], expected)
            actual = np.load(folder/'decrypted.npy', allow_pickle=False)
            result = compare(actual, expected, 1e-5, 1e-4)
            self.assertEqual(result, row['comparison']); self.assertTrue(result['passed'])
            self.assertEqual((row['exit_code'], row['status']), (0,'passed'))
            execution = json.loads((folder/'execution.json').read_text())
            self.assertEqual(execution, row['execution'])
            self.assertTrue(execution['encrypted_execution'])
            self.assertFalse(execution['bootstrap_executed'])
            self.assertEqual(execution['input_batches'], 4)
            self.assertEqual(execution['encrypted_input_count'], binding['input_count'])
            gate = inspect_artifacts((folder/'lowered._hecate_golden.hevm').read_bytes(),
                    (folder/'_hecate_golden.cst').read_bytes(), expected_inputs=binding['input_count'])
            self.assertEqual(len(gate['res_dst']), independent.shape[1])
            if row['case'] == 'array':
                # Sensitivity check on saved decrypted values, not a new FHE run.
                self.assertFalse(compare(actual[:, [1,0,3,2]], expected, 1e-5, 1e-4)['passed'])
            values += result['compared_values']; comparisons.append(result)
        self.assertEqual(values, 544)
        self.assertEqual(report['compared_values'], values)
        self.assertEqual(report['max_absolute_error'], max(c['max_absolute_error'] for c in comparisons))
        self.assertEqual(report['weighted_mae'], sum(c['mae']*c['compared_values'] for c in comparisons)/values)
        cleanup = json.loads((root/'key-cleanup-outcome.json').read_text())
        self.assertTrue(cleanup['complete']); self.assertGreater(cleanup['freed_bytes'], 0)
        self.assertFalse((root/'private-keys').exists())


if __name__ == '__main__': unittest.main()
