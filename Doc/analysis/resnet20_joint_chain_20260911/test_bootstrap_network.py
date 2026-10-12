"""Metadata contracts and optional evidence checks for the full-network probe."""
import os
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
BINARY = os.environ.get("POSEIDON_BOOTSTRAP_NETWORK_BINARY")
GPU_LOG = os.environ.get("POSEIDON_BOOTSTRAP_NETWORK_LOG")


class NetworkSourceContracts(unittest.TestCase):
    def test_full_topology_and_no_intermediate_encryptor(self):
        source = (HERE / "bootstrap_network_precision.cpp").read_text()
        self.assertIn('completed_blocks=9 bootstraps=', source)
        self.assertIn('intermediate_reencryptions=0', source)
        self.assertIn('max_blocks=9', source)
        self.assertIn('logit_error<=0.1', source)
        self.assertIn('gpu_prediction==plain_prediction', source)
        runtime = (HERE / "conv_probe_runtime.h").read_text()
        self.assertIn('intermediate encryption forbidden in Conv probe', runtime)


@unittest.skipUnless(BINARY, "set POSEIDON_BOOTSTRAP_NETWORK_BINARY")
class NetworkMetadataTests(unittest.TestCase):
    def test_full_network_metadata(self):
        result = subprocess.run(
            [BINARY, "--metadata-only", str(HERE / "relu_precision_fixture.txt"), "0"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('mode=lcdnn_s2c_first_real', result.stdout)
        self.assertIn('NETWORK_METADATA topology_blocks=9 bootstrap_points=18', result.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', result.stdout)
        self.assertNotIn('SECRET ', result.stdout)

    def test_invalid_staging_rejected(self):
        result = subprocess.run(
            [BINARY, "--metadata-only", str(HERE / "relu_precision_fixture.txt"),
             "0", "--max-blocks", "10"], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 2)
        self.assertIn('max-blocks must be 0..9', result.stderr)


@unittest.skipUnless(GPU_LOG, "set POSEIDON_BOOTSTRAP_NETWORK_LOG after GPU execution")
class NetworkGpuEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = Path(GPU_LOG).read_text().splitlines()

    def test_complete_continuous_chain(self):
        self.assertEqual(sum(line.startswith('PHASE bootstrap_existing_ciphertext ')
                             for line in self.lines), 18)
        self.assertEqual(sum(line == 'BOOTSTRAP_OFFLINE_CACHE matrices=miss'
                             for line in self.lines), 1)
        self.assertEqual(sum(line == 'BOOTSTRAP_OFFLINE_CACHE matrices=hit'
                             for line in self.lines), 17)
        blocks = [line for line in self.lines if line.startswith('NETWORK_BLOCK_RESULT ')]
        self.assertEqual(len(blocks), 9)
        self.assertTrue(all('q=9 log_scale=40' in line for line in blocks))
        self.assertFalse(any(line.startswith(('ERROR ', 'STOP ')) for line in self.lines))

    def test_final_correctness_result(self):
        result = next(line for line in self.lines if line.startswith('NETWORK_RESULT '))
        for expected in (
            'integration=PASS', 'staged=false', 'completed_blocks=9',
            'bootstraps=18', 'convolutions=18', 'residual_adds=9',
            'input_encryptions=27', 'intermediate_reencryptions=0',
            'plain_prediction=3', 'gpu_prediction=3', 'first_failure=none',
            'full_network_tested=true'):
            self.assertIn(expected, result)
        fields = dict(item.split('=', 1) for item in result.split()[1:])
        self.assertLessEqual(float(fields['max_logit_error']), 0.1)
        self.assertLessEqual(float(fields['max_boundary_error']), 1e-3)
        self.assertEqual(sum(line.startswith('NETWORK_LOGITS ') for line in self.lines), 1)
        self.assertEqual(sum(line.startswith('NETWORK_REFERENCE_LOGITS ')
                             for line in self.lines), 1)


if __name__ == "__main__":
    unittest.main()
