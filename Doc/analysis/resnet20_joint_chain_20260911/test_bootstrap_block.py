"""Metadata/negative checks, plus optional read-only complete GPU log checks."""
import os
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
BINARY = os.environ.get('POSEIDON_BOOTSTRAP_BLOCK_BINARY')
GPU_LOG = os.environ.get('POSEIDON_BOOTSTRAP_BLOCK_LOG')


@unittest.skipUnless(BINARY, 'set POSEIDON_BOOTSTRAP_BLOCK_BINARY')
class BlockMetadataTests(unittest.TestCase):
    def probe(self, image='0', mode='--metadata-only'):
        return subprocess.run([BINARY, mode, str(HERE/'relu_precision_fixture.txt'), image],
                              capture_output=True, text=True, timeout=120)

    def test_first_block_original_reference_and_residual(self):
        r = self.probe()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('layer=layer1_0.conv1+BN image=0', r.stdout)
        self.assertIn('layer=layer1_0.conv2+BN image=0', r.stdout)
        self.assertIn('diagnostic_stem_bootstraps=1 block_bootstraps=2 real_convolutions=2 residual_adds=1 final_q=9', r.stdout)
        self.assertIn('main_Q=6 shortcut_Q=9 aligned_Q=6 rescale=0 shortcut_preserved=true', r.stdout)
        self.assertIn('scale_mismatch=REJECTED layout_mismatch=REJECTED', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)
        self.assertNotIn('SECRET ', r.stdout)
        self.assertNotIn('MEMORY ', r.stdout)

    def test_second_real_image(self):
        r = self.probe('1')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('BLOCK_PLAN name=layer1_0 image=1', r.stdout)

    def test_lcdnn_s2c_first_real_contract(self):
        r = subprocess.run(
            [BINARY, '--metadata-only', str(HERE/'relu_precision_fixture.txt'),
             '0', '--lcdnn-s2c-first-real'],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(
            'BOOTSTRAP_BRANCH mode=lcdnn_s2c_first_real active_log_slots=15 '
            'evalmod_calls=2 output_real_projection=true', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)
        self.assertNotIn('SECRET ', r.stdout)

    def test_unknown_bootstrap_branch_rejected(self):
        r = subprocess.run(
            [BINARY, '--metadata-only', str(HERE/'relu_precision_fixture.txt'),
             '0', '--unknown'], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertIn('unknown bootstrap branch', r.stderr)
        self.assertNotIn('BASELINE ', r.stdout)

    def test_bootstrap_only_metadata_is_accepted(self):
        r = subprocess.run(
            [BINARY, '--metadata-only', str(HERE/'relu_precision_fixture.txt'),
             '0', '--lcdnn-s2c-first-real', '--bootstrap-only'],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('mode=lcdnn_s2c_first_real', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)

    def test_relu_only_metadata_is_accepted(self):
        r = subprocess.run(
            [BINARY, '--metadata-only', str(HERE/'relu_precision_fixture.txt'),
             '0', '--lcdnn-s2c-first-real', '--relu-only'],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('mode=lcdnn_s2c_first_real', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)

    def test_invalid_image(self):
        for image in ('-1', '10000', '1oops'):
            with self.subTest(image=image):
                r = self.probe(image)
                self.assertEqual(r.returncode, 2)
                self.assertNotIn('BASELINE ', r.stdout)

    def test_explicit_mode_required(self):
        r = self.probe(mode='--run')
        self.assertEqual(r.returncode, 2)
        self.assertNotIn('BASELINE ', r.stdout)


@unittest.skipUnless(GPU_LOG, 'set POSEIDON_BOOTSTRAP_BLOCK_LOG after GPU execution')
class BlockGpuEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.lines = Path(GPU_LOG).read_text().splitlines()

    def test_existing_ciphertext_chain_completed(self):
        self.assertEqual(sum(x.startswith('PHASE bootstrap_existing_ciphertext ') for x in self.lines), 3)
        secrets = [x for x in self.lines if x.startswith('SECRET ')]
        self.assertEqual(len(secrets), 1)
        self.assertIn('h=192 positive=96 negative=96 source=OS_random nonzero=true', secrets[0])
        self.assertEqual(sum(x == 'PUBLIC_KEYS count=39 host_cache_hit=1 double_hoist=1' for x in self.lines), 2)
        self.assertEqual(sum(x == 'PUBLIC_KEYS count=21 host_cache_hit=1 double_hoist=0' for x in self.lines), 1)
        self.assertEqual(sum(x.startswith('CONV_RESULT integration=PASS') for x in self.lines), 2)
        result = next(x for x in self.lines if x.startswith('BLOCK_RESULT '))
        for expected in ('integration=PASS', 'boundary_target_1e_4=PASS', 'first_target_failure=none',
                         'final_q=9', 'bootstraps=3 real_convolutions=2 residual_adds=1',
                         'completed_blocks=1 encryptions=1 no_reencryption=true', 'full_network_tested=false'):
            self.assertIn(expected, result)
        self.assertFalse(any(x.startswith(('ERROR ', 'STOP ')) for x in self.lines))

    def test_all_application_boundaries_and_shortcut(self):
        rows = [dict(v.split('=', 1) for v in x.split()[1:]) for x in self.lines if x.startswith('BLOCK_BOUNDARY ')]
        self.assertEqual([(x['stage'], int(x['q'])) for x in rows], [
            ('stem.relu', 9), ('conv1', 6), ('act1.native_bootstrap', 31), ('act1.relu', 9),
            ('conv2', 6), ('residual', 6), ('act2.native_bootstrap', 31), ('act2.relu', 9)])
        for row in rows:
            self.assertLessEqual(float(row['max_abs']), 1e-4)
            self.assertLessEqual(float(row['domain']), 1)
        self.assertTrue(any(x.startswith('CHECK block/shortcut_retained ') for x in self.lines))
        self.assertTrue(any(x.startswith('RESIDUAL_RESULT ') and 'rescale=0' in x for x in self.lines))


if __name__ == '__main__':
    unittest.main()
