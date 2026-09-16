"""Original-source Conv metadata/layout/negative tests; never initialize CUDA."""
import os
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
BINARY = os.environ.get('POSEIDON_BOOTSTRAP_RELU_CONV_BINARY')


@unittest.skipUnless(BINARY, 'set POSEIDON_BOOTSTRAP_RELU_CONV_BINARY')
class BootstrapReluConvTests(unittest.TestCase):
    def run_probe(self, image='0', mode='--metadata-only'):
        return subprocess.run([BINARY, mode, str(HERE/'relu_precision_fixture.txt'), image],
                              text=True, capture_output=True, timeout=120)

    def test_original_conv_layout_and_real_weights(self):
        r = self.run_probe()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('q_in=9 q_out=6 Q_consumed=3 log_scale=40 original_source=true', r.stdout)
        self.assertIn('layer=layer1_0.conv1+BN image=0', r.stdout)
        self.assertIn('CONV_ROTATIONS q=7 steps=15360', r.stdout)
        self.assertIn('CONV_ROTATIONS q=9 steps=', r.stdout)
        self.assertIn('rescale=3 rotate=49', r.stdout)
        self.assertIn('CONV_NEGATIVE_CHECKS encryption=REJECTED decryption=REJECTED modraise=REJECTED misalignment=REJECTED', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)
        self.assertNotIn('SECRET ', r.stdout)
        self.assertNotIn('MEMORY ', r.stdout)

    def test_second_real_image(self):
        r = self.run_probe('1')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('layer=layer1_0.conv1+BN image=1', r.stdout)

    def test_invalid_image_rejected_before_key_generation(self):
        for image in ('-1', '10000', '0oops'):
            with self.subTest(image=image):
                r = self.run_probe(image)
                self.assertEqual(r.returncode, 2)
                self.assertIn('invalid CIFAR image id', r.stderr)
                self.assertNotIn('BASELINE ', r.stdout)

    def test_explicit_accuracy_only_mode_required(self):
        r = self.run_probe(mode='--run')
        self.assertEqual(r.returncode, 2)
        self.assertNotIn('BASELINE ', r.stdout)


if __name__ == '__main__':
    unittest.main()
