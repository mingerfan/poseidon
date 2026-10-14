"""Metadata/negative tests; no keys, CUDA initialization or HE execution."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BINARY = os.environ.get('POSEIDON_BOOTSTRAP_RELU_BINARY')


@unittest.skipUnless(BINARY, 'set POSEIDON_BOOTSTRAP_RELU_BINARY to the compiled probe')
class BootstrapReluContractTests(unittest.TestCase):
    def run_metadata(self, fixture=None, dataset='grid-near-zero', mode='--metadata-only'):
        return subprocess.run([BINARY, mode, str(fixture or HERE/'relu_precision_fixture.txt'), dataset],
                              capture_output=True, text=True, timeout=120)

    def test_fixed_period_fold_and_original_relu_contract(self):
        r = self.run_metadata()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn('SECRET ', r.stdout)
        self.assertNotIn('MEMORY ', r.stdout)
        self.assertIn('keys_generated=false gpu_executed=false', r.stdout)
        self.assertIn('c2s_log_scale=45 periods_per_integer=1', r.stdout)
        lines = r.stdout.splitlines()
        fold = dict(v.split('=', 1) for v in next(x for x in lines if x.startswith('FOLD_PLAN ')).split()[1:])
        self.assertEqual((fold['q_in'], fold['q_out'], fold['application_log_scale']), ('46', '31', '40'))
        self.assertEqual((fold['same_DAG'], fold['extra_Q']), ('true', '0'))
        self.assertLess(float(fold['identity_max']), 1e-10)
        self.assertLess(float(fold['source_reference_max']), 1e-10)
        self.assertAlmostEqual(math.log2(float(fold['alpha'])), 40-53.03094629532368, places=8)
        self.assertAlmostEqual(float(fold['seed'])**4, float(fold['alpha']), places=15)
        rows = [dict(v.split('=', 1) for v in x.split()[1:]) for x in lines if x.startswith('RELU_PLAN ')]
        self.assertEqual([(x['degree'], x['q_in'], x['q_out']) for x in rows],
                         [('15', '31', '25'), ('15', '25', '19'), ('27', '19', '12')])
        self.assertIn('bootstrap_out=31 relu_out=9 relu_Q_consumed=22 output_log_scale=40', r.stdout)
        q = list(map(int, next(x for x in lines if x.startswith('Q_PRIMES ')).split()[1:]))
        self.assertEqual(q, json.loads((HERE/'q50.json').read_text())['q_bottom_first'])

    def test_complex_dataset_rejected_before_keys(self):
        r = self.run_metadata(dataset='historical')
        self.assertEqual(r.returncode, 2)
        self.assertIn('real grid dataset required', r.stderr)
        self.assertNotIn('BASELINE ', r.stdout)

    def test_explicit_mode_required(self):
        r = self.run_metadata(mode='--run')
        self.assertEqual(r.returncode, 2)
        self.assertNotIn('BASELINE ', r.stdout)

    def test_wrong_tail_rejected_before_context(self):
        raw = (HERE/'relu_precision_fixture.txt').read_text()
        self.assertIn('\n19 3\n', raw)
        with tempfile.TemporaryDirectory(prefix='bootstrap-relu-test-') as directory:
            fixture = Path(directory)/'bad.txt'
            fixture.write_text(raw.replace('\n19 3\n', '\n18 3\n'))
            r = self.run_metadata(fixture)
        self.assertEqual(r.returncode, 2)
        self.assertIn('invalid ReLU fixture tail', r.stderr)
        self.assertNotIn('BASELINE ', r.stdout)

    def test_changed_coefficient_rejected_by_original_reference(self):
        tokens = (HERE/'relu_precision_fixture.txt').read_text().split()
        index = tokens.index('leaf')+6
        tokens[index] = str(float(tokens[index])+0.001)
        with tempfile.TemporaryDirectory(prefix='bootstrap-relu-test-') as directory:
            fixture = Path(directory)/'bad.txt'
            fixture.write_text(' '.join(tokens))
            r = self.run_metadata(fixture)
        self.assertEqual(r.returncode, 2)
        self.assertIn('original ReLU reference mismatch', r.stderr)
        self.assertNotIn('SECRET ', r.stdout)


if __name__ == '__main__':
    unittest.main()
