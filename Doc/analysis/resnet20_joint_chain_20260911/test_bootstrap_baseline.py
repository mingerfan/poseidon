"""Run only metadata mode; no CUDA initialization or keys in these tests."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from joint import HISTORICAL

HERE=Path(__file__).resolve().parent
BINARY=os.environ.get('POSEIDON_BASELINE_BINARY')


@unittest.skipUnless(BINARY, 'set POSEIDON_BASELINE_BINARY to the compiled diagnostic')
class BaselineContractTests(unittest.TestCase):
    def run_profile(self, profile):
        cmd=[BINARY,'--metadata-only',profile,'historical']
        if profile=='q50-fixed45': cmd.append(str(HERE/'relu_precision_fixture.txt'))
        r=subprocess.run(cmd,text=True,capture_output=True,timeout=120,check=True)
        lines=r.stdout.splitlines()
        self.assertIn('RESULT metadata=PASS keys_generated=false gpu_executed=false',lines)
        self.assertIn('secret_hamming_weight=192',lines[0])
        self.assertIn('output_fold=false',lines[0])
        c=next(x for x in lines if x.startswith('CONTRACT '))
        self.assertIn('preparation_target_log_scale=59',c)
        self.assertIn('c2s_log_scale=45',c)
        periods=float(c.split('periods_per_integer=')[1])
        self.assertAlmostEqual(periods,1,places=12)
        primes=list(map(int,next(x for x in lines if x.startswith('Q_PRIMES')).split()[1:]))
        plan=next(x for x in lines if x.startswith('PLAN EvalMod '))
        self.assertIn('q_out='+str(len(primes)-19),plan)
        scale=float(plan.split('output_log_scale=')[1].split()[0])
        self.assertAlmostEqual(scale,53.03094629532368,places=8)
        return primes

    def test_historical_exact_primes_and_native_contract(self):
        self.assertEqual(self.run_profile('historical-q34'),HISTORICAL)

    def test_q50_preserves_saved_primes_but_fixes_contract(self):
        q=json.loads((HERE/'q50.json').read_text())['q_bottom_first']
        self.assertEqual(self.run_profile('q50-fixed45'),q)

    def test_unknown_profile_rejected_before_keys(self):
        r=subprocess.run([BINARY,'--metadata-only','unknown','historical'],text=True,capture_output=True,timeout=10)
        self.assertEqual(r.returncode,2)
        self.assertNotIn('SECRET ',r.stdout)

    def test_explicit_accuracy_only_flag_required(self):
        r=subprocess.run([BINARY,'--run','historical-q34','historical'],text=True,capture_output=True,timeout=10)
        self.assertEqual(r.returncode,2)
        self.assertIn('explicit accuracy-only flag required',r.stderr)
        self.assertNotIn('SECRET ',r.stdout)

    def test_truncated_primes_rejected_before_context(self):
        with tempfile.TemporaryDirectory(prefix='bootstrap-baseline-test-') as directory:
            fixture=Path(directory)/'truncated.txt'
            fixture.write_text('RELU_PRECISION_V1 50 25\n4255252481\n')
            r=subprocess.run([BINARY,'--metadata-only','q50-fixed45','historical',str(fixture)],
                             text=True,capture_output=True,timeout=10)
        self.assertEqual(r.returncode,2)
        self.assertIn('truncated fixture primes',r.stderr)
        self.assertNotIn('SECRET ',r.stdout)
        self.assertNotIn('BASELINE profile=',r.stdout)


if __name__=='__main__':unittest.main()
