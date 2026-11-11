import tempfile
from pathlib import Path
import re
import unittest

from bench.resnet20_s2c_first.compile_prepared import spec_from_mlir


class TraceContextTest(unittest.TestCase):
    def test_q49_and_q50_physical_contexts(self):
        root = Path(__file__).resolve().parents[2]
        original = (root / 'test-results/manual-mlir-20261110/resnet20-manual.mlir').read_text()
        match = re.search(r'ckks\.rns_moduli\s*=\s*array<i64:\s*([^>]*)>', original)
        self.assertIsNotNone(match)
        q50 = [int(x) for x in match.group(1).split(',')]
        self.assertEqual(len(q50), 50)
        for primes in (q50, q50[:8] + q50[9:]):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'trace.mlir'
                path.write_text('module attributes {ckks.rns_moduli = array<i64: ' +
                                ', '.join(map(str, primes)) + '>} {}')
                spec = spec_from_mlir(path)
                self.assertEqual(spec['context']['rns_moduli'], primes)
                self.assertEqual(spec['context']['context_id'], f'manual-resnet20-q{len(primes)}')
                self.assertEqual(spec['levels']['upper_bound'], len(primes) - 1)
                self.assertFalse(spec['operators']['boot']['supported'])
                bad = primes[:-1] + [primes[0]]
                path.write_text('module attributes {ckks.rns_moduli = array<i64: ' +
                                ', '.join(map(str, bad)) + '>} {}')
                with self.assertRaises(ValueError):
                    spec_from_mlir(path)


if __name__ == '__main__':
    unittest.main()
