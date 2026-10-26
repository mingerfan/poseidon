"""S1 §8: explicit platform accumulation and compiler constant reproduction."""

import ctypes
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import toolchain
from pt_agent.manage.earth_emit import RMSCalculator, read
from pt_agent.paths import RESULTS_DIR
from pt_agent.tests._artifacts import require


PROOF = RESULTS_DIR/'s1-cnn_plain-20261025T184900572487/FMA_PROOF.json'
ARTIFACT_RESULTS = {}


def bits(value):
    return struct.pack('>d', float(value)).hex()


class RMSUnitTests(unittest.TestCase):
    def test_profile_selects_fma(self):
        self.assertEqual(toolchain.management_profile()['rms_accumulation'], 'fma')
        with self.assertRaisesRegex(ValueError, 'fma or separate'):
            RMSCalculator('guess')

    def test_fma_and_payload_cache(self):
        data = np.linspace(-1, 2, 1024)
        library = ctypes.CDLL('libm.so.6')
        fma = library.fma
        fma.argtypes = [ctypes.c_double]*3
        fma.restype = ctypes.c_double
        total = 0.
        for d in data:
            total = fma(float(d), float(d), total)
        calculator = RMSCalculator('fma')
        self.assertEqual(bits(calculator(data)), bits(math.sqrt(total/len(data))))
        self.assertEqual(bits(calculator(data.copy())), bits(math.sqrt(total/len(data))))
        self.assertEqual((calculator.hits, calculator.misses), (1, 1))

    def test_missing_fma_is_an_error(self):
        with patch('pt_agent.manage.earth_emit.ctypes.CDLL', return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, 'requires system libm fma'):
                RMSCalculator('fma')

    def test_separate_does_not_load_fma(self):
        with patch('pt_agent.manage.earth_emit.ctypes.CDLL', side_effect=OSError('missing')):
            self.assertEqual(RMSCalculator('separate')(np.array([3., 4.])), math.sqrt(12.5))


def compiler_suite(root):
    """Run against seven newly prepared compiler references; no hidden recompilation."""
    root = Path(root)

    @require(PROOF, root/'REFERENCES.json')
    class CompilerRMSTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            references = json.loads((root/'REFERENCES.json').read_text())
            cls.rows = {}
            for label, path in references.items():
                fma, separate = RMSCalculator('fma'), RMSCalculator('separate')
                function = read(Path(path)/'lowered.earth.mlir', toolchain.management_profile())
                cls.rows[label] = [dict(operation=n.id, compiler=bits(n.attrs['rms_var']),
                                       fma=bits(fma(n.data)), separate=bits(separate(n.data)))
                                   for n in function.nodes if n.kind == 'constant']
            ARTIFACT_RESULTS['constants'] = cls.rows

        def test_all_compiler_constants(self):
            self.assertEqual(set(self.rows), {'mlp', 'cnn_plain', 'unit-add', 'unit-chain',
                'unit-residual', 'unit-constants', 'unit-no-final-mul'})
            for label, rows in self.rows.items():
                for row in rows:
                    with self.subTest(model=label, operation=row['operation']):
                        self.assertEqual(row['fma'], row['compiler'])

        def test_sixteen_historical_values(self):
            proof = json.loads(PROOF.read_text())['comparisons']
            self.assertEqual(len(proof), 16)
            by_id = {r['operation']:r for r in self.rows['cnn_plain']}
            for row in proof:
                with self.subTest(operation=row['operation']):
                    self.assertEqual(by_id[row['operation']]['fma'], bits(row['compiler_rms']))

        def test_separate_is_distinguished(self):
            differences = {label:[r['operation'] for r in rows if r['separate'] != r['compiler']]
                           for label, rows in self.rows.items()}
            ARTIFACT_RESULTS['separate_differences'] = differences
            self.assertGreater(sum(map(len, differences.values())), 0)

    return unittest.defaultTestLoader.loadTestsFromTestCase(CompilerRMSTests)
