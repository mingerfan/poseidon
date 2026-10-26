"""S1 bounded constant comparison keeps full Emit behavior under local memory."""

import unittest

import numpy as np

from pt_agent.backends import EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.experiments.s1_acceptance import emit_comparison
from pt_agent.lower import lower


SOURCE = '''@fhe.program(slots=1024)
def dense(x: Cipher[1,4]):
 a = pack(x, layout="flat")
 y = linear(a, w=W["w"])
 return y
'''


class CompatibilityStreamTests(unittest.TestCase):
    def test_emit_and_each_constant_are_compared(self):
        p = parse(SOURCE)
        result = emit_comparison(EmitBackend,lower,p,EmitBackend,lower,p,{'w':np.eye(4)})
        self.assertTrue(result['emit_text_equal'])
        self.assertTrue(result['constants_equal'])
        self.assertGreater(result['constant_calls_compared'],0)

    def test_numeric_difference_is_reported(self):
        p = parse(SOURCE)
        def changed(program,weights,backend):
            return lower(program,{'w':weights['w']+1.},backend)
        with self.assertRaisesRegex(AssertionError,'Constant payload differs'):
            emit_comparison(EmitBackend,changed,p,EmitBackend,lower,p,{'w':np.eye(4)})
