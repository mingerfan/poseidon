"""Precise bounded native diagnostics; accepted language and limits unchanged."""
import unittest
from decorated_functions import validate
def program(count):
    names=[f"u{i}" for i in range(count)]
    return ('@hc.func("'+','.join(['c']*count)+'")\ndef helper('+','.join(names)+'):\n    return u0\n'
            '@hc.func("c")\ndef golden(x):\n    return helper('+','.join(['x']*count)+')\n')
class NativeDiagnosticTests(unittest.TestCase):
    def test_sixteen_parameters_remain_accepted(self):
        self.assertEqual(validate(program(16),{})["functions"]["golden"]["result"],"c")
    def test_seventeen_parameters_report_capacity_not_name_collision(self):
        with self.assertRaisesRegex(ValueError,"parameter limit: helper declares 17 parameters; maximum 16"):
            validate(program(17),{})
    def test_duplicate_and_constant_collision_are_distinct(self):
        with self.assertRaisesRegex(ValueError,"Duplicate native function parameter"):
            validate(program(2).replace("helper(u0,u1)","helper(u0,u0)"),{})
        with self.assertRaisesRegex(ValueError,"Invalid or colliding parameter name"):
            validate(program(1),{"u0":[.5]})
    def test_plain_negation_still_rejected_cipher_expression_accepted(self):
        bad='@hc.func("c")\ndef golden(x):\n    return -w*x\n'
        with self.assertRaisesRegex(ValueError,"bound Plain cannot be negated"):validate(bad,{"w":[.5]})
        self.assertEqual(validate(bad.replace("-w*x","-(w*x)"),{"w":[.5]})["functions"]["golden"]["result"],"c")
if __name__=="__main__":unittest.main(verbosity=2)
