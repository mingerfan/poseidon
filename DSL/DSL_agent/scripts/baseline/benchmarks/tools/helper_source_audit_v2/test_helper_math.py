import unittest
from helper_math import evaluate,Unsupported
class HelperMathTests(unittest.TestCase):
    def test_nested_helpers_and_rotation_have_independent_expected_values(self):
        source='@hc.func("c")\ndef roll(x):\n return x.rotate(1)\n@hc.func("c")\ndef helper(y):\n z=roll(y)+y\n return z*p0\n@hc.func("c")\ndef golden(x):\n return helper(x)\n'
        self.assertEqual(evaluate(source,[[1.,2.,4.,8.]],{"p0":.5},4),[(1.5,3.,6.,4.5)])
    def test_helper_reads_global_instead_of_callers_shadow(self):
        source='@hc.func("c")\ndef helper(x):\n return x*p0\n@hc.func("c")\ndef golden(x):\n p0=9.\n return helper(x)\n'
        self.assertEqual(evaluate(source,[[1.,2.]],{"p0":.5},2),[(.5,1.)])
    def test_function_locals_do_not_leak_between_calls(self):
        source='@hc.func("c")\ndef helper(x):\n y=x+x\n return y\n@hc.func("c,c")\ndef golden(x,z):\n a=helper(x)\n b=helper(z)\n return [a,b]\n'
        self.assertEqual(evaluate(source,[[1.,2.],[3.,4.]],{},2),[(2.,4.),(6.,8.)])
    def test_public_vector_mask_and_positive_negative_rotation(self):
        source='@hc.func("c")\ndef helper(x):\n return (x*mask).rotate(-1)\n@hc.func("c")\ndef golden(x):\n return helper(x)\n'
        self.assertEqual(evaluate(source,[[2.,3.,4.,5.]],{"mask":[1.,0.,0.,0.]},4),[(0.,2.,0.,0.)])
    def test_direct_and_indirect_recursion_rejected(self):
        for expression in ("helper(x)","other(x)"):
            source='@hc.func("c")\ndef helper(x):\n return '+expression+'\n@hc.func("c")\ndef other(x):\n return helper(x)\n@hc.func("c")\ndef golden(x):\n return helper(x)\n'
            with self.assertRaisesRegex(Unsupported,"Recursive"):evaluate(source,[[1.,2.]],{},2)
    def test_unknown_call_and_arbitrary_method_rejected(self):
        for value in ("open(x)","x.decrypt()","__import__(x)"):
            source='@hc.func("c")\ndef golden(x):\n return '+value+'\n'
            with self.assertRaises(Unsupported):evaluate(source,[[1.,2.]],{},2)
    def test_dynamic_control_flow_rejected(self):
        source='@hc.func("c")\ndef golden(x):\n if x:\n  return x\n return x+x\n'
        with self.assertRaisesRegex(Unsupported,"Unsupported statement"):evaluate(source,[[1.,2.]],{},2)
    def test_wrong_arity_and_kwargs_rejected(self):
        for call in ("helper(x,x)","helper(x=x)"):
            source='@hc.func("c")\ndef helper(x):\n return x\n@hc.func("c")\ndef golden(x):\n return '+call+'\n'
            with self.assertRaises(Unsupported):evaluate(source,[[1.,2.]],{},2)
    def test_decorators_and_function_defaults_rejected(self):
        for source in ('@untrusted("c")\ndef golden(x):\n return x\n',
                       '@hc.func("c")\ndef golden(x=2.):\n return x\n'):
            with self.assertRaises(Unsupported):evaluate(source,[[1.,2.]],{},2)
    def test_shape_and_nonfinite_values_rejected(self):
        source='@hc.func("c")\ndef golden(x):\n return x\n'
        for inputs in ([[1.]],[[float("nan"),2.]]):
            with self.assertRaises((Unsupported,ValueError)):evaluate(source,inputs,{},2)
    def test_legacy_straight_arithmetic_retains_same_values(self):
        from audit import evaluate as old
        source='@hc.func("c")\ndef golden(x):\n a=x*x\n return [a*p0+x.rotate(1),-a]\n'
        self.assertEqual(evaluate(source,[[1.,-2.,3.,-4.]],{"p0":.125},4),
                         old(source,[[1.,-2.,3.,-4.]],{"p0":.125},4))
if __name__=="__main__":unittest.main(verbosity=2)
