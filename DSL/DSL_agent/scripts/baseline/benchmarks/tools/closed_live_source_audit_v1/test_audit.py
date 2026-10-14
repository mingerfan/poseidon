import unittest
from audit import Unsupported, difference, evaluate

class DiagnosticTests(unittest.TestCase):
    def run_source(self, source, constants=None):
        return evaluate(source, [[1.,2.,3.,4.]], constants or {},4)
    def test_signed_rotations(self):
        self.assertEqual(self.run_source("def f(x):\n return x.rotate(1)"), [(2.,3.,4.,1.)])
        self.assertEqual(self.run_source("def f(x):\n return x.rotate(-1)"), [(4.,1.,2.,3.)])
    def test_scalar_broadcast_is_distinct_from_slot_mask(self):
        scalar=self.run_source("def f(x):\n return x*p", {"p":.25})
        masked=self.run_source("def f(x):\n return x*p", {"p":[.25,0,0,0]})
        self.assertEqual(scalar,[(.25,.5,.75,1.)])
        self.assertEqual(masked,[(.25,0,0,0)])
    def test_assignment_keeps_prior_value(self):
        self.assertEqual(self.run_source("def f(x):\n y=x\n x=x*2\n return (y,x)"),[(1.,2.,3.,4.),(2.,4.,6.,8.)])
    def test_nested_rotation_and_output(self):
        self.assertEqual(self.run_source("def f(x):\n return (x.rotate(1).rotate(2), x-x)"),[(4.,1.,2.,3.),(0.,0.,0.,0.)])
    def test_arbitrary_call_rejected(self):
        with self.assertRaises(Unsupported):
            self.run_source("def f(x):\n return open('/etc/passwd')")
    def test_loop_rejected(self):
        with self.assertRaises(Unsupported):
            self.run_source("def f(x):\n for i in range(2):\n  x=x+1\n return x")
    def test_nonfinite_and_wrong_period_rejected(self):
        with self.assertRaises(Unsupported):
            self.run_source("def f(x):\n return x+p",{"p":[1,2]})
        with self.assertRaises(Unsupported):
            self.run_source("def f(x):\n return x+p",{"p":float('inf')})
    def test_frozen_threshold_and_shape(self):
        self.assertTrue(difference([[1.0001]],[[1.]])["matches_frozen_tolerance"])
        self.assertFalse(difference([[1.01]],[[1.]])["matches_frozen_tolerance"])
        with self.assertRaises(ValueError):
            difference([[1.,2.]],[[1.]])

if __name__ == "__main__":
    unittest.main(verbosity=2)
