import unittest
from interpreter import evaluate,Unsupported,legacy
class ArithmeticHelpers(unittest.TestCase):
 def test_helper_is_evaluated_as_public_arithmetic(self):
  s='@hc.func("c")\ndef sum2(v):\n w=v+v.rotate(1)\n return w\n@hc.func("c")\ndef golden(x):\n return sum2(x)*gain\n'
  self.assertEqual(evaluate(s,[[1,2,3,4]],{"gain":.5},4),[(1.5,2.5,3.5,2.5)])
 def test_plain_and_multiport_helper(self):
  s='def pair(a):\n return (a,a*a)\ndef golden(x):\n return pair(x)\n'
  self.assertEqual(evaluate(s,[[1,2]],{},2),[(1.,2.),(1.,4.)])
 def test_public_globals_and_local_scope(self):
  s='def f(a):\n gain=a+a\n return gain\ndef golden(x):\n y=f(x)\n return y*gain\n'
  self.assertEqual(evaluate(s,[[1,2]],{"gain":3},2),[(6.,12.)])
 def test_old_supported_math_unchanged(self):
  for s in ['def f(x):\n return x+x.rotate(1)\n','def f(x):\n y=x*.5\n return (y,y*y)\n']:
   self.assertEqual(evaluate(s,[[1,2,3,4]],{},4),legacy.evaluate(s,[[1,2,3,4]],{},4))
 def test_reject_recursive_helper(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n return f(x)\n',[[1,2]],{},2)
 def test_reject_io(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n y=open("secret")\n return x\n',[[1,2]],{},2)
 def test_reject_import(self):
  with self.assertRaises(Unsupported):evaluate('import os\ndef f(x):\n return x\n',[[1,2]],{},2)
 def test_reject_side_effect_statement(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n print(x)\n return x\n',[[1,2]],{},2)
 def test_reject_arbitrary_decorator(self):
  with self.assertRaises(Unsupported):evaluate('@open("secret")\ndef f(x):\n return x\n',[[1,2]],{},2)
 def test_reject_wrong_declared_signature(self):
  with self.assertRaises(Unsupported):evaluate('@hc.func("c,c")\ndef f(x):\n return x\n',[[1,2]],{},2)
 def test_reject_wrong_call_arity(self):
  with self.assertRaises(Unsupported):evaluate('def f(x,y):\n return x+y\ndef golden(x):\n return f(x)\n',[[1,2]],{},2)
 def test_reject_keyword_calls(self):
  with self.assertRaises(Unsupported):evaluate('def f(y):\n return y\ndef golden(x):\n return f(y=x)\n',[[1,2]],{},2)
 def test_reject_return_of_scalar(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n return 1\n',[[1,2]],{},2)
 def test_reject_wrong_period(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n return x\n',[[1]],{},2)
 def test_reject_nonfinite_constant(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n return x*gain\n',[[1,2]],{"gain":float("inf")},2)
 def test_reject_shadowed_helper(self):
  with self.assertRaises(Unsupported):evaluate('def h(a):\n return a\ndef golden(x):\n h=x\n return h(x)\n',[[1,2]],{},2)
 def test_reject_after_return(self):
  with self.assertRaises(Unsupported):evaluate('def f(x):\n return x\n y=x+x\n',[[1,2]],{},2)
 def test_explicit_entry(self):
  s='def a(x):\n return x+x\ndef b(x):\n return a(x)\n'
  self.assertEqual(evaluate(s,[[1,2]],{},2,entry="b"),[(2.,4.)])
  with self.assertRaises(Unsupported):evaluate(s,[[1,2]],{},2)
if __name__=="__main__":unittest.main(verbosity=2)
