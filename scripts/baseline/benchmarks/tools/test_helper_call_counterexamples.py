"""Regression gates for helper-negative generation; no frontend/FHE claim."""
import ast
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_helper_call_counterexamples import omit_calls

class HelperOmissionTests(unittest.TestCase):
    def test_nested_calls_removed_and_unrelated_calls_preserved(self):
        source="def golden(x):\n    return combine(HE_BN(HE_BN(x, a), b), HE_SiLU(x))\n"
        negative,count=omit_calls(source,"HE_BN")
        self.assertEqual(count,2)
        calls=[n.func.id for n in ast.walk(ast.parse(negative))
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertCountEqual(calls,["combine","HE_SiLU"])
        self.assertEqual(ast.dump(ast.parse(negative)),
                         ast.dump(ast.parse("def golden(x):\n    return combine(x, HE_SiLU(x))\n")))
        self.assertIn("HE_BN(HE_BN",source)

    def test_multiple_outputs_and_first_argument_expression_preserved(self):
        source="def golden(x,y):\n    a=HE_Linear(x+y,w)\n    b=HE_Linear(y,w)\n    return a,b\n"
        negative,count=omit_calls(source,"HE_Linear")
        self.assertEqual(count,2)
        self.assertEqual(ast.dump(ast.parse(negative)),
                         ast.dump(ast.parse("def golden(x,y):\n    a=x+y\n    b=y\n    return a,b\n")))

    def test_absent_or_unsupported_target_rejected(self):
        for source,reason in [("HE_Other(x)","absent"),("HE_BN()","argument form"),
                              ("HE_BN(*xs)","argument form"),("HE_BN(x=x)","argument form")]:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError,reason):omit_calls(source,"HE_BN")

    def test_names_strings_attributes_are_not_direct_helper_calls(self):
        source="def golden(x):\n    label='HE_BN(x)'\n    return module.HE_BN(x), HE_BN(x)\n"
        negative,count=omit_calls(source,"HE_BN")
        self.assertEqual(count,1)
        tree=ast.parse(negative)
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                            and n.func.attr=="HE_BN" for n in ast.walk(tree)))
        self.assertTrue(any(isinstance(n,ast.Constant) and n.value=="HE_BN(x)" for n in ast.walk(tree)))

if __name__=="__main__":unittest.main()
