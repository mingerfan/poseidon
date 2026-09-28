import json
from pathlib import Path
import unittest
from earth_failure_diagnostic import diagnose
PROFILE=Path(__file__).resolve().parents[2]/"third_party/dacapo/profiled_SEAL_CPU.json"
class EarthDiagnosticTests(unittest.TestCase):
    def log(self,left=(1,"ci",90,12),right=(1,"pl",45,12)):
        def ty(v):return "tensor<%sx!earth.%s<%s * %s>>"%v
        return "PRIVATE error: 'earth.mul' op failed to infer returned types\\n"+'%22 = "earth.mul"(%20, %21) : ('+ty(left)+", "+ty(right)+') -> tensor<1x!earth.ci<0 * 0>>'
    def check(self,*args):return diagnose(self.log(*args),PROFILE.read_bytes())
    def test_consumed_level_budget(self):
        d=self.check()
        self.assertEqual(d["failed_conditions"],["compiler_accumulated_scale_budget"])
        self.assertEqual((d["compiler_accumulated_scale"],d["compiler_budget"]),(810,780))
        self.assertFalse(d["seal_capacity_checked"])
        self.assertFalse(d["proves_all_equivalent_programs_impossible"])
        self.assertNotIn("PRIVATE",json.dumps(d))
    def test_boundary_is_inclusive(self):
        self.assertIsNone(self.check((1,"ci",60,12),(1,"ci",90,12)))
    def test_left_operand_rule_matches_fixed_source(self):
        self.assertIsNone(self.check((1,"ci",30,12),(1,"ci",100,12)))
    def test_level_mismatch(self):
        self.assertEqual(self.check((1,"ci",45,1),(1,"ci",45,2))["failed_conditions"],
                         ["operand_level_alignment"])
    def test_shape_mismatch(self):
        self.assertEqual(self.check((1,"ci",45,1),(2,"ci",45,1))["failed_conditions"],
                         ["operand_tensor_shape"])
    def test_180_is_not_a_diagnostic_cap(self):
        self.assertIsNone(self.check((1,"ci",240,1),(1,"ci",240,1)))
    def test_no_unverified_profile(self):
        self.assertIsNone(diagnose(self.log(),b'{}'))
    def test_no_runtime_or_unknown_failure(self):
        for text in ("",self.log().replace("error:","warning:"),None,"a"*4001):
            self.assertIsNone(diagnose(text,PROFILE.read_bytes()))
    def test_ambiguous_multiple_operations(self):
        self.assertIsNone(diagnose(self.log()+"\\n"+self.log(),PROFILE.read_bytes()))
    def test_no_arbitrary_numeric_token(self):
        self.assertIsNone(diagnose(self.log().replace("90 *","99999999999999999999 *"),PROFILE.read_bytes()))
if __name__=="__main__":unittest.main()
