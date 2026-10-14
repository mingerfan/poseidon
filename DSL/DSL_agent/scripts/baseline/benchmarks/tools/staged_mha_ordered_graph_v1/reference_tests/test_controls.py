"""Standard-library fixture boundaries, no Torch import or environment entry."""
import copy,unittest
from cases import CASES,validate_case,signatures,output_names,masks
class Controls(unittest.TestCase):
 def test_six_structural_contexts(self):
  self.assertEqual(len(CASES),6);self.assertEqual(len({c["id"] for c in CASES}),6)
  for c in CASES:validate_case(c)
 def test_small_ports_and_elements(self):
  import math
  for c in CASES:
   self.assertLessEqual(sum(math.prod(x["shape"]) for x in signatures(c)),64)
   self.assertLessEqual(len(output_names(c)),3)
 def test_all_required_compositions(self):
  self.assertTrue(any(c.get("pre_width") for c in CASES));self.assertTrue(any(c.get("post_width") and c.get("residual") for c in CASES))
  self.assertTrue(any(c.get("projected_output") for c in CASES));self.assertTrue(any(not c["need"] for c in CASES))
 def test_bool_direction_declared(self):
  a,p=masks(CASES[2]);self.assertEqual(a["dtype"],"bool");self.assertTrue(a["values"][0][1]);self.assertTrue(p["values"][0][1])
 def test_additive_uses_finite_values(self):
  import math
  a,p=masks(CASES[3]);self.assertEqual(len(a["values"]),4)
  self.assertTrue(all(math.isfinite(x) for row in p["values"] for x in row));self.assertIsNone(a["negative_infinity"])
 def test_residual_dimension_rejected(self):
  c=copy.deepcopy(CASES[0]);c["post_width"]=3
  with self.assertRaisesRegex(ValueError,"residual"):validate_case(c)
 def test_head_dimension_rejected(self):
  c=copy.deepcopy(CASES[0]);c["h"]=3
  with self.assertRaisesRegex(ValueError,"head"):validate_case(c)
 def test_false_self_role_dimensions_rejected(self):
  c=copy.deepcopy(CASES[0]);c["s"]=3
  with self.assertRaisesRegex(ValueError,"self"):validate_case(c)
if __name__=="__main__":unittest.main()
