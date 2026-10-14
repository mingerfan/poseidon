"""Draft overlay-only contract regression; production files are read, not replaced."""
import copy,importlib.util,json,sys,unittest
from pathlib import Path
import public_math_context as m
from test_controls import fixture,poly
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class OverlayControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.original=load("original_public_weight_context",ROOT/"scripts/baseline/public_weight_context.py")
  cls.overlay=load("overlay_public_weight_context",Path(__file__).resolve().parent/"overlay/public_weight_context.py")
 def test_v1_bindings_identical(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):
   q=fixture(n);self.assertEqual(self.original.binding(q),self.overlay.binding(q))
 def test_v1_projected_bytes_identical(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):
   q=fixture(n);self.assertEqual(m.canonical(self.original.project(q)),m.canonical(self.overlay.project(q)))
 def test_v1_request_hash_identical(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):
   q=fixture(n);self.assertEqual(q["request_id"],self.overlay.request_digest(q));self.assertEqual(self.original.request_digest(q),self.overlay.request_digest(q))
 def test_v1_full_binding_validates(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):self.overlay.verify_binding(fixture(n))
 def test_v2_full_binding_validates(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):
   q=m.upgrade(fixture(n));self.assertEqual(self.overlay.binding(q,policy=self.overlay.MATH_POLICY),q["public_weight_binding"]);self.overlay.verify_binding(q)
 def test_v2_context_validates(self):
  for n in ("tanh_module","tanh_functional","tanh_input_keyword"):
   q=m.upgrade(fixture(n));context=self.overlay.project(q);self.assertEqual(self.overlay.verify_context(context),context)
 def test_v2_context_hash_changes(self):
  q=fixture();new=m.upgrade(q);self.assertEqual(new["request_id"],self.overlay.request_digest(new));self.assertNotEqual(new["request_id"],q["request_id"])
 def test_v1_not_implicitly_upgraded(self):
  q=fixture();self.assertEqual(self.overlay.binding(q)["policy"],"descriptors-v1");self.assertNotIn("math_roles",self.overlay.binding(q))
 def test_v2_tampered_role_rejected(self):
  q=m.upgrade(fixture());q["public_weight_binding"]["math_roles"]["constants"][0]["elements"][0]["value"]=100.
  with self.assertRaises(ValueError):self.overlay.verify_binding(q)
if __name__=="__main__":unittest.main()
