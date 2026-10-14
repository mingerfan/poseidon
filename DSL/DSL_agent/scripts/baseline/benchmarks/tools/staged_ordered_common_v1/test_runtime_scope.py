import copy,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_scope as s
class ScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.p=s.plan()
 def rebind(self,value):
  value.pop("binding",None);value["binding"]=hashlib.sha256(s.canonical(value)).hexdigest()
  return value
 def test_exact_original_and_staged_source_identity(self):
  p=self.p
  self.assertEqual(len(p["bindings"]["original_production"]),704)
  self.assertEqual(len(p["bindings"]["inside_sources"]),705)
  changed={n for n,h in p["bindings"]["inside_sources"].items()
           if p["bindings"]["original_production"].get(n)!=h}
  self.assertEqual(len(changed),8)
 def test_exact_model_source_and_no_provider_scope(self):
  self.assertEqual(self.p["cases"],["alpha","dropout","boolean"])
  self.assertEqual(self.p["scope"]["provider_calls"],0)
  self.assertEqual(self.p["scope"]["http_attempts"],0)
  self.assertFalse(self.p["production_applied"])
  self.assertTrue(all(v["groups"]==16 for v in self.p["bindings"]["source_reference"].values()))
 def test_matching_plan_passes(self):s.check_plan(self.p)
 def test_rehashed_larger_budget_is_rejected(self):
  changed=copy.deepcopy(self.p);changed["scope"]["artifact_bytes"]+=1
  with self.assertRaisesRegex(ValueError,"Finite offline scope"):s.check_plan(self.rebind(changed))
 def test_rehashed_case_substitution_is_rejected(self):
  changed=copy.deepcopy(self.p);changed["cases"][0]="replacement"
  with self.assertRaisesRegex(ValueError,"Finite offline scope"):s.check_plan(self.rebind(changed))
 def test_unknown_binding_is_rejected(self):
  changed=copy.deepcopy(self.p);changed["binding"]="0"*64
  with self.assertRaisesRegex(ValueError,"plan integrity"):s.check_plan(changed)
 def test_changed_production_is_rejected_before_launch(self):
  actual=s.sha
  name=next(iter(self.p["bindings"]["original_production"]))
  path=s.ROOT/name
  with patch.object(s,"sha",side_effect=lambda p:"0"*64 if Path(p)==path else actual(p)):
   with self.assertRaisesRegex(ValueError,"Frozen production"):s.source_binding()
 def test_changed_fixture_is_rejected(self):
  actual=s.sha;path=s.HERE/"fixtures"/"alpha"/"common_models.py"
  with patch.object(s,"sha",side_effect=lambda p:"0"*64 if Path(p)==path else actual(p)):
   with self.assertRaisesRegex(ValueError,"Fixture source hash"):s.source_binding()
 def test_changed_driver_is_rejected_after_freeze(self):
  actual=s.sha;path=s.HERE/"runtime_acceptance.py"
  with patch.object(s,"sha",side_effect=lambda p:"0"*64 if Path(p)==path else actual(p)):
   with self.assertRaisesRegex(ValueError,"Frozen tool/fixture"):s.check_plan(self.p)
if __name__=="__main__":unittest.main(verbosity=2)
