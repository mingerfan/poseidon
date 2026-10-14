import unittest,copy,json
import scope
class ScopeTests(unittest.TestCase):
 def original(self):return scope.read(scope.PROPOSAL)
 def test_frozen_scope(self):
  value=self.original();scope.validate(value);view=scope.verify(value)
  self.assertEqual(len(view["cases"]),3);self.assertEqual(len(view["source_contexts"]),6)
  self.assertEqual(scope.BOUNDS["max_generations"],12);self.assertEqual(scope.BOUNDS["max_http_attempts"],48)
 def test_rebound_changes_rejected(self):
  for field,value in (("status","passed"),("provider",{}),("numeric_tolerance",dict(atol=.1,rtol=.1))):
   with self.subTest(field=field):
    changed=copy.deepcopy(self.original());changed[field]=value
    changed["binding"]=scope.digest({k:v for k,v in changed.items()if k!="binding"})
    with self.assertRaises(ValueError):scope.validate(changed)
 def test_source_format_remains_explicit(self):
  value=self.original();value["cases"][0]["source_format"]="poseidon-trusted-pytorch-v27"
  value["binding"]=scope.digest({k:v for k,v in value.items()if k!="binding"})
  with self.assertRaises(ValueError):scope.validate(value)
 def test_limits_and_authorization(self):
  v=self.original();scope.checked_authorization(scope.read(scope.AUTHORIZATION),v,scope.PROPOSAL_SHA)
  self.assertEqual(scope.BOUNDS["global_result_bytes"],32*1024**3)
  self.assertEqual(scope.PROVIDER["transport_retries"],3)
if __name__=="__main__":unittest.main()
