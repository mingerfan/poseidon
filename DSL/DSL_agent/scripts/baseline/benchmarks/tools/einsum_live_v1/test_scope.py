"""Offline integrity/budget boundaries for the frozen eight-request proposal."""
import copy,hashlib,json,pathlib,tempfile,unittest
import scope
def rebound(value):
 value["binding"]=scope.digest({k:v for k,v in value.items()if k!="binding"});return value
class ScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.proposal=scope.read(scope.PROPOSAL)
 def test_current_prepared_scope(self):
  result=scope.verify(self.proposal)
  self.assertEqual(result["status"],"prepared_waiting_for_separate_approval")
  self.assertFalse(result["approved"]);self.assertEqual(result["provider_calls"],0)
 def test_changed_bound_document(self):
  v=copy.deepcopy(self.proposal);v["revision"]="changed"
  with self.assertRaisesRegex(ValueError,"proposal_binding"):scope.fields(v)
 def test_cannot_treat_plan_as_authorization(self):
  v=copy.deepcopy(self.proposal);v["approved"]=True
  with self.assertRaisesRegex(ValueError,"planning_is_not_authorization"):scope.fields(rebound(v))
 def test_cannot_mark_execution_started(self):
  v=copy.deepcopy(self.proposal);v["executed"]=True
  with self.assertRaisesRegex(ValueError,"planning_is_not_authorization"):scope.fields(rebound(v))
 def test_no_automatic_budget_growth(self):
  for key,increment in [("max_generations",1),("max_http_attempts",1),("batch_wall_seconds",1),
                       ("new_batch_bytes",1),("native_concurrency",1)]:
   with self.subTest(key=key):
    v=copy.deepcopy(self.proposal);v["budget"][key]+=increment
    with self.assertRaisesRegex(ValueError,"reviewed_budget"):scope.fields(rebound(v))
 def test_no_model_replacement(self):
  v=copy.deepcopy(self.proposal);v["provider"]["model"]="other"
  with self.assertRaisesRegex(ValueError,"reviewed_budget"):scope.fields(rebound(v))
 def test_no_unknown_call_retry(self):
  v=copy.deepcopy(self.proposal);v["previous_unknown_http_retried"]=True
  with self.assertRaisesRegex(ValueError,"unknown_call_not_retried"):scope.fields(rebound(v))
 def test_all_eight_original_cases(self):
  for changed in ("missing","duplicate","reordered"):
   v=copy.deepcopy(self.proposal)
   if changed=="missing":v["cases"].pop()
   elif changed=="duplicate":v["cases"][-1]=v["cases"][0]
   else:v["cases"].reverse()
   with self.assertRaisesRegex(ValueError,"reviewed_cases"):scope.fields(rebound(v))
 def test_existing_numerical_standard(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["validation_level"]="compile"
  with self.assertRaisesRegex(ValueError,"case_contract"):scope.fields(rebound(v))
 def test_context_byte_ceiling(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["public_context_bytes"]=131073
  with self.assertRaisesRegex(ValueError,"context_budget"):scope.fields(rebound(v))
 def test_source_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);sources=v["source_stage_identity"]["generation"]["sources"]
  name=next(iter(sources));sources[name]="0"*64
  with self.assertRaisesRegex(ValueError,"source_identity_changed"):scope.verify(rebound(v))
 def test_weight_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);c=next(c for c in v["cases"]if c["weight_files"])
  c["weight_files"][next(iter(c["weight_files"]))]="0"*64
  with self.assertRaisesRegex(ValueError,"source_or_weight_changed"):scope.verify(rebound(v))
 def test_file_byte_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["public_request_file_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"frozen_request_changed"):scope.verify(rebound(v))
 def test_redigested_public_body_is_rejected(self):
  v=copy.deepcopy(self.proposal);c=v["cases"][0];body=scope.read(c["public_request_file"])
  body["rules"]+="\nunauthorized"
  with tempfile.TemporaryDirectory(prefix="scope-control-",dir=scope.HERE)as tmp:
   p=pathlib.Path(tmp)/"body.json";p.write_text(json.dumps(body,sort_keys=True))
   c["public_request_file"]=str(p);c["public_request_file_sha256"]=scope.sha(p);c["public_context_bytes"]=p.stat().st_size
   with self.assertRaisesRegex(ValueError,"lossless_public_request_projection"):scope.verify(rebound(v))
 def test_symlink_paths_are_rejected(self):
  with tempfile.TemporaryDirectory(prefix="scope-control-",dir=scope.HERE)as tmp:
   p=pathlib.Path(tmp)/"link.json";p.symlink_to(scope.PROPOSAL)
   with self.assertRaisesRegex(ValueError,"frozen_file_path"):scope.rooted(p,scope.ROOT)
if __name__=="__main__":unittest.main(verbosity=2)
