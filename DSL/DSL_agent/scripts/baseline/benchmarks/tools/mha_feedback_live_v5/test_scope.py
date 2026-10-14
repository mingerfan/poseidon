"""Unpaid exact authority, request scope and call-accounting controls."""
import copy,unittest,tempfile
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
import scope
class ScopeTests(unittest.TestCase):
 def setUp(self):self.p=scope.read(scope.PROPOSAL)
 def changed(self,fn):
  p=copy.deepcopy(self.p);fn(p);p["binding"]=scope.digest({k:v for k,v in p.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.validate(p)
 def test_original_proposal(self):self.assertEqual(scope.validate(self.p),self.p)
 def test_source_case_order(self):self.changed(lambda p:p["cases"].reverse())
 def test_add_case(self):self.changed(lambda p:p["cases"].append(p["cases"][0]))
 def test_remove_case(self):self.changed(lambda p:p["cases"].pop())
 def test_model_name(self):self.changed(lambda p:p["provider"].update(model="other"))
 def test_budget(self):self.changed(lambda p:p["bounds"].update(max_generations=13))
 def test_budget_bool(self):self.changed(lambda p:p["bounds"].update(compiler_jobs=True))
 def test_threshold(self):self.changed(lambda p:p["numeric_tolerance"].update(atol=1e-4))
 def test_validation_grade(self):self.changed(lambda p:p["cases"][0].update(validation_level="compiled"))
 def test_missing_self_binding(self):
  p=copy.deepcopy(self.p);p.pop("binding")
  with self.assertRaises(ValueError):scope.validate(p)
 def test_duplicate_json(self):
  with self.assertRaises(ValueError):scope.decode(b'{"a":1,"a":2}')
 def test_nonfinite_json(self):
  with self.assertRaises(ValueError):scope.decode(b'{"a":NaN}')
 def test_missing_approval(self):
  with self.assertRaises(ValueError):scope.authority({},self.p,scope.PROPOSAL_SHA)
 def test_pending_calls(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0)])
  self.assertFalse(scope.totals([s])["calls_resolved"])
 def test_resolved_calls(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0),dict(event="provider_checkpoint",outcomes=[dict(index=0,status="response_received")])])
  self.assertEqual(scope.totals([s]),dict(generations=1,http_attempts=1,calls_resolved=True))
 def test_unknown_outcome(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0),dict(event="provider_checkpoint",outcomes=[dict(index=0,status="unknown")])])
  self.assertFalse(scope.totals([s])["calls_resolved"])
 def test_http_bound(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=1,http_attempts=17,ledger=[])])
 def test_generation_bound(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=5,http_attempts=0,ledger=[])])
 def test_ledger_gap(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=1)])])
 def test_boolean_accounting(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=True,http_attempts=0,ledger=[])])
 def test_no_calls(self):self.assertTrue(scope.totals([])["calls_resolved"])
 def test_failure_not_pass(self):
  r=scope.terminal([dict(case=scope.CASES[0],status="failed")],[])
  self.assertEqual((r["passed"],r["failed"],r["not_run"],r["status"]),(0,1,1,"stopped"))
 def test_duplicate_terminal_case(self):
  with self.assertRaises(ValueError):scope.terminal([dict(case=scope.CASES[0],status="passed")]*2,[])
 def test_illegal_terminal_case(self):
  with self.assertRaises(ValueError):scope.terminal([dict(case="linear",status="passed")],[])
 def test_regular_symlink_scope(self):
  with self.assertRaises(ValueError):scope.regular(Path("/etc/passwd"))
 def test_tool_manifest_cannot_add_file(self):
  m=dict(format="poseidon-mha-feedback-live-runner-v1",proposal_binding=scope.PROPOSAL_BINDING,proposal_sha256=scope.PROPOSAL_SHA,tool_hashes={"x":"y"})
  m["binding"]=scope.digest(m)
  with patch.object(scope,"tools",return_value={}):
   with self.assertRaises(ValueError):scope.check_manifest(m)

 def accounted(self,g,h):
  return dict(generations=g,http_attempts=h,ledger=[dict(event="http_reserved",index=i) for i in range(h)]+
   [dict(event="provider_checkpoint",outcomes=[dict(index=i,status="failed") for i in range(h)])])
 def test_two_cases_exact_maximum(self):
  r=scope.totals([self.accounted(4,16),self.accounted(4,16)])
  self.assertEqual(r,dict(generations=8,http_attempts=32,calls_resolved=True))
 def test_third_generation_scope_rejected(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_ceiling"):
   scope.totals([self.accounted(4,16),self.accounted(4,16),self.accounted(1,1)])
 def test_complete_means_both_reviewed_cases(self):
  rows=[dict(case=name,status="passed") for name in scope.CASES]
  self.assertEqual(scope.terminal(rows,[])["status"],"complete")
 def test_original_three_case_approval_is_rejected(self):
  old=scope.read(scope.ROOT/"docs/baseline/agent-mha-live-authorization-r1313.json")
  with self.assertRaises(ValueError):scope.checked_authorization(old,self.p,scope.PROPOSAL_SHA)
 def test_unapproved_record_rejected(self):
  with self.assertRaises(ValueError):scope.checked_authorization({},self.p,scope.PROPOSAL_SHA)

 def test_correction_matches_already_reviewed_execution_identity(self):
  old=scope.corrected_review_equivalence(self.p)
  self.assertNotEqual(old["source_stage_identity"],self.p["source_stage_identity"])
  for stage in ("generation","compilation","execution","validation"):
   self.assertEqual(old["source_stage_identity"][stage]["backend"],self.p["source_stage_identity"][stage]["backend"])
  self.assertEqual(old["cases"],self.p["cases"])
  self.assertEqual(old["bounds"],self.p["bounds"])
 def test_correction_cannot_change_request_or_budget(self):
  for mutate in (lambda p:p["bounds"].update(max_generations=9),
                 lambda p:p["cases"][0].update(request_id="0"*64)):
   p=copy.deepcopy(self.p);mutate(p)
   with self.assertRaises(ValueError):scope.corrected_review_equivalence(p)
 def test_correction_cannot_add_unreviewed_source(self):
  p=copy.deepcopy(self.p);p["source_capture_implementation"]["outside.py"]="0"*64
  with self.assertRaises(ValueError):scope.corrected_review_equivalence(p)

 def synthetic_authorization(self,reviewed=False):
  # Pure data fixture only; no production approval file is ever created.
  value=dict(format="poseidon-mha-feedback-human-authorization-v1",
   approved=True,authorization="human_explicit_case_budget_approval",bounds=scope.BOUNDS,
   cases=list(scope.CASES),human_message="SYNTHETIC UNIT TEST; no execution authority",
   proposal_binding=scope.REVIEWED_BINDING if reviewed else scope.PROPOSAL_BINDING,
   proposal_sha256=scope.REVIEWED_SHA if reviewed else scope.PROPOSAL_SHA,
   provider=scope.PROVIDER,question_item_id="SYNTHETIC UNIT TEST")
  value["binding"]=scope.digest(value)
  return value
 def test_matching_authorization_data_schema(self):
  value=self.synthetic_authorization()
  self.assertEqual(scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA),value)
 def test_previous_proposal_authorization_cannot_authorize_new_source(self):
  value=self.synthetic_authorization(reviewed=True)
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_authorization_sha_pair_cannot_be_mixed(self):
  value=self.synthetic_authorization(reviewed=True)
  value["proposal_sha256"]=scope.PROPOSAL_SHA
  value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_authorization_cannot_grant_extra_budget(self):
  value=self.synthetic_authorization()
  value["bounds"]=dict(value["bounds"],max_http_attempts=33)
  value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_authorization_cannot_use_integer_approval(self):
  value=self.synthetic_authorization();value["approved"]=1
  value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_boolean_budget_is_distinct_from_integer_budget(self):
  value=self.synthetic_authorization()
  value["bounds"]=dict(value["bounds"],api_concurrency=True)
  value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_authorization_cannot_include_private_or_extra_fields(self):
  value=self.synthetic_authorization();value["reference"]="private"
  value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)
 def test_authorization_requires_human_record_metadata(self):
  for field in ("human_message","question_item_id"):
   value=self.synthetic_authorization();value[field]=""
   value["binding"]=scope.digest({k:v for k,v in value.items() if k!="binding"})
   with self.subTest(field=field),self.assertRaises(ValueError):
    scope.checked_authorization(value,self.p,scope.PROPOSAL_SHA)

 @contextmanager
 def authority_fixture(self):
  # Local synthetic data cannot authorize the fixed production AUTHORIZATION path.
  with tempfile.TemporaryDirectory(dir=scope.HERE,prefix="authority-unit-") as folder:
   path=Path(folder)/"synthetic-human.json"
   path.write_bytes(scope.canonical(self.synthetic_authorization()))
   manifest=scope.check_manifest()
   receipt=dict(format="poseidon-mha-feedback-live-approval-v1",approved=True,
    proposal_binding=scope.PROPOSAL_BINDING,proposal_sha256=scope.PROPOSAL_SHA,
    runner_manifest_sha256=scope.sha(scope.MANIFEST),tool_hashes=scope.tools(),
    authorization="human_explicit_case_budget_approval",authorization_record=str(path),
    authorization_sha256=scope.sha(path))
   with patch.object(scope,"AUTHORIZATION",path):
    yield receipt,scope.execution_view(self.p,manifest)
 def test_authority_accepts_actual_normalized_execution_view(self):
  with self.authority_fixture() as (receipt,view):
   self.assertEqual(view,scope.verify(self.p))
   scope.authority(receipt,view,scope.PROPOSAL_SHA)
 def test_authority_refuses_changed_derived_budget(self):
  with self.authority_fixture() as (receipt,view):
   view["budget"]["max_http_attempts"]=33
   with self.assertRaisesRegex(ValueError,"execution_view_changed"):
    scope.authority(receipt,view,scope.PROPOSAL_SHA)
 def test_authority_refuses_changed_case_execution_fields(self):
  for key,value in (("full_request_file","/unapproved"),
   ("current_validation_sources",{}),("request_id","0"*64)):
   with self.subTest(key=key),self.authority_fixture() as (receipt,view):
    view["cases"][0][key]=value
    with self.assertRaisesRegex(ValueError,"execution_view_changed"):
     scope.authority(receipt,view,scope.PROPOSAL_SHA)
 def test_authority_refuses_changed_application_identity(self):
  with self.authority_fixture() as (receipt,view):
   view["application_stage_identity"]["generation"]["sources"]={}
   with self.assertRaisesRegex(ValueError,"execution_view_changed"):
    scope.authority(receipt,view,scope.PROPOSAL_SHA)
 def test_authority_refuses_unknown_execution_field(self):
  with self.authority_fixture() as (receipt,view):
   view["additional_calls"]=1
   with self.assertRaisesRegex(ValueError,"execution_view_changed"):
    scope.authority(receipt,view,scope.PROPOSAL_SHA)
 def test_authority_refuses_raw_proposal_as_execution_view(self):
  with self.authority_fixture() as (receipt,view):
   raw=dict(self.p,tool_hashes=view["tool_hashes"])
   with self.assertRaisesRegex(ValueError,"execution_view_changed"):
    scope.authority(receipt,raw,scope.PROPOSAL_SHA)

if __name__=="__main__":unittest.main()
