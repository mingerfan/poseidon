"""Offline accounting, authority, tool binding and unchanged request checks."""
import copy,json,os,pathlib,tempfile,unittest
from unittest.mock import patch
import runner_scope as scope
class RunnerScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.proposal=scope.read(scope.PROPOSAL)
 def authorized(self):
  a=dict(format="poseidon-einsum-human-authorization-v1",approved=True,
    authorization="human_explicit_case_budget_approval",proposal_binding=scope.PROPOSAL_BINDING,
    proposal_sha256=scope.PROPOSAL_SHA,budget=copy.deepcopy(scope.BUDGET),provider=copy.deepcopy(scope.PROVIDER),
    cases=list(scope.CASES),human_message="offline schema fixture only",question_item_id="fixture-never-authorizes-execution")
  a["binding"]=scope.digest(a);return a
 def check_auth(self,a):
  a["binding"]=scope.digest({k:v for k,v in a.items()if k!="binding"})
  return scope.checked_authorization(a,self.proposal,scope.PROPOSAL_SHA)
 def state(self,g=1,h=1,status="response_received"):
  return dict(generations=g,http_attempts=h,ledger=[
    *[dict(event="http_reserved",index=i)for i in range(h)],
    dict(event="provider_checkpoint",outcomes=[dict(index=i,status=status)for i in range(h)])])
 def test_original_proposal_and_view(self):
  v=scope.verify()
  self.assertEqual(v["application_stage_identity"],self.proposal["source_stage_identity"])
  self.assertEqual(v["tool_hashes"],scope.tools());self.assertEqual(len(v["cases"]),8)
 def test_offline_authorization_schema(self):
  self.assertEqual(self.check_auth(self.authorized())["budget"],scope.BUDGET)
 def test_old_scope_authorization_does_not_apply(self):
  a=self.authorized();a["proposal_binding"]="0"*64
  with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_wrong_proposal_sha_does_not_apply(self):
  a=self.authorized();a["proposal_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_case_scope_cannot_shrink_or_expand(self):
  for change in("less","more","reorder"):
   with self.subTest(change=change):
    a=self.authorized()
    if change=="less":a["cases"].pop()
    elif change=="more":a["cases"].append(a["cases"][0])
    else:a["cases"].reverse()
    with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_boolean_is_not_numeric_budget(self):
  a=self.authorized();a["budget"]["link_jobs"]=True
  with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_budget_cannot_grow(self):
  for key in("max_generations","max_http_attempts","batch_wall_seconds","native_address_bytes"):
   with self.subTest(key=key):
    a=self.authorized();a["budget"][key]+=1
    with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_approval_must_be_literal_true(self):
  for v in(False,1,"true"):
   a=self.authorized();a["approved"]=v
   with self.assertRaisesRegex(ValueError,"human_authorization_scope"):self.check_auth(a)
 def test_authorization_record_needs_human_provenance(self):
  for key in("human_message","question_item_id"):
   a=self.authorized();a[key]=""
   with self.assertRaisesRegex(ValueError,"human_authorization_record"):self.check_auth(a)
 def test_authorization_extra_data_is_rejected(self):
  a=self.authorized();a["private_reference"]=[1]
  with self.assertRaisesRegex(ValueError,"human_authorization_fields"):self.check_auth(a)
 def test_duplicate_json_fields(self):
  with self.assertRaisesRegex(ValueError,"duplicate_json_field"):scope.decode('{"x":1,"x":2}')
 def test_nonfinite_json(self):
  for bad in("NaN","Infinity","-Infinity"):
   with self.assertRaisesRegex(ValueError,"nonfinite_json"):scope.decode('{"x":'+bad+'}')
 def test_empty_accounting(self):
  self.assertEqual(scope.totals([]),dict(generations=0,http_attempts=0,calls_resolved=True))
 def test_eight_cases_exact_ceiling(self):
  self.assertEqual(scope.totals([self.state(4,16)for _ in scope.CASES]),
    dict(generations=32,http_attempts=128,calls_resolved=True))
 def test_extra_scope_calls_are_rejected(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_ceiling"):
   scope.totals([self.state(4,16)for _ in range(9)])
 def test_per_case_calls_are_bounded(self):
  for g,h in((5,1),(1,17),(-1,0),(0,-1),(True,1),(1,True)):
   with self.assertRaisesRegex(ValueError,"case_call_ceiling"):scope.totals([self.state(g,h)])
 def test_unknown_call_is_not_resolved(self):
  self.assertFalse(scope.totals([self.state(status="unknown")])["calls_resolved"])
 def test_missing_call_outcome_is_not_resolved(self):
  v=self.state();v["ledger"][-1]["outcomes"]=[]
  self.assertFalse(scope.totals([v])["calls_resolved"])
 def test_failed_http_is_resolved_not_a_pass(self):
  self.assertTrue(scope.totals([self.state(status="failed")])["calls_resolved"])
 def test_ledger_gap_is_integrity_failure(self):
  v=self.state();v["ledger"][0]["index"]=2
  with self.assertRaisesRegex(ValueError,"reservation_ledger"):scope.totals([v])
 def test_missing_reservation_is_integrity_failure(self):
  v=self.state();v["ledger"].pop(0)
  with self.assertRaisesRegex(ValueError,"reservation_ledger"):scope.totals([v])
 def test_completion_requires_eight_results(self):
  result=scope.terminal([dict(case=scope.CASES[0],status="passed")],[])
  self.assertEqual(result["status"],"stopped");self.assertEqual(result["not_run"],7)
 def test_complete_failure_counts_are_separate(self):
  rows=[dict(case=c,status="passed"if i<7 else"failed")for i,c in enumerate(scope.CASES)]
  result=scope.terminal(rows,[]);self.assertEqual((result["status"],result["passed"],result["failed"]),("complete",7,1))
 def test_complete_unresolved_batch_is_stopped(self):
  rows=[dict(case=c,status="passed")for c in scope.CASES]
  self.assertEqual(scope.terminal(rows,[self.state(status="unknown")])["status"],"stopped")
 def test_duplicate_result_rejected(self):
  row=dict(case=scope.CASES[0],status="passed")
  with self.assertRaisesRegex(ValueError,"result_case_scope"):scope.terminal([row,row],[])
 def test_unknown_case_rejected(self):
  with self.assertRaisesRegex(ValueError,"result_case_scope"):scope.terminal([dict(case="other",status="passed")],[])
 def test_skip_cannot_count_as_pass(self):
  with self.assertRaisesRegex(ValueError,"result_case_scope"):
   scope.terminal([dict(case=scope.CASES[0],status="skipped")],[])
 def test_manifest_fields_bound(self):
  value=scope.read(scope.MANIFEST);value["extra"]=True
  with self.assertRaisesRegex(ValueError,"runner_manifest_fields"):scope.check_manifest(value)
 def test_manifest_tool_bytes_bound(self):
  value=scope.read(scope.MANIFEST);key=next(iter(value["tool_hashes"]));value["tool_hashes"][key]="0"*64
  value["binding"]=scope.digest({k:v for k,v in value.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"runner_manifest_binding"):scope.check_manifest(value)
 def test_lossless_request_decode(self):
  p=self.proposal["cases"][0]["full_request_file"]
  raw=scope.read(p);request=scope.request_read(p)
  self.assertEqual(request["request_id"],raw["request_id"])
  self.assertNotEqual(request["constant_origins"],raw["constant_origins"])
  from public_weight_context import project
  self.assertEqual(scope.digest({k:v for k,v in project(request).items()if k!="request_id"}),request["request_id"])
 def test_exact_proposal_required(self):
  p=copy.deepcopy(self.proposal);p["cases"][0]["options"]["chunk_period"]=8
  with self.assertRaisesRegex(ValueError,"exact_original_proposal"):scope.verify(p)
 def test_original_planning_documents_not_approved(self):
  self.assertFalse(self.proposal["approved"]);self.assertFalse(self.proposal["executed"])
  self.assertFalse(scope.AUTHORIZATION.exists())
 def test_authority_absent_stops_before_provider(self):
  with patch.dict("sys.modules",{"agent_credentials":None,"deepseek_provider":None}):
   with self.assertRaisesRegex(ValueError,"approval_record_fields"):scope.authority({},scope.verify(),scope.PROPOSAL_SHA)
if __name__=="__main__":unittest.main()
