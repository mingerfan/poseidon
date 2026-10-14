"""Offline integrity/budget boundaries for the frozen thirty-six-request proposal."""
import copy,hashlib,json,pathlib,tempfile,unittest
import scope as preparation
def rebound(value):
 value["binding"]=preparation.digest({k:v for k,v in value.items()if k!="binding"});return value
class ScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.proposal=preparation.read(preparation.PROPOSAL)
 def test_current_prepared_scope(self):
  result=preparation.verify(self.proposal)
  self.assertEqual(result["status"],"prepared_waiting_for_separate_approval")
  self.assertFalse(result["approved"]);self.assertEqual(result["provider_calls"],0)
 def test_changed_bound_document(self):
  v=copy.deepcopy(self.proposal);v["revision"]="changed"
  with self.assertRaisesRegex(ValueError,"proposal_binding"):preparation.fields(v)
 def test_cannot_treat_plan_as_authorization(self):
  v=copy.deepcopy(self.proposal);v["approved"]=True
  with self.assertRaisesRegex(ValueError,"planning_is_not_authorization"):preparation.fields(rebound(v))
 def test_cannot_mark_execution_started(self):
  v=copy.deepcopy(self.proposal);v["executed"]=True
  with self.assertRaisesRegex(ValueError,"planning_is_not_authorization"):preparation.fields(rebound(v))
 def test_no_automatic_budget_growth(self):
  for key,increment in [("max_generations",1),("max_http_attempts",1),("batch_wall_seconds",1),
                       ("new_batch_bytes",1),("native_concurrency",1)]:
   with self.subTest(key=key):
    v=copy.deepcopy(self.proposal);v["budget"][key]+=increment
    with self.assertRaisesRegex(ValueError,"reviewed_budget"):preparation.fields(rebound(v))
 def test_no_model_replacement(self):
  v=copy.deepcopy(self.proposal);v["provider"]["model"]="other"
  with self.assertRaisesRegex(ValueError,"reviewed_budget"):preparation.fields(rebound(v))
 def test_no_unknown_call_retry(self):
  v=copy.deepcopy(self.proposal);v["previous_unknown_http_retried"]=True
  with self.assertRaisesRegex(ValueError,"unknown_call_not_retried"):preparation.fields(rebound(v))
 def test_all_original_cases(self):
  for changed in ("missing","duplicate","reordered"):
   v=copy.deepcopy(self.proposal)
   if changed=="missing":v["cases"].pop()
   elif changed=="duplicate":v["cases"][-1]=v["cases"][0]
   else:v["cases"].reverse()
   with self.assertRaisesRegex(ValueError,"reviewed_cases"):preparation.fields(rebound(v))
 def test_existing_numerical_standard(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["validation_level"]="compile"
  with self.assertRaisesRegex(ValueError,"case_contract"):preparation.fields(rebound(v))
 def test_context_byte_ceiling(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["public_context_bytes"]=131073
  with self.assertRaisesRegex(ValueError,"context_budget"):preparation.fields(rebound(v))
 def test_source_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);sources=v["source_stage_identity"]["generation"]["sources"]
  name=next(iter(sources));sources[name]="0"*64
  with self.assertRaisesRegex(ValueError,"source_identity_changed"):preparation.verify(rebound(v))
 def test_weight_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);c=next(c for c in v["cases"]if c["weight_files"])
  c["weight_files"][next(iter(c["weight_files"]))]="0"*64
  with self.assertRaisesRegex(ValueError,"source_or_weight_changed"):preparation.verify(rebound(v))
 def test_file_byte_changes_invalidate_scope(self):
  v=copy.deepcopy(self.proposal);v["cases"][0]["public_request_file_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"frozen_request_changed"):preparation.verify(rebound(v))
 def test_redigested_public_body_is_rejected(self):
  v=copy.deepcopy(self.proposal);c=v["cases"][0];body=preparation.read(c["public_request_file"])
  body["rules"]+="\nunauthorized"
  with tempfile.TemporaryDirectory(prefix="scope-control-",dir=preparation.HERE)as tmp:
   p=pathlib.Path(tmp)/"body.json";p.write_text(json.dumps(body,sort_keys=True))
   c["public_request_file"]=str(p);c["public_request_file_sha256"]=preparation.sha(p);c["public_context_bytes"]=p.stat().st_size
   with self.assertRaisesRegex(ValueError,"public_request_identity"):preparation.verify(rebound(v))
 def test_symlink_paths_are_rejected(self):
  with tempfile.TemporaryDirectory(prefix="scope-control-",dir=preparation.HERE)as tmp:
   p=pathlib.Path(tmp)/"link.json";p.symlink_to(preparation.PROPOSAL)
   with self.assertRaisesRegex(ValueError,"frozen_file_path"):preparation.rooted(p,preparation.ROOT)

"""Offline accounting, authority, tool binding and unchanged request checks."""
import copy,json,os,pathlib,tempfile,unittest
from unittest.mock import patch
import runner_scope as scope
class RunnerScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.proposal=scope.read(scope.PROPOSAL)
 def authorized(self):
  a=dict(format="poseidon-source-human-authorization-v1",approved=True,
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
  self.assertEqual(v["tool_hashes"],scope.tools());self.assertEqual(len(v["cases"]),36)
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
 def test_all_cases_exact_ceiling(self):
  self.assertEqual(scope.totals([self.state(4,16)for _ in scope.CASES]),
    dict(generations=144,http_attempts=576,calls_resolved=True))
 def test_extra_scope_calls_are_rejected(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_ceiling"):
   scope.totals([self.state(4,16)for _ in range(len(scope.CASES)+1)])
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
 def test_completion_requires_all_results(self):
  result=scope.terminal([dict(case=scope.CASES[0],status="passed")],[])
  self.assertEqual(result["status"],"stopped");self.assertEqual(result["not_run"],35)
 def test_complete_failure_counts_are_separate(self):
  rows=[dict(case=c,status="passed"if i<35 else"failed")for i,c in enumerate(scope.CASES)]
  result=scope.terminal(rows,[]);self.assertEqual((result["status"],result["passed"],result["failed"]),("complete",35,1))
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

"""Unpaid supervisor and argument controls. No Provider constructed."""
import unittest,sys,os,time,subprocess,tempfile
from unittest.mock import patch
from pathlib import Path
import run,runner_scope as scope
class RunTests(unittest.TestCase):
 def proposal(self):
  return dict(binding=scope.PROPOSAL_BINDING,provider=scope.PROVIDER,budget=scope.BUDGET)
 def plan(self):
  public=run.public_plan(self.proposal());now=time.time()
  p=dict(public,approved=True,approval_record=str(scope.AUTHORIZATION),approval_sha256="0"*64,
   created=now,deadline_wall=now+scope.BUDGET["batch_wall_seconds"],initial_result_directories=[],owner=dict(pid=os.getpid(),start=1))
  p["binding"]=scope.digest(p);return p
 def test_no_approval_no_execution(self):
  with patch.object(run,"verify",side_effect=AssertionError("must not inspect production")),patch.dict(sys.modules,{"agent_credentials":None,"deepseek_provider":None}):
   with self.assertRaisesRegex(ValueError,"approval_record_required"):run.execute(scope.RUN,None)
 def test_wrong_root_rejected(self):
  with self.assertRaisesRegex(ValueError,"fixed_campaign_root"):run.execute(scope.RUN.parent/"other",scope.AUTHORIZATION)
 def test_existing_root_rejected(self):
  with patch.object(Path,"exists",return_value=True):
   with self.assertRaisesRegex(ValueError,"fresh_campaign_root"):run.execute(scope.RUN,scope.AUTHORIZATION)
 def test_public_cases(self):self.assertEqual(run.public_plan(self.proposal())["cases"],list(scope.CASES))
 def test_public_budget(self):self.assertEqual(run.public_plan(self.proposal())["budget"],scope.BUDGET)
 def test_hidden_approval_path(self):
  with self.assertRaisesRegex(ValueError,"approval_record_path"):run.approval_path(Path("/tmp/.hidden.json"))
 def test_plan_unknown_field(self):
  p=self.plan();p["extra"]=1
  with self.assertRaisesRegex(ValueError,"execution_plan_fields"):run.validate_plan(p,self.proposal())
 def test_plan_tamper(self):
  p=self.plan();p["budget"]=dict(p["budget"],max_generations=13)
  with self.assertRaisesRegex(ValueError,"execution_plan_binding"):run.validate_plan(p,self.proposal())
 def test_plan_extended_deadline(self):
  p=self.plan();p["deadline_wall"]+=1;p["binding"]=scope.digest({k:v for k,v in p.items() if k!="binding"})
  with self.assertRaisesRegex(ValueError,"original_batch_deadline"):run.validate_plan(p,self.proposal())
 def test_pid_identity(self):
  info=run.process_identity(os.getpid());self.assertTrue(run.same_process(info))
  info["start"]+=1;self.assertFalse(run.same_process(info))
 def test_owned_tree_only_owned_child(self):
  tree=run.OwnedTree().start()
  child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],start_new_session=True)
  try:
   time.sleep(.15);self.assertIn(child.pid,tree.living())
   self.assertTrue(tree.finish(exclude_reap=(child.pid,)))
   child.wait(timeout=3);self.assertIsNotNone(child.returncode)
  finally:
   if child.poll() is None:child.kill();child.wait()

 def test_approval_file_absent_does_not_construct_provider(self):
  with patch.dict(sys.modules,{"agent_credentials":None,"deepseek_provider":None}):
   with self.assertRaisesRegex(ValueError,"approval_record_path"):
    run.execute(scope.RUN,scope.AUTHORIZATION)

 def test_transport_receipt_normalized_roundtrip_preserves_diagnostics(self):
  # Validate the sidecar; the older billing format intentionally omits stream counters.
  sys.path.insert(0,str(scope.BASE))
  from application_transport_receipt import snapshot
  raw=dict(provider="deepseek_api",calls=[dict(index=0,generation_index=0,retry_index=0,status="failed",
   error="transport_invalid_stream",transport_diagnostics=dict(received_bytes=123,sse_events=4))],
   request_attempts=1,generation_attempts=1,transport_retries=0)
  receipt=snapshot(raw)
  self.assertEqual(receipt["calls"][0]["transport_diagnostics"],dict(received_bytes=123,sse_events=4))
  self.assertEqual(snapshot(receipt),receipt)



from collections import Counter
class CurrentSourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.proposal=preparation.read(preparation.PROPOSAL)
 def test_all_source_groups(self):
  self.assertEqual(dict(Counter(c["group"]for c in self.proposal["cases"])),
   dict(einsum=8,static_partition=6,einsum_views=6,scalar_partition=6,exact_arithmetic=8,mha=2))
 def test_six_concrete_source_versions(self):
  self.assertEqual({c["source_format"]for c in self.proposal["cases"]},
   {"poseidon-trusted-pytorch-v"+str(v)for v in range(16,22)})
 def test_source_manifest_format_is_original(self):
  for case in self.proposal["cases"]:
   with self.subTest(case=case["id"]):
    self.assertEqual(preparation.read(case["source_manifest"])["format"],case["source_format"])
 def test_source_identity_cannot_be_switched(self):
  value=copy.deepcopy(self.proposal);value["cases"][0]["source_format"]="poseidon-trusted-pytorch-v21"
  with self.assertRaisesRegex(ValueError,"original_source_case"):preparation.fields(rebound(value))
 def test_reference_preparation_does_not_claim_execution(self):
  report=scope.read(self.proposal["baseline_report"])
  self.assertEqual((report["status"],report["prepared"],report["failed"]),("prepared",36,0))
  self.assertEqual(report["independent_references_verified"],36)
  self.assertTrue(report["preparation_tasks_closed"])
  self.assertEqual([report[k]for k in ["provider_calls","http_attempts","compiler_calls","encrypted_calls"]],[0]*4)
  self.assertFalse(report["new_agent_generation"])
 def test_new_tasks_are_required(self):
  self.assertTrue(all(c["new_live_task_required"]is True and c["baseline_task_is_closed"]is True
   for c in self.proposal["cases"]))
  value=copy.deepcopy(self.proposal);value["cases"][0]["new_live_task_required"]=False
  value=rebound(value)
  with self.assertRaisesRegex(ValueError,"reviewed_provider_or_budget|public_generation_boundary"):scope.validate(value)
 def test_public_context_no_test_or_reference_fields(self):
  forbidden={"test_inputs","reference_outputs","test_cases","expected_outputs","api_key","hecate_source","baseline_source"}
  for case in self.proposal["cases"]:
   raw=preparation.read(case["public_request_file"])
   self.assertFalse(forbidden.intersection(raw))
 def test_public_source_context_ceiling(self):
  self.assertEqual(max(c["public_context_bytes"]for c in self.proposal["cases"]),119342)
 def test_structurally_distinct_requests(self):
  report=scope.read(self.proposal["baseline_report"])
  self.assertEqual([report[k]for k in ["unique_generation_requests","unique_semantic_signatures","unique_topology_signatures"]],[36]*3)
 def test_old_paid_scope_is_not_retried(self):
  self.assertFalse(self.proposal["previous_unknown_http_retried"])
  self.assertNotIn("r1333",str(scope.RUN));self.assertNotIn("r1341",str(scope.RUN))
 def test_plan_command_no_provider(self):
  command=[sys.executable,"-B",str(scope.HERE/"run.py"),"--plan"]
  result=subprocess.run(command,text=True,capture_output=True,timeout=40)
  self.assertEqual(result.returncode,0,result.stderr[:300])
  value=scope.decode(result.stdout)
  self.assertEqual(value["cases"],list(scope.CASES))
  self.assertFalse(value["approved"]);self.assertEqual(value["provider_calls"],0)
  self.assertFalse(scope.RUN.exists());self.assertFalse(scope.AUTHORIZATION.exists())
 def test_cli_execute_requires_approval_before_creation(self):
  result=subprocess.run([sys.executable,"-B",str(scope.HERE/"run.py"),"--execute",str(scope.RUN)],
   text=True,capture_output=True,timeout=10)
  self.assertNotEqual(result.returncode,0)
  self.assertIn("approval_record_required",result.stderr)
  self.assertFalse(scope.RUN.exists())
 def test_numeric_boolean_budget_rejected(self):
  value=copy.deepcopy(self.proposal);value["budget"]["api_concurrency"]=True
  with self.assertRaisesRegex(ValueError,"reviewed_provider_or_budget"):scope.validate(rebound(value))
 def test_output_receipt_requires_new_agent(self):
  plan=dict(binding="x");case=dict(request_id="r");name=scope.CASES[0]
  with tempfile.TemporaryDirectory(prefix="receipt-control-",dir=scope.HERE)as folder:
   p=Path(folder)
   run.write(p/"case-started.json",dict(binding="m"))
   row=dict(format="poseidon-source-live-case-report-v2",case=name,request_id="r",
    plan_binding="x",case_started_binding="m",status="passed",new_agent_generation=False)
   run.write(p/"case-report.json",row)
   with self.assertRaisesRegex(ValueError,"case_receipt_identity"):run.checked_receipt(p,name,plan,case)
 def test_budget_does_not_extend_native_resources(self):
  self.assertEqual(scope.BUDGET["native_address_bytes"],4*1024**3)
  self.assertEqual(scope.BUDGET["native_cpu_seconds"],150)
  self.assertEqual((scope.BUDGET["compile_jobs"],scope.BUDGET["link_jobs"],scope.BUDGET["native_concurrency"]),(2,1,1))

if __name__=="__main__":unittest.main(verbosity=2)
