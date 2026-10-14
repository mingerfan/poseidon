"""Accounting, authority, process and evidence checks for a never-started-only continuation."""
import copy,importlib.util,json,os,subprocess,sys,time,unittest
import continuation_scope as scope
spec=importlib.util.spec_from_file_location("continuation_runtime_under_test",scope.HERE/"run.py")
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ScopeTests(unittest.TestCase):
 def original(self):return scope.original.verify()
 def test_actual_retired_evidence_and_counters(self):
  p=scope.retired_snapshot();self.assertEqual(p["counts"],dict(generations=34,http_attempts=35,calls_resolved=False))
  self.assertEqual(len(p["remaining"]),7);self.assertFalse(scope.legacy.same_process(p["plan"]["owner"]))
 def test_subset_excludes_all_started_cases(self):
  v=self.original();selected=scope.select_cases(v["cases"],scope.STARTED)
  self.assertEqual(tuple(c["id"]for c in selected),scope.CASES)
  self.assertFalse(set(scope.CASES)&set(scope.STARTED));self.assertEqual(len(scope.CASES),7)
 def test_reordered_or_added_original_case_rejected(self):
  for alter in(lambda v:v["cases"].reverse(),lambda v:v["cases"].append(copy.deepcopy(v["cases"][0]))):
   v=self.original();alter(v)
   with self.assertRaisesRegex(ValueError,"original_case_inventory"):scope.select_cases(v["cases"],scope.STARTED)
 def test_omitted_started_case_is_not_resumable(self):
  for name in scope.STARTED:
   with self.subTest(case=name),self.assertRaisesRegex(ValueError,"retired_started_case_inventory"):
    scope.select_cases(self.original()["cases"],tuple(c for c in scope.STARTED if c!=name))
 def test_full_projection_preserves_every_request_and_backend(self):
  original=self.original();new=scope.verify()
  self.assertEqual([c for c in original["cases"]if c["id"]not in scope.STARTED],new["cases"])
  for k in("provider","budget","source_stage_identity","application_stage_identity"):self.assertEqual(original[k],new[k])
 def test_original_human_approval_authorizes_exact_subset(self):
  p=scope.verify();scope.authority(scope.read(scope.EXECUTION_APPROVAL),p,scope.original.PROPOSAL_SHA)
 def test_changed_model_does_not_reuse_approval(self):
  p=scope.verify();p["cases"][0]["request_id"]="0"*64
  with self.assertRaisesRegex(ValueError,"original_human_approval_subset"):
   scope.authority(scope.read(scope.EXECUTION_APPROVAL),p,scope.original.PROPOSAL_SHA)
 def test_changed_provider_or_budget_does_not_reuse_approval(self):
  for k,field,value in(("provider","stream",False),("budget","api_concurrency",2)):
   p=scope.verify();p[k][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):
    scope.authority(scope.read(scope.EXECUTION_APPROVAL),p,scope.original.PROPOSAL_SHA)
 def test_unapproved_receipt_rejected(self):
  receipt=scope.read(scope.EXECUTION_APPROVAL);receipt["approved"]=False
  with self.assertRaises(ValueError):scope.authority(receipt,scope.verify(),scope.original.PROPOSAL_SHA)
 def test_manifest_cannot_authorize_changed_tool(self):
  m=scope.read(scope.MANIFEST);k=next(iter(m["tool_hashes"]));m["tool_hashes"][k]="0"*64
  m["binding"]=scope.digest({k:v for k,v in m.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"continuation_tool_manifest"):scope.check_manifest(m)
 def test_manifest_cannot_replace_retired_evidence(self):
  m=scope.read(scope.MANIFEST);m["retired_audit_sha256"]="0"*64
  m["binding"]=scope.digest({k:v for k,v in m.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"continuation_tool_manifest"):scope.check_manifest(m)
 def test_original_deadline_and_storage_baseline_retained(self):
  p=scope.prior_context();old=scope.read(scope.OLD_RUN/"plan.json")
  self.assertEqual(p["created"],old["created"]);self.assertEqual(p["deadline_wall"],old["deadline_wall"])
  self.assertEqual(p["deadline_wall"],p["created"]+scope.original.BUDGET["batch_wall_seconds"])
  self.assertEqual(p["initial_result_directories"],old["initial_result_directories"])
 def test_plan_records_parked_unknown_call(self):
  p=run.preview();self.assertEqual(p["counter_offset"],dict(generations=34,http_attempts=35,calls_resolved=False))
  self.assertEqual(p["parked_http_attempts"],[dict(case="einsum_public_weight",index=1,status="unknown"),dict(case="scalar_shared_power_division",index=0,status="unknown")])
  self.assertTrue(p["original_budget_kept"]);self.assertEqual(len(p["cases"]),7)
 def test_retired_owner_is_never_accepted(self):
  with self.assertRaisesRegex(ValueError,"execution_owner_identity"):run.owner_current(scope.retired_snapshot()["plan"]["owner"])
class AccountingTests(unittest.TestCase):
 def counted(self,g=0,h=0,resolved=True,prior=None):
  return scope.counts(dict(generations=g,http_attempts=h,calls_resolved=resolved),
   prior or dict(generations=34,http_attempts=35,calls_resolved=False),scope.original.BUDGET)
 def test_unknown_reservation_remains_charged(self):
  c=self.counted();self.assertEqual(c["cumulative_generations"],34);self.assertEqual(c["cumulative_http_attempts"],35)
  self.assertFalse(c["all_campaign_calls_resolved"]);self.assertFalse(c["prior_calls_resolved"])
 def test_remaining_worst_case_stays_in_original_budget(self):
  c=self.counted(7*4,7*16);self.assertEqual((c["cumulative_generations"],c["cumulative_http_attempts"]),(62,147))
 def test_global_exact_boundary_and_one_more(self):
  c=self.counted(110,541);self.assertEqual((c["cumulative_generations"],c["cumulative_http_attempts"]),(144,576))
  for g,h in((111,541),(110,542)):
   with self.subTest(g=g,h=h),self.assertRaisesRegex(ValueError,"cumulative_original_call_ceiling"):self.counted(g,h)
 def test_prior_calls_cannot_be_zeroed_or_settled(self):
  for p in(dict(generations=0,http_attempts=0,calls_resolved=False),dict(generations=34,http_attempts=35,calls_resolved=True)):
   with self.assertRaisesRegex(ValueError,"fixed_prior_call_reservations"):self.counted(prior=p)
 def test_invalid_counter_types_rejected(self):
  for g,h,r in((True,0,True),(-1,0,True),(0,False,True),(0,-1,True),(0,0,1)):
   with self.subTest(g=g,h=h),self.assertRaisesRegex(ValueError,"continuation_call_counts"):self.counted(g,h,r)
 def test_complete_subset_does_not_claim_historical_call_settled(self):
  rows=[dict(case=c,status="passed")for c in scope.CASES]
  r=scope.terminal(rows,[]);self.assertEqual(r["status"],"complete")
  self.assertEqual(r["result_scope"],"original_never_started_after_second_restart")
  self.assertEqual((r["overall_campaign_passed"],r["overall_campaign_failed"],r["overall_campaign_not_run"]),(34,1,0))
  self.assertEqual(r["overall_campaign_status"],"unresolved_prior_http");self.assertFalse(r["all_campaign_calls_resolved"])
  self.assertEqual(r["overall_campaign_interrupted"],1);self.assertEqual(r["historical_interrupted"],1)
 def test_duplicate_or_started_case_in_results_rejected(self):
  for rows in([dict(case=scope.STARTED[0],status="passed")],
   [dict(case=scope.CASES[0],status="passed")]*2):
   with self.assertRaisesRegex(ValueError,"continuation_case_scope"):scope.terminal(rows,[])
class ExecutionGuards(unittest.TestCase):
 def test_new_calls_cannot_target_started_case(self):
  for name in scope.STARTED:
   with self.subTest(case=name),self.assertRaisesRegex(ValueError,"reviewed_cases"):run.one(scope.RUN,name)
 def test_live_execution_requires_approval_and_fresh_root(self):
  with self.assertRaisesRegex(ValueError,"approval_record_required"):run.execute(scope.RUN,None)
  with self.assertRaisesRegex(ValueError,"fixed_campaign_root"):run.execute(scope.OLD_RUN,scope.EXECUTION_APPROVAL)
 def test_live_owner_and_lock_are_both_required(self):
  identity=run.process_identity(os.getpid())
  with self.assertRaisesRegex(ValueError,"execution_owner_lock"):run.owner_current(identity)
  with scope.owned_lock(run.LOCK):run.owner_current(identity)
 def test_child_cannot_be_accepted_as_owner_of_parent(self):
  child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(5)"])
  try:
   with self.assertRaisesRegex(ValueError,"execution_owner_ancestry"):run.owner_current(run.process_identity(child.pid))
  finally:child.terminate();child.wait(timeout=5)
 def test_concurrent_legacy_runner_is_rejected(self):
  child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(5)",str(scope.OLDHERE/"run.py")])
  try:
   with self.assertRaisesRegex(ValueError,"active_project_execution"):run.no_active()
  finally:child.terminate();child.wait(timeout=5)

class RebootBoundaries(unittest.TestCase):
 def test_interrupted_case_is_excluded_and_old_report_unchanged(self):
  name="scalar_shared_power_division"
  self.assertIn(name,scope.STARTED);self.assertNotIn(name,scope.CASES)
  p=scope.retired_snapshot()
  self.assertFalse((scope.OLD_RUN/name/"case-report.json").exists())
  self.assertEqual(scope.sha(scope.OLD_RUN/"report.json"),p["evidence"][str(scope.OLD_RUN/"report.json")])
 def test_previous_continuation_process_is_rejected(self):
  child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(5)",str(scope.PREVIOUS_HERE/"run.py")])
  try:
   with self.assertRaisesRegex(ValueError,"active_project_execution"):run.no_active()
  finally:child.terminate();child.wait(timeout=5)
