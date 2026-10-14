"""Pure cutover gates. No source mutation or execution."""
import copy,unittest
from guarded_apply import eligible,closed_approval
class BoundaryTests(unittest.TestCase):
 def setUp(self):
  self.root=dict(status="complete",planned=32,groups=[
   dict(id="matrix_instance60",status="complete",exit_code=0),dict(id="attention1",status="complete",exit_code=0)])
  self.groups={name:dict(status="complete",planned=n,not_run=0,passed=n-1,failed=1,
     rows=[dict(status="failed" if i==n-1 else "passed",result=dict(phase="complete")) for i in range(n)]) for name,n in (("matrix_instance60",31),("attention1",1))}
 def test_closed_scope_can_reach_source_checks(self):self.assertTrue(eligible(self.root,self.groups))
 def test_running_batch_never_applies(self):
  self.root["status"]="running";self.assertFalse(eligible(self.root,self.groups))
 def test_provider_stop_with_pending_scope_never_applies(self):
  self.root["status"]="stopped";self.assertFalse(eligible(self.root,self.groups))
 def test_unstarted_attention_never_applies(self):
  del self.groups["attention1"];self.assertFalse(eligible(self.root,self.groups))
 def test_active_task_even_in_closed_report_rejected(self):
  self.groups["matrix_instance60"]["rows"][0]["result"]["phase"]="validating"
  self.assertFalse(eligible(self.root,self.groups))
 def test_duplicate_group_record_rejected(self):
  self.root["groups"].append(copy.deepcopy(self.root["groups"][0]));self.assertFalse(eligible(self.root,self.groups))
 def test_bad_denominator_rejected(self):
  self.groups["matrix_instance60"]["not_run"]=1;self.assertFalse(eligible(self.root,self.groups))

 def test_real_row_counts_must_match(self):
  self.groups["matrix_instance60"]["passed"]=31
  self.groups["matrix_instance60"]["failed"]=0
  self.assertFalse(eligible(self.root,self.groups))
 def test_latest_25_scope(self):
  self.root["planned"]=25
  self.groups["matrix_instance60"].update(planned=24,passed=23,failed=1,
    rows=[dict(status="failed" if i==23 else "passed",result=dict(phase="complete")) for i in range(24)])
  self.assertTrue(eligible(self.root,self.groups,{"matrix_instance60":24,"attention1":1}))
 def test_latest_14_scope(self):
  self.root["planned"]=14
  self.groups["matrix_instance60"].update(planned=13,passed=12,failed=1,
    rows=[dict(status="failed" if i==12 else "passed",result=dict(phase="complete")) for i in range(13)])
  self.assertTrue(eligible(self.root,self.groups,{"matrix_instance60":13,"attention1":1}))
class WholeApprovalTests(unittest.TestCase):
 def setUp(self):
  all_cases=[dict(id="case_"+str(i),request_id="hash_"+str(i),options=dict(fixed=True)) for i in range(76)]
  self.original=dict(groups=dict(full=dict(cases=[dict(case=x) for x in all_cases])),
    prior_generation_reservations=1,prior_http_attempts=1)
  self.closed=[dict(id=x["id"],request_id=x["request_id"],status="passed" if i<42 else "failed",
    calls_resolved=True,not_retried=True,generations=1,http_attempts=1) for i,x in enumerate(all_cases[:51])]
  self.current=dict(groups=dict(matrix_instance60=dict(cases=[dict(case=x) for x in all_cases[51:75]]),
    attention1=dict(cases=[dict(case=all_cases[75])])),
    closed_contexts=self.closed,historical_passed=42,historical_failed=9,
    prior_generation_reservations=52,prior_http_attempts=52)
  self.groups={k:dict(rows=[dict(id=x["case"]["id"],request_id=x["case"]["request_id"]) for x in v["cases"]])
    for k,v in self.current["groups"].items()}
 def test_exact_76_closed(self):self.assertTrue(closed_approval(self.current,self.groups,self.original))
 def test_pending_original_not_ignored(self):
  self.groups["attention1"]["rows"]=[]
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_duplicate_history(self):
  self.current["closed_contexts"][50]=copy.deepcopy(self.current["closed_contexts"][0])
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_unresolved_history(self):
  self.current["closed_contexts"][0]["calls_resolved"]=False
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_history_binding_change(self):
  self.current["closed_contexts"][0]["request_id"]="wrong"
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_models_and_options_fixed(self):
  self.current["groups"]["attention1"]["cases"][0]["case"]=dict(id="case_75",request_id="hash_75",options=dict(fixed=False))
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_calls_not_reset(self):
  self.current["prior_generation_reservations"]=0
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_history_count_not_relabeled(self):
  self.current["historical_passed"]=51
  self.assertFalse(closed_approval(self.current,self.groups,self.original))
 def test_closed_failure_may_remain_failed(self):
  self.assertEqual(self.current["historical_failed"],9)
  self.assertTrue(closed_approval(self.current,self.groups,self.original))


 def test_closed_history_count_follows_exact_remaining_scope(self):
  all_cases=[entry["case"] for entry in self.original["groups"]["full"]["cases"]]
  for closed_count in (44,51,62):
   current=copy.deepcopy(self.current)
   history=[dict(id=x["id"],request_id=x["request_id"],status="passed" if i<closed_count-13 else "failed",
      calls_resolved=True,not_retried=True,generations=1,http_attempts=1)
      for i,x in enumerate(all_cases[:closed_count])]
   current.update(closed_contexts=history,historical_passed=closed_count-13,historical_failed=13,
      prior_generation_reservations=closed_count+1,prior_http_attempts=closed_count+1,
      groups=dict(matrix_instance60=dict(cases=[dict(case=x) for x in all_cases[closed_count:75]]),
                  attention1=dict(cases=[dict(case=all_cases[75])])))
   groups={k:dict(rows=[dict(id=x["case"]["id"],request_id=x["case"]["request_id"]) for x in v["cases"]])
       for k,v in current["groups"].items()}
   with self.subTest(history=closed_count):
    self.assertTrue(closed_approval(current,groups,self.original))
    missing=copy.deepcopy(current);missing["closed_contexts"].pop()
    self.assertFalse(closed_approval(missing,groups,self.original))

if __name__=="__main__":unittest.main(verbosity=2)
