"""Unpaid denominator and transport-outcome controls."""
import copy,unittest
from acceptance_outcomes import terminal_counts,receipt_coverage
class OutcomeTests(unittest.TestCase):
 cases=("a","b","c")
 def report(self,status="complete",rows=None):
  rows=rows if rows is not None else[dict(case=n,status="passed")for n in self.cases]
  return dict(status=status,planned=3,passed=sum(r["status"]=="passed"for r in rows),
   failed=sum(r["status"]=="failed"for r in rows),not_run=3-len(rows),generations=3,http_attempts=4,calls_resolved=True,rows=rows)
 def calls(self):return dict(generations=3,http_attempts=4,calls_resolved=True)
 def reject(self,change):
  r=self.report();change(r)
  with self.assertRaises(ValueError):terminal_counts(r,self.calls(),self.cases)
 def test_all_passed(self):self.assertEqual(terminal_counts(self.report(),self.calls(),self.cases)["benchmark_status"],"passed")
 def test_known_failure_is_counted(self):
  r=self.report(rows=[dict(case="a",status="passed"),dict(case="b",status="failed"),dict(case="c",status="passed")])
  v=terminal_counts(r,self.calls(),self.cases);self.assertEqual((v["passed"],v["failed"],v["not_run"],v["benchmark_status"]),(2,1,0,"completed_with_failures"))
 def test_stopped_missing_case_remains_not_run(self):
  v=terminal_counts(self.report("stopped",[dict(case="a",status="failed")]),self.calls(),self.cases)
  self.assertEqual((v["passed"],v["failed"],v["not_run"],v["benchmark_status"]),(0,1,2,"incomplete"))
 def test_inflated_pass_count_rejected(self):self.reject(lambda r:r.update(passed=3,failed=0,rows=[dict(case=n,status="failed")for n in self.cases]))
 def test_boolean_count_rejected(self):self.reject(lambda r:r.update(failed=False))
 def test_duplicate_case_rejected(self):self.reject(lambda r:r["rows"].__setitem__(1,r["rows"][0]))
 def test_reordered_case_rejected(self):self.reject(lambda r:r["rows"].reverse())
 def test_complete_cannot_omit_case(self):
  r=self.report(rows=[dict(case="a",status="passed")])
  with self.assertRaises(ValueError):terminal_counts(r,self.calls(),self.cases)
 def test_unknown_http_outcome_rejected(self):
  c=self.calls();c["calls_resolved"]=False
  with self.assertRaises(ValueError):terminal_counts(self.report(),c,self.cases)
 def test_mismatched_call_totals_rejected(self):self.reject(lambda r:r.update(http_attempts=3))
 def test_skip_is_not_passed(self):self.reject(lambda r:r["rows"][0].update(status="skipped"))
 def test_failed_http_is_terminal_but_usage_unknown(self):
  v=receipt_coverage(dict(calls=[dict(index=0,status="response_received",usage=dict(prompt_tokens=2,completion_tokens=3,total_tokens=5)),dict(index=1,status="failed")]))
  self.assertEqual((v["terminal_http_outcomes"],v["terminal_failed_http_attempts"],v["unmeasured_http_attempts"],v["usage_complete"]),(2,1,1,False))
 def test_received_without_usage_is_unknown(self):
  self.assertEqual(receipt_coverage(dict(calls=[dict(index=0,status="response_received")]))["unmeasured_http_attempts"],1)
 def test_empty_usage_is_unknown(self):
  self.assertEqual(receipt_coverage(dict(calls=[dict(index=0,status="response_received",usage={})]))["unmeasured_http_attempts"],1)
 def test_partial_usage_is_unknown(self):
  v=receipt_coverage(dict(calls=[dict(index=0,status="response_received",usage=dict(prompt_tokens=2))]))
  self.assertEqual((v["received_responses_with_partial_usage"],v["unmeasured_http_attempts"],v["usage_complete"]),(1,1,False))
 def test_invalid_usage_is_rejected(self):
  with self.assertRaises(ValueError):receipt_coverage(dict(calls=[dict(index=0,status="response_received",usage=dict(prompt_tokens=True))]))
 def test_inflight_receipt_rejected(self):
  with self.assertRaises(ValueError):receipt_coverage(dict(calls=[dict(index=0,status="in_flight")]))
 def test_receipt_reordered_rejected(self):
  with self.assertRaises(ValueError):receipt_coverage(dict(calls=[dict(index=1,status="failed")]))
if __name__=="__main__":unittest.main()
