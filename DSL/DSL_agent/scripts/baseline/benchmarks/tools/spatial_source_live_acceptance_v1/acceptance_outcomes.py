"""Terminal accounting for acceptance audits; no provider or backend imports."""
def require(ok,code):
 if not ok:raise ValueError(code)
def terminal_counts(report,calls,cases):
 rows=report["rows"]
 require(report["status"]in("complete","stopped")and calls["calls_resolved"]is True and
  report["calls_resolved"]is True,"audit_terminal_ledger")
 require(type(report["planned"])is int and report["planned"]==len(cases),"audit_case_denominator")
 names=[r["case"]for r in rows]
 require(names==list(cases[:len(names)])and len(names)<=len(cases)and
  len(names)==len(set(names)),"audit_case_denominator")
 require(all(r.get("status")in("passed","failed")for r in rows),"audit_terminal_case_status")
 passed=sum(r["status"]=="passed"for r in rows);failed=len(rows)-passed;not_run=len(cases)-len(rows)
 for key,value in (("passed",passed),("failed",failed),("not_run",not_run),
                   ("generations",calls["generations"]),("http_attempts",calls["http_attempts"])):
  require(type(report[key])is int and report[key]==value,"audit_terminal_count")
 require(report["status"]!="complete"or not_run==0,"audit_complete_missing_case")
 return dict(planned=len(cases),passed=passed,failed=failed,not_run=not_run,
  benchmark_status="passed"if passed==len(cases)else"incomplete"if not_run else"completed_with_failures")
def receipt_coverage(metrics):
 calls=metrics["calls"]
 require(type(calls)is list and len(calls)<=16 and
  [c["index"]for c in calls]==list(range(len(calls))),"audit_http_receipt_order")
 require(all(c["status"]in("response_received","failed")for c in calls),"audit_http_receipt_outcome")
 received=[c for c in calls if c["status"]=="response_received"]
 for c in received:
  if "usage"in c:require(type(c["usage"])is dict and
   all(type(v)is int and v>=0 for v in c["usage"].values()),"audit_usage_fields")
 without=sum(not c.get("usage")for c in received);failed=len(calls)-len(received)
 partial=sum(bool(c.get("usage"))and not{"prompt_tokens","completion_tokens","total_tokens"}.issubset(c["usage"])for c in received)
 return dict(terminal_http_outcomes=len(calls),received_responses=len(received),
  terminal_failed_http_attempts=failed,received_responses_without_usage=without,
  received_responses_with_partial_usage=partial,unmeasured_http_attempts=failed+without+partial,
  usage_complete=failed==without==partial==0)
