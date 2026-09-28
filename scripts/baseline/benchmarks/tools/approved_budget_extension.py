"""Exact user-authorized extension of six exhausted shard clocks, no call retries."""
from pathlib import Path
from stage2_agent_campaign import ROOT
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_runner import strict_file
from workspace_paths import RESULTS
from semantic_benchmark_execution import runtime_sources

def checked_document(path,expected):
 p=Path(path)
 if p.is_symlink() or not any(p.resolve().is_relative_to(b.resolve()) for b in (ROOT,RESULTS)):
  raise ValueError("Unsafe authorization path")
 if sha(p)!=expected:raise ValueError("Authorization evidence changed")
 d=strict_file(p,8*1024**2);check_binding(d);return d

def check_selection(cases,prepared,spent,limit):
 if cases!=prepared["cases"]:raise ValueError("Cases or requests outside explicit authorization")
 if spent!=prepared["limits"]["original_seconds_used"]:raise ValueError("Original clock reset")
 if limit!=prepared["limits"]["cumulative_wall_limit"] or limit!=7200:raise ValueError("Unauthorized cumulative limit")
 if not 3600<=spent<limit:raise ValueError("Exhausted original clock required")
 if prepared["limits"]["remaining_seconds"]!=limit-spent:raise ValueError("Remaining clock changed")

def verify_extension(plan):
 ref=plan["budget_extension"]
 prepared=checked_document(ref["preparation"],ref["preparation_sha256"])
 if prepared["format"]!="poseidon-approved-budget-supplement-preparation-v1" or prepared["executable"] is not False:
  raise ValueError("Expected non-executable approved preparation")
 auth=checked_document(prepared["authorization"],prepared["authorization_sha256"])
 proposal=checked_document(auth["proposal"],auth["proposal_sha256"])
 if auth["proposal_binding"]!=proposal["binding"] or auth["user_answer"]!="批准上述有限追加":
  raise ValueError("Explicit approval missing")
 if (auth["authorized_cases"]!=47 or auth["authorized_original_shards"]!=["002","003","007","009","010","011"]
     or auth["cumulative_wall_limit_per_shard"]!=7200 or auth["maximum_added_wall_budget_seconds"]!=21600
     or auth["maximum_new_generations"]!=188 or auth["maximum_new_http_attempts"]!=752
     or auth["retry_failed_or_uncertain_calls"] is not False):
  raise ValueError("Approval scope changed")
 original=checked_document(prepared["original_plan"],prepared["original_plan_sha256"])
 report=checked_document(prepared["original_report"],prepared["original_report_sha256"])
 item=next((s for s in proposal["shards"] if s["shard"]==prepared["shard"]),None)
 if item is None:raise ValueError("Unapproved shard")
 batch=Path(item["original_batch"])
 if (str(batch/"plan.json")!=prepared["original_plan"] or str(batch/"report.json")!=prepared["original_report"]
     or original["binding"]!=item["original_plan_binding"] or report["plan_binding"]!=original["binding"]
     or report["failure"]!="TimeoutError: Cumulative wall budget exhausted"):
  raise ValueError("Original exhausted batch mismatch")
 ids={t["id"] for t in item["tasks"]}
 expected=[c for c in original["cases"] if c["id"] in ids]
 if len(expected)!=len(ids) or prepared["cases"]!=expected:raise ValueError("Prepared subset differs from approved original tasks")
 done={r["id"] for r in report["rows"]}
 if ids&done or any((batch/i/"launch.json").exists() for i in ids):raise ValueError("Previously attempted case cannot be retried")
 if report["seconds"]!=ref["seconds_already_used"]:raise ValueError("Reported elapsed time reset")
 check_selection(plan["cases"],prepared,ref["seconds_already_used"],plan["limits"]["max_wall_seconds"])
 n=len(expected)
 if (plan["limits"]["maximum_generations"]!=4*n or plan["limits"]["maximum_http_attempts"]!=16*n
     or prepared["limits"]["maximum_generations"]!=4*n or prepared["limits"]["maximum_http_attempts"]!=16*n):
  raise ValueError("Subset API budget changed")
 if plan["source_hashes"]!=original["source_hashes"] or runtime_sources()!=original["source_hashes"]:
  raise ValueError("Runtime drift")
 if plan["paid_configuration"]!=original["paid_configuration"]:raise ValueError("Paid configuration changed")
 if plan["inventory_binding"]!=original["inventory_binding"] or plan["definition_parents"]!=original["definition_parents"]:
  raise ValueError("Definition drift")
 return prepared

def retained_paths(prepared):
 batch=Path(prepared["original_plan"]).parent
 report=strict_file(Path(prepared["original_report"]),8*1024**2)
 paths=[batch]+[Path(r["evidence"]) for r in report["rows"] if "evidence" in r]
 from run_stage2_guidance_retest import evidence_path
 for log in batch.glob("*/run.log"):
  p=evidence_path(log,RESULTS)
  if p:paths.append(p)
 return list(dict.fromkeys(paths))
