"""Second continuation: preserve r131 execution clocks and all earlier evidence."""
from pathlib import Path
from stage2_agent_campaign import ROOT,LIMITS
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file
from campaign_live_state import sealed
from approved_budget_extension import checked_document
from workspace_paths import RESULTS
import remaining_campaign_policy as prior_policy
from remaining_campaign_policy import check_clock,unclaimed,assert_idle

INDEX_BINDING="5a463b4e368fd5985ba0f486bad0507e21b58402cab2fa6f7c37d4ef68653264"
FILES=prior_policy.FILES+("remaining_campaign_policy_v2.py","run_stage2_remaining_campaign_v2.py","remaining_campaign_queue_v2.py")

def index_document(path,expected=None):
 p=Path(path);d=checked_document(p,expected or sha(p))
 if d["binding"]!=INDEX_BINDING:raise ValueError("Unapproved continuation index")
 prior_policy.index_document(d["base_index"],d["base_index_sha256"])
 for name,h in d["parents"].items():checked_document(name,h)
 used=sum(checked_document(r["path"],r["sha256"])["seconds"] for r in d["prior_queue_reports"])
 if d["aggregate_seconds_already_used"]!=used or d["original_total_queue_seconds"]!=163590 or d["aggregate_remaining_seconds"]!=163590-used:
  raise ValueError("Aggregate clock changed")
 if d["maximum_continuation_seconds_including_audits"]>d["aggregate_remaining_seconds"]:raise ValueError("Time budget")
 return d

def prepared_document(path,hsh):
 p=Path(path);d=checked_document(p,hsh);index=index_document(p.parent/"index.json")
 row=next((r for r in index["shards"] if r["original_shard"]==d["original_shard"]),None)
 if row is None or row["file"]!=p.name or row["sha256"]!=hsh or row["binding"]!=d["binding"]:
  raise ValueError("Preparation outside frozen continuation")
 base,original=prior_policy.prepared_document(d["base_preparation"],d["base_preparation_sha256"])
 if base["original_shard"]!=d["original_shard"] or d["source_hashes"]!=base["source_hashes"]:
  raise ValueError("Original identity changed")
 ids={c["id"] for c in d["cases"]}
 if len(ids)!=len(d["cases"]) or [c for c in base["cases"] if c["id"] in ids]!=d["cases"]:
  raise ValueError("Continuation task/request changed")
 spent=base["clock"]["seconds_already_used"]
 if d["prior"]:
  ref=d["prior"];plan=checked_document(ref["plan"],ref["plan_sha256"])
  if plan!=prior_policy.build_shard(Path(d["base_preparation"]),d["base_preparation_sha256"]):
   raise ValueError("Previous plan changed")
  report=checked_document(ref["report"],ref["report_sha256"])
  if report["plan_binding"]!=plan["binding"] or report["seconds"]<spent:raise ValueError("Previous clock reset")
  if ids&{r["id"] for r in report["rows"]}:raise ValueError("Attempted case cannot be retried")
  batch=Path(ref["plan"]).parent
  if any((batch/i/"launch.json").exists() or (batch/i/"launch.json").is_symlink() for i in ids):
   raise ValueError("Uncertain launch cannot be retried")
  spent=report["seconds"]
 check_clock(d["clock"],spent,base["clock"]["cumulative_limit_seconds"])
 if (d["space_authorization"]!=base["space_authorization"] or d["space_authorization_sha256"]!=base["space_authorization_sha256"]):
  raise ValueError("Space authorization changed")
 return d,original

def build_shard(prep_path,prep_hash):
 prep,original=prepared_document(prep_path,prep_hash)
 value=dict(original);value.pop("binding")
 n=len(prep["cases"]);value["cases"]=prep["cases"]
 value["limits"]=dict(LIMITS,max_wall_seconds=prep["clock"]["cumulative_limit_seconds"],maximum_generations=4*n,maximum_http_attempts=16*n)
 value["cumulative_remainder"]=dict(preparation=str(prep_path),preparation_sha256=prep_hash,seconds_already_used=prep["clock"]["seconds_already_used"])
 value["proposal_files"]=dict(original["proposal_files"])
 for name in FILES:
  p=Path(__file__).with_name(name);value["proposal_files"][str(p.relative_to(ROOT))]=sha(p)
 return sealed(value)

def verify_remainder(plan):
 ref=plan["cumulative_remainder"]
 if build_shard(Path(ref["preparation"]),ref["preparation_sha256"])!=plan:raise ValueError("Continuation plan drift")
 return prepared_document(ref["preparation"],ref["preparation_sha256"])[0]

def retained_paths(prepared):
 base,_=prior_policy.prepared_document(prepared["base_preparation"],prepared["base_preparation_sha256"])
 paths=set(prior_policy.retained_paths(base))
 if prepared["prior"]:
  from run_stage2_guidance_retest import evidence_path
  ref=prepared["prior"];batch=Path(ref["plan"]).parent
  report=checked_document(ref["report"],ref["report_sha256"]);paths.add(batch)
  paths.update(Path(r["evidence"]) for r in report["rows"] if "evidence" in r)
  for log in batch.glob("*/run.log"):
   p=evidence_path(log,RESULTS)
   if p:paths.add(p)
 return sorted(paths)

def cumulative_elapsed(prior,started,now):
 if now<started:raise ValueError("Monotonic clock went backwards")
 return prior+now-started
