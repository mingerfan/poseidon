"""Second continuation: preserve r131 execution clocks and all earlier evidence."""
from pathlib import Path
from stage2_agent_campaign import ROOT,LIMITS
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file
from campaign_live_state import sealed
from approved_budget_extension import checked_document
from workspace_paths import RESULTS
import remaining_campaign_policy_v2 as prior_policy
from remaining_campaign_policy_v2 import unclaimed,assert_idle

INDEX_BINDING="34e64c0917c8de6e811c21f6d201a3675d3bae56b8069a76ae5dc12d07fa23bc"
FILES=prior_policy.FILES+("remaining_campaign_policy_v3.py","run_stage2_remaining_campaign_v3.py","remaining_campaign_queue_v3.py")

def verify_authorizations(d):
 time=checked_document(d["time_authorization"],d["time_authorization_sha256"])
 space=checked_document(d["space_authorization"],d["space_authorization_sha256"])
 if time["binding"]!="983a5a773ed754e4072fbcec740c7c2b2d8807862a6d4ca1216c4e9db58ac1f7":
  raise ValueError("Unapproved time expansion")
 if space["binding"]!="486c2a2ba1fb59fd65b107efc89f8a1b5a12a9105f449c4736dbd40fd7525102":
  raise ValueError("Unapproved shard space expansion")
 if (time["selected_per_original_shard_cumulative_seconds"]!=43200 or time["selected_campaign_cumulative_seconds"]!=259200
     or space["per_shard_retained_mib"]!=4096 or space["total_retained_mib"]!=32768):
  raise ValueError("Authorization scope changed")

def require_headroom(current_bytes,limit_mib,reserve_mib=1024):
 if current_bytes+reserve_mib*1024**2>limit_mib*1024**2:
  raise ValueError("Pre-dispatch artifact headroom inadequate")

def index_document(path,expected=None):
 p=Path(path);d=checked_document(p,expected or sha(p))
 if d["binding"]!=INDEX_BINDING:raise ValueError("Unapproved continuation index")
 verify_authorizations(d)
 prior_policy.index_document(d["base_index"],d["base_index_sha256"])
 for name,h in d["parents"].items():checked_document(name,h)
 used=sum(checked_document(r["path"],r["sha256"])["seconds"] for r in d["prior_queue_reports"])
 if d["aggregate_seconds_already_used"]!=used or d["original_total_queue_seconds"]!=259200 or d["aggregate_remaining_seconds"]!=259200-used:
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
 verify_authorizations(d)
 if d["clock"]!={"seconds_already_used":spent,"cumulative_limit_seconds":43200,"remaining_seconds":43200-spent} or not 0<=spent<43200:
  raise ValueError("Authorized cumulative clock changed")
 return d,original

def build_value(prep,original,prep_path,prep_hash):
 value=dict(original);value.pop("binding")
 n=len(prep["cases"]);value["cases"]=prep["cases"]
 value["limits"]=dict(LIMITS,max_retained_mib=4096,max_wall_seconds=prep["clock"]["cumulative_limit_seconds"],maximum_generations=4*n,maximum_http_attempts=16*n)
 value["cumulative_remainder"]=dict(preparation=str(prep_path),preparation_sha256=prep_hash,seconds_already_used=prep["clock"]["seconds_already_used"])
 value["proposal_files"]=dict(original["proposal_files"])
 for name in FILES:
  p=Path(__file__).with_name(name);value["proposal_files"][str(p.relative_to(ROOT))]=sha(p)
 return sealed(value)

def build_shard(prep_path,prep_hash):
 prep,original=prepared_document(prep_path,prep_hash)
 return build_value(prep,original,prep_path,prep_hash)

def verify_remainder(plan):
 ref=plan["cumulative_remainder"]
 prep,original=prepared_document(ref["preparation"],ref["preparation_sha256"])
 if build_value(prep,original,ref["preparation"],ref["preparation_sha256"])!=plan:raise ValueError("Continuation plan drift")
 return prep

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
