"""Frozen remaining tasks: preserve original clocks, claims and artifact accounting."""
from pathlib import Path
from stage2_agent_campaign import ROOT, LIMITS
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file
from campaign_live_state import sealed
from approved_budget_extension import checked_document
from workspace_paths import RESULTS

INDEX_BINDING="49f568ae59e46208fd95d44d1b2b1aadda4f2dde447d7ef3abf1626cdbfefd9d"
SPACE_BINDING="e0d92a5ce3c54aef86e86d0ea0bf13e7c5120d3688a2e3f1f38af40ced9189b8"
FILES=("remaining_campaign_policy.py","run_stage2_remaining_campaign.py","remaining_campaign_queue.py","retained_artifact_usage.py")

def index_document(path,expected=None):
 p=Path(path)
 d=checked_document(p,expected or sha(p))
 if d["binding"]!=INDEX_BINDING:raise ValueError("Unapproved remainder index")
 for name,h in d["parents"].items():checked_document(name,h)
 used=sum(checked_document(r["path"],r["sha256"])["seconds"] for r in d["prior_queue_reports"])
 if (used!=d["aggregate_seconds_already_used"] or d["original_total_queue_seconds"]!=163590
     or d["aggregate_remaining_seconds"]!=163590-used):
  raise ValueError("Aggregate elapsed clock changed")
 if d["maximum_continuation_seconds_including_audits"]>d["aggregate_remaining_seconds"]:
  raise ValueError("Aggregate time ceiling")
 return d

def check_clock(clock,spent,ceiling):
 if clock!={"seconds_already_used":spent,"cumulative_limit_seconds":ceiling,"remaining_seconds":ceiling-spent}:
  raise ValueError("Original cumulative clock changed")
 if not 0<=spent<ceiling or ceiling not in (3600,7200):raise ValueError("Clock outside authorization")

def prepared_document(path,hsh):
 p=Path(path);d=checked_document(p,hsh)
 idx=index_document(p.parent/"index.json")
 row=next((r for r in idx["shards"] if r["original_shard"]==d["original_shard"]),None)
 if row is None or row["file"]!=p.name or row["sha256"]!=hsh or row["binding"]!=d["binding"]:
  raise ValueError("Preparation outside frozen index")
 original=checked_document(d["original_plan"],d["original_plan_sha256"])
 if original["binding"]!=d["original_binding"] or original["shard_index"]!=d["original_shard"]:
  raise ValueError("Original shard identity")
 selected={s["id"] for s in d["cases"]}
 if [s for s in original["cases"] if s["id"] in selected]!=d["cases"] or len(selected)!=len(d["cases"]):
  raise ValueError("Original request/case mismatch")
 prior=d["prior"];spent=0
 if prior:
  old=checked_document(prior["plan"],prior["plan_sha256"])
  report=checked_document(prior["report"],prior["report_sha256"])
  if old["cases"]!=original["cases"] or report["plan_binding"]!=old["binding"]:
   raise ValueError("Prior batch identity")
  if selected & {r["id"] for r in report["rows"]}:raise ValueError("Previously completed task")
  batch=Path(prior["plan"]).parent
  if any((batch/i/"launch.json").exists() or (batch/i/"launch.json").is_symlink() for i in selected):
   raise ValueError("Previously launched task")
  spent=report["seconds"]
 ceiling=3600
 if d["time_extension"]:
  ext=d["time_extension"];auth=checked_document(ext["authorization"],ext["sha256"])
  if auth["binding"]!="212a92205435856145876c355edcf1cb1ee38a09763cbadb0692afe881685193":
   raise ValueError("Unapproved time extension")
  proposal=checked_document(auth["proposal"],auth["proposal_sha256"])
  item=next((r for r in proposal["shards"] if r["shard"]=="%03d"%d["original_shard"]),None)
  if item is None or {r["id"] for r in item["tasks"]}!=selected or not prior:
   raise ValueError("Time extension task set")
  if str(Path(prior["plan"]).parent)!=item["original_batch"] or spent<3600:
   raise ValueError("Time extension original batch")
  ceiling=7200
 check_clock(d["clock"],spent,ceiling)
 space=checked_document(d["space_authorization"],d["space_authorization_sha256"])
 if space["binding"]!=SPACE_BINDING or space["selected_total_limit_mib"]!=32768 or space["min_free_disk_mib"]!=4096:
  raise ValueError("Unapproved space policy")
 return d,original

def build_shard(prep_path,prep_hash):
 prep,original=prepared_document(prep_path,prep_hash)
 value=dict(original);value.pop("binding")
 n=len(prep["cases"]);value["cases"]=prep["cases"]
 value["limits"]=dict(LIMITS,max_wall_seconds=prep["clock"]["cumulative_limit_seconds"],
  maximum_generations=4*n,maximum_http_attempts=16*n)
 value["cumulative_remainder"]={"preparation":str(prep_path),"preparation_sha256":prep_hash,
  "seconds_already_used":prep["clock"]["seconds_already_used"]}
 value["proposal_files"]=dict(original["proposal_files"])
 for name in FILES:
  p=Path(__file__).with_name(name);value["proposal_files"][str(p.relative_to(ROOT))]=sha(p)
 return sealed(value)

def verify_remainder(plan):
 ref=plan["cumulative_remainder"]
 expected=build_shard(Path(ref["preparation"]),ref["preparation_sha256"])
 if expected!=plan:raise ValueError("Cumulative continuation plan changed")
 return prepared_document(ref["preparation"],ref["preparation_sha256"])[0]

def unclaimed(cases):
 for spec in cases:
  p=RESULTS/"stage2-agent-evaluation-claims"/(spec["evaluation_identity"]+".json")
  if p.exists() or p.is_symlink():raise ValueError("Previously claimed task; no implicit retry: "+spec["id"])

def retained_paths(prepared):
 if not prepared["prior"]:return []
 from run_stage2_guidance_retest import evidence_path
 prior=prepared["prior"];batch=Path(prior["plan"]).parent
 report=checked_document(prior["report"],prior["report_sha256"])
 paths={batch}|{Path(r["evidence"]) for r in report["rows"] if "evidence" in r}
 for log in batch.glob("*/run.log"):
  p=evidence_path(log,RESULTS)
  if p:paths.add(p)
 return sorted(paths)

def assert_idle():
 names={b"run_candidate.py",b"parallel_campaign_queue_v3.py",b"parallel_campaign_queue_v2.py",
        b"budget_supplement_queue.py",b"budget_supplement_queue_v2.py",b"run_stage2_remaining_campaign.py"}
 for p in Path("/proc").glob("[0-9]*"):
  try:args=(p/"cmdline").read_bytes().split(bytes([0]))
  except (FileNotFoundError,ProcessLookupError):continue
  if any(a.rsplit(b"/",1)[-1] in names for a in args):raise ValueError("Existing paid worker; do not duplicate")
