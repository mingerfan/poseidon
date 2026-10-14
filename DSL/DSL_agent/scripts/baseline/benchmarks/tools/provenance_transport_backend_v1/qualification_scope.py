"""Frozen developer qualification scope; application execution stays in Application."""
import copy,hashlib,json,os,shutil,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
OFFLINE_CASES=("wide_tail_linear","weights_linear_10k","rank_permute_32")
LIVE_CASES=OFFLINE_CASES
REUSE_CASES=("wide_tail_linear","weights_linear_10k")
def reuse_cases(mode):return tuple(c for c in REUSE_CASES if c in (OFFLINE_CASES if mode=="offline"else LIVE_CASES))
def options_for(case):
 return copy.deepcopy(case["options"])
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def pairs(items):
  out={}
  for k,v in items:require(k not in out,"duplicate_field");out[k]=v
  return out
 return json.loads(Path(p).read_bytes(),object_pairs_hook=pairs,
  parse_constant=lambda _:(_ for _ in()).throw(ValueError("nonfinite_json")))
def write(p,v):
 p=Path(p);require(not p.exists(),"record_exists")
 with p.open("xb")as s:os.chmod(p,0o600);s.write(canonical(v))
def bound(v):require(v.get("binding")==digest({k:x for k,x in v.items()if k!="binding"}),"binding")
def regular(p):
 p=Path(p);require(p.is_absolute()and p==p.resolve()and p.is_file()and not p.is_symlink()
  and (p.is_relative_to(ROOT)or p.is_relative_to(RESULTS)),"regular_project_path");return p
def verify(mode):
 require(mode in ("offline","live"),"mode")
 p=read(HERE/(mode+"-plan.json"));bound(p)
 require(p["mode"]==mode and tuple(c["id"]for c in p["cases"])==(OFFLINE_CASES if mode=="offline"else LIVE_CASES),"case_scope")
 require(tuple(p["package_new_input_cases"])==reuse_cases(mode),"reuse_case_scope")
 require(p["authorization"]=="human_explicit_current_plan_test_approval","authorization")
 require(p["storage_activation"]=="docs/baseline/agent-result-storage-activation-r1442.json","storage_activation")
 a=read(ROOT/p["storage_activation"]);bound(a)
 require(a["approved"]is True and p["storage_activation_binding"]==a["binding"],"storage_approval")
 require(p["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"frozen_tolerance")
 b=p["budget"]
 require(b["global_result_bytes"]==48*1024**3 and b["new_artifact_bytes"]==4*1024**3 and
  b["per_case_bytes"]==4*1024**3 and b["reserve_bytes"]==4*1024**3 and
  b["compiler_jobs"]==2 and b["link_jobs"]==1 and b["encrypted_concurrency"]==b["api_concurrency"]==1,"resource_scope")
 require((b["wall_seconds"],b["case_seconds"],b["max_generations"],b["max_http"],
  b["max_compiler_calls"],b["max_encrypted_calls"])==
  ((1800,180,3,0,3,7)if mode=="offline"else(43200,14400,12,48,12,16)),"call_or_time_scope")
 for rel,want in p["frozen_files"].items():require(sha(regular(ROOT/rel))==want,"frozen_file:"+rel)
 require(read(HERE/"identity-snapshot.json")["manifest_sha256"]==p["sdk"],"sdk_snapshot")
 for c in p["cases"]:
  require(sha(regular(c["request_file"]))==c["request_file_sha256"]and
   read(c["request_file"])["request_id"]==c["request_id"],"frozen_request")
  require(sha(regular(c["public_request_file"]))==c["public_request_file_sha256"],"frozen_public_request")
 return p
def calls(states,max_generations,max_http):
 g=h=0;resolved=True
 for s in states:
  require(type(s["generations"])is int and type(s["http_attempts"])is int and
   0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_calls")
  r=[e for e in s["ledger"]if e.get("event")=="http_reserved"]
  require([e["index"]for e in r]==list(range(s["http_attempts"])),"reservation_ledger")
  ck=[e for e in s["ledger"]if e.get("event")=="provider_checkpoint"]
  outcomes=ck[-1].get("outcomes",[])if ck else []
  resolved=resolved and len(outcomes)==s["http_attempts"]and all(o.get("index")==i and
   o.get("status")in("response_received","failed")for i,o in enumerate(outcomes))
  g+=s["generations"];h+=s["http_attempts"]
 require(g<=max_generations and h<=max_http,"campaign_call_budget")
 return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def states(out):
 return [read(p)for p in Path(out).glob("*/tasks/*/state.json")]
_last_resource_check = 0.
def boundary(p,out,owner,case_deadline=None,*,force=False):
 global _last_resource_check
 import run as live
 require(time.time()<min(owner["deadline"],case_deadline or owner["deadline"]),"wall_budget")
 info=live.process_info(owner["pid"])
 require(info is not None and str(info["start"])==owner["start"]and info["state"]not in("Z","X"),"controller_identity")
 if force or time.monotonic()-_last_resource_check >= 10:
  require(live.measure(out)<=p["budget"]["new_artifact_bytes"]and
   live.measure(RESULTS)<=p["budget"]["global_result_bytes"],"artifact_budget")
  _last_resource_check=time.monotonic()
 require(shutil.disk_usage(RESULTS).free>=p["budget"]["reserve_bytes"],"disk_reserve")
 return calls(states(out),p["budget"]["max_generations"],p["budget"]["max_http"])
