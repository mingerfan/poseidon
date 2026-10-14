"""Frozen offline source-intake acceptance scope; provider access is disabled."""
import hashlib,json,subprocess,shutil,time
from pathlib import Path
from benchmark_graph import require,digest
from hecate_python_env import WORK
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OUT=WORK/"results/agent-public-dtype-cast-cpu-v24-v1"
DRAFT=ROOT/"docs/baseline/agent-public-dtype-offline-proposal-r1375.json"
PROPOSAL=ROOT/"docs/baseline/agent-public-dtype-offline-proposal-r1376.json"
AUTH=ROOT/"docs/baseline/agent-public-dtype-offline-authorization-r1376.json"
SCOPE=dict(wall_seconds=1200,case_seconds=180,new_bytes=512*1024**2,global_bytes=32*1024**3,
 reserve_bytes=4*1024**3,compiler_jobs=2,link_jobs=1,native_concurrency=1,
 native_address_bytes=4*1024**3,native_cpu_seconds=150,provider_calls=0,http_attempts=0,
 maximum_compiler_calls=36,maximum_encrypted_calls=42)
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",
 public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
REUSE=["f64_to_f32_c0","f64_roundtrip_c2","compat_v23_named"]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open("x")as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def measure(p):
 return max(int(subprocess.check_output(["du",flag,"-s","-B1",str(p)],text=True,timeout=20).split()[0])
  for flag in("--apparent-size","--count-links"))
def current():
 from application_identity import current_identity_manifest
 return current_identity_manifest()
def verify(plan):
 require(type(plan)is dict and plan.get("binding")==digest({k:v for k,v in plan.items()if k!="binding"}),"dtype_proposal_binding")
 require(plan["format"]=="poseidon-public-dtype-offline-ready-v1"and plan["scope"]==SCOPE and
  plan["options"]==OPTIONS and plan["reuse_cases"]==REUSE and len(plan["cases"])==36 and
  plan["unique_requests"]==12 and plan["backend"]=="poseidon_cpu"and
  plan["candidate_origin"]=="deterministic_rule"and plan["approved"]is False and
  plan["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"dtype_exact_scope")
 old=read(DRAFT)
 require(old["binding"]==plan["prepared_draft_binding"]==digest({k:v for k,v in old.items()if k!="binding"})and
  old["cases"]==plan["cases"],"dtype_prepared_scope_unchanged")
 require(len({x["case"]for x in plan["cases"]})==36 and
  len({x["request_id"]for x in plan["cases"]})==12,"dtype_exact_cases")
 sdk=current()
 require(sdk["manifest_sha256"]==plan["sdk_identity"]and
  {k:sdk[k]["sha256"]for k in("generation","compilation","execution","validation")}==plan["stage_identities"],
  "dtype_current_identity")
 for p,h in plan["tools"].items():require(sha(ROOT/p)==h,"dtype_tool_binding")
 for item in plan["cases"]:
  for p,h in item["files"].items():require(sha(p)==h,"dtype_frozen_case_file")
  require(sha(Path(item["task_store"])/item["task_id"]/"task.json")==plan["task_metadata"][item["case"]],"dtype_task_binding")
 return plan
def authorization(plan):
 receipt=read(AUTH)
 require(receipt.get("format")=="poseidon-public-dtype-offline-authorization-v1"and
  receipt.get("proposal_binding")==plan["binding"]and
  receipt.get("human_reply")=="批准这36项离线验收","dtype_human_authorization")
 return receipt
def boundary(plan,launch):
 verify(plan)
 require(time.time()<launch["deadline"],"dtype_wall_budget")
 total=measure(WORK/"results")
 require(total-launch["baseline_results_bytes"]<=SCOPE["new_bytes"]and
  total<=SCOPE["global_bytes"]and shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"dtype_disk_budget")
 return dict(results_bytes=total,new_bytes=total-launch["baseline_results_bytes"],
             free_bytes=shutil.disk_usage(WORK).free)
