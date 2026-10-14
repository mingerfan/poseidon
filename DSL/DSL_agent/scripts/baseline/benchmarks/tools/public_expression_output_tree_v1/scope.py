"""One unchanged failing source case, new execution binding, API disabled."""
import hashlib,json,time,subprocess,shutil
from pathlib import Path
from benchmark_graph import require,digest
from hecate_python_env import WORK
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OUT=WORK/"results/agent-public-expression-output-tree-v23-v1"
PROPOSAL=ROOT/"docs/baseline/agent-public-expression-output-offline-proposal-r1372.json"
AUTH=ROOT/"docs/baseline/agent-public-expression-output-offline-authorization-r1372.json"
ORIGINAL=ROOT/"docs/baseline/agent-public-expression-offline-proposal-r1369.json"
SCOPE=dict(wall_seconds=240,case_seconds=180,new_bytes=128*1024**2,global_bytes=32*1024**3,
 reserve_bytes=4*1024**3,compiler_jobs=2,link_jobs=1,native_concurrency=1,
 native_address_bytes=4*1024**3,native_cpu_seconds=150,provider_calls=0,http_attempts=0,
 maximum_compiler_calls=1,maximum_encrypted_calls=3)
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",
 public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open("x")as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def measure(p):
 return max(int(subprocess.check_output(["du",flag,"-s","-B1",str(p)],text=True,timeout=20).split()[0])
  for flag in ("--apparent-size","--count-links"))
def current():
 from application_identity import current_identity_manifest
 return current_identity_manifest()
def verify(plan):
 require(type(plan)is dict and plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"}),"output_requalification_binding")
 require(plan["format"]=="poseidon-public-expression-output-offline-proposal-v1"and
   plan["scope"]==SCOPE and plan["options"]==OPTIONS and plan["case"]=="named"and
   plan["source_contract"]=="poseidon-trusted-pytorch-v23"and plan["backend"]=="poseidon_cpu"and
   plan["candidate_origin"]=="unchanged_previous_rule"and
   plan["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"output_exact_scope")
 require(current()["manifest_sha256"]==plan["sdk_identity"],"output_new_sdk_binding")
 for p,h in plan["tools"].items():require(sha(ROOT/p)==h,"output_tool_binding")
 for p,h in plan["files"].items():require(sha(p)==h,"output_frozen_file_binding")
 old=read(ORIGINAL)
 require(old["binding"]==digest({k:v for k,v in old.items()if k!="binding"})==
  plan["original_proposal_binding"]=="faa04c4a585dfca471c8eca23484a149002c6d24d7d2b6bfef21c416afb854c1","output_original_scope")
 item=next(x for x in old["cases"]if x["case"]=="named")
 require(read(plan["request"])==read(item["request"])and
  Path(plan["candidate"]).read_bytes()==Path(item["candidate"]).read_bytes(),"output_original_model_and_source_unchanged")
 return plan
def authorization(plan):
 receipt=read(AUTH)
 require(receipt.get("format")=="poseidon-public-expression-output-offline-authorization-v1"and
  receipt.get("proposal_binding")==plan["binding"]and
  receipt.get("human_reply")=="批准这1项离线补验","output_exact_human_authorization")
 return receipt
def boundary(plan,deadline,baseline):
 verify(plan)
 require(time.time()<deadline,"output_deadline")
 total=measure(WORK/"results")
 require(total-baseline<=SCOPE["new_bytes"]and total<=SCOPE["global_bytes"]and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"output_total_artifact_budget")
