
"""Read-only audit of the approved named-output qualification."""
import ast,hashlib,json,math,os,struct,subprocess,sys,time,shutil
from pathlib import Path
os.environ.update(POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(HERE));sys.path.insert(1,str(ROOT/"scripts/baseline"))
from scope import read,sha,verify,PROPOSAL,AUTH,OUT,WORK,measure
from benchmark_graph import digest
plan=verify(read(PROPOSAL));raw=read(OUT/"report.json");launch=read(OUT/"launch.json")
assert raw["binding"]==digest({k:v for k,v in raw.items()if k!="binding"})
assert raw["status"]=="passed"and raw["compiler_calls"]==1 and raw["encrypted_calls"]==3
assert not raw["agent_generated"]and raw["provider_calls"]==raw["http_attempts"]==0
assert raw["seconds"]<=plan["scope"]["wall_seconds"]
auth=read(AUTH);assert auth["binding"]==digest({k:v for k,v in auth.items()if k!="binding"})
assert auth["proposal_binding"]==plan["binding"]and auth["question_call_id"]=="call_39be14decd274eb482c02bd90f904422"
ps=subprocess.check_output(["ps","-eo","pid,ppid,comm,args"],text=True,timeout=10)
assert not any("public_expression_output_tree_v1/run.py"in l and"python3 -B -"not in l for l in ps.splitlines())
td=Path(plan["task_store"])/plan["task_id"];task=read(td/"task.json");folder=Path(raw["package"])
manifest=read(folder/"manifest.json")
assert manifest["binding"]==raw["package_binding"]
assert manifest["agent_generated"]is False and manifest["validation"]["level"]=="numerical"
assert manifest["validation"]["encrypted_execution"]and manifest["validation"]["numerically_validated"]
for name,h in manifest["files"].items():assert sha(folder/name)==h,(name,"package integrity")
assert sha(folder/"candidate.py")==sha(plan["candidate"])and read(folder/"request.json")==read(plan["request"])
identity=read(folder/"artifacts/application-identities.json")
assert identity["manifest_sha256"]==plan["sdk_identity"]
assert {k:identity[k]["sha256"] for k in plan["stage_identities"]}==plan["stage_identities"]
old=read(ROOT/"docs/baseline/agent-public-expression-offline-results-r1370.json")
assert old["binding"]=="43f485c321dce9572e9d4abbd5b61b5122d9fb40af282f5574232b18d174145f"
assert old["complete_passed"]==17 and old["complete_failed"]==1
for name,h in old["evidence_hashes"].items():assert sha(name)==h,(name,"historical integrity")
repair=read(ROOT/"docs/baseline/agent-public-expression-output-repair-r1371.json")
for name,h in repair["evidence_hashes"].items():assert sha(name)==h,(name,"repair integrity")
orig=task["model_provenance"]["original"]
assert [x["op"]for x in orig["nodes"]]==["reshape","reshape","multiply","reshape","reshape"]
def flatten(v):
 if isinstance(v,list):return [z for w in v for z in flatten(w)]
 assert type(v)in(int,float)and math.isfinite(v);return[float(v)]
coeff=flatten(orig["constants"]["public_6"]);divisors=flatten(orig["constants"]["public_4"])
assert len(coeff)==len(divisors)==6 and coeff==[1.0/x for x in divisors]
def npy(path):
 data=Path(path).read_bytes();assert data[:6]==b"\x93NUMPY"
 version=data[6];n=2 if version==1 else 4
 size=struct.unpack("<H"if n==2 else"<I",data[8:8+n])[0]
 header=ast.literal_eval(data[8+n:8+n+size].decode("latin1"))
 assert header["descr"]=="<f8"and header["fortran_order"]is False
 values=list(struct.unpack("<"+"d"*(math.prod(header["shape"])),data[8+n+size:]))
 assert len(header["shape"])==2 and header["shape"][1]==12
 return[values[i:i+12]for i in range(0,len(values),12)]
validation=read(td/"validation-0.json")
assert validation["status"]=="passed"and all(validation["observed_stages"].values())
assert validation["encrypted_execution"]and validation["numerically_validated"]and validation["paid_calls"]==0
qout=Path(validation["evidence"])/"attempt-00/output"
paths=[("qualification",qout,None,4),("new_inputs",Path(read(OUT/"new-inputs.json")["evidence"])/"output",read(OUT/"new-inputs.json"),3),("repeat",Path(read(OUT/"repeat.json")["evidence"])/"output",read(OUT/"repeat.json"),1)]
evidence={}
def record(p):evidence[str(p)]=sha(p)
for name in ["report.json","launch.json","qualification.json","new-inputs.json","repeat.json","new-inputs-comparison.json","repeat-comparison.json"]:record(OUT/name)
for p in [PROPOSAL,AUTH,td/"task.json",td/"validation-0.json",folder/"manifest.json"]:record(p)
rows=[];all_errors=[];inputs_by_run={}
for name,p,returned,count in paths:
 meta=read(p/"execution-metadata.json")
 assert meta["backend"]=="poseidon_cpu"and meta["encrypted_execution"]and not meta["executor_secret_key"]and not meta["bootstrap"]
 assert meta["degree"]==32768 and meta["data_moduli"]==13 and meta["security"]=="tc128"
 precheck=read(p/"resource-precheck.json");gate=precheck["artifact_gate"]
 assert meta["q"]==gate["q"] and meta["p"]==gate["p"] and meta["security"]==gate["security"]
 assert precheck["backend_identity"]==manifest["dependencies"]["backend_identity"]
 assert gate["plan_sha256"]=="sha256:"+sha(p/"bound/plan.json")
 assert gate["spec_sha256"]=="sha256:"+sha(folder/"artifacts/operator-spec.json")
 assert sha(p/"bound/plan.json")==manifest["files"]["artifacts/bound/plan.json"]
 packets=read(p/"inputs.json");actual=npy(p/"decrypted.npy")
 assert len(packets)==len(actual)==count
 logical=[[z for x in b[:2]for z in x][:6]for b in packets]
 inputs_by_run[name]=logical
 expected=[[x*c for x,c in zip(b,coeff)]+b for b in logical]
 err=[abs(a-r)for av,rv in zip(actual,expected)for a,r in zip(av,rv)]
 assert all(abs(a-r)<=1e-5+1e-4*abs(r)for av,rv in zip(actual,expected)for a,r in zip(av,rv))
 if returned is not None:
  assert returned["status"]=="executed"and returned["encrypted_execution"]and not returned["executor_secret_key"]
  assert returned["compiler_calls"]==returned["agent_calls"]==0
  for i,s in enumerate(returned["structured_outputs"]):
   assert list(s)==["primary","saved"]and type(s["saved"])is list and len(s["saved"])==2 and s["saved"][1]is None
   assert s["primary"]==returned["outputs"][i]["y"]and s["saved"][0]==returned["outputs"][i]["saved"]
   assert flatten(s["primary"])+flatten(s["saved"][0])==actual[i]
 for f in ["execution-metadata.json","inputs.json","decrypted.npy","candidate_trace.mlir","lowered.earth.mlir","lowered.ckks.mlir","lowered._hecate_golden.runtime-plan.json","bound/plan.json","bound/layout-binding.json","bound/bundle/manifest.json"]:
  record(p/f)
 for p2 in(p/"bound/bundle").rglob("*.bin"):record(p2)
 rows.append(dict(stage=name,groups=count,values=count*12,maximum_absolute_error=max(err),mae=sum(err)/len(err),passed=True))
 all_errors+=err
assert inputs_by_run["repeat"]==inputs_by_run["new_inputs"][:1]
assert sum(r["groups"]for r in rows)==8 and sum(r["values"]for r in rows)==96
total=measure(WORK/"results");free=shutil.disk_usage(WORK).free;delta=total-launch["baseline_results_bytes"]
assert delta<=plan["scope"]["new_bytes"]and total<=plan["scope"]["global_bytes"]and free>=plan["scope"]["reserve_bytes"]
result=dict(format="poseidon-public-expression-output-cpu-audit-v1",proposal_binding=plan["binding"],raw_report_binding=raw["binding"],
 authorization_binding=auth["binding"],sdk_identity=plan["sdk_identity"],stage_identities=plan["stage_identities"],
 task_id=plan["task_id"],request_id=plan["request_id"],status="passed",passed=1,failed=0,skipped=0,not_run=0,
 candidate_origin="unchanged_previous_rule",agent_generated=False,provider_calls=0,http_attempts=0,compiler_calls=1,encrypted_calls=3,
 input_groups=8,compared_values=96,maximum_absolute_error=max(all_errors),stages=rows,package_binding=manifest["binding"],
 package_files_checked=len(manifest["files"]),reloaded=True,new_inputs_executed=True,repeated_input_executed=True,
 static_none_output_restored=True,executor_secret_key=False,bootstrap=False,cuda=False,
 original_model_request_candidate_unchanged=True,historical_evidence_unchanged=True,historical_complete_passed=17,historical_complete_failed=1,
 independent_math_audit="stdlib multiplication by fixed reciprocal constants; no DSL converter or helper",
 resource_observation=dict(seconds=raw["seconds"],seconds_limit=240,new_results_bytes=delta,new_bytes_limit=134217728,
  total_results_bytes=total,total_bytes_limit=34359738368,free_bytes=free,reserve_bytes=4294967296),
 evidence_hashes=evidence,audit_source_sha256=sha(__file__))
result["binding"]=digest(result)
destination=OUT/"audit.json"
with destination.open("x")as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k not in{"evidence_hashes","stage_identities"}},ensure_ascii=False))
