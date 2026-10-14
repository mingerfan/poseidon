"""Representative V22 component-to-Poseidon acceptance; scripted candidates, zero API."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import require,digest
from hecate_python_env import WORK,VENV,enter_nix
OUT=WORK/"results/agent-public-tensor-v22-cpu-v1"
SCOPE=dict(wall_seconds=900,new_bytes=256*1024**2,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
 compiler_jobs=2,link_jobs=1,native_concurrency=1,native_address_bytes=4*1024**3,native_cpu_seconds=150)
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",
 public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
NAMES=("bool_multiply","mask_index_residual","float32_batch_norm","method_channel_mask","divided_gelu_linear")
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 Path(p).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
def measure(p):
 return max(int(subprocess.check_output(["du",flag,"-s","-B1",str(p)],text=True,timeout=20).split()[0])
  for flag in ("--apparent-size","--count-links"))
def identity():
 from application_identity import current_identity_manifest
 return current_identity_manifest()
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"offline_deadline")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(WORK/"results")<=SCOPE["global_bytes"],"global_artifact_budget")
 require(measure(OUT)<=SCOPE["new_bytes"],"new_artifact_budget")
 for rel,h in plan["tools"].items():require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,"offline tool integrity")
 require(identity()["manifest_sha256"]==plan["source_identity"],"frozen SDK identity")
def outer():
 require(not OUT.exists(),"fresh acceptance results")
 source_root=WORK/"results/agent-public-tensor-v22-source-v2";source=read(source_root/"report.json")
 require(source["passed"]==8 and source["failed"]==0 and source["controls"]==dict(tests=146,failed=0,errors=0,skipped=0),
  "source qualification required")
 require(measure(WORK/"results")+SCOPE["new_bytes"]<=SCOPE["global_bytes"] and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"offline preflight space")
 now=time.time();OUT.mkdir(mode=0o700)
 tools={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob("*.py")}
 cases=[dict(case=n,manifest=str(HERE/"fixtures"/n/"manifest.json"),
  captured_provenance=str(source_root/n/"model-provenance.json"),source_entry=n in NAMES[:3])for n in NAMES]
 plan=dict(format="poseidon-public-tensor-v22-cpu-plan-v1",created_wall=now,deadline_wall=now+SCOPE["wall_seconds"],
  tools=tools,scope=SCOPE,cases=cases,source_report_sha256=hashlib.sha256((source_root/"report.json").read_bytes()).hexdigest(),
  source_identity=identity()["manifest_sha256"],paid_calls=0,http_attempts=0,candidate_origin="deterministic_rule",
  tolerance=dict(atol=1e-5,rtol=1e-4),source_contract="poseidon-trusted-pytorch-v22")
 plan["binding"]=digest(plan);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
def inside():
 import numpy as np
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from candidate_bundle import check_bundle
 from benchmark_graph import samples
 from operator_reference import evaluate as original_math
 from numerical_results import compare
 plan=read(OUT/"plan.json");boundary(plan)
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned acceptance runtime")
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 report=dict(format="poseidon-public-tensor-v22-cpu-report-v1",plan_binding=plan["binding"],status="running",
  rows=rows,paid_calls=0,http_attempts=0,new_agent_generation=False,candidate_origin="deterministic_rule",
  tolerance=dict(atol=1e-5,rtol=1e-4))
 def save(status):
  report.update(status=status,seconds=time.time()-plan["created_wall"]);write(OUT/"report.json",report)
 save("running")
 for item in plan["cases"]:
  boundary(plan);dest=OUT/item["case"];dest.mkdir()
  row=dict(case=item["case"],status="running",source_entry=item["source_entry"],compiled=False,
   encrypted_execution=False,reloaded=False,repeated=False,failure_layer=None)
  rows.append(row);save("running")
  try:
   limits=dict(generations=1,http_attempts=1,wall_seconds=max(1,int(plan["deadline_wall"]-time.time())),
    api_timeout_seconds=1200,max_tokens=384000)
   if item["source_entry"]:
    task=app.submit_source(dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
     manifest=item["manifest"],trust_local_code=True),resource_profile="general-tensors-v1",options=OPTIONS,limits=limits)
    app.prepare_source(task)
   else:
    provenance=read(item["captured_provenance"]);model=read(Path(item["captured_provenance"]).parent/"model.json")
    task=app.submit(model,model_provenance=provenance,options=OPTIONS,limits=limits)
   row["task_id"]=task;require(app.status(task)["status"]=="queued","component source rejected")
   meta=app._checked_task_metadata(task);request=meta["request"];model=request["model"]
   provenance=meta["model_provenance"]
   source=candidate_source(request)[0];(dest/"rule-candidate.py").write_text(source)
   row["candidate_sha256"]=hashlib.sha256(source.encode()).hexdigest()
   class Rule:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     require(actual==request and feedback==[],"One scripted candidate")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
   result=app.run(task,Rule());write(dest/"qualification.json",result)
   accepted=result.get("acceptance")or{}
   row.update(acceptance=accepted,compiled=accepted.get("compiled")is True,
    encrypted_execution=accepted.get("encrypted_execution")is True,failure_layer=(result.get("failure")or{}).get("layer"))
   require(result["status"]=="succeeded"and accepted.get("numerically_validated")is True,"numerical acceptance")
   require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"Scripted origin")
   package=app._dir(task)/"program";manifest=check_bundle(package)
   require(manifest["validation"]["level"]=="numerical","Package validation level")
   require(any(n.endswith("runtime-plan.json")for n in manifest["files"]),"RuntimePlan artifact")
   files={n:hashlib.sha256((package/n).read_bytes()).hexdigest()for n in manifest["files"]}
   probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in x.items()}for x in samples(model,4)[:3]]
   wire=[{k:v.tolist()for k,v in x.items()}for x in probes]
   expected=np.asarray([np.concatenate([original_math(provenance["original"],x)[o["name"]].reshape(-1)
                                       for o in model["outputs"]])for x in probes])
   for suffix in ("new-inputs","repeat"):
    execution=app.execute(task,wire);write(dest/(suffix+".json"),execution)
    require(execution["status"]=="executed"and execution["encrypted_execution"]is True
     and execution["agent_calls"]==execution["compiler_calls"]==0,"Package execution without regeneration")
    actual=np.asarray([np.concatenate([np.asarray(x[o["name"]],dtype=np.float64).reshape(-1)
                                      for o in model["outputs"]])for x in execution["outputs"]])
    comparison=compare(actual,expected,1e-5,1e-4);write(dest/(suffix+"-comparison.json"),comparison)
    require(comparison["passed"],"Original graph package numerical comparison")
    row[suffix.replace("-","_")+"_max_absolute_error"]=comparison["max_absolute_error"]
    row[suffix.replace("-","_")+"_values"]=comparison["compared_values"]
   require(check_bundle(package)==manifest and all(hashlib.sha256((package/n).read_bytes()).hexdigest()==h for n,h in files.items()),
    "Package content changed after reuse")
   row.update(status="passed",reloaded=True,repeated=True,package=str(package),package_binding=manifest["binding"],
    reuse_groups=6,reuse_values=row["new_inputs_values"]+row["repeat_values"])
  except Exception as error:
   (dest/"failure-private.txt").write_text(traceback.format_exc())
   row.update(status="failed",error=type(error).__name__,reason=str(error)[:160])
  save("running");print("CASE",row["case"],row["status"],row.get("reason",""),flush=True)
  boundary(plan)
  require(row["failure_layer"]not in("environment","integrity","resources"),"Environment or integrity stops batch")
 report.update(passed=sum(x["status"]=="passed"for x in rows),failed=sum(x["status"]=="failed"for x in rows),
  compiled=sum(x["compiled"]for x in rows),encrypted_initial=sum(x["encrypted_execution"]for x in rows),
  encrypted_reuse_calls=2*sum(x["repeated"]for x in rows),package_reloads=sum(x["reloaded"]for x in rows),
  total_groups=sum(x.get("acceptance",{}).get("input_groups",0)+x.get("reuse_groups",0)for x in rows),
  compared_values=sum(x.get("acceptance",{}).get("compared_values",0)+x.get("reuse_values",0)for x in rows))
 save("passed"if report["passed"]==len(NAMES)else"failed")
 report["binding"]=digest(report);write(OUT/"report.json",report)
 print("REPORT",json.dumps({k:v for k,v in report.items()if k!="rows"}),flush=True)
 return 0 if report["status"]=="passed"else 1
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
 raise SystemExit(inside()if a.inside else outer())
