"""Three fixed manual-equivalent graphs through the existing CPU chain; no model API."""
import hashlib,json,os,shlex,shutil,sys,time,traceback,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
LIVE=HERE.parent/"source_live_v2";sys.path.insert(0,str(LIVE));sys.path.insert(0,str(BASE))
import runner_scope as frozen
from run import OwnedTree
OUT=frozen.RESULTS/"agent-public-divisor-equivalent-cpu-v1"
SCOPE=dict(wall_seconds=480,new_bytes=256*1024**2,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
 compiler_jobs=2,link_jobs=1,native_concurrency=1,native_address_bytes=4*1024**3,native_cpu_seconds=150)
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",
 public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+".next")
 with tmp.open("x")as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write("\n")
 tmp.replace(p)
def measure(p):
 return max(int(subprocess.check_output(["du",flag,"-s","-B1",str(p)],text=True,timeout=20).split()[0])
  for flag in ("--apparent-size","--count-links"))
def boundary(plan):
 frozen.require(time.time()<plan["deadline_wall"],"offline_deadline")
 frozen.require(shutil.disk_usage(frozen.WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 frozen.require(measure(frozen.RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 added=[p for p in frozen.RESULTS.iterdir()if p.name not in plan["initial_results"]]
 frozen.require(sum(measure(p)for p in added)<=SCOPE["new_bytes"],"new_artifact_budget")
 for path,h in plan["tools"].items():frozen.require(frozen.sha(ROOT/path)==h,"frozen_offline_tools")
 frozen.verify()
def outer():
 frozen.require(not OUT.exists(),"fresh_offline_output");frozen.verify()
 controls=frozen.read(HERE/"controls-v3.json")
 frozen.require(controls["status"]=="passed"and controls["tests_run"]==52,"prototype_controls_required")
 for name,h in controls["tools"].items():frozen.require(frozen.sha(HERE/name)==h,"prototype_control_source_changed")
 frozen.require(measure(frozen.RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"]and
  shutil.disk_usage(frozen.WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"offline_space_precheck")
 now=time.time()
 tools={str(p.relative_to(ROOT)):frozen.sha(p)for p in HERE.iterdir()if p.is_file()and(p.suffix==".py"or p.name=="backend-cases.json")}
 plan=dict(format="poseidon-public-divisor-equivalent-plan-v1",created_wall=now,deadline_wall=now+SCOPE["wall_seconds"],
  tools=tools,scope=SCOPE,cases=frozen.read(HERE/"backend-cases.json"),source_identity=frozen.verify()["identity_sha256"],
  live_proposal_binding=frozen.PROPOSAL_BINDING,initial_results=[p.name for p in frozen.RESULTS.iterdir()],
  provider_calls=0,compiler_semantic_changes=False,source_intake_activated=False)
 plan["binding"]=frozen.digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 tree=OwnedTree().start()
 try:code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 finally:background=tree.finish()
 if code not in(0,1)or background:
  write(OUT/"termination.json",dict(code=code,background=background,automatic_restart=False,provider_calls=0))
 return code if code else int(background)
def inside():
 import numpy as np
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from candidate_bundle import check_bundle
 from benchmark_graph import samples
 from numerical_results import compare
 from hecate_python_env import VENV
 frozen.require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"locked_offline_worker")
 plan=frozen.read(OUT/"plan.json");boundary(plan);identity=current_identity_manifest()
 frozen.require(identity["manifest_sha256"]==plan["source_identity"],"frozen_sdk_identity")
 write(OUT/"identity.json",identity)
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 report=dict(format="poseidon-public-divisor-equivalent-cpu-report-v1",plan_binding=plan["binding"],
  status="running",rows=rows,provider_calls=0,http_attempts=0,new_agent_generation=False,source_intake_activated=False,
  source_captures=0,candidate_origin="deterministic_rule_on_manual_equivalent_graph",tolerance=dict(atol=1e-5,rtol=1e-4))
 def save(status):report.update(status=status,seconds=time.time()-plan["created_wall"]);write(OUT/"report.json",report)
 save("running")
 for item in plan["cases"]:
  boundary(plan);dest=OUT/item["case"];dest.mkdir(mode=0o700)
  row=dict(case=item["case"],status="running",compiled=False,encrypted_execution=False,reload_passed=False,
   request_id=item["request_id"],failure_layer=None);rows.append(row);save("running")
  try:
   task=app.submit(item["model"],options=OPTIONS,limits=dict(generations=1,http_attempts=1,
    wall_seconds=max(1,int(plan["deadline_wall"]-time.time())),api_timeout_seconds=1200,max_tokens=384000))
   row["task_id"]=task;state=app.status(task)
   frozen.require(state["status"]=="queued","equivalent_graph_rejected")
   meta=app._checked_task_metadata(task);request=meta["request"];model=request["model"]
   frozen.require(request["request_id"]==item["request_id"],"exact_equivalent_request")
   source=candidate_source(request)[0];(dest/"rule-candidate.py").write_text(source)
   row["candidate_sha256"]=hashlib.sha256(source.encode()).hexdigest()
   class Rule:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     frozen.require(actual==request and feedback==[],"single_rule_candidate")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
   result=app.run(task,Rule());write(dest/"qualification.json",result)
   accepted=result.get("acceptance")or{}
   row.update(acceptance=accepted,compiled=accepted.get("compiled")is True,encrypted_execution=accepted.get("encrypted_execution")is True,
    failure_layer=(result.get("failure")or{}).get("layer"))
   frozen.require(result["status"]=="succeeded"and accepted.get("numerically_validated")is True,"equivalent_numerical_qualification")
   frozen.require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"rule_origin")
   package=app._dir(task)/"program";manifest=check_bundle(package)
   frozen.require(manifest["validation"]["level"]=="numerical","numerical_bundle")
   files={name:frozen.sha(package/name)for name in manifest["files"]}
   probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in x.items()}for x in samples(model,4)]
   execution=app.execute(task,[{k:v.tolist()for k,v in x.items()}for x in probes]);write(dest/"reloaded-execution.json",execution)
   frozen.require(execution["status"]=="executed"and execution["encrypted_execution"]is True and
    execution["agent_calls"]==execution["compiler_calls"]==0,"real_reload_without_generation")
   divisor=np.asarray(item["divisor"],dtype=np.float64).reshape(item["divisor_shape"])
   def original_divide(value):
    a,b=np.broadcast_arrays(value,divisor);output=np.empty(a.shape,dtype=np.float64)
    for idx in np.ndindex(a.shape):output[idx]=float(a[idx])/float(b[idx])
    return output
   expected=[]
   for values in probes:
    math_outputs={"out":original_divide(values["x"])}
    if item["mode"]=="dual_snapshot":
     math_outputs={"snapshot":values["x"].copy(),"out":original_divide(values["x"])+original_divide(values["y"])}
    expected.append(np.concatenate([math_outputs[o["name"]].reshape(-1)for o in model["outputs"]]))
   actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)
    for o in model["outputs"]])for v in execution["outputs"]])
   compared=compare(actual,np.asarray(expected),1e-5,1e-4);write(dest/"original-division-comparison.json",compared)
   frozen.require(compared["passed"],"original_public_division_numerical")
   frozen.require(check_bundle(package)==manifest and all(frozen.sha(package/name)==h for name,h in files.items()),"package_reloaded_integrity")
   row.update(status="passed",reload_passed=True,reload_groups=4,reload_values=compared["compared_values"],
    original_division_max_absolute_error=compared["max_absolute_error"],package=str(package),package_binding=manifest["binding"],
    artifacts=[n for n in manifest["files"]if n.startswith("artifacts/")])
  except Exception as error:
   (dest/"failure-private.txt").write_text(traceback.format_exc())
   row.update(status="failed",error_type=type(error).__name__,code=str(error)[:120])
  save("running");print(json.dumps({k:row[k]for k in["case","status","compiled","encrypted_execution","reload_passed"]}),flush=True)
  boundary(plan)
  frozen.require(row["failure_layer"]not in("environment","resources","integrity"),"related_batch_stopped")
 report.update(planned=3,passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),
  compilations=sum(r["compiled"]for r in rows),encrypted_calls=sum(int(r["encrypted_execution"])+int(r["reload_passed"])for r in rows),
  input_groups=sum(r.get("acceptance",{}).get("input_groups",0)+r.get("reload_groups",0)for r in rows),
  compared_values=sum(r.get("acceptance",{}).get("compared_values",0)+r.get("reload_values",0)for r in rows),
  original_division_max_absolute_error=max((r.get("original_division_max_absolute_error",0)for r in rows),default=None))
 save("passed"if report["passed"]==3 else"failed");report["binding"]=frozen.digest(report);write(OUT/"report.json",report)
 print(json.dumps({k:v for k,v in report.items()if k!="rows"}),flush=True)
 return 0 if report["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
