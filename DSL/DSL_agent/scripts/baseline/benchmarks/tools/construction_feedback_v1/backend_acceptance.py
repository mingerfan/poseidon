"""Finite v13 construction-guidance compatibility acceptance; no provider calls or backend changes."""
import copy,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import WORK,VENV,enter_nix
RESULTS=WORK/"results";OUT=RESULTS/"agent-construction-guidance-acceptance-v1"
ORIGINAL=RESULTS/"agent-mha-live-delivery-recovery-v2/tasks/5384076bf9c04dc097254d1b1db30886/program"
SCOPE=dict(wall_seconds=300,new_bytes=128*1024**2,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
 provider_calls=0,native_workers=1,compile_jobs=2,link_jobs=1,native_address_bytes=4*1024**3,native_cpu_seconds=150)
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_bytes(canonical(v))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for block in iter(lambda:f.read(1024**2),b""):h.update(block)
 return h.hexdigest()
def measure(p):return int(subprocess.check_output(["du","--apparent-size","-s","-B1",str(p)],text=True,timeout=30).split()[0])
def protected():
 inventory=read(ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json")["protected_sources"]
 for name,want in inventory.items():
  p=Path(name)if Path(name).is_absolute()else ROOT/name
  require(p.name!=".env"and p.suffix not in(".sk",".pem")and"private-keys"not in p.parts,"protected_scope")
  require(sha(p)==want,"protected_backend_source_changed")
 return len(inventory)
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"offline_wall_budget")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 new=[p for p in RESULTS.iterdir()if p.name not in plan["existing_results"]]
 require(sum(measure(p)for p in new)<=SCOPE["new_bytes"],"new_artifact_budget")
 for name,want in plan["sources"].items():require(sha(ROOT/name)==want,"frozen_source_changed")
 for name,want in plan["historical_files"].items():require(sha(name)==want,"historical_evidence_changed")
def same_math_binding(old,new):
 from component_contract import reconstruct_request
 require(reconstruct_request(old)==old and reconstruct_request(new)==new,"valid_requests_required")
 require(old["generation_guidance"]["version"]=="explicit-v11"and
         new["generation_guidance"]["version"]=="explicit-v13","reviewed_guidance_transition")
 left=copy.deepcopy(old);right=copy.deepcopy(new)
 for value in (left,right):
  value.pop("request_id");value.pop("generation_guidance")
 require(canonical(left)==canonical(right),"historical_math_binding_changed")

def outer():
 require(not OUT.exists(),"fresh_output_directory")
 ancestors={os.getpid()};pid=os.getpid()
 while pid>1:
  try:pid=int(Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[1])
  except (OSError,ValueError,IndexError):break
  ancestors.add(pid)
 workers=[]
 for line in subprocess.check_output(["ps","-eo","pid,comm,args"],text=True).splitlines()[1:]:
  parts=line.split(None,2)
  if len(parts)!=3:continue
  pid,comm,args=parts
  if int(pid) in ancestors:continue
  if (comm.startswith("python") or comm in ("ninja","frontend","ckks-runtime-validate-poseidon")) and any(x in args for x in (
   "/Code/Poseidon/scripts/", "construction_feedback_v1", "candidate_worker", "run_agent_batch", "run_candidate", "component_worker")):
   workers.append(dict(pid=pid,comm=comm))
 require(not workers,"no_active_project_worker")
 controls=read(HERE/"controls.json")
 require(controls["status"]=="passed" and controls["failed"]==controls["errors"]==controls["skipped"]==0,"controls_required")
 require(controls["tests_run"]==103,"qualified_control_scope")
 require(measure(RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"] and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"offline_space_boundary")
 identity=controls["identity"]
 names=sorted({"scripts/baseline/"+n for s in ("generation","compilation","execution","validation")
  for n in identity[s]["sources"]})
 names += ["scripts/baseline/test_unified_construction_guidance.py",
  "scripts/baseline/test_unified_tensor_axis_guidance.py",str(Path(__file__).relative_to(ROOT))]
 history=[ORIGINAL/"request.json",ORIGINAL/"candidate.py",ORIGINAL/"manifest.json",
  ROOT/"docs/baseline/agent-mha-live-results-r1315.json",
  ROOT/"docs/baseline/agent-mha-feedback-live-results-r1331.json",
  ROOT/"docs/baseline/agent-mha-tensor-axis-live-results-r1335.json", HERE/"controls.json",HERE/"baseline.json"]
 history += [Path(c["request_file"]) for c in controls["frozen_new_contexts"]]
 plan=dict(format="poseidon-construction-guidance-offline-plan-v1",scope=SCOPE,
  cases=["group_mean_center","leading_linear_transpose_sum","historical_masked_mha"],
  controls_binding=controls["binding"],stage_identity_manifest_sha256=identity["manifest_sha256"],
  sources={n:sha(ROOT/n) for n in names},historical_files={str(p):sha(p) for p in history},
  protected_sources_checked=protected(),existing_results=sorted(p.name for p in RESULTS.iterdir()),
  deadline_wall=time.time()+SCOPE["wall_seconds"],created_wall=time.time(),
  continuation=dict(new_offline_window=True,authorization="Agent-only offline implementation and representative backend checks; API0, no dependencies, backend changes, parameter or native-resource expansion.",
   reason="Qualify explicit-v13 normal preparation/feedback and sandbox imports with two rule graphs and one unchanged historical Agent source. No current Agent generation claim."))
 plan["binding"]=hashlib.sha256(canonical(plan)).hexdigest()
 OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 code=enter_nix(cmd,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 if code not in (0,1):
  write(OUT/"termination.json",dict(status="timed_out_or_worker_exit",exit_code=code,provider_calls=0,
   automatic_retry=False,plan_binding=plan["binding"]))
 return code
def cases():
 from benchmark_suite import Builder
 from benchmark_graph import CAPACITY_FORMAT
 b=Builder([(2,2)]);m=b.node("mean",["input0"],axes=[-1],keepdims=True)
 y=b.node("subtract",["input0",m]);first=b.finish(y)
 b=Builder([(2,2)]);h=b.node("linear",["input0",b.const([[.11,-.07],[.23,.19]]),b.const([.03,-.05])])
 h=b.node("transpose",[h],dim0=-2,dim1=-1);y=b.node("sum",[h],axes=[-1],keepdims=False);second=b.finish(y)
 for name,g in (("group_mean_center",first),("leading_linear_transpose_sum",second)):
  g.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1")
  yield name,g,None,dict(candidate_origin="deterministic_rule_baseline")
 from component_contract import reconstruct_request
 old=reconstruct_request(read(ORIGINAL/"request.json"))
 yield "historical_masked_mha",copy.deepcopy(old["model"]),(ORIGINAL/"candidate.py").read_text(),dict(
  candidate_origin="historical_real_agent_source_unmodified",historical_request=old,
  original_package=str(ORIGINAL),historical_candidate_sha256=sha(ORIGINAL/"candidate.py"))
def inside():
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"pinned_worker_required")
 import numpy as np,torch
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from benchmark_graph import samples,digest
 from benchmark_math import evaluate
 from benchmark_torch import evaluate as torch_evaluate
 from candidate_bundle import check_bundle
 from numerical_results import compare
 torch.set_num_threads(2)
 plan=read(OUT/"plan.json");boundary(plan)
 identity=current_identity_manifest();require(identity["manifest_sha256"]==plan["stage_identity_manifest_sha256"],"pre_run_identity_changed");write(OUT/"identity.json",identity)
 context=WorkerContext.capture(OUT/"tasks");app=application(context)
 rows=[];started=time.monotonic()
 report=dict(format="poseidon-construction-guidance-offline-report-v1",status="running",plan_binding=plan["binding"],rows=rows,
  provider_calls=0,http_attempts=0,new_agent_generation=False,generation_guidance="explicit-v13",tolerance=dict(atol=1e-5,rtol=1e-4),
  identity_sha256=identity["manifest_sha256"],stage_source_counts={s:len(identity[s]["sources"])for s in ("generation","compilation","execution","validation")})
 def save(status):
  report.update(status=status,seconds=time.monotonic()-started);write(OUT/"report.json",report)
 def stable():
  boundary(plan);require(current_identity_manifest()==identity,"active_stage_source_changed")
 save("running")
 for name,model,source,origin in cases():
  if name not in plan["cases"]:continue
  stable();row=dict(case=name,status="running",compiled=False,encrypted_execution=False,reuse_checks=[],
    candidate_origin=origin["candidate_origin"])
  rows.append(row);save("running")
  try:
   dest=OUT/name;dest.mkdir(mode=0o700);write(dest/"model.json",model)
   probes=list(samples(model,4))
   fresh=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in p.items()}for p in probes]
   for probe in probes+fresh:
    a=evaluate(model,probe);b=torch_evaluate(model,probe)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
   row.update(independent_reference_groups=8,model_sha256=digest(model))
   options=dict(chunk_period=4,generation_guidance="explicit-v13")
   if source is not None:
    from component_contract import request_options
    from unified_capacity_constants import request_version
    old=origin["historical_request"];options=request_options(old)
    options["generation_guidance"]="explicit-v13"
    options["capacity_packing_version"]=request_version(old)

   task=app.submit(model,options=options,limits=dict(generations=1,http_attempts=1,wall_seconds=min(170,max(1,int(plan["deadline_wall"]-time.time()))),
    api_timeout_seconds=1200,max_tokens=384000))
   row["task_id"]=task
   require(app.status(task)["status"]=="queued","application_input_rejected")
   meta=app._checked_task_metadata(task);request=meta["request"]
   if source is None:source=candidate_source(request)[0]
   else:
    old=origin["historical_request"]
    same_math_binding(old,request)
    row.update(original_package=origin["original_package"],historical_candidate_sha256=origin["historical_candidate_sha256"])
    require(hashlib.sha256(source.encode()).hexdigest()==origin["historical_candidate_sha256"],"historical_source_changed")
   class Replay:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     require(actual==request and feedback==[],"single_bound_offline_replay")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
   result=app.run(task,Replay());write(dest/"qualification.json",result)
   accepted=result.get("acceptance")or{}
   row.update(request_id=request["request_id"],candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    qualification_status=result["status"],acceptance=accepted,
    compiled=accepted.get("compiled")is True,encrypted_execution=accepted.get("encrypted_execution")is True)
   require(result["status"]=="succeeded"and accepted.get("numerically_validated")is True,"qualification_rejected")
   require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"offline_provenance")
   package=app._dir(task)/"program";manifest=check_bundle(package)
   require(manifest["validation"]["level"]=="numerical","numerical_package")
   hashes={n:sha(package/n)for n in manifest["files"]}
   row.update(package=str(package),package_binding=manifest["binding"],
    artifacts=[n for n in manifest["files"]if n.startswith("artifacts/")])
   stable()
   execution=app.execute(task,[{k:v.tolist()for k,v in p.items()}for p in fresh])
   write(dest/"new-input-execution.json",execution)
   require(execution.get("status")=="executed"and execution.get("encrypted_execution")is True,"reloaded_execution_rejected")
   require(execution["agent_calls"]==execution["compiler_calls"]==0,"reuse_recompiled")
   expected=np.asarray([np.concatenate([x.reshape(-1)for x in evaluate(model,p).values()])for p in fresh])
   observed=np.asarray([np.concatenate([np.asarray(x).reshape(-1)for x in p.values()])for p in execution["outputs"]])
   comparison=compare(observed,expected,1e-5,1e-4);write(dest/"new-input-comparison.json",comparison)
   row["reuse_checks"].append(dict(input_groups=4,compared_values=comparison["compared_values"],
    maximum_absolute_error=comparison["max_absolute_error"],passed=comparison["passed"],
    compiler_calls=0,agent_calls=0,evidence=execution["evidence"]))
   require(comparison["passed"],"new_input_numerical_rejected")
   require(all(sha(package/n)==h for n,h in hashes.items()),"package_changed_during_reuse")
   require(app.result(task)["status"]=="succeeded","package_delivery_rejected")
   row["status"]="passed"
  except Exception as error:
   code=str(error)
   row.update(status="failed",error_type=type(error).__name__,
    code=code if code.replace("_","").isalnum()and len(code)<120 else"offline_case_exception")
   if row["code"]in("active_stage_source_changed","frozen_source_changed","historical_evidence_changed",
    "offline_wall_budget","disk_reserve","global_artifact_budget","new_artifact_budget"):
    save("failed");raise
  stable();save("running")
  print(json.dumps({k:row[k]for k in("case","status","compiled","encrypted_execution")}),flush=True)
 stable()
 report.update(passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),
  protected_sources_checked=protected(),protected_sources_changed=0,controls_binding=plan["controls_binding"],
  compilations=sum(r["compiled"]for r in rows),encrypted_calls=sum((r["encrypted_execution"]is True)+len(r["reuse_checks"])for r in rows),
  input_groups=sum(r.get("acceptance",{}).get("input_groups",0)+sum(t["input_groups"]for t in r["reuse_checks"])for r in rows),
  compared_values=sum(r.get("acceptance",{}).get("compared_values",0)+sum(t["compared_values"]for t in r["reuse_checks"])for r in rows),
  maximum_absolute_error=max([r.get("acceptance",{}).get("maximum_absolute_error",0)for r in rows]+[
   t["maximum_absolute_error"]for r in rows for t in r["reuse_checks"]]),
  new_artifact_bytes=measure(OUT),old_evidence_unchanged=True)
 save("passed"if report["failed"]==0 else"failed")
 print(json.dumps({k:report[k]for k in("status","passed","failed","compilations","encrypted_calls","input_groups","compared_values","maximum_absolute_error","seconds","provider_calls")}),flush=True)
 return 0 if report["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
