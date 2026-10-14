"""Real package reuse and repeated Poseidon execution; no generation/recompile."""
import hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import WORK,VENV,enter_nix
RESULTS=WORK/"results";OUT=RESULTS/"agent-staged-context-package-acceptance-r1329"
PREVIOUS=RESULTS/"agent-parameter-boundary-acceptance-r1328-v2"
SCOPE=dict(wall_seconds=600,new_bytes=128*1024**2,global_bytes=32*1024**3,
 reserve_bytes=4*1024**3,provider_calls=0,native_workers=1,compiler_jobs=2,link_jobs=1,
 native_address_bytes=4*1024**3,native_cpu_seconds=150)
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_bytes(canonical(v))
def measure(p):return int(subprocess.check_output(["du","--apparent-size","-s","-B1",str(p)],text=True,timeout=30).split()[0])
def retained_files():
 folders=[PREVIOUS/"tasks"/r["task_id"]for r in read(PREVIOUS/"report.json")["rows"]]
 result={}
 for folder in folders:
  for p in folder.rglob("*"):
   require(not p.is_symlink(),"regular_source_evidence")
   if not p.is_file():continue
   require(p.name!=".env"and p.suffix not in (".sk",".pem")and"private-keys"not in p.parts,"no_key_transfer")
   result[str(p)]=sha(p)
 result[str(PREVIOUS/"tasks/worker-context.json")]=sha(PREVIOUS/"tasks/worker-context.json")
 return result
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"offline_time_budget")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 new=[p for p in RESULTS.iterdir()if p.name not in plan["existing_results"]]
 require(sum(measure(p)for p in new)<=SCOPE["new_bytes"],"new_artifact_budget")
 require(all(sha(ROOT/n)==h for n,h in plan["sources"].items()),"frozen_source_changed")
 require(all(sha(p)==h for p,h in plan["retained_files"].items()),"retained_evidence_changed")
def outer():
 require(not OUT.exists(),"fresh_output_directory")
 before=read(PREVIOUS/"report.json")
 require(before["status"]=="passed"and len(before["rows"])==2,"two_checked_unchanged_packages")
 require(measure(RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"]and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"offline_space_boundary")
 names=["application_execution/context.py","application_execution/context_identity.py",
        "application_component.py","application_reuse.py","test_staged_application_reuse.py",
        "golden_cases/fixtures.json","golden_cases/linear4x2/model.py","golden_cases/mlp4x4x2/model.py"]
 plan=dict(format="poseidon-staged-context-package-plan-v1",scope=SCOPE,
  previous_report=str(PREVIOUS/"report.json"),previous_report_sha256=sha(PREVIOUS/"report.json"),
  sources={str(Path("scripts/baseline")/n):sha(BASE/n)for n in names},
  tool_sha256=sha(__file__),retained_files=retained_files(),
  controls_binding=read(HERE/"controls.initial.json")["binding"],
  existing_results=sorted(p.name for p in RESULTS.iterdir()),deadline_wall=time.time()+SCOPE["wall_seconds"])
 plan["binding"]=hashlib.sha256(canonical(plan)).hexdigest()
 OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=SCOPE["wall_seconds"],keep_env=(),environment="poseidon-cpu")
def golden(name):
 p=BASE/"golden_cases"/name/"model.py"
 spec=importlib.util.spec_from_file_location("_staged_golden_"+name,p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.build_model()
def inside():
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"pinned_sdk")
 import numpy as np,torch
 from application_identity import current_identity_manifest,compatible
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from component_contract import request_options
 from benchmark_math import evaluate
 from numerical_results import compare
 from candidate_bundle import check_bundle
 torch.set_num_threads(2)
 plan=read(OUT/"plan.json");boundary(plan)
 original=read(PREVIOUS/"report.json");old_identity=read(PREVIOUS/"identity.json")
 identity=current_identity_manifest();require(compatible(old_identity,identity),"reused_package_stages_changed")
 require(old_identity["generation"]["sha256"]!=identity["generation"]["sha256"],"real_generation_identity_change_required")
 write(OUT/"identity.json",identity)
 tasks=OUT/"tasks";tasks.mkdir(mode=0o700)
 for row in original["rows"]:
  shutil.copytree(PREVIOUS/"tasks"/row["task_id"],tasks/row["task_id"])
 context=WorkerContext.capture(tasks);require(context.record()["format"]=="poseidon-worker-context-v2","new_context_policy")
 app=application(context);rows=[];started=time.monotonic()
 report=dict(format="poseidon-staged-context-package-report-v1",binding=plan["binding"],status="running",
  rows=rows,candidate_origin=original["candidate_origin"],new_agent_generation=False,
  generation_identity_changed=True,delivery_stages_unchanged=True,current_identity_sha256=identity["manifest_sha256"],
  worker_binding=context.binding,worker_format=context.record()["format"],
  compiler_calls=0,provider_calls=0,http_attempts=0,encrypted_calls=0)
 def save(status):
  report.update(status=status,seconds=time.monotonic()-started);write(OUT/"report.json",report)
 def stable():
  boundary(plan);require(current_identity_manifest()==identity,"stage_identity_drift")
  context.check_current()
 save("running")
 for original_row in original["rows"]:
  stable();name=original_row["case"]
  row=dict(case=name,status="running",reuse_checks=[]);rows.append(row);save("running")
  try:
   old_task=original_row["task_id"];old_result=app.result(old_task)
   require(old_result["status"]=="succeeded","old_package_not_deliverable")
   meta=app._checked_task_metadata(old_task);request=meta["request"]
   task=app.submit(request["model"],options=request_options(request),limits=dict(
    generations=1,http_attempts=1,wall_seconds=170,api_timeout_seconds=1200,max_tokens=384000))
   require(app._checked_task_metadata(task)["request"]==request,"same_request_bytes")
   class NoGeneration:
    kind="scripted_replay";agent_calls=0
    def generate(self,*_):raise AssertionError("Package reuse must not generate")
   result=app.run(task,NoGeneration())
   write(OUT/(name+"-cache.json"),result)
   require(result["status"]=="succeeded"and result["package"]["reused"]is True,"stage_cache_rejected")
   require(result["package"]["task_id"]==old_task and result["generations"]==result["http_attempts"]==0,"same_package_zero_generation")
   package=tasks/old_task/"program";manifest=check_bundle(package)
   require(manifest["binding"]==original_row["package_binding"],"original_package_binding")
   package_hashes={n:sha(package/n)for n in manifest["files"]}
   row.update(old_task=old_task,new_task=task,package_binding=manifest["binding"],
    old_acceptance_preserved=result["acceptance"],request_id=request["request_id"],reused=True)
   source=golden(name)
   probes=[dict(x=np.asarray(x,dtype=np.float64))for x in
    ([.19,-.27,.41,-.13],[-.53,.37,-.29,.11],[.47,.23,-.17,-.31],[-.07,-.43,.59,.17])]
   for q in probes:
    with torch.no_grad():expected=source(torch.tensor(q["x"],dtype=torch.float64)).numpy()
    np.testing.assert_allclose(evaluate(request["model"],q)["output"],expected,atol=1e-12,rtol=1e-12)
   payload=[dict(x=q["x"].tolist())for q in probes]
   expected=np.asarray([evaluate(request["model"],q)["output"].reshape(-1)for q in probes])
   for index in range(2):
    stable();execution=app.execute(task,payload)
    write(OUT/("%s-execution-%d.json"%(name,index)),execution)
    require(execution.get("status")=="executed"and execution.get("encrypted_execution")is True,"poseidon_execution_rejected")
    require(execution["agent_calls"]==execution["compiler_calls"]==0,"unexpected_generation_or_compile")
    require(execution["run_context"]["worker_binding"]==context.binding,"execution_context_not_bound")
    observed=np.asarray([q["output"]for q in execution["outputs"]],dtype=np.float64).reshape(expected.shape)
    comparison=compare(observed,expected,1e-5,1e-4)
    write(OUT/("%s-comparison-%d.json"%(name,index)),comparison)
    row["reuse_checks"].append(dict(index=index,evidence=execution["evidence"],groups=4,
      values=comparison["compared_values"],max_absolute_error=comparison["max_absolute_error"],passed=comparison["passed"]))
    report["encrypted_calls"]+=1;save("running")
    require(comparison["passed"],"numeric_failure")
   require(all(sha(package/n)==h for n,h in package_hashes.items()),"package_mutation")
   require(app.result(task)["status"]=="succeeded","repeat_delivery_rejected")
   row["status"]="passed"
  except Exception as error:
   row.update(status="failed",error_type=type(error).__name__,code=str(error)[:200])
   save("failed");print(json.dumps(dict(case=name,status=row["status"],code=row["code"])),flush=True);return 1
  stable();save("running");print(json.dumps(dict(case=name,status="passed",reused=True,encrypted_calls=2)),flush=True)
 report.update(input_groups=sum(t["groups"]for r in rows for t in r["reuse_checks"]),
  compared_values=sum(t["values"]for r in rows for t in r["reuse_checks"]),
  max_absolute_error=max(t["max_absolute_error"]for r in rows for t in r["reuse_checks"]),
  retained_evidence_unchanged=True,backend_parameters_unchanged=True)
 stable();save("passed")
 print(json.dumps({k:report[k]for k in ("status","compiler_calls","encrypted_calls","provider_calls","input_groups","compared_values","max_absolute_error","seconds")}),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
