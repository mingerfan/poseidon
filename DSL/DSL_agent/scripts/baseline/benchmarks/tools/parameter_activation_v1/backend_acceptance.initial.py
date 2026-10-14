"""Finite offline acceptance of the activated Agent parameter boundary."""
import hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
HERE=Path(__file__).resolve().parent
os.environ["POSEIDON_PLATFORM"]="aarch64-linux"
os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import WORK,VENV,enter_nix
RESULTS=WORK/"results";OUT=RESULTS/"agent-parameter-boundary-acceptance-r1328"
SCOPE=dict(wall_seconds=600,new_bytes=128*1024**2,global_bytes=32*1024**3,
 reserve_bytes=4*1024**3,api_calls=0,native_workers=1,compile_jobs=2,link_jobs=1,
 native_address_bytes=4*1024**3,native_cpu_seconds=150)
def require(ok,code):
 if not ok:raise ValueError(code)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def write(p,x):Path(p).write_bytes(canonical(x))
def read(p):return json.loads(Path(p).read_text())
def measure(p):return int(subprocess.check_output(["du","--apparent-size","-s","-B1",str(p)],text=True,timeout=30).split()[0])
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"offline_wall_budget")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 new=[p for p in RESULTS.iterdir() if p.name not in plan["existing_results"]]
 require(sum(measure(p) for p in new)<=SCOPE["new_bytes"],"new_artifact_budget")
 require(all(sha(ROOT/n)==h for n,h in plan["sources"].items()),"reviewed_source_drift")
def outer():
 require(not OUT.exists(),"fresh_result_directory")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"disk_reserve")
 require(measure(RESULTS)<SCOPE["global_bytes"]-SCOPE["new_bytes"],"global_artifact_budget")
 names=["scripts/baseline/application_planning/backend_parameters.py",
 "scripts/baseline/application_planning/resources.py",
 "scripts/baseline/application_planning/compiled_resources.py",
 "scripts/baseline/test_compiled_execution_precheck.py",
 "scripts/baseline/golden_cases/fixtures.json",
 "scripts/baseline/golden_cases/linear4x2/model.py",
 "scripts/baseline/golden_cases/mlp4x4x2/model.py",
 str(Path(__file__).relative_to(ROOT))]
 plan=dict(format="poseidon-parameter-activation-offline-plan-v1",scope=SCOPE,
 cases=["linear4x2","mlp4x4x2"],candidate_origin="deterministic_rule_baseline",
 source_transition_sha256=sha(HERE/"source-transition.json"),
 source_after_sha256=sha(HERE/"source-after.json"),
 controls_binding=read(HERE/"controls.final.json")["binding"],
 sources={n:sha(ROOT/n)for n in names},
 existing_results=sorted(p.name for p in RESULTS.iterdir()),deadline_wall=time.time()+SCOPE["wall_seconds"])
 plan["binding"]=hashlib.sha256(canonical(plan)).hexdigest()
 OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
 [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=SCOPE["wall_seconds"],keep_env=(),environment="poseidon-cpu")
def graph(name,f):
 constants=dict(w2=f["linear_weight"],b2=f["linear_bias"])
 nodes=[]
 if name=="mlp4x4x2":
  constants.update(w1=f["mlp_hidden_weight"],b1=f["mlp_hidden_bias"])
  nodes.extend([dict(id="hidden",op="linear",inputs=["x","w1","b1"],outputs=["h"],attrs={}),
   dict(id="activation",op="multiply",inputs=["h","h"],outputs=["h2"],attrs={})])
 nodes.append(dict(id="output",op="linear",inputs=["h2" if name=="mlp4x4x2" else "x","w2","b2"],
  outputs=["y"],attrs={}))
 return dict(format="poseidon-model-graph-v1",id="parameter_boundary_"+name,
  inputs=[dict(name="x",shape=[4])],constants=constants,nodes=nodes,outputs=[dict(name="output",value="y")])
def golden(name):
 p=BASE/"golden_cases"/name/"model.py"
 spec=importlib.util.spec_from_file_location("_trusted_boundary_"+name,p)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.build_model()
def inside():
 require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"pinned_worker_required")
 import numpy as np,torch
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_planning.backend_parameters import parameters,native_address_limit
 from unified_graph_lowering import candidate_source
 from benchmark_graph import samples,digest
 from benchmark_math import evaluate
 from candidate_bundle import check_bundle
 from numerical_results import compare
 torch.set_num_threads(2)
 plan=read(OUT/"plan.json");boundary(plan)
 identity=current_identity_manifest();write(OUT/"identity.json",identity)
 context=WorkerContext.capture(OUT/"tasks");app=application(context)
 identity_sha=identity["manifest_sha256"];rows=[];started=time.monotonic()
 report=dict(format="poseidon-parameter-activation-offline-report-v1",binding=plan["binding"],status="running",
  identity_sha256=identity_sha,stage_source_counts={s:len(identity[s]["sources"])for s in ("generation","compilation","execution","validation")},
  rows=rows,provider_calls=0,http_attempts=0,new_agent_generation=False,candidate_origin="deterministic_rule_baseline")
 def save(status):
  report.update(status=status,seconds=time.monotonic()-started)
  write(OUT/"report.json",report)
 def stable():
  boundary(plan)
  require(current_identity_manifest()==identity,"active_stage_identity_drift")
 save("running")
 f=read(BASE/"golden_cases/fixtures.json")
 for name in plan["cases"]:
  stable();row=dict(case=name,status="running",compiled=False,encrypted_execution=False,reuse_checks=[])
  rows.append(row);save("running")
  try:
   model=graph(name,f);trusted=golden(name)
   dest=OUT/name;dest.mkdir(mode=0o700);write(dest/"model.json",model)
   probes=list(samples(model,4))
   fresh=[dict(x=np.asarray(x,dtype=np.float64))for x in
    ([0.31,-0.47,0.23,-0.11],[-0.62,0.41,-0.18,0.37],[0.13,0.29,-0.51,0.43],[-0.21,-0.33,0.17,0.59])]
   for probe in probes+fresh:
    with torch.no_grad():expected=trusted(torch.tensor(probe["x"],dtype=torch.float64)).numpy()
    np.testing.assert_allclose(evaluate(model,probe)["output"],expected,atol=1e-12,rtol=1e-12)
   row.update(independent_golden_reference_groups=8,model_sha256=digest(model))
   limits=dict(generations=1,http_attempts=1,wall_seconds=170,api_timeout_seconds=1200,max_tokens=384000)
   task=app.submit(model,options=dict(chunk_period=4,generation_guidance="explicit-v11"),limits=limits)
   row["task_id"]=task
   meta=app._checked_task_metadata(task);request=meta["request"];src=candidate_source(request)[0]
   class RuleReplay:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     require(actual==request and feedback==[],"single_bound_rule_replay")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=src))
   result=app.run(task,RuleReplay());write(dest/"qualification.json",result)
   row.update(qualification_status=result["status"],acceptance=result.get("acceptance"),
    request_id=request["request_id"],candidate_sha256=hashlib.sha256(src.encode()).hexdigest())
   accepted=result.get("acceptance")or{}
   row.update(compiled=accepted.get("compiled")is True,encrypted_execution=accepted.get("encrypted_execution")is True)
   require(result["status"]=="succeeded" and accepted.get("numerically_validated")is True,"qualification_rejected")
   require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"offline_provenance")
   package=app._dir(task)/"program";manifest=check_bundle(package)
   require(manifest["validation"]["level"]=="numerical","numerical_package")
   package_before={n:sha(package/n)for n in manifest["files"]}
   row.update(package=str(package),package_binding=manifest["binding"],parameters=parameters(request),
    native_address_limit_bytes=native_address_limit(),
    artifacts=[n for n in manifest["files"]if n.startswith("artifacts/")])
   payload=[dict(x=p["x"].tolist())for p in fresh]
   expected=np.asarray([evaluate(model,p)["output"].reshape(-1)for p in fresh])
   for index in range(2):
    stable()
    execution=app.execute(task,payload);write(dest/("new-inputs-%d.json"%index),execution)
    require(execution.get("status")=="executed" and execution.get("encrypted_execution")is True,"reload_execution_rejected")
    require(execution["agent_calls"]==execution["compiler_calls"]==0,"reuse_recompiled")
    observed=np.asarray([p["output"]for p in execution["outputs"]],dtype=np.float64).reshape(expected.shape)
    comparison=compare(observed,expected,1e-5,1e-4)
    write(dest/("new-inputs-comparison-%d.json"%index),comparison)
    row["reuse_checks"].append(dict(index=index,evidence=execution["evidence"],input_groups=4,
     compared_values=comparison["compared_values"],max_absolute_error=comparison["max_absolute_error"],
     passed=comparison["passed"],compiler_calls=0,agent_calls=0))
    require(comparison["passed"],"new_input_numerical_rejected")
   require(all(sha(package/n)==h for n,h in package_before.items()),"package_changed_during_reuse")
   require(app.result(task)["status"]=="succeeded","package_delivery_after_reuse")
   row["status"]="passed"
  except Exception as error:
   row.update(status="failed",error_type=type(error).__name__,code=str(error)[:240])
   save("failed");print(json.dumps(dict(case=name,status="failed",code=row["code"])),flush=True);return 1
  stable();save("running")
  print(json.dumps(dict(case=name,status=row["status"],compiled=row["compiled"],encrypted_execution=row["encrypted_execution"],
   reload_checks=len(row["reuse_checks"]))),flush=True)
 stable()
 report.update(compilations=len(rows),encrypted_calls=sum(1+len(r["reuse_checks"])for r in rows),
  input_groups=sum(r["acceptance"]["input_groups"]+sum(t["input_groups"]for t in r["reuse_checks"])for r in rows),
  compared_values=sum(r["acceptance"]["compared_values"]+sum(t["compared_values"]for t in r["reuse_checks"])for r in rows),
  maximum_absolute_error=max([r["acceptance"]["maximum_absolute_error"]for r in rows]+[t["max_absolute_error"]for r in rows for t in r["reuse_checks"]]),
  source_unchanged=True,backend_parameters_unchanged=True)
 save("passed")
 print(json.dumps({k:report[k]for k in ("status","compilations","encrypted_calls","input_groups","compared_values","maximum_absolute_error","seconds","provider_calls")}),flush=True)
 return 0
if __name__=="__main__":
 raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
