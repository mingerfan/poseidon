"""Nine new mathematical contexts through the unchanged Poseidon CPU backend."""
import hashlib,json,os,pathlib,shlex,shutil,sys,time,traceback,subprocess
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE));import planner
sys.path.insert(0,str(HERE.parent/"einsum_live_v1"));import runner_scope as guard
from run import OwnedTree
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
OUT=guard.RESULTS/"agent-einsum-extension-backend-v3"
PRIOR=guard.RESULTS/"agent-einsum-extension-backend-v2"
CASES=[
 ("diagonal_vector","ii->i",[[2,2]],"diagonal"),
 ("diagonal_batch","bii->bi",[[2,2,2]],"diagonal"),
 ("diagonal_matrix_fusion","bii,ij->bj",[[1,2,2],[2,2]],"diagonal"),
 ("trace_scalar","ii->",[[2,2]],"trace"),
 ("trace_batch","...ii->...",[[2,2,2]],"trace"),
 ("trace_scalar_factor","ii,->",[[2,2],[]],"trace"),
 ("scalar_matrix_factor",",ij->ij",[[],[2,2]],"scalar_tensor"),
 ("scalar_vector_reduction","i,->",[[2],[]],"scalar_tensor"),
 ("three_scalar_factors",",,->",[[],[],[]],"scalar_tensor")]
PROFILE="general-tensors-v1"
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
 with pathlib.Path(p).open("x")as stream:json.dump(v,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
def save(p,v):
 temp=pathlib.Path(p).with_suffix(".next")
 with temp.open("x")as stream:json.dump(v,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
 temp.replace(p)
def measure(path):return int(subprocess.check_output(["du","-sb",str(path)],text=True,timeout=20).split()[0])
def sources():
 paths=[HERE/"planner.py",HERE/"test_extension.py",HERE/"controls.py",pathlib.Path(__file__)]
 return {str(p.relative_to(ROOT)):guard.sha(p)for p in paths}
def boundary(plan):
 guard.require(time.time()<plan["deadline_wall"],"offline_time_boundary")
 guard.require(measure(OUT)<=1024**3 and measure(guard.RESULTS)<=32*1024**3 and
  shutil.disk_usage(guard.WORK).free>=4*1024**3,"offline_resource_boundary")
 guard.require(sources()==plan["sources"],"draft_source_frozen");guard.verify()
 before=read(HERE/"before.json")
 for name,expected in before["current_source_hashes"].items():guard.require(guard.sha(ROOT/name)==expected,"frozen_production_source")
 for key in["protected_sources","historical_request_files"]:
  for name,expected in before[key].items():
   p=pathlib.Path(name);p=p if p.is_absolute()else ROOT/p
   guard.require(guard.sha(p)==expected,"protected_backend_reference_history")
def outer():
 guard.verify();guard.require(not OUT.exists(),"fresh_backend_acceptance")
 controls=read(HERE/"controls-round1.json")
 guard.require(controls["status"]=="passed"and controls["sources"]["planner.py"]==guard.sha(HERE/"planner.py"),
  "new_semantics_controls_required")
 guard.require(measure(guard.RESULTS)+1024**3<=32*1024**3 and
  shutil.disk_usage(guard.WORK).free>=5*1024**3,"offline_space_precheck")
 rows=[]
 prior=read(PRIOR/"report.json");guard.require(prior["status"]=="failed","prior_batch_terminal")
 selected_cases={r["case"]for r in prior["rows"]if r["status"]=="failed"and not r["compiled"]and not r["encrypted_execution"]}
 guard.require(len(selected_cases)==6,"exact_six_precompile_failures")
 for name,eq,shapes,partition in CASES:
  if name not in selected_cases:continue
  selected=planner.plan(eq,shapes);high,low,provenance=planner.graph(selected,model_id=name)
  rows.append(dict(case=name,partition=partition,equation=eq,logical_shapes=shapes,plan=selected,high_graph=high,
   lowered_graph=low,provenance=provenance))
 created=time.time();plan=dict(format="poseidon-einsum-extension-backend-plan-v1",cases=rows,
  created_wall=created,deadline_wall=read(PRIOR/"plan.json")["deadline_wall"],prior_plan_binding=read(PRIOR/"plan.json")["binding"],sources=sources(),scope=dict(new_bytes=1024**3,global_bytes=32*1024**3,
  reserve_bytes=4*1024**3,compiler_jobs=2,link_jobs=1,native_concurrency=1,native_address_bytes=4*1024**3,native_cpu_seconds=150),
  provider_calls=0,new_agent_generation=False,production_source_policy=False)
 plan["binding"]=guard.digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([
  str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 tree=OwnedTree().start()
 try:code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 finally:background=tree.finish()
 if code or background:write(OUT/"termination.json",dict(exit_code=code,background_processes=background,automatic_restart=False))
 return code if code else int(background)
def independent(equation,shapes,sample):
 import numpy as np,torch
 values=[torch.from_numpy(np.asarray(sample["input_"+str(i)],dtype=np.float64).reshape(shape))for i,shape in enumerate(shapes)]
 return torch.einsum(equation,*values).numpy().reshape(-1)
def inside():
 guard.require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"pinned_worker")
 import numpy as np,torch
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from candidate_bundle import check_bundle
 from benchmark_graph import samples
 from benchmark_math import evaluate
 from numerical_results import compare
 torch.set_num_threads(1);plan=read(OUT/"plan.json");boundary(plan)
 identity=current_identity_manifest();proposal=guard.verify()
 guard.require(identity==proposal["application_stage_identity"],"actual_unchanged_sdk")
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[];started=time.monotonic()
 report=dict(format="poseidon-einsum-extension-backend-report-v1",status="running",plan_binding=plan["binding"],
  identity_sha256=identity["manifest_sha256"],rows=rows,provider_calls=0,http_attempts=0,new_agent_generation=False,
  production_source_policy=False,tolerance=dict(atol=1e-5,rtol=1e-4))
 def persist(status):report.update(status=status,seconds=time.monotonic()-started);save(OUT/"report.json",report)
 persist("running")
 for item in plan["cases"]:
  boundary(plan);dest=OUT/item["case"];dest.mkdir(mode=0o700)
  row=dict(case=item["case"],partition=item["partition"],status="running",compiled=False,encrypted_execution=False,
   reuse_passed=False,independent_reference=False,candidate_origin="deterministic_rule_baseline")
  rows.append(row);persist("running")
  try:
   selected=planner.plan(item["equation"],item["logical_shapes"]);high,model,origin=planner.graph(selected,model_id=item["case"])
   guard.require(model==item["lowered_graph"]and origin==item["provenance"],"exact_frozen_decomposition")
   reference_probes=samples(model,4)
   for sample in reference_probes:
    expected=independent(item["equation"],item["logical_shapes"],sample)
    np.testing.assert_allclose(evaluate(model,sample)["out"].reshape(-1),expected,atol=1e-10,rtol=1e-10)
   row["independent_reference"]=True
   task=app.submit(model,options=OPTIONS,limits=dict(generations=1,http_attempts=4,wall_seconds=max(1,min(300,int(plan["deadline_wall"]-time.time()))),api_timeout_seconds=1200,max_tokens=384000))
   row["task_id"]=task;state=app.status(task);guard.require(state["status"]=="queued","application_graph_submission")
   request=app._checked_task_metadata(task)["request"];code=candidate_source(request)[0]
   (dest/"rule_candidate.py").write_text(code);row["candidate_sha256"]=guard.sha(dest/"rule_candidate.py")
   class Replay:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     guard.require(actual==request and feedback==[],"single_offline_candidate")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=code))
   result=app.run(task,Replay());write(dest/"qualification.json",result);accepted=result.get("acceptance")or{}
   row.update(qualification_status=result["status"],acceptance=accepted,compiled=accepted.get("compiled")is True,
    encrypted_execution=accepted.get("encrypted_execution")is True)
   guard.require(result["status"]=="succeeded"and accepted.get("numerically_validated")is True,"backend_qualification")
   guard.require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"offline_origin")
   program=app._dir(task)/"program";manifest=check_bundle(program)
   guard.require(manifest["validation"]["level"]=="numerical","numerical_program_package")
   hashes={name:guard.sha(program/name)for name in manifest["files"]}
   row.update(package=str(program),package_binding=manifest["binding"],
    artifacts=[name for name in manifest["files"]if name.startswith("artifacts/")])
   probes=[{name:np.asarray(v,dtype=np.float64)*.5+.015 for name,v in sample.items()}for sample in reference_probes]
   execution=app.execute(task,[{name:v.tolist()for name,v in sample.items()}for sample in probes])
   write(dest/"new-input-execution.json",execution)
   guard.require(execution.get("status")=="executed"and execution.get("encrypted_execution")is True,"reloaded_execution")
   guard.require(execution["agent_calls"]==execution["compiler_calls"]==0,"program_reuse_no_generation")
   expected=np.asarray([independent(item["equation"],item["logical_shapes"],sample)for sample in probes])
   actual=np.asarray([np.asarray(sample["out"],dtype=np.float64).reshape(-1)for sample in execution["outputs"]])
   comparison=compare(actual,expected,1e-5,1e-4);write(dest/"independent-numerical-comparison.json",comparison)
   guard.require(comparison["passed"],"independent_numerical_result")
   guard.require(all(guard.sha(program/name)==expected_hash for name,expected_hash in hashes.items()),"package_unchanged")
   guard.require(app.result(task)["status"]=="succeeded","package_delivery")
   row.update(status="passed",reuse_passed=True,reuse_input_groups=4,reuse_values=comparison["compared_values"],
    reuse_maximum_absolute_error=comparison["max_absolute_error"],reuse_compiler_calls=0,reuse_agent_calls=0)
  except Exception as error:
   write(dest/"failure.json",dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
    provider_calls=0,automatic_retry=False))
   row.update(status="failed",error_type=type(error).__name__,code=str(error)if len(str(error))<200 else"offline_exception")
  persist("running");print(json.dumps({key:row[key]for key in["case","status","compiled","encrypted_execution"]}),flush=True)
  boundary(plan);guard.require(current_identity_manifest()==identity,"sdk_identity_changed")
 report.update(planned=len(rows),passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),
  source_captures=0,compilations=sum(r["compiled"]for r in rows),
  encrypted_calls=sum(int(r["encrypted_execution"])+int(r["reuse_passed"])for r in rows),
  input_groups=sum(r.get("acceptance",{}).get("input_groups",0)+r.get("reuse_input_groups",0)for r in rows),
  compared_values=sum(r.get("acceptance",{}).get("compared_values",0)+r.get("reuse_values",0)for r in rows),
  maximum_absolute_error=max([r.get("acceptance",{}).get("maximum_absolute_error",0)for r in rows]+
   [r.get("reuse_maximum_absolute_error",0)for r in rows]),protected_sources_changed=0)
 persist("passed"if report["failed"]==0 else"failed")
 report["binding"]=guard.digest({k:v for k,v in report.items()if k!="binding"});save(OUT/"report.json",report)
 print(json.dumps({key:report[key]for key in["status","passed","failed","compilations","encrypted_calls","input_groups",
  "compared_values","maximum_absolute_error","seconds","provider_calls"]}),flush=True)
 return 0 if report["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
