"""Frozen offline source/compile/numerical/package acceptance; no API calls."""
import json,os,pathlib,hashlib,shlex,shutil,sys,time,traceback,subprocess
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE));os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
WORK=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux");RESULTS=WORK/"results"
PREVIOUS=RESULTS/"agent-einsum-source-acceptance-v1"
OUT=RESULTS/"agent-exact-arithmetic-v21-source-acceptance-v1"
PRIOR=RESULTS/"agent-static-partition-acceptance-v2"
SCOPE=dict(wall_seconds=900,new_bytes=1024**3,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
 compiler_jobs=2,link_jobs=1,native_concurrency=1,native_address_bytes=4*1024**3,native_cpu_seconds=150)
OPTIONS=dict(capacity_packing_version="capacity-public-tensors-v9",generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
PROFILE="general-tensors-v1"
def require(ok,code):
 if not ok:raise ValueError(code)
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write(p,v):
 raw=json.dumps(v,sort_keys=True,allow_nan=False).encode();path=pathlib.Path(p);temp=path.with_suffix(path.suffix+".next")
 with temp.open("xb")as f:f.write(raw)
 temp.replace(path)
def measure(p):
 if not pathlib.Path(p).exists():return 0
 return int(subprocess.run(["du","-sb",str(p)],capture_output=True,text=True,check=True,timeout=20).stdout.split()[0])
def protected():
 prior=read(HERE/"before.json")
 for key in ["protected_sources","historical_request_files"]:
  for n,h in prior[key].items():
   p=pathlib.Path(n)if pathlib.Path(n).is_absolute()else ROOT/n
   require(sha(p)==h,"protected_backend_reference_or_request_changed")
 return len(prior["protected_sources"])
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"offline_window_exhausted")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 added=[p for p in RESULTS.iterdir()if p.name not in plan["existing_results"]]
 require(sum(measure(p)for p in added)<=SCOPE["new_bytes"],"new_artifact_budget")
 for n,h in plan["sources"].items():require(sha(ROOT/n)==h,"frozen_agent_source_changed")
 for n,h in plan["fixtures"].items():require(sha(ROOT/n)==h,"frozen_fixture_changed")
 protected()
def outer():
 require(not OUT.exists(),"fresh_offline_directory")
 for name,count in [("fixed",72),("compatibility",225)]:
  v=read(HERE/("controls-"+name+".json"))
  require(v["status"]=="passed"and v["tests_run"]==count and v["failed"]==v["errors"]==v["skipped"]==0,"controls_required")
  for n,h in v["tested_sources"].items():require(sha(BASE/n)==h,"controls_source_changed")
 old=read(PRIOR/"identity.json")
 names={"scripts/baseline/"+n for stage in ["generation","compilation","execution","validation"]for n in old[stage]["sources"]}
 names.add("scripts/baseline/pytorch_einsum_normalization.py")
 names.add("scripts/baseline/pytorch_einsum_extension.py")
 names.update(str(p.relative_to(ROOT))for p in BASE.glob("*.py"))
 names.add("scripts/prepare_pytorch_source.py")
 names.add(str(pathlib.Path(__file__).relative_to(ROOT)))
 for stage in ["generation","compilation","execution","validation"]:names.update(old[stage]["configuration_files"])
 fixtures={str(p.relative_to(ROOT)):sha(p)for p in(HERE/"fixtures").rglob("*")if p.is_file()}
 require(measure(RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"]and
         shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"space_precheck")
 cli=read(BASE/"benchmarks/tools/source_manifest_v1/controls-v21-compatibility.json")
 require(cli["status"]=="passed"and cli["tests_run"]==59 and cli["failed"]==cli["errors"]==cli["skipped"]==0,"cli_controls_required")
 require(cli["source_sha256"]==sha(ROOT/"scripts/prepare_pytorch_source.py"),"cli_source_changed")
 created=time.time()
 plan=dict(format="poseidon-exact-arithmetic-offline-plan-v1",scope=SCOPE,cases=read(HERE/"fixtures.json"),
  sources={n:sha(ROOT/n)for n in sorted(names)},fixtures=fixtures,
   existing_results=[p.name for p in RESULTS.iterdir()],created_wall=created,
   deadline_wall=created+SCOPE["wall_seconds"],
  protected_sources_checked=protected(),provider_calls=0,candidate_origin="rule_baseline",
  controls_sha256={n:sha(HERE/("controls-"+n+".json"))for n in["fixed","compatibility"]})
 plan["binding"]=digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 code=enter_nix(cmd,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 if code not in(0,1):write(OUT/"termination.json",dict(status="timed_out_or_worker_exit",code=code,automatic_retry=False,provider_calls=0))
 return code
def inside():
 require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"locked_worker")
 import numpy as np,torch
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from unified_graph_lowering import candidate_source
 from candidate_bundle import check_bundle
 from benchmark_graph import samples
 from benchmark_math import evaluate
 from numerical_results import compare
 from application_policy import budget
 torch.set_num_threads(1);plan=read(OUT/"plan.json");boundary(plan)
 identity=current_identity_manifest();write(OUT/"identity.json",identity)
 actual={"scripts/baseline/"+n for stage in ["generation","compilation","execution","validation"]for n in identity[stage]["sources"]}
 require(actual<=set(plan["sources"]),"unfrozen_identity_source")
 budget(dict(generations=1,http_attempts=1,wall_seconds=600,api_timeout_seconds=1200,max_tokens=384000))
 app=application(WorkerContext.capture(OUT/"tasks"));started=time.monotonic();rows=[]
 report=dict(format="poseidon-exact-arithmetic-offline-report-v1",status="running",plan_binding=plan["binding"],
  identity_sha256=identity["manifest_sha256"],rows=rows,provider_calls=0,http_attempts=0,new_agent_generation=False,
  source_format="explicit v21 with legacy v20/v18 regressions",tolerance=dict(atol=1e-5,rtol=1e-4))
 def save(status):report.update(status=status,seconds=time.monotonic()-started);write(OUT/"report.json",report)
 save("running")
 for item in plan["cases"]:
  boundary(plan);dest=OUT/item["case"];dest.mkdir(mode=0o700)
  row=dict(case=item["case"],status="running",source_captured=False,compiled=False,encrypted_execution=False,
           reuse_passed=False,candidate_origin="deterministic_rule_baseline");rows.append(row);save("running")
  try:
   source=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",manifest=item["manifest"],trust_local_code=True)
   task=app.submit_source(source,resource_profile=PROFILE,options=OPTIONS,
                         limits=dict(generations=1,http_attempts=1,wall_seconds=max(1,min(600,int(plan["deadline_wall"]-time.time()))),
                                     api_timeout_seconds=1200,max_tokens=384000))
   row["task_id"]=task
   state=app.prepare_source(task);write(dest/"source-preparation.json",state)
   require(state["status"]=="queued"and state["phase"]=="queued"and
           state.get("intake",{}).get("reference_verified")is True,"source_intake_rejected")
   row["source_captured"]=True;meta=app._checked_task_metadata(task);request=meta["request"];model=request["model"]
   code=candidate_source(request)[0];(dest/"rule_candidate.py").write_text(code)
   row.update(request_id=request["request_id"],candidate_sha256=hashlib.sha256(code.encode()).hexdigest())
   class Replay:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     require(actual==request and feedback==[],"single_scripted_candidate")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=code))
   result=app.run(task,Replay());write(dest/"qualification.json",result);accepted=result.get("acceptance")or{}
   row.update(qualification_status=result["status"],acceptance=accepted,
              compiled=accepted.get("compiled")is True,encrypted_execution=accepted.get("encrypted_execution")is True)
   require(result["status"]=="succeeded"and accepted.get("numerically_validated")is True,"qualification_rejected")
   require(result["generation"]["agent_generated"]is False and result["http_attempts"]==0,"offline_origin")
   package=app._dir(task)/"program";manifest=check_bundle(package)
   require(manifest["validation"]["level"]=="numerical","numerical_package_required")
   hashes={n:sha(package/n)for n in manifest["files"]}
   row.update(package=str(package),package_binding=manifest["binding"],
              artifacts=[n for n in manifest["files"]if n.startswith("artifacts/")])
   probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in x.items()}for x in samples(model,4)]
   execution=app.execute(task,[{k:v.tolist()for k,v in x.items()}for x in probes]);write(dest/"new-input-execution.json",execution)
   require(execution.get("status")=="executed"and execution.get("encrypted_execution")is True,"reload_execution_rejected")
   require(execution["agent_calls"]==execution["compiler_calls"]==0,"reuse_recompiled")
   expected=np.asarray([np.concatenate([v.reshape(-1)for v in evaluate(model,x).values()])for x in probes])
   observed=np.asarray([np.concatenate([np.asarray(v).reshape(-1)for v in x.values()])for x in execution["outputs"]])
   comparison=compare(observed,expected,1e-5,1e-4);write(dest/"new-input-comparison.json",comparison)
   require(comparison["passed"],"new_input_numerical_rejected")
   require(all(sha(package/n)==h for n,h in hashes.items()),"package_changed_during_reuse")
   require(app.result(task)["status"]=="succeeded","program_delivery_rejected")
   row.update(status="passed",reuse_passed=True,reuse_input_groups=4,reuse_values=comparison["compared_values"],
              reuse_maximum_absolute_error=comparison["max_absolute_error"],reuse_compiler_calls=0,reuse_agent_calls=0)
  except Exception as error:
   with(dest/"failure-private.txt").open("x")as f:f.write(traceback.format_exc())
   code=str(error)
   row.update(status="failed",error_type=type(error).__name__,
              code=code if code.replace("_","").isalnum()and len(code)<120 else"offline_case_exception")
  save("running");print(json.dumps({k:row[k]for k in["case","status","source_captured","compiled","encrypted_execution"]}),flush=True)
  boundary(plan);require(current_identity_manifest()==identity,"SDK_source_changed")
 report.update(passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),
  planned_cases=len(rows),protected_sources_checked=protected(),protected_sources_changed=0,
  source_captures=sum(r["source_captured"]for r in rows),compilations=sum(r["compiled"]for r in rows),
  encrypted_calls=sum(int(r["encrypted_execution"])+int(r["reuse_passed"])for r in rows),
  input_groups=sum(r.get("acceptance",{}).get("input_groups",0)+r.get("reuse_input_groups",0)for r in rows),
  compared_values=sum(r.get("acceptance",{}).get("compared_values",0)+r.get("reuse_values",0)for r in rows),
  maximum_absolute_error=max([r.get("acceptance",{}).get("maximum_absolute_error",0)for r in rows]+
                            [r.get("reuse_maximum_absolute_error",0)for r in rows]))
 save("passed"if report["failed"]==0 else"failed")
 report["binding"]=digest({k:v for k,v in report.items()if k!="binding"});write(OUT/"report.json",report)
 print(json.dumps({k:report[k]for k in["status","passed","failed","source_captures","compilations","encrypted_calls",
                  "input_groups","compared_values","maximum_absolute_error","seconds","provider_calls"]}),flush=True)
 return 0 if report["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
