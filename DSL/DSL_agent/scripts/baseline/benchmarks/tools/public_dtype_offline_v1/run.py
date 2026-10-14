"""One pinned environment, independent sandbox per fixed rule candidate."""
import os,sys,shlex,time,argparse,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parents[2]))
from hecate_python_env import enter_nix,VENV
from scope import *
def inside():
 import numpy as np
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from candidate_bundle import check_bundle
 from benchmark_graph import samples
 from operator_reference import evaluate
 from numerical_results import compare
 plan=verify(read(PROPOSAL));authorization(plan);launch=read(OUT/"launch.json")
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"dtype_pinned_environment")
 rows=[dict(case=x["case"],status="not_run",compiled=False,encrypted_execution=False,
  compiler_attempts_reserved=0,encrypted_attempts_reserved=0,encrypted_calls_confirmed=0,
  package_reloaded=False,new_inputs_executed=False,repeated_input_executed=False)for x in plan["cases"]]
 report=dict(format="poseidon-public-dtype-cpu-report-v1",proposal_binding=plan["binding"],rows=rows,
  provider_calls=0,http_attempts=0,agent_generated=False,candidate_origin="deterministic_rule")
 def save(status):
  report.update(status=status,seconds=time.time()-launch["started_wall"],
   passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),
   not_run=sum(r["status"]=="not_run"for r in rows),skipped=0,
   compiler_calls_confirmed=sum(r["compiled"]for r in rows),
   encrypted_calls_confirmed=sum(r["encrypted_calls_confirmed"]for r in rows))
  report.pop("binding",None);report["binding"]=digest(report)
  (OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 def limits():
  require(sum(r["compiler_attempts_reserved"]for r in rows)<=SCOPE["maximum_compiler_calls"]and
   sum(r["encrypted_attempts_reserved"]for r in rows)<=SCOPE["maximum_encrypted_calls"],"dtype_call_budget")
  return boundary(plan,launch)
 save("running");stopped=False
 for item,row in zip(plan["cases"],rows):
  directory=OUT/item["case"];task=item["task_id"]
  try:limits()
  except Exception as exc:
   report["stop_reason"]=dict(error=type(exc).__name__,reason=str(exc)[:180]);stopped=True;break
  try:
   row["status"]="running";save("running")
   app=application(WorkerContext.capture(item["task_store"]))
   meta=app._checked_task_metadata(task);request=meta["request"];source=Path(item["candidate"]).read_text()
   require(request==read(item["request"])and request["request_id"]==item["request_id"],"dtype_frozen_request")
   class Rule:
    kind="scripted_replay";agent_calls=0
    def generate(self,actual,feedback=None):
     require(actual==request and feedback==[],"dtype_single_unchanged_rule")
     return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
   row.update(compiler_attempts_reserved=1,encrypted_attempts_reserved=1);limits()
   q=app.run(task,Rule());write(directory/"qualification.json",q)
   validation_path=app._dir(task)/"validation-0.json"
   if validation_path.exists():
    v=read(validation_path);row["observed_stages"]=v.get("observed_stages",{})
    row["validation_evidence"]=v.get("evidence")
    row["compiled"]=row["observed_stages"].get("compiled")is True
    row["encrypted_execution"]=v.get("encrypted_execution")is True
    row["encrypted_calls_confirmed"]=int(row["encrypted_execution"])
   row["failure_layer"]=(q.get("failure")or{}).get("layer")
   row["acceptance"]=q.get("acceptance")or{}
   require(q["status"]=="succeeded"and all(row["acceptance"].get(k)is True for k in
    ("compiled","artifact_checked","encrypted_execution","numerically_validated"))and
    q["acceptance"]["input_groups"]==4,"dtype_full_numerical_acceptance")
   require(q["http_attempts"]==0 and not q["generation"]["agent_generated"]and
    not q["package"]["reused"],"dtype_fresh_rule_origin")
   row.update(compiled=True,encrypted_execution=True,encrypted_calls_confirmed=1)
   folder=app._dir(task)/"program";manifest=check_bundle(folder)
   require(manifest["validation"]["level"]=="numerical"and
    any(n.endswith("runtime-plan.json")for n in manifest["files"]),"dtype_runtime_plan_package")
   frozen={n:sha(folder/n)for n in manifest["files"]}
   row.update(package_reloaded=True,package=str(folder),package_binding=manifest["binding"])
   if item["case"]in REUSE:
    probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in b.items()}for b in samples(request["model"],4)[:3]]
    for suffix,batches in(("new-inputs",probes),("repeat",probes[:1])):
     row["encrypted_attempts_reserved"]+=1;limits()
     output=app.execute(task,[{k:v.tolist()for k,v in b.items()}for b in batches])
     write(directory/(suffix+".json"),output)
     row["encrypted_calls_confirmed"]+=int(output.get("encrypted_execution")is True)
     require(output["status"]=="executed"and output["encrypted_execution"]is True and
      output["agent_calls"]==output["compiler_calls"]==0 and not output["executor_secret_key"],"dtype_package_execution")
     if len(request["model"]["outputs"])==2:
      require(all(list(o)==["primary","saved"]and o["saved"][1]is None for o in output["structured_outputs"]),
       "dtype_named_none_result")
     expected=np.asarray([np.concatenate([evaluate(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1)
      for o in request["model"]["outputs"]])for b in batches])
     actual=np.asarray([np.concatenate([np.asarray(b[o["name"]],dtype=np.float64).reshape(-1)
      for o in request["model"]["outputs"]])for b in output["outputs"]])
     comparison=compare(actual,expected,1e-5,1e-4);write(directory/(suffix+"-comparison.json"),comparison)
     require(comparison["passed"],"dtype_independent_package_comparison")
     row[suffix.replace("-","_")]=dict(groups=len(batches),values=comparison["compared_values"],
      maximum_absolute_error=comparison["max_absolute_error"])
    row.update(new_inputs_executed=True,repeated_input_executed=True)
   require(check_bundle(folder)==manifest and all(sha(folder/n)==h for n,h in frozen.items()),"dtype_immutable_package")
   row["status"]="passed";report["resources"]=limits()
  except Exception as exc:
   row.update(status="failed",error=type(exc).__name__,reason=str(exc)[:180])
   (directory/"failure-private.txt").write_text(traceback.format_exc())
   stopped=(row.get("failure_layer")in("environment","integrity","resources","worker")or
    str(exc).startswith(("dtype_current","dtype_tool","dtype_frozen","dtype_task","dtype_wall","dtype_disk","dtype_call")))
  save("running")
  if stopped:break
 save("complete"if all(r["status"]=="passed"for r in rows)else"failed")
 return 0 if report["status"]=="complete"else 1
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--plan",action="store_true");parser.add_argument("--inside",action="store_true")
 args=parser.parse_args();plan=verify(read(PROPOSAL))
 if args.plan:
  print(json.dumps(dict(binding=plan["binding"],cases=len(plan["cases"]),unique_requests=plan["unique_requests"],
   input_groups=plan["planned_input_groups"],output_values=plan["planned_output_values"],scope=SCOPE,approved=AUTH.exists())))
  return 0
 if args.inside:return inside()
 authorization(plan)
 require(not(OUT/"launch.json").exists()and not(OUT/"report.json").exists(),"dtype_no_duplicate_or_unknown_retry")
 for item in plan["cases"]:
  require(read(Path(item["task_store"])/item["task_id"]/"state.json")["status"]=="queued","dtype_prepared_queued_tasks")
 now=time.time();baseline=measure(WORK/"results")
 require(baseline+SCOPE["new_bytes"]<=SCOPE["global_bytes"]and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"dtype_disk_preflight")
 launch=dict(proposal_binding=plan["binding"],started_wall=now,deadline=now+SCOPE["wall_seconds"],
  baseline_results_bytes=baseline)
 write(OUT/"launch.json",launch)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=max(1,int(launch["deadline"]-time.time())),keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(main())
