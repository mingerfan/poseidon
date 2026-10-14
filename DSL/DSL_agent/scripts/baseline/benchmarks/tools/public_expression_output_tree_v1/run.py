"""One approved requalification and two verified-package executions."""
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
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned one-case qualification")
 app=application(WorkerContext.capture(plan["task_store"]));task=plan["task_id"]
 request=app._checked_task_metadata(task)["request"];source=Path(plan["candidate"]).read_text()
 report=dict(format="poseidon-public-expression-output-cpu-report-v1",proposal_binding=plan["binding"],
  status="running",provider_calls=0,http_attempts=0,agent_generated=False,compiler_calls=0,encrypted_calls=0)
 def boundary_now():boundary(plan,launch["deadline"],launch["baseline_results_bytes"])
 try:
  boundary_now()
  class Rule:
   kind="scripted_replay";agent_calls=0
   def generate(self,actual,feedback=None):
    require(actual==request and feedback==[],"One unchanged rule candidate")
    return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
  q=app.run(task,Rule());write(OUT/"qualification.json",q)
  require(q["status"]=="succeeded"and q["acceptance"]["numerically_validated"]is True and
   q["acceptance"]["encrypted_execution"]is True and q["http_attempts"]==0 and
   not q["package"]["reused"]and not q["generation"]["agent_generated"],"One-case fresh numerical qualification")
  report.update(compiler_calls=1,encrypted_calls=1,acceptance=q["acceptance"])
  folder=app._dir(task)/"program";manifest=check_bundle(folder)
  fixed={n:sha(folder/n)for n in manifest["files"]}
  meta=app._checked_task_metadata(task)
  probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in b.items()}for b in samples(request["model"],4)[:3]]
  for name,batches in (("new-inputs",probes),("repeat",probes[:1])):
   boundary_now()
   output=app.execute(task,[{k:v.tolist()for k,v in b.items()}for b in batches])
   write(OUT/(name+".json"),output)
   require(output["status"]=="executed"and output["encrypted_execution"]is True and
    output["agent_calls"]==output["compiler_calls"]==0 and not output["executor_secret_key"],
    "Fixed program returned executed")
   require(all(list(v)==["primary","saved"]and v["saved"][1]is None for v in output["structured_outputs"]),
    "Bound dictionary/list/None returned")
   report["encrypted_calls"]+=1
   expected=np.asarray([np.concatenate([evaluate(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1)
    for o in request["model"]["outputs"]])for b in batches])
   actual=np.asarray([np.concatenate([np.asarray(b[o["name"]],dtype=np.float64).reshape(-1)
    for o in request["model"]["outputs"]])for b in output["outputs"]])
   comparison=compare(actual,expected,1e-5,1e-4);write(OUT/(name+"-comparison.json"),comparison)
   require(comparison["passed"],"Independent unchanged numerical comparison")
   report[name.replace("-","_")]=dict(groups=len(batches),values=comparison["compared_values"],
    maximum_absolute_error=comparison["max_absolute_error"])
  require(check_bundle(folder)==manifest and all(sha(folder/n)==h for n,h in fixed.items()),"Fixed program immutable")
  report.update(status="passed",package_binding=manifest["binding"],package=str(folder),reloaded=True)
 except Exception as exc:
  report.update(status="failed",error=type(exc).__name__,reason=str(exc)[:160])
  (OUT/"failure-private.txt").write_text(traceback.format_exc())
 report.update(seconds=time.time()-launch["started_wall"]);report["binding"]=digest(report)
 write(OUT/"report.json",report);boundary_now()
 return 0 if report["status"]=="passed"else 1
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--plan",action="store_true");parser.add_argument("--inside",action="store_true")
 args=parser.parse_args();plan=verify(read(PROPOSAL))
 if args.plan:
  print(json.dumps(dict(binding=plan["binding"],case=plan["case"],scope=SCOPE,approved=AUTH.exists())));return 0
 if args.inside:return inside()
 authorization(plan)
 require(not(OUT/"launch.json").exists()and not(OUT/"report.json").exists(),"No duplicate or historic retry")
 now=time.time();baseline=measure(WORK/"results");boundary(plan,now+60,baseline)
 write(OUT/"launch.json",dict(proposal_binding=plan["binding"],started_wall=now,deadline=now+240,
  baseline_results_bytes=baseline))
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=240,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(main())
