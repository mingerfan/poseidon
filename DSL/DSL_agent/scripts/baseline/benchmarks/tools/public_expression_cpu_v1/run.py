"""Exact approved CPU batch. Rule candidates are never reported as Agent output."""
import os,sys,shlex,json,time,traceback,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
from scope import *
def inside():
    import numpy as np
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from candidate_bundle import check_bundle
    from benchmark_graph import samples
    from operator_reference import evaluate as original_math
    from numerical_results import compare
    plan=verify(read(PROPOSAL));authorization(plan)
    launch=read(OUT/"launch.json");deadline=launch["deadline"]
    require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned CPU batch")
    rows=[];report=dict(format="poseidon-public-expression-cpu-report-v1",proposal_binding=plan["binding"],
                        rows=rows,provider_calls=0,http_attempts=0,new_agent_generation=False,
                        candidate_origin="deterministic_rule",status="running")
    def save(status):
        report.update(status=status,seconds=time.time()-launch["started_wall"],
                      passed=sum(r["status"]=="passed"for r in rows),
                      failed=sum(r["status"]=="failed"for r in rows),not_run=len(CASES)-len(rows))
        path=OUT/"report.json";path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    save("running")
    for item in plan["cases"]:
        boundary(plan,deadline);task=item["task_id"];directory=OUT/item["case"]
        app=application(WorkerContext.capture(item["task_store"]))
        row=dict(case=item["case"],status="running",compiled=False,encrypted_execution=False,
                 package_reloaded=False,repeated=False,failure_layer=None);rows.append(row);save("running")
        try:
            meta=app._checked_task_metadata(task);request=meta["request"];model=request["model"]
            require(digest(request)==item["request_sha256"],"Frozen component request")
            source=Path(item["candidate"]).read_text()
            class Rule:
                kind="scripted_replay";agent_calls=0
                def generate(self,actual,feedback=None):
                    require(actual==request and feedback==[],"One frozen rule candidate")
                    return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
            result=app.run(task,Rule());write(directory/"qualification.json",result)
            acceptance=result.get("acceptance")or{}
            row.update(acceptance=acceptance,compiled=acceptance.get("compiled")is True,
                       encrypted_execution=acceptance.get("encrypted_execution")is True,
                       failure_layer=(result.get("failure")or{}).get("layer"))
            require(result["status"]=="succeeded"and acceptance.get("numerically_validated")is True,"CPU numerical acceptance")
            require(result["http_attempts"]==0 and result["generation"]["agent_generated"]is False,"Scripted origin")
            require(result["package"]["reused"]is False,"Fresh case compile and numerical qualification")
            package=app._dir(task)/"program";manifest=check_bundle(package)
            require(manifest["validation"]["level"]=="numerical"and
                    any(n.endswith("runtime-plan.json")for n in manifest["files"]),"Numerical RuntimePlan package")
            package_files={n:sha(package/n)for n in manifest["files"]}
            row.update(package=str(package),package_binding=manifest["binding"],package_reloaded=True)
            if item["case"]in REUSE:
                probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in b.items()}for b in samples(model,4)[:3]]
                for suffix,selected in(("new-inputs",probes),("repeat",probes[:1])):
                    boundary(plan,deadline)
                    expected=np.asarray([np.concatenate([original_math(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1)
                                                         for o in model["outputs"]])for b in selected])
                    execution=app.execute(task,[{k:v.tolist()for k,v in b.items()}for b in selected])
                    write(directory/(suffix+".json"),execution)
                    require(execution["status"]=="executed"and execution["encrypted_execution"]is True and
                            execution["agent_calls"]==execution["compiler_calls"]==0,"Verified package execution")
                    actual=np.asarray([np.concatenate([np.asarray(b[o["name"]],dtype=np.float64).reshape(-1)
                                                       for o in model["outputs"]])for b in execution["outputs"]])
                    comparison=compare(actual,expected,1e-5,1e-4);write(directory/(suffix+"-comparison.json"),comparison)
                    require(comparison["passed"],"Independent package numerical comparison")
                    row[suffix.replace("-","_")]=dict(groups=len(selected),values=comparison["compared_values"],
                                                      maximum_absolute_error=comparison["max_absolute_error"])
                row["repeated"]=True
            require(check_bundle(package)==manifest and all(sha(package/n)==h for n,h in package_files.items()),"Package immutable")
            row["status"]="passed"
        except Exception as error:
            row.update(status="failed",error=type(error).__name__,reason=str(error)[:180])
            (directory/"failure-private.txt").write_text(traceback.format_exc())
        save("running")
        boundary(plan,deadline)
        require(row["failure_layer"]not in("environment","integrity","resources"),"Shared failure stops related batch")
    save("complete"if all(r["status"]=="passed"for r in rows)else"failed")
    report["binding"]=digest(report);(OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return 0 if report["status"]=="complete"else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",action="store_true");parser.add_argument("--inside",action="store_true")
    args=parser.parse_args()
    if args.plan:
        plan=verify(read(PROPOSAL),sdk=False)
        print(json.dumps(dict(binding=plan["binding"],cases=list(CASES),scope=SCOPE,approved=AUTHORIZATION.exists())));return 0
    if args.inside:return inside()
    plan=verify(read(PROPOSAL));authorization(plan)
    require(not(OUT/"launch.json").exists()and not(OUT/"report.json").exists(),"No duplicate batch or old evidence reuse")
    now=time.time();write(OUT/"launch.json",dict(proposal_binding=plan["binding"],started_wall=now,deadline=now+SCOPE["wall_seconds"]))
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    return enter_nix(command,seconds=SCOPE["wall_seconds"],keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(main())
