"""Bounded developer acceptance of reusable packages and pre-encryption domains.

No provider calls. Application execution is used unchanged; experiment identities
and example names remain in this developer tool, never in the application path.
"""
import argparse,contextlib,hashlib,io,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(index):
    from benchmark_runner import strict_file
    from benchmark_graph import NAME,require
    raw=strict_file(index,131072)
    require(type(raw) is dict and set(raw)=={"format","cases"} and
            raw["format"]=="poseidon-execution-domain-cases-v1","Domain case index")
    require(type(raw["cases"]) is list and 1<=len(raw["cases"])<=48,"Domain case count")
    cases=[]
    for value in raw["cases"]:
        require(type(value) is dict and set(value)=={"id","source_kind","source","options"} and
                type(value["id"]) is str and NAME.fullmatch(value["id"]),"Domain case fields")
        require(value["source_kind"] in ("model_import","graph") and
                type(value["source"]) is str and type(value["options"]) is dict,"Domain case source")
        source=Path(value["source"]).resolve()
        if value["source_kind"]=="model_import":
            require(source.is_dir() and not Path(value["source"]).is_symlink(),"Domain import location")
            files={str(p.relative_to(source)):sha(p) for p in source.rglob("*")
                   if p.is_file() and p.suffix in (".json",".py",".npy")}
        else:
            require(source.is_file() and not Path(value["source"]).is_symlink(),"Domain graph location")
            files={source.name:sha(source)}
        cases.append(dict(**value,source_hashes=files))
    require(len({v["id"] for v in cases})==len(cases),"Domain case identifiers")
    return cases


def run(index,output,parent_result=None,max_seconds=None):
    import copy,numpy as np
    from hecate_python_env import WORK
    from benchmark_graph import canonical,digest,require
    from semantic_benchmark_execution import runtime_sources
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from candidate_bundle import load_model_provenance,check_execution_dependencies,replay_bundle
    from candidate_bundle import main as bundle_main
    from model_import_cli import main as import_main
    from component_contract import prepare_task
    from unified_graph_lowering import candidate_source
    from benchmark_math import evaluate as math_reference
    from benchmark_torch import evaluate as torch_reference
    from numerical_results import compare
    from program_execution import normalize_samples
    from execution_input_domain import check_approximation_domains,ApproximationDomainRejected
    from check_pytorch_sources import check_source_boundary
    results=WORK/"results";cases=inventory(index)
    require(not output.exists() and output.parent.resolve()==results.resolve(),"Fresh domain result")
    output.mkdir(mode=0o700)
    started=time.monotonic();before_dirs=set(results.iterdir());artifacts={output}
    context=WorkerContext.capture(output/"tasks");app=application(context)
    source_hashes=runtime_sources();identity=app.backend.identity()
    parent=None
    if parent_result is not None:
        from benchmark_runner import strict_file
        old_plan=strict_file(parent_result/"plan.json",4*1024**2)
        old_report=strict_file(parent_result/"report.json",4*1024**2)
        terminal=strict_file(parent_result/"terminal-observation.json",131072)
        require(old_report["status"]=="stopped" and old_report["paid_calls"]==0 and
                old_report["seconds"]<600 and terminal["outer_exit_code"]==1 and
                terminal["active_processes"]==[] and
                terminal["parent_report_sha256"]==sha(parent_result/"report.json") and
                old_plan["cases"]==cases and old_plan["backend_identity"]["tools"]==identity["tools"] and
                sha(parent_result/"offline-tool-original.py")==old_plan["tool_sha256"],
                "Domain corrected-run lineage")
        parent=dict(result=str(parent_result),binding=old_plan["binding"],
                    plan_sha256=sha(parent_result/"plan.json"),
                    report_sha256=sha(parent_result/"report.json"),
                    execution_seconds=old_report["seconds"],
                    reason="execution response field correction; historical outcomes retained")
    plan=dict(format="poseidon-execution-domain-offline-plan-v1",cases=cases,
        index_sha256=sha(index),tool_sha256=sha(Path(__file__)),source_hashes=source_hashes,
        backend_identity=identity,worker_binding=context.binding,max_seconds=max_seconds if max_seconds is not None else (2700 if parent else 3300),
        artifact_bytes=4*1024**3,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
        compiler_jobs=2,link_jobs=1,encrypted_jobs=1,source="rule_baseline",
        agent_generated=False,paid_calls=0,atol=1e-5,rtol=1e-4)
    if parent is not None:plan["parent"]=parent
    plan["binding"]=digest(plan);(output/"plan.json").write_bytes(canonical(plan));context.persist()
    rows=[]
    def boundary():
        check_source_boundary(context,app.backend,source_hashes,identity)
        require(sha(index)==plan["index_sha256"] and
                inventory(index)==cases and sha(Path(__file__))==plan["tool_sha256"],"Domain source drift")
        require(time.monotonic()-started<plan["max_seconds"],"Domain wall budget")
        artifacts.update(set(results.iterdir())-before_dirs)
        require(sum(p.stat().st_size for d in artifacts for p in d.rglob("*") if p.is_file())
                <=plan["artifact_bytes"],"Domain artifact budget")
        require(shutil.disk_usage(WORK).free>=plan["reserve_bytes"],"Domain disk reserve")
        require(int(subprocess.check_output(["du","-sb",str(results)],text=True,timeout=30).split()[0])
                <=plan["global_bytes"],"Domain global budget")
    def save(status):
        record=dict(format="poseidon-execution-domain-offline-report-v1",binding=plan["binding"],
            status=status,planned=len(cases),passed=sum(v["status"]=="passed" for v in rows),
            failed=sum(v["status"]=="failed" for v in rows),not_run=len(cases)-len(rows),
            source="rule_baseline",agent_generated=False,paid_calls=0,rows=rows,
            seconds=time.monotonic()-started)
        (output/"report.json").write_bytes(canonical(record))
    def invoke(main,arguments):
        old=sys.argv
        try:
            sys.argv=["offline-acceptance",*arguments]
            return main()
        finally:sys.argv=old
    class Rule:
        kind="scripted_replay";agent_calls=0
        def generate(self,request,history):
            require(not history,"Domain rule candidate cannot repair")
            return json.dumps(dict(schema=1,request_id=request["request_id"],
                                   hecate_source=candidate_source(request)[0]))
    save("running")
    try:
        for item in cases:
            boundary();row=dict(id=item["id"],status="running");rows.append(row);save("running")
            source=Path(item["source"])
            if item["source_kind"]=="graph":
                source=output/("import-"+item["id"])
                require(invoke(import_main,["--graph",item["source"],"--resource-profile",
                    "general-tensors-v1","--write","--inside","--output",str(source)])==0,"Domain graph import")
            model=json.loads((source/"model.json").read_text())
            provenance=load_model_provenance(source,model,max_bytes=4*1024**2)
            original=provenance["original"];request=prepare_task(model,item["options"])
            row.update(model_sha256=digest(model),original_sha256=digest(original),
                       request_id=request["request_id"],model_import=str(source))
            task=app.submit_import(source,options=item["options"],limits=dict(generations=1,
                http_attempts=1,wall_seconds=3600,api_timeout_seconds=1200,max_tokens=384000))
            row["task_id"]=task;result=app.run(task,Rule());row["qualification"]=result;save("running")
            boundary()
            if result["status"]!="succeeded":
                row.update(status="failed",failure=result.get("failure"));save("running")
                if result["status"] in ("service_error","cancelled","timed_out"):
                    save("stopped");return 1
                continue
            package=context.task_store/task/"program";manifest,checked=check_execution_dependencies(package)
            require(checked==request and manifest["agent_generated"] is False and
                    result["http_attempts"]==0,"Domain qualification scope")
            row.update(package=str(package),package_binding=manifest["binding"],
                candidate_sha256=sha(package/"candidate.py"))
            boundary();before=set(results.iterdir());require(replay_bundle(package)==0,"Domain package reload")
            replay_dirs=[p for p in set(results.iterdir())-before if p.name.startswith("poseidon-package-replay-")]
            require(len(replay_dirs)==1,"Domain replay record")
            row["reload"]=json.loads((replay_dirs[0]/"report.json").read_text())
            valid=[];expected=[]
            for seed in (20261012,20261013,20261014):
                rng=np.random.default_rng(seed)
                args={s["name"]:rng.uniform(-.2,.2,s["shape"]) for s in model["inputs"]}
                independent=math_reference(model,args);torch_values=torch_reference(model,args)
                for name in independent:
                    np.testing.assert_allclose(independent[name],torch_values[name],atol=1e-12,rtol=1e-12)
                valid.append({k:v.tolist() for k,v in args.items()})
                expected.append(np.concatenate([independent[o["name"]].reshape(-1) for o in model["outputs"]]))
            row["new_input_runs"]=[]
            for repetition in range(2):
                boundary();executed=app.execute(task,valid)
                require(executed["status"]=="executed" and executed["encrypted_execution"] and
                        executed["executor_secret_key"] is False and
                        executed["agent_calls"]==executed["compiler_calls"]==0,"Domain valid execution")
                require(executed["input_domain_check"]["status"]=="passed" and
                        executed["input_domain_check"]["source_sha256"]==digest(original) and
                        executed["numerically_validated_on_supplied_inputs"] is False and
                        executed["application_accuracy_certified"] is False,"Domain preflight claim")
                actual=np.asarray([np.concatenate([np.asarray(v[o["name"]]).reshape(-1)
                    for o in model["outputs"]]) for v in executed["outputs"]])
                compared=compare(actual,np.asarray(expected),1e-5,1e-4)
                require(compared["passed"],"Domain new-input comparison")
                row["new_input_runs"].append(dict(repetition=repetition,evidence=executed["evidence"],
                    comparison=compared,domain_check=executed["input_domain_check"]))
                save("running")
            invalid=None;diagnostic=None
            for magnitude in (2.,8.,32.,128.):
                for seed in (19,31):
                    rng=np.random.default_rng(seed)
                    probe={s["name"]:rng.uniform(-magnitude,magnitude,s["shape"]).tolist()
                           for s in model["inputs"]}
                    logical=normalize_samples(request,[probe])
                    try:check_approximation_domains(original,logical)
                    except ApproximationDomainRejected as exc:
                        invalid=probe;diagnostic=exc.public_failure;break
                if invalid is not None:break
            require(invalid is not None,"Domain discriminating rejection probe unavailable")
            boundary();before=set(results.iterdir());rejected=app.execute(task,[invalid])
            require(rejected["status"]=="validation_failed" and rejected["failure"]==diagnostic and
                    rejected["encrypted_execution"] is False and
                    rejected["agent_calls"]==rejected["compiler_calls"]==0 and
                    set(results.iterdir())==before,"Domain application rejection side effects")
            row["invalid_application"]=rejected
            inputs=output/(item["id"]+"-rejected-input.json")
            inputs.write_bytes(canonical(dict(format="poseidon-local-inputs-v1",samples=[invalid])))
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code=invoke(bundle_main,["--inside","execute","--bundle",str(package),"--inputs",str(inputs)])
            cli=json.loads(stdout.getvalue())
            require(exit_code==1 and cli==rejected and set(results.iterdir())==before,
                    "Domain CLI rejection")
            row["invalid_cli"]=dict(exit_code=exit_code,result=cli)
            cached_task=app.submit_import(source,options=item["options"])
            class NeverGenerate:
                kind="scripted_replay";agent_calls=0
                def generate(self,*args):raise AssertionError("Domain package must be reused")
            cached=app.run(cached_task,NeverGenerate())
            require(cached["status"]=="succeeded" and cached["package"]["reused"] and
                    cached["generations"]==cached["http_attempts"]==0,"Domain cache reuse")
            row["cache_reuse"]=dict(status="succeeded",reused=True,generations=0,http_attempts=0)
            row["status"]="passed";save("running")
            print(json.dumps(dict(id=item["id"],status="passed",rejection=diagnostic["location"])),flush=True)
        boundary();save("complete")
        return 0 if all(v["status"]=="passed" for v in rows) else 1
    except BaseException:
        save("stopped");raise


def main():
    from hecate_python_env import VENV,enter_nix
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--parent-result",type=Path)
    p.add_argument("--max-seconds",type=int)
    p.add_argument("--plan",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.max_seconds is not None and not 60<=args.max_seconds<=3300:
        p.error("--max-seconds must be 60..3300")
    if args.plan:
        print(json.dumps(dict(mode="offline_domain_plan",cases=inventory(args.index),paid_calls=0)));return 0
    if args.output is None:p.error("Fresh --output required")
    if os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV:
        return run(args.index.resolve(),args.output.resolve(),args.parent_result.resolve() if args.parent_result else None,args.max_seconds)
    if args.inside:p.error("Pinned pure environment required")
    command=shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),
        "--inside","--index",str(args.index.resolve()),"--output",str(args.output.resolve())])
    if args.parent_result:command+=" --parent-result "+shlex.quote(str(args.parent_result.resolve()))
    if args.max_seconds is not None:command+=" --max-seconds "+str(args.max_seconds)
    return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+command,
                     seconds=args.max_seconds+200 if args.max_seconds is not None else (2900 if args.parent_result else 3500),environment="poseidon-cpu")

if __name__=="__main__":raise SystemExit(main())
