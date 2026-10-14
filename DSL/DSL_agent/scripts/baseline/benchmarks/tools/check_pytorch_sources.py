"""Developer-only, bounded qualification of explicitly trusted source cases.

The application path remains Application. This runner records isolated reference
checks, submits exact rule candidates, and exercises selected reusable packages.
It never calls a provider or consumes a live-evaluation proposal.
"""
import argparse, hashlib, json, re, shlex, shutil, subprocess, sys, time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def case_inventory(path):
    from benchmark_graph import require
    from benchmark_runner import strict_file
    from trusted_pytorch_import import prepare
    value=strict_file(path,131072)
    require(type(value) is dict and set(value)=={"format","cases"} and
            value["format"]=="poseidon-pytorch-offline-cases-v1","Offline source index")
    cases=value["cases"]
    require(type(cases) is list and 1<=len(cases)<=48,"Offline source case count")
    records=[]
    for item in cases:
        require(type(item) is dict and set(item)=={"id","manifest","options","exercise_reuse"},
                "Offline source case fields")
        require(type(item["id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,80}",item["id"]),
                "Offline source case name")
        require(type(item["manifest"]) is str and type(item["options"]) is dict
                and type(item["exercise_reuse"]) is bool,"Offline source options")
        path=(ROOT/item["manifest"]).resolve()
        require(path.is_relative_to(ROOT/"scripts/baseline/cases"),"Offline source location")
        spec,_=prepare(path)
        records.append(dict(**item,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            source_sha256=spec["files"]))
    require(len({r["id"] for r in records})==len(records),"Duplicate offline source cases")
    return records


def check_source_boundary(context, backend, sources, expected_identity):
    """ContextBackend.identity checks the worker before reading current identity.

    Delegate this exact check once; no snapshot or source hash is cached here.
    """
    from application_execution.backend import ContextBackend
    from benchmark_graph import require
    from semantic_benchmark_execution import runtime_sources
    require(isinstance(backend, ContextBackend) and backend.context is context,
            "Offline context backend mismatch")
    current_identity = backend.identity()
    require(runtime_sources() == sources and current_identity == expected_identity,
            "Offline source/dependency drift")


def select_resume_rows(parent_plan,parent_report,proposal,cases,sources,identity,*,terminal_observed=False):
    """Carry exact original evidence; authorize only unfinished original rows."""
    import copy
    from benchmark_graph import require,digest
    require(parent_plan.get("format")=="poseidon-pytorch-offline-plan-v1" and
            parent_plan.get("paid_calls")==0 and parent_plan.get("source")=="rule baseline",
            "Resume original plan format/source")
    require(parent_plan.get("binding")==digest({k:v for k,v in parent_plan.items() if k!="binding"}),
            "Resume original plan binding")
    require(parent_report.get("binding")==parent_plan["binding"] and
            (parent_report.get("status")=="stopped" or
             (parent_report.get("status")=="running" and terminal_observed is True)) and parent_report.get("paid_calls")==0 and
            parent_report.get("agent_generated") is False,"Resume original report boundary")
    require(parent_plan["source_hashes"]==sources and parent_plan["backend_identity"]==identity and
            parent_plan["cases"]==cases and parent_plan["capture_only"] is False,"Resume source/identity drift")
    require(proposal.get("format")=="poseidon-offline-continuation-proposal-v1" and
            proposal.get("parent_binding")==parent_plan["binding"] and proposal.get("api_calls")==0 and
            proposal.get("downloads")==0 and proposal.get("parameters_changed") is False,
            "Resume approved scope")
    require(proposal["bounds"]==dict(original_wall_seconds=3600,additional_wall_seconds=3600,
            total_wall_seconds=7200,aggregate_added_artifact_bytes=4*1024**3,
            global_artifact_bytes=32*1024**3,free_disk_bytes=4*1024**3,
            compiler_jobs=2,link_jobs=1,encrypted_jobs=1,sandbox_address_space_bytes=4*1024**3),
            "Resume approved resource scope")
    rows=copy.deepcopy(parent_report["rows"])
    require([r["id"] for r in rows]==[c["id"] for c in cases] and
            all(r.get("capture",{}).get("reference_verified") is True for r in rows),
            "Resume captured inventory")
    require(parent_report["passed"]==sum(r["status"]=="passed" for r in rows) and
            parent_report["failed"]==sum(r["status"]=="failed" for r in rows),
            "Resume report counters")
    allowed=proposal["candidate_ids"]
    require(type(allowed) is list and len(set(allowed))==len(allowed) and
            set(allowed)<=set(r["id"] for r in rows),"Resume approved identifiers")
    selected=[r["id"] for r in rows if r["status"] not in ("passed","failed")]
    require(set(selected)<=set(allowed),"Resume unapproved original task")
    return rows,selected

def run(index,output,*,capture_only=False,resume_from=None,approved_continuation=None,approved_sha=None):
    import numpy as np
    from hecate_python_env import WORK
    from benchmark_graph import canonical,digest,require
    from semantic_benchmark_execution import runtime_sources
    from trusted_pytorch_import import run as capture
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from component_contract import prepare_task
    from candidate_bundle import check_execution_dependencies,replay_bundle
    from unified_graph_lowering import candidate_source
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from numerical_results import compare
    results=WORK/"results"
    require(not output.exists() and output.parent.resolve()==results.resolve(),"Fresh offline output")
    cases=case_inventory(index);index_sha=hashlib.sha256(index.read_bytes()).hexdigest()
    output.mkdir(mode=0o700);(output/"imports").mkdir()
    context=WorkerContext.capture((resume_from if resume_from is not None else output)/"tasks")
    app=application(context)
    tool_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources=runtime_sources();identity=app.backend.identity()
    plan=dict(format="poseidon-pytorch-offline-plan-v1",index_sha256=index_sha,cases=cases,
        source_hashes=sources,backend_identity=identity,worker_binding=context.binding,
        tool_sha256=tool_sha,capture_only=capture_only,source="rule baseline",
        paid_calls=0,max_seconds=3600,case_bytes=4*1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,
        reference_probes=16,qualification_probes=4,new_inputs=3,repetitions=2,atol=1e-5,rtol=1e-4)
    rows=[];selected=None;parent_report_sha=None;parent_plan_sha=None
    artifacts={output}
    if resume_from is not None:
        from benchmark_runner import strict_file
        from candidate_bundle import load_model_provenance
        require(not capture_only and approved_continuation is not None and
                approved_sha is not None,"Resume explicit approval")
        require(resume_from.parent.resolve()==results.resolve() and resume_from.is_dir()
                and not resume_from.is_symlink(),"Resume original result location")
        approved_raw=approved_continuation.read_bytes()
        require(len(approved_raw)<=4*1024**2 and hashlib.sha256(approved_raw).hexdigest()==approved_sha,
                "Resume approval changed")
        proposal=json.loads(approved_raw)
        parent_plan_path=resume_from/"plan.json";parent_report_path=resume_from/"report.json"
        old_plan=strict_file(parent_plan_path,4*1024**2)
        old_report=strict_file(parent_report_path,4*1024**2)
        parent_plan_sha=hashlib.sha256(parent_plan_path.read_bytes()).hexdigest()
        parent_report_sha=hashlib.sha256(parent_report_path.read_bytes()).hexdigest()
        require(parent_plan_sha==proposal["parent_plan_sha256"] and
                context.binding==old_plan["worker_binding"] and old_plan["index_sha256"]==index_sha,
                "Resume original worker/index binding")
        archived=resume_from/"offline-tool-original.py"
        require(archived.is_file() and hashlib.sha256(archived.read_bytes()).hexdigest()==old_plan["tool_sha256"],
                "Resume original tool evidence")
        terminal=strict_file(resume_from/"terminal-observation.json",131072)
        require(terminal.get("format")=="poseidon-offline-terminal-observation-v1" and
                terminal.get("outer_exit_code")==124 and terminal.get("active_processes")==[] and
                terminal.get("parent_report_sha256")==parent_report_sha,"Resume terminal observation")
        rows,selected=select_resume_rows(old_plan,old_report,proposal,cases,sources,identity,
            terminal_observed=True)
        require(selected,"Resume has no remaining task")
        plan.update(format="poseidon-pytorch-offline-plan-v2",
            parent_result=str(resume_from),parent_binding=old_plan["binding"],
            parent_plan_sha256=parent_plan_sha,parent_report_sha256=parent_report_sha,
            approved_continuation_sha256=approved_sha,selected_ids=selected,
            preexisting_passed_ids=[r["id"] for r in rows if r["status"]=="passed"],
            max_seconds=3300)
        # Include all original artifacts conservatively, including incomplete jobs.
        cutoff=parent_plan_path.stat().st_mtime-60
        artifacts.update(p for p in results.iterdir() if p.is_dir() and p.stat().st_mtime>=cutoff)
        for item,row in zip(cases,rows):
            folder=Path(row["model_import"])
            require(folder==resume_from/"imports"/item["id"] and folder.is_dir(),
                    "Resume original import location")
            model=json.loads((folder/"model.json").read_text())
            request=prepare_task(model,item["options"])
            load_model_provenance(folder,model)
            require(digest(model)==row["model_sha256"] and request["request_id"]==row["request_id"]
                    and json.loads((folder/"request.json").read_text())==request,
                    "Resume original model/request changed")
    else:
        require(approved_continuation is None and approved_sha is None,"Unexpected resume approval")
    plan["binding"]=digest(plan);(output/"plan.json").write_bytes(canonical(plan))
    context.persist()
    started=time.monotonic()
    def collect(before):artifacts.update(set(results.iterdir())-before)
    def boundary():
        check_source_boundary(context, app.backend, sources, identity)
        require(hashlib.sha256(index.read_bytes()).hexdigest()==index_sha and case_inventory(index)==cases
                and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==tool_sha,"Offline source index drift")
        if resume_from is not None:
            require(hashlib.sha256((resume_from/"plan.json").read_bytes()).hexdigest()==parent_plan_sha and
                    hashlib.sha256((resume_from/"report.json").read_bytes()).hexdigest()==parent_report_sha,
                    "Resume original evidence changed")
        require(time.monotonic()-started<plan["max_seconds"],"Offline wall budget")
        require(shutil.disk_usage(WORK).free>=plan["free_bytes"],"Offline disk reserve")
        require(int(subprocess.check_output(["du","-sb",str(results)],text=True,timeout=30).split()[0])
                <=plan["global_bytes"],"Offline global evidence budget")
        require(sum(p.stat().st_size for d in artifacts for p in d.rglob("*") if p.is_file())
                <=plan["case_bytes"],"Offline batch evidence budget")
    def save(status):
        report=dict(format="poseidon-pytorch-offline-report-v1",binding=plan["binding"],status=status,
            planned=len(cases),reference_passed=sum(r.get("capture",{}).get("reference_verified") is True for r in rows),
            passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
            not_run=len(cases)-len(rows),qualification_not_run=sum(r.get("result") is None for r in rows),
            preexisting_passed=len(plan.get("preexisting_passed_ids",[])),
            qualification_reused=sum(r.get("qualification_reused") is True for r in rows),
            parent_binding=plan.get("parent_binding"),rows=rows,seconds=time.monotonic()-started,
            source="rule baseline",agent_generated=False,paid_calls=0,capture_only=capture_only)
        (output/"report.json").write_bytes(canonical(report))
    class Rule:
        kind="scripted_replay";agent_calls=0
        def generate(self,request,history):
            require(not history,"Offline rule must not repair an original model")
            source=candidate_source(request)[0]
            return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
    save("running")
    try:
        # Prepare the full inventory before compiling any candidate.
        for item in ([] if resume_from is not None else cases):
            boundary()
            row=dict(id=item["id"],status="running",new_input_runs=[]);rows.append(row);save("running")
            folder=output/"imports"/item["id"]
            row["capture"]=capture(ROOT/item["manifest"],folder,
                                  resource_profile="general-tensors-v1",trusted=True)
            row["model_import"]=str(folder)
            model=json.loads((folder/"model.json").read_text())
            request=prepare_task(model,item["options"])
            row.update(model_sha256=digest(model),request_id=request["request_id"],status="reference_passed")
            row["reference"]=json.loads((folder/"reference-report.json").read_text())
            (folder/"request.json").write_bytes(canonical(request))
            save("running")
        if capture_only:
            boundary();save("reference_complete");return 0
        for item,row in zip(cases,rows):
            if selected is not None and item["id"] not in selected:continue
            boundary();row["status"]="running";save("running")
            folder=Path(row["model_import"]);model=json.loads((folder/"model.json").read_text())
            before=set(results.iterdir())
            if resume_from is not None and row.get("result",{}).get("status")=="succeeded":
                task=row["task_id"];result=app.result(task)
                require(result["status"]=="succeeded","Resume qualified package unavailable")
                row["qualification_reused"]=True
            else:
                # The original batch may stop after submission but before saving result.
                old_task=row.get("task_id")
                prior=app.status(old_task) if old_task is not None else None
                if prior is not None and prior["status"]=="succeeded":
                    task=old_task;result=app.result(task);row["qualification_reused"]=True
                else:
                    require(prior is None,"Resume interrupted task needs explicit review")
                    task=app.submit_import(folder,options=item["options"],
                        limits=dict(generations=1,http_attempts=1,wall_seconds=3600,
                                    api_timeout_seconds=1200,max_tokens=384000))
                    row["task_id"]=task;result=app.run(task,Rule())
            collect(before)
            row["result"]=result;save("running");boundary()
            if result["status"]!="succeeded":
                row["status"]="failed";save("running")
                print(json.dumps(dict(id=item["id"],status="failed",failure=result.get("failure"))),flush=True)
                if result["status"] in ("service_error","cancelled","timed_out"):
                    save("stopped");return 1
                continue
            package=context.task_store/task/"program"
            manifest,checked=check_execution_dependencies(package)
            request=json.loads((folder/"request.json").read_text())
            require(checked==request and manifest["agent_generated"] is False and result["http_attempts"]==0,
                    "Exact offline request/package scope")
            row.update(package=str(package),package_binding=manifest["binding"],
                       candidate_sha256=hashlib.sha256((package/"candidate.py").read_bytes()).hexdigest())
            if item["exercise_reuse"]:
                boundary();before=set(results.iterdir())
                if not row.get("reload"):
                    require(replay_bundle(package)==0,"Offline package replay")
                fresh=set(results.iterdir())-before;collect(before)
                replays=[p for p in fresh if p.name.startswith("poseidon-package-replay-")]
                if not row.get("reload"):
                    require(len(replays)==1,"Unique replay evidence")
                    row["reload"]=json.loads((replays[0]/"report.json").read_text())
                require(row["reload"].get("encrypted_execution") is True and
                        row["reload"].get("numerically_validated") is True,"Resume reload evidence")
                logical=[];expected=[]
                for probe in range(3):
                    rng=np.random.default_rng(20261009+probe)
                    args={v["name"]:rng.uniform(-.3,.3,v["shape"]) for v in model["inputs"]}
                    values=mathematical(model,args);torch_values=tensor_reference(model,args)
                    for name in values:np.testing.assert_allclose(values[name],torch_values[name],atol=1e-12,rtol=1e-12)
                    logical.append({k:v.tolist() for k,v in args.items()})
                    expected.append(np.concatenate([np.asarray(values[o["name"]]).reshape(-1) for o in model["outputs"]]))
                (output/(item["id"]+"-new-inputs.json")).write_bytes(canonical(logical))
                for repetition in range(len(row["new_input_runs"]),2):
                    boundary();before=set(results.iterdir())
                    executed=app.execute(task,logical);collect(before)
                    require(executed["status"]=="executed" and executed["encrypted_execution"]
                            and executed["agent_calls"]==executed["compiler_calls"]==0
                            and executed["executor_secret_key"] is False,"Offline reusable execution scope")
                    actual=np.asarray([np.concatenate([np.asarray(values[o["name"]]).reshape(-1)
                                        for o in model["outputs"]]) for values in executed["outputs"]])
                    compared=compare(actual,np.asarray(expected),1e-5,1e-4)
                    require(compared["passed"],"Offline new-input numerical comparison")
                    row["new_input_runs"].append(dict(repetition=repetition,evidence=executed["evidence"],
                        comparison=compared,agent_calls=0,compiler_calls=0,output_structure=executed.get("output_structure")))
                    save("running")
                cache_task=app.submit_import(folder,options=item["options"])
                class NeverGenerate:
                    kind="scripted_replay";agent_calls=0
                    def generate(self,*args):raise AssertionError("Qualified source must be reused")
                cached=app.run(cache_task,NeverGenerate())
                require(cached["status"]=="succeeded" and cached["package"]["reused"]
                        and cached["generations"]==cached["http_attempts"]==0,"Offline cache reuse")
                row["cache_reuse"]=dict(status=cached["status"],reused=True,generations=0,http_attempts=0)
            row["status"]="passed";save("running")
            print(json.dumps(dict(id=item["id"],status="passed",
                max_absolute_error=result["acceptance"]["maximum_absolute_error"])),flush=True)
        boundary();save("complete")
        return 0 if all(r["status"]=="passed" for r in rows) else 1
    except BaseException:
        save("stopped");raise

def main():
    from hecate_python_env import VENV,enter_nix
    from benchmark_graph import require
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trust-local-code",action="store_true")
    parser.add_argument("--capture-only",action="store_true")
    parser.add_argument("--resume-from",type=Path)
    parser.add_argument("--approved-continuation",type=Path)
    parser.add_argument("--approved-continuation-sha256")
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    require(args.trust_local_code is True,"Explicit trusted source authorization")
    if not args.inside:
        command=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--trust-local-code",
                 "--index",str(args.index.resolve()),"--output",str(args.output.resolve())]
        if args.capture_only:command.append("--capture-only")
        if args.resume_from:
            require(args.approved_continuation is not None and args.approved_continuation_sha256 is not None,
                    "Resume requires exact authorization")
            command.extend(["--resume-from",str(args.resume_from.resolve()),
                "--approved-continuation",str(args.approved_continuation.resolve()),
                "--approved-continuation-sha256",args.approved_continuation_sha256])
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join(command),seconds=600 if args.capture_only else (3500 if args.resume_from else 3630),environment="poseidon-cpu")
    return run(args.index.resolve(),args.output.resolve(),capture_only=args.capture_only,
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        approved_continuation=args.approved_continuation.resolve() if args.approved_continuation else None,
        approved_sha=args.approved_continuation_sha256)
if __name__=="__main__":raise SystemExit(main())
