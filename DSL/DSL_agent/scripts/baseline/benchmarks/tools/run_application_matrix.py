"""Developer-only bounded application evaluation. --prepare never calls a provider.

The normal application workflow lives in Application. This tool freezes imported
cases, delegates generation and qualification to that component, then reloads
successful packages. It never resumes uncertain transport attempts.
"""
import argparse, copy, hashlib, json, os, re, shlex, subprocess, sys, time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

FORMAT="poseidon-application-evaluation-plan-v1"
PROVIDER=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",max_tokens=384000,
              max_repairs=3,api_timeout_seconds=1200,provider_retries=3,loopback_proxy_port=0,stream=True)
BOUNDS=dict(batch_seconds=43200,case_seconds=14400,case_bytes=4*1024**3,global_bytes=32*1024**3,
            free_bytes=4*1024**3,api_concurrency=1,native_concurrency=1,compile_jobs=2,link_jobs=1)

def check_envelope(value,binding):
    from benchmark_graph import canonical,digest,require
    require(type(value) is dict and set(value)=={"format","cases","source_hashes","backend_identity",
             "application_identities","tool_sha256","provider","limits","max_generations","max_http_attempts","binding"},
            "Evaluation plan fields")
    require(value["format"]==FORMAT and value["binding"]==binding==digest({k:v for k,v in value.items() if k!="binding"}),
            "Evaluation approval binding")
    require(value["provider"]==PROVIDER and value["limits"]==BOUNDS,"Evaluation budget changed")
    require(type(value["cases"]) is list and 1<=len(value["cases"])<=48,"Evaluation case count")
    require(value["max_generations"]==4*len(value["cases"]) and value["max_http_attempts"]==16*len(value["cases"]),
            "Evaluation call budget")
    require(len(canonical(value))<=4*1024**2,"Evaluation plan size")
    return value

def snapshot_case(item):
    from benchmark_graph import digest,require,canonical
    from candidate_bundle import load_model_provenance
    from component_contract import prepare_task
    from benchmark_runner import strict_file
    from deepseek_provider import public_request
    from public_weight_context import TRUSTED_BYTES
    require(type(item) is dict and set(item)=={"id","imported_directory","options"},"Evaluation case fields")
    require(type(item["id"]) is str and re.fullmatch("[A-Za-z0-9_-]{1,80}",item["id"]),"Evaluation case name")
    path=Path(item["imported_directory"])
    require(path.is_absolute() and path.is_dir() and not path.is_symlink(),"Evaluation import directory")
    model=strict_file(path/"model.json",TRUSTED_BYTES if item["options"].get("public_weight_policy") else 131072)
    provenance=load_model_provenance(path,model,max_bytes=TRUSTED_BYTES if item["options"].get("public_weight_policy") else 131072)
    request=prepare_task(model,item["options"])
    context=public_request(request)
    record=dict(**copy.deepcopy(item),model_sha256=digest(model),provenance_sha256=digest(provenance),
                request_id=request["request_id"],public_context_sha256=digest(context),
                public_context_bytes=len(canonical(context)),inputs=model["inputs"],outputs=model["outputs"],
                operators=sorted({n["op"] for n in model["nodes"]}))
    return record,request

def freeze(items):
    from application_backend import PoseidonBackend
    from semantic_benchmark_execution import runtime_sources
    from benchmark_graph import require,digest
    records=[snapshot_case(item)[0] for item in items]
    require(len({r["id"] for r in records})==len(records),"Duplicate evaluation cases")
    backend=PoseidonBackend()
    value=dict(format=FORMAT,cases=records,source_hashes=runtime_sources(),backend_identity=backend.identity(),
               application_identities=backend.identity_manifest(),
               tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               provider=PROVIDER,limits=BOUNDS,max_generations=4*len(records),max_http_attempts=16*len(records))
    value["binding"]=digest(value)
    check_envelope(value,value["binding"])
    return value

def items_from_plan(value):
    return [{k:c[k] for k in ("id","imported_directory","options")} for c in value["cases"]]

def execute(value,output):
    from benchmark_graph import canonical,require
    from application_component import Application
    from application_backend import PoseidonBackend
    from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport
    from candidate_bundle import check_bundle,replay_bundle
    from hecate_python_env import WORK
    from agent_credentials import load_api_key
    require(not output.exists() and output.resolve().parent==(WORK/"results").resolve(),"Fresh evaluation directory")
    # Validate frozen cases/code and resource allowance before touching credentials.
    require(freeze(items_from_plan(value))==value,"Evaluation sources or dependencies changed")
    output.mkdir()
    (output/"plan.json").write_bytes(canonical(value))
    started=time.monotonic();rows=[];paths={output}
    def boundary():
        require(time.monotonic()-started<BOUNDS["batch_seconds"],"Evaluation batch time budget")
        require(freeze(items_from_plan(value))==value,"Evaluation binding changed")
        import shutil
        require(shutil.disk_usage(WORK).free>=BOUNDS["free_bytes"],"Evaluation disk reserve")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
        require(total<=BOUNDS["global_bytes"],"Evaluation global space budget")
        require(sum(p.stat().st_size for d in paths for p in d.rglob("*") if p.is_file())<=BOUNDS["case_bytes"],
                "Evaluation case space budget")
    def save(status):
        report=dict(format="poseidon-application-evaluation-report-v1",binding=value["binding"],status=status,
             rows=rows,planned=len(value["cases"]),passed=sum(r["status"]=="passed" for r in rows),
             failed=sum(r["status"]=="failed" for r in rows),not_run=len(value["cases"])-len(rows),
             api_attempts=sum(r.get("result",{}).get("http_attempts",0) for r in rows),
             generations=sum(r.get("result",{}).get("generations",0) for r in rows),
             cost_usd=None,seconds=time.monotonic()-started)
        (output/"report.json").write_bytes(canonical(report))
    boundary();save("running")
    key=load_api_key(ROOT,provider=PROVIDER["provider"])
    # Remove inherited credentials before any backend child process.
    os.environ.pop("DEEPSEEK_API_KEY",None)
    class BoundedBackend(PoseidonBackend):
        def qualify(self,request,candidate,level):
            boundary()
            previous=set((WORK/"results").iterdir())
            result=super().qualify(request,candidate,level)
            paths.update(set((WORK/"results").iterdir())-previous)
            boundary()
            return result
        def export(self,result,destination,level,**requirements):
            boundary()
            super().export(result,destination,level,**requirements)
            boundary()
    app=Application(output/"tasks",BoundedBackend())
    c=Config(model=PROVIDER["model"],service_provider=PROVIDER["provider"],reasoning_effort=PROVIDER["reasoning_effort"],
        max_tokens=PROVIDER["max_tokens"],max_calls=4,timeout_seconds=PROVIDER["api_timeout_seconds"],
        provider_retries=PROVIDER["provider_retries"],loopback_proxy_port=0,stream=True)
    try:
        for item in value["cases"]:
            boundary()
            before=set((WORK/"results").iterdir())
            paths={output}
            row=dict(id=item["id"],request_id=item["request_id"],status="not_run");rows.append(row);save("running")
            limits=dict(generations=4,http_attempts=16,wall_seconds=min(BOUNDS["case_seconds"],
                int(BOUNDS["batch_seconds"]-(time.monotonic()-started))),api_timeout_seconds=1200,max_tokens=384000)
            transport=HTTPSTransport(api_key=key,approved_request_id=item["request_id"],approved_config=c,enabled=True)
            provider=DeepSeekProvider(c,transport=transport)
            task=app.submit_import(item["imported_directory"],options=item["options"],limits=limits)
            row["task_id"]=task
            paths={output/"tasks"/task}
            result=app.run(task,provider);row["result"]=result
            paths|=set((WORK/"results").iterdir())-before
            if result["status"]=="succeeded":
                package=output/"tasks"/result["package"]["task_id"]/"program"
                manifest=check_bundle(package)
                require(manifest["agent_generated"] is True and result["http_attempts"]>0 and
                        not result["package"]["reused"],"Missing new exact Agent generation evidence")
                require(replay_bundle(package)==0,"Application package reload failed")
                row.update(status="passed",package=str(package),package_binding=manifest["binding"],
                           generation=result["generation"],reloaded=True)
            else:
                row["status"]="failed"
            paths|=set((WORK/"results").iterdir())-before
            boundary();save("running")
            print("APPLICATION",item["id"],row["status"],flush=True)
            if result["status"] in ("service_error","cancelled","timed_out"):
                save("stopped");return 1
        save("complete");return 0 if all(r["status"]=="passed" for r in rows) else 1
    except BaseException:
        save("stopped");raise
    finally:
        key=""

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",type=Path,help="Explicit imported-case list; offline")
    g.add_argument("--execute",type=Path,help="Frozen plan; requires scoped paid authorization")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--approved-binding")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    from hecate_python_env import VENV,enter_nix
    from benchmark_runner import strict_file
    from benchmark_graph import canonical,require
    if not a.inside:
        argv=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output.resolve())]
        argv += ["--prepare",str(a.prepare.resolve())] if a.prepare else ["--execute",str(a.execute.resolve()),
                  "--approved-binding",a.approved_binding or ""]
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(argv)
        return enter_nix(cmd,seconds=600 if a.prepare else 43230,environment="poseidon-cpu")
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned application worker")
    if a.prepare:
        items=strict_file(a.prepare,131072)
        require(type(items) is list,"Evaluation input list")
        require(not a.output.exists(),"Preserve existing evaluation plan")
        value=freeze(items)
        a.output.write_bytes(canonical(value))
        print(json.dumps(dict(mode="prepared",binding=value["binding"],cases=len(items),paid_calls=0)))
        return 0
    value=check_envelope(strict_file(a.execute,4*1024**2),a.approved_binding)
    return execute(value,a.output.resolve())

if __name__=="__main__":
    raise SystemExit(main())
