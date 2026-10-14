"""Three-case post-cutover acceptance, using existing application/backend APIs.

Developer tool only. No Provider imports, downloads, SDK rebuilds or backend
changes. It cannot run before the guarded production cutover is complete.
"""
import argparse,fcntl,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
FIXTURES=HERE.parent/"staged_ordered_intake_v1"/"fixtures"
CASES=("linear","norm","split_weight")
RESOURCE_PROFILE="general-tensors-v1"
PHASES=("qualification","new_inputs","cache")
SCOPE=dict(wall_seconds=1800,phase_seconds=180,artifact_bytes=4*1024**3,
           global_bytes=32*1024**3,reserve_bytes=4*1024**3,
           provider_calls=0,native_workers=1,compile_jobs=2,link_jobs=1)
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def write(path,value):
    Path(path).write_bytes(canonical(value))
def hashes():
    files={str(Path(__file__).resolve()):sha(Path(__file__))}
    for name in CASES:
        for p in sorted((FIXTURES/name).iterdir()):
            require(p.is_file() and not p.is_symlink(),"Regular reviewed fixture")
            files[str(p)]=sha(p)
    return files
def preview():
    import guarded_apply_joint
    plan=read(HERE/"plan.json")
    sys.path.insert(0,str(BASE))
    from model_resources import identity
    resource=identity(RESOURCE_PROFILE)
    return dict(format="poseidon-joint-cutover-offline-scope-v1",
        resource_profile=RESOURCE_PROFILE,logical_resource_identity=resource,
        guard=guarded_apply_joint.check(),cases=list(CASES),phases=list(PHASES),scope=SCOPE,
        files=hashes(),joint_target_hashes={name:i["sha256"] for name,i in plan["source_overlays"].items()},
        candidate_origin="rule_baseline",new_agent_generation=False,
        request_preflight=plan["post_cutover_request_preflight"],
        bootstrap=False,cuda=False,production_cutover_required=True)
def measure(folder):
    return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(folder)],text=True,timeout=30).split()[0])
               for arg in ("--apparent-size","--count-links"))
def verify_applied():
    import guarded_apply_joint
    state=guarded_apply_joint.check()
    require(state.get("applied") is True and not state.get("changed") and
            state["status"]=="applied_pending_backend_acceptance","Reviewed joint cutover required")
    return read(HERE/"plan.json")
def outer(output):
    applied=verify_applied()
    from guarded_apply_joint import existing_guard
    require(not existing_guard.active_processes(),"Existing live or candidate process")
    lock=WORK/"deps/nix-portable-v012/launcher.lock"
    with lock.open("r+") as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    output=Path(output)
    require(output.is_absolute() and output==output.resolve() and output.parent==RESULTS
            and not output.exists(),"Fresh platform result directory")
    require(measure(RESULTS)<=SCOPE["global_bytes"] and shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],
            "Existing artifact/reserve boundary")
    public=preview()
    output.mkdir(mode=0o700)
    plan=dict(public,cutover_record=applied["cutover_record"],
              cutover_record_sha256=applied["cutover_record_sha256"],
              launcher_sha256=sha(Path(__file__)),initial_result_directories=sorted(p.name for p in RESULTS.iterdir()),
              deadline_wall=time.time()+SCOPE["wall_seconds"])
    plan["binding"]=hashlib.sha256(canonical(plan)).hexdigest()
    write(output/"plan.json",plan)
    sys.path.insert(0,str(BASE))
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",str(output)])
    return enter_nix(command,seconds=SCOPE["wall_seconds"],keep_env=())
def boundary(output,plan):
    require(time.time()<plan["deadline_wall"],"Offline wall budget")
    require(all(sha(p)==h for p,h in plan["files"].items()),"Frozen reviewed sources")
    cutover=read(plan["cutover_record"])
    require(sha(plan["cutover_record"])==plan["cutover_record_sha256"],"Cutover record changed")
    require(all(sha(ROOT/p)==h for p,h in cutover["targets"].items()),"Production targets changed")
    for record in read(HERE/"plan.json")["protected_records"].values():
        require(all(sha(ROOT/p)==h for p,h in read(record).items()),"Backend/reference changed")
    require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"Disk reserve")
    new=[p for p in RESULTS.iterdir() if p.name not in plan["initial_result_directories"] or p==output]
    require(sum(measure(p) for p in new)<=SCOPE["artifact_bytes"],"New artifact boundary")
    require(measure(RESULTS)<=SCOPE["global_bytes"],"Global artifact boundary")
def inside(output):
    output=Path(output);plan=read(output/"plan.json")
    require(plan["binding"]==hashlib.sha256(canonical({k:v for k,v in plan.items() if k!="binding"})).hexdigest(),
            "Offline plan binding")
    sys.path.insert(0,str(BASE))
    from hecate_python_env import VENV
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Actual pinned Nix worker required")
    from semantic_benchmark_execution import runtime_sources
    sources=runtime_sources()
    write(output/"production-sources.json",sources)
    rows=[];started=time.monotonic()
    def save(status):
        write(output/"report.json",dict(format="poseidon-joint-cutover-offline-report-v1",binding=plan["binding"],
            status=status,planned_models=3,rows=rows,seconds=time.monotonic()-started,
            provider_calls=0,http_attempts=0,new_agent_generation=False,candidate_origin="rule_baseline",
            compilations=sum(r.get("compiled") is True for r in rows),
            completed_phases=sum(r.get("status")=="passed" for r in rows)))
    save("running")
    for name in CASES:
        for phase in PHASES:
            if phase!="qualification" and not any(r["case"]==name and r["phase"]=="qualification"
                                                    and r["status"]=="passed" for r in rows):continue
            boundary(output,plan)
            require(runtime_sources()==sources,"Current production source binding")
            destination=output/name;destination.mkdir(exist_ok=True)
            child_report=destination/(phase+".json");require(not child_report.exists(),"Fresh phase only")
            command=[sys.executable,"-B",str(Path(__file__).resolve()),"--case",str(output),name,phase]
            allowed=("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
                     "HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG")
            env={k:os.environ[k] for k in allowed if k in os.environ}
            env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(BASE),
                       OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HOME="/nonexistent")
            code=0
            with (destination/(phase+".log")).open("xb") as stream:
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:code=proc.wait(timeout=min(SCOPE["phase_seconds"],max(1,int(plan["deadline_wall"]-time.time()))))
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
            row=read(child_report) if child_report.exists() else dict(case=name,phase=phase,status="failed",
                failure_layer="worker",code="phase_timeout" if code==124 else "missing_phase_receipt")
            row["exit_code"]=code
            require(row["case"]==name and row["phase"]==phase,"Phase identity")
            if code!=0:row["status"]="failed"
            rows.append(row);save("running")
            boundary(output,plan)
            if row.get("failure_layer") in ("integrity","environment","resources"):save("stopped");return 1
    require(runtime_sources()==sources,"Final production source binding")
    status="passed" if len(rows)==9 and all(r["status"]=="passed" for r in rows) else "failed"
    save(status);print(json.dumps(dict(status=status,rows=len(rows),evidence=str(output))),flush=True)
    return 0 if status=="passed" else 1
def one(output,name,phase):
    sys.path.insert(0,str(BASE))
    import numpy as np
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from application_model_source import FORMAT
    from hecate_python_env import VENV
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned case worker")
    require(name in CASES and phase in PHASES,"Reviewed case/phase")
    directory=Path(output)/name
    context=WorkerContext.capture(directory/"tasks")
    app=application(context)
    from candidate_bundle import check_bundle
    row=dict(case=name,phase=phase,status="failed",provider_calls=0,new_agent_generation=False)
    try:
        if phase=="qualification":
            limits=dict(generations=1,http_attempts=1,wall_seconds=170,api_timeout_seconds=1200,max_tokens=384000)
            options=dict(chunk_period=4,generation_guidance="explicit-v11",
                capacity_packing_version="capacity-public-tensors-v9",public_weight_policy="descriptors-v1")
            source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(FIXTURES/name/"manifest.json"),trust_local_code=True)
            task=app.submit_source(source,resource_profile=RESOURCE_PROFILE,options=options,limits=limits)
            state=app.prepare_source(task)
            require(state["status"]=="queued" and state["phase"]=="queued","Actual source capture/admission")
            meta=app._checked_task_metadata(task)
            from unified_graph_lowering import candidate_source
            source_text=candidate_source(meta["request"])[0]
            preflight=read(read(Path(output)/"plan.json")["request_preflight"]["report"])
            expected=[row for row in preflight["rows"] if row["case"]==name]
            require(len(expected)==1 and expected[0]["request_id"]==meta["request"]["request_id"] and
                    expected[0]["rule_source_sha256"]==hashlib.sha256(source_text.encode()).hexdigest(),
                    "Exact tested mathematical request and rule source")
            class RuleReplay:
                kind="scripted_replay";agent_calls=0
                def generate(self,request,feedback=None):
                    require(feedback==[] and request==meta["request"],"One fixed rule expression")
                    return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source_text))
            result=app.run(task,RuleReplay())
            write(directory/"task-id.json",dict(task_id=task))
            row.update(result=result,source_capture_supervisor_executed=True,
                request_id=meta["request"]["request_id"],candidate_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                compiled=bool((result.get("acceptance") or {}).get("compiled")),
                encrypted_execution=bool((result.get("acceptance") or {}).get("encrypted_execution")))
            if result["status"]!="succeeded":
                row["failure_layer"]=(result.get("failure") or {}).get("layer","case_acceptance")
            require(result["status"]=="succeeded" and result["acceptance"]["numerically_validated"],"Real numerical qualification")
            require(result["generation"]["agent_generated"] is False and result["http_attempts"]==0,"Offline rule provenance")
            package=app._dir(task)/"program";m=check_bundle(package)
            require(m["validation"]["level"]=="numerical","Checked numerical package")
            row.update(status="passed",package=str(package),package_binding=m["binding"],
                       input_groups=result["acceptance"]["input_groups"],compared_values=result["acceptance"]["compared_values"],
                       maximum_absolute_error=result["acceptance"]["maximum_absolute_error"])
        else:
            task=read(directory/"task-id.json")["task_id"]
            meta=app._checked_task_metadata(task);package=app._dir(task)/"program"
            before=check_bundle(package);model=meta["request"]["model"]
            if phase=="new_inputs":
                from benchmark_math import evaluate as math_ref
                from benchmark_torch import evaluate as torch_ref
                from numerical_results import compare
                samples=[];reference=[]
                for i in range(3):
                    rng=np.random.default_rng(20261007+i)
                    values={s["name"]:rng.uniform(-.2,.2,s["shape"]) for s in model["inputs"]}
                    a,b=math_ref(model,values),torch_ref(model,values)
                    for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
                    samples.append({k:v.tolist() for k,v in values.items()})
                    reference.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
                execution=app.execute(task,samples)
                actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1) for o in model["outputs"]])
                                   for s in execution["outputs"]])
                compared=compare(actual,np.asarray(reference),1e-5,1e-4)
                require(execution["encrypted_execution"] is True and execution["executor_secret_key"] is False
                        and execution["agent_calls"]==execution["compiler_calls"]==0,"Real package execution")
                require(compared["passed"],"Frozen numerical tolerance")
                require(before==check_bundle(package),"Reloaded package integrity")
                row.update(status="passed",execution=execution,comparison=compared,input_groups=3,
                           compiled=False,encrypted_execution=True,package_reloaded=True)
            else:
                origin=read(app._dir(task)/"source.json")
                source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(FIXTURES/name/"manifest.json"),trust_local_code=True)
                again=app.submit_source(source,resource_profile=RESOURCE_PROFILE,options=origin["requirements"]["options"],
                                        limits=origin["budget"])
                class NeverGenerate:
                    kind="scripted_replay";agent_calls=0
                    def generate(self,*args):raise AssertionError("Cached package required")
                result=app.run(again,NeverGenerate())
                require(result["status"]=="succeeded" and result["package"]["reused"] is True
                        and result["package"]["binding"]==before["binding"]
                        and result["generations"]==result["http_attempts"]==0,"Reusable source package")
                row.update(status="passed",result=result,package_reused=True,compiled=False,encrypted_execution=False)
    except Exception as error:
        row.update(error_type=type(error).__name__,diagnostic=str(error)[:300])
        row.setdefault("failure_layer","case_acceptance")
    write(directory/(phase+".json"),row)
    return 0 if row["status"]=="passed" else 1
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan",action="store_true")
    group.add_argument("--execute",type=Path)
    group.add_argument("--inside",type=Path,help=argparse.SUPPRESS)
    group.add_argument("--case",nargs=3,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.plan:print(json.dumps(preview(),indent=2,sort_keys=True));raise SystemExit(0)
    if args.execute:raise SystemExit(outer(args.execute))
    if args.inside:raise SystemExit(inside(args.inside))
    raise SystemExit(one(*args.case))
