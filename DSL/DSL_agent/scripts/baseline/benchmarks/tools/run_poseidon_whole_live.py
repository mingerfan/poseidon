"""Frozen, serial execution of the three previously approved Poseidon models.

Preparation and check never read credentials or send HTTP. Existing approved
cases, public requests, backend and budgets must remain identical to r368.
A new source binding does not permit extra evaluations or automatic restarts.
"""
import argparse,copy,fcntl,hashlib,json,os,re,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]
PRIOR=ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r368.json"
IDS=("mlp","rmsnorm","attention_decode")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":")).encode()
def require(ok,message):
    if not ok:raise ValueError(message)
def seal(value):
    value=dict(value);value["binding"]=hashlib.sha256(canonical(value)).hexdigest();return value
def document(path):
    value=json.loads(Path(path).read_text());binding=value.pop("binding")
    require(hashlib.sha256(canonical(value)).hexdigest()==binding,"Frozen document integrity")
    return dict(value,binding=binding)
def write(path,value,exclusive=False):
    with Path(path).open("xb" if exclusive else "wb") as stream:stream.write(canonical(value)+b"\n")
def provider_counts(metrics):
    g=metrics.get("generation_attempts",0);h=metrics.get("request_attempts",0)
    require(type(g) is int and type(h) is int and 0<=g<=4 and 0<=h<=16 and g<=h<=g*4,
            "Provider call ledger exceeds original per-case budget")
    require(len(metrics.get("calls",[]))==h,"Incomplete HTTP ledger")
    return g,h
def fixed_scope(plan,prior):
    for k in ("provider","maximum_generations","maximum_http_attempts","limits","numeric_tolerance","outbound","excluded","bootstrapping","seal_fallback","money_cap"):
        require(plan[k]==prior[k],"Original approved scope changed: "+k)
    require(tuple(c["id"] for c in plan["cases"])==IDS,"Exactly three approved models")
    for a,b in zip(plan["cases"],prior["cases"]):
        for k in ("id","model_path","model_sha256","model_file_sha256","request_id","request_sha256","live_command"):
            require(a[k]==b[k],"Original approved case changed: "+k)
def verify(plan):
    from semantic_benchmark_execution import runtime_sources
    from check_poseidon_helpers import snapshot
    from poseidon_backend_identity import identity
    prior=document(PRIOR);fixed_scope(plan,prior)
    require(sha(PRIOR)==plan["authorization_reuse"]["proposal_sha256"],"Approved original proposal drift")
    require(runtime_sources()==plan["runtime_source_hashes"] and snapshot()==plan["source_hashes"],"Source freeze changed")
    require(identity()==plan["backend_identity"]==prior["backend_identity"],"Approved backend changed")
    require(sha(Path(__file__))==plan["tool_sha256"],"Runner changed")
    for c in plan["cases"]:
        require(sha(ROOT/c["model_path"])==c["model_file_sha256"],"Approved model changed")
        require(sha(Path(c["prepare_evidence"])/"request.json")==c["request_sha256"],"Public request file changed")
    return prior
def prepare(output):
    from hecate_python_env import WORK
    from benchmark_graph import digest,samples
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    from component_contract import prepare_task
    from semantic_benchmark_execution import runtime_sources
    from check_poseidon_helpers import snapshot
    from poseidon_backend_identity import identity
    import numpy as np
    require(not output.exists() and output.parent==WORK/"results","Fresh work-root preparation")
    prior=document(PRIOR);backend=identity()
    require(backend==prior["backend_identity"],"Approved backend drift")
    # Search existing live evidence. Any matching attempted evaluation consumes
    # authorization, including uncertain/in-flight calls; never reset it.
    consumed=[]
    expected={c["request_id"] for c in prior["cases"]}
    for p in (WORK/"results").glob("agent-deepseek-*/report.json"):
        require(p.stat().st_size<=8*1024**2,"Oversized prior paid report")
        o=json.loads(p.read_text())
        if o.get("request_id") in expected and o.get("backend")=="poseidon_cpu" and (o.get("agent_calls",0) or o.get("provider_metrics",{}).get("request_attempts",0)):
            consumed.append(str(p))
    require(not consumed,"Original approved evaluations already attempted; preserve their ledger")
    output.mkdir();plan=copy.deepcopy(prior);plan.pop("binding")
    plan.update(status="authorized_rebound_not_started",evidence_root=str(output),source_hashes=snapshot(),
                runtime_source_hashes=runtime_sources(),tool_sha256=sha(__file__),
                authorization_reuse=dict(proposal=str(PRIOR),proposal_sha256=sha(PRIOR),original_binding=prior["binding"],
                user_answer="批准这3项有界测试",scope_unchanged=True,original_paid_attempts=0),
                official_documentation=dict(chat="https://api-docs.deepseek.com/api/create-chat-completion/",
                    models="https://api-docs.deepseek.com/api/list-models/",checked_date="2026-10-02",
                    max_output_limit_verified=393216,configured_max_tokens=384000))
    changes={p:dict(before=prior["source_hashes"].get(p),after=h) for p,h in plan["source_hashes"].items() if prior["source_hashes"].get(p)!=h}
    plan["source_changes_since_approval"]=changes;references=[]
    for c in plan["cases"]:
        model=json.loads((ROOT/c["model_path"]).read_text())
        require(digest(model)==c["model_sha256"] and sha(ROOT/c["model_path"])==c["model_file_sha256"],"Original model identity")
        request=prepare_task(model,dict(compiler_configuration="poseidon-cpu-eva-w45-v1",generation_guidance="explicit-v10"))
        original=Path(c["prepare_evidence"])/"request.json";raw=original.read_bytes()
        require(sha(original)==c["request_sha256"] and canonical(request)==canonical(json.loads(raw)) and request["request_id"]==c["request_id"],"Original public request changed")
        error=0.
        for values in samples(model,16):
            a,b=math_ref(model,values),torch_ref(model,values)
            for name in a:
                np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                error=max(error,float(np.max(np.abs(a[name]-b[name]))))
        folder=output/c["id"];folder.mkdir();(folder/"request.json").write_bytes(raw)
        c["prepare_evidence"]=str(folder)
        references.append(dict(id=c["id"],reference_groups=16,max_absolute_error=error))
    fixed_scope(plan,prior);plan=seal(plan);write(output/"plan.json",plan,True)
    write(output/"preparation.json",dict(public_requests_unchanged=True,models_unchanged=True,backend_unchanged=True,
         references=references,paid_calls=0,source_changes=len(changes),binding=plan["binding"]),True)
    print(json.dumps(dict(binding=plan["binding"],models=list(IDS),requests_unchanged=True,backend_unchanged=True,source_changes=len(changes),paid_calls=0)),flush=True)
    return 0
def case_child(plan,case_id):
    from run_candidate import parse_args,inside
    from agent_credentials import load_api_key
    from benchmark_graph import canonical as graph_canonical
    verify(plan)
    case=next(c for c in plan["cases"] if c["id"]==case_id)
    request=json.loads((Path(case["prepare_evidence"])/"request.json").read_text())
    # Only the existing trusted credential loader reads .env. No key reaches
    # logs, manifests, source snapshots, candidates or native worker environments.
    os.environ["DEEPSEEK_API_KEY"]=load_api_key(ROOT,provider="deepseek")
    try:
        args=parse_args(["--inside",*case["live_command"][3:]])
        return inside(args,expected_request=request)
    finally:os.environ.pop("DEEPSEEK_API_KEY",None)
def evidence_path(log):
    matches=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
    require(len(matches)<=1,"Multiple candidate evidence paths")
    return Path(matches[0]) if matches else None
def stop(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGINT)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
def run(output,binding):
    from hecate_python_env import WORK,VENV
    from stage3_operator_run_r227 import assert_idle,serial_native_reservation
    from semantic_benchmark_execution import verify_numerical_report
    from audit_unified_candidate import verify_candidate
    from candidate_bundle import export_bundle,replay_bundle
    from retained_artifact_usage import size_bytes
    from run_stage2_guidance_retest import terminal_failure_layer
    plan=document(output/"plan.json");require(binding==plan["binding"],"Exact authorized binding")
    verify(plan);assert_idle()
    require(not (output/"dispatch.json").exists(),"Already dispatched; no automatic restart or duplicate calls")
    locks=[]
    for name in ("stage2-agent-paid-queue.lock","stage2-agent-campaign-execution.lock"):
        fd=os.open(WORK/"results"/name,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
    start=time.monotonic();rows=[dict(id=x,status="not_run") for x in IDS];failure=None
    paths={output};current_paths=set();last_storage=0.;last_sources=0.;started_cases=set()
    def boundary(case_begin=None):
        nonlocal last_storage,last_sources
        now=time.monotonic()
        require(now-start<plan["limits"]["max_wall_seconds"],"Batch time exhausted")
        if case_begin is not None:require(now-case_begin<plan["limits"]["per_case_wall_seconds"],"Case time exhausted")
        require(shutil.disk_usage(WORK).free>=plan["limits"]["min_free_disk_bytes"],"Free disk reserve")
        if now-last_sources>=10:verify(plan);last_sources=now
        if now-last_storage>=10:
            require(size_bytes([WORK/"results"])<=plan["limits"]["max_total_artifact_bytes"],"Total artifact budget")
            require(size_bytes(current_paths)<=plan["limits"]["max_case_artifact_bytes"],"Per-case artifact budget")
            last_storage=now
    def save():
        g=h=0
        for row in rows:
            g+=row.get("generation_attempts",0);h+=row.get("http_attempts",0)
        require(g<=12 and h<=48,"Original aggregate call budget")
        write(output/"report.json",seal(dict(format="poseidon-whole-models-live-v1",plan_binding=binding,rows=rows,
            passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"] not in ("passed","not_run","running") for r in rows),
            not_run=sum(r["status"]=="not_run" for r in rows),seconds=time.monotonic()-start,
            generation_attempts=g,http_attempts=h,maximum_generations=12,maximum_http_attempts=48,
            failure=failure,uncertain_calls_possible=bool(failure),automatic_restart=False,current_agent_generated=True)))
    try:
        with serial_native_reservation():
            boundary();write(output/"dispatch.json",seal(dict(plan_binding=binding,status="dispatching",automatic_restart=False)),True);save()
            for i,c in enumerate(plan["cases"]):
                boundary();case_begin=time.monotonic();folder=output/c["id"];current_paths={folder}
                write(folder/"claim.json",seal(dict(plan_binding=binding,request_id=c["request_id"],started_at=time.time(),automatic_restart=False)),True)
                rows[i]["status"]="running";started_cases.add(c["id"]);save()
                log=folder/"live.log";cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__)),"case","--inside","--output",str(output),"--case-id",c["id"]]
                with log.open("x") as stream:child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while child.poll() is None:
                        ev=evidence_path(log)
                        if ev:paths.add(ev);current_paths.add(ev);rows[i]["evidence"]=str(ev)
                        boundary(case_begin)
                        try:child.wait(timeout=1)
                        except subprocess.TimeoutExpired:pass
                finally:stop(child)
                ev=evidence_path(log);require(ev is not None,"Missing candidate evidence")
                paths.add(ev);current_paths.add(ev);report=json.loads((ev/"report.json").read_text())
                require(report["request_id"]==c["request_id"] and report["backend_identity"]==plan["backend_identity"],"Executed request/backend drift")
                require(report["status"]!="running","Uncertain live result; never restart")
                require(sha(ev/"model.json")==c["model_file_sha256"] or json.loads((ev/"model.json").read_text())==json.loads((ROOT/c["model_path"]).read_text()),"Executed model changed")
                g,h=provider_counts(report.get("provider_metrics",{}))
                verify_numerical_report(report,"poseidon-cpu-eva-w45-v1")
                row=dict(id=c["id"],status="failed",candidate_status=report["status"],evidence=str(ev),exit_code=child.returncode,
                    generation_attempts=g,http_attempts=h,report_sha256=sha(ev/"report.json"),failure_layer=terminal_failure_layer(report))
                rows[i]=row;save()
                if report["status"]=="passed":
                    require(report["llm_generation_validated"] and report["provider"]=="deepseek_api" and h>0,"Not a real Agent result")
                    audit=verify_candidate(ev);row["comparison"]={k:audit["comparison"][k] for k in ("compared_values","max_absolute_error")}
                    package=folder/"program";manifest=export_bundle(ev,package)
                    before=set((WORK/"results").glob("poseidon-package-replay-*"))
                    with (folder/"reload.log").open("x") as stream:
                        import contextlib
                        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):require(replay_bundle(package)==0,"Live program reload")
                    added=set((WORK/"results").glob("poseidon-package-replay-*"))-before;require(len(added)==1,"Unique reload evidence")
                    reload=next(iter(added));paths.add(reload);current_paths.add(reload)
                    rr=json.loads((reload/"report.json").read_text())
                    require(rr["status"]=="passed" and rr["encrypted_execution"] and rr["agent_calls"]==rr["compiler_calls"]==0,"Repeated package numerical execution")
                    row.update(status="passed",program=str(package),package_binding=manifest["binding"],reload_evidence=str(reload),
                        reload_comparison={k:rr["comparison"][k] for k in ("compared_values","max_absolute_error")},
                        package_generation_calls=0,package_compiler_calls=0)
                print("LIVE_WHOLE",c["id"],row["status"],row["failure_layer"],"generations",g,"HTTP",h,flush=True)
                save();boundary()
                metrics=report.get("provider_metrics",{});last=metrics.get("calls",[])[-1] if metrics.get("calls") else {}
                require(last.get("error") not in ("http_status_401","http_status_403","response_model_mismatch"),"Provider access/identity fault")
                require(row["failure_layer"] not in ("environment","harness_launch","key_setup","integrity","sandbox"),"Shared infrastructure fault")
            verify(plan)
    except BaseException as e:
        failure=type(e).__name__+": "+str(e)
        for row in rows:
            if row["status"]=="running":
                row["status"]="interrupted"
                if row.get("evidence"):
                    p=Path(row["evidence"])/"report.json"
                    if p.exists():
                        o=json.loads(p.read_text());g,h=provider_counts(o.get("provider_metrics",{}))
                        row.update(generation_attempts=g,http_attempts=h)
        raise
    finally:
        save()
        for fd in locks:os.close(fd)
    return int(any(r["status"]!="passed" for r in rows))
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","check","run","case"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--binding");p.add_argument("--inside",action="store_true");p.add_argument("--case-id",choices=IDS)
    a=p.parse_args()
    from hecate_python_env import WORK,VENV,enter_nix
    a.output=a.output.resolve();require(a.output.parent==WORK/"results","Only current native work root")
    if not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"])
        return enter_nix(cmd,seconds=43300 if a.mode=="run" else 300,environment="poseidon-cpu")
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned Nix/venv")
    if a.mode=="prepare":return prepare(a.output)
    plan=document(a.output/"plan.json");verify(plan)
    if a.mode=="check":print(json.dumps(dict(verified=True,binding=plan["binding"],paid_calls=0)));return 0
    if a.mode=="case":
        require(a.case_id and (a.output/"dispatch.json").exists() and (a.output/a.case_id/"claim.json").exists(),"Claimed parent dispatch required")
        claim=document(a.output/a.case_id/"claim.json")
        require(claim["plan_binding"]==plan["binding"],"Claim binding drift")
        write(a.output/a.case_id/"execution-started.json",seal(dict(plan_binding=plan["binding"],pid=os.getpid(),started_at=time.time(),automatic_restart=False)),True)
        return case_child(plan,a.case_id)
    return run(a.output,a.binding)
if __name__=="__main__":raise SystemExit(main())
