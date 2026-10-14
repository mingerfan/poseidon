"""Poseidon helper execution and RuntimePlan audit for the public benchmark CLI.

Developer tools only. All generation is frozen scripted source, not Agent output.
"""
import argparse,copy,hashlib,json,math,os,re,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest,require
from benchmark_runner import DEFAULT,dump,strict_file
from hecate_python_env import WORK,VENV,enter_nix
from semantic_benchmark_execution import CPU_CONFIGURATIONS,cpu_configuration,load_execution_suite,runtime_sources,reference_check,verify_numerical_report
from upstream_helper_directed_cases import cases, tasks
from unified_graph_contract import prepare,validate_candidate
from upstream_candidate_helpers import BN_PROFILE,verify_sources
from poly_dependencies import verify as helper_dependency
from poseidon_backend_identity import identity
from audit_unified_candidate import verify_candidate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pinned():return os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV
def enter(argv,seconds):
    cmd=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/benchmark.py"),*argv,"--inside"]
    return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(cmd),
                     seconds=seconds,environment="poseidon-cpu")
def sources():
    out=runtime_sources()
    out[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    return out
def result_path(path):
    require(path.resolve().is_relative_to((WORK/"results").resolve()) and not path.is_symlink(),"Platform-local direct evidence path")
def storage(out,rows):
    used=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    used+=sum(r.get("evidence_bytes",0) for r in rows)
    require(used<=512*1024**2,"Helper shard artifact budget")
    total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],timeout=30,text=True).split()[0])
    require(total<=32*1024**3 and shutil.disk_usage(WORK).free>=4*1024**3,"Global artifact/free-space boundary")

def helper_selection(suite,index_sha,wanted,limit,override):
    _,index=load_execution_suite(suite,index_sha)
    frozen=strict_file(suite/"coverage.json",4*1024**2)["helper_directed_tasks"]
    require(frozen==tasks(),"Frozen helper definitions changed")
    for task in frozen:
        require(digest({k:v for k,v in task.items() if k!="task_sha256"})==task["task_sha256"],"Frozen helper task hash")
    names=wanted or [t["id"] for t in frozen]
    require(len(names)==len(set(names)) and set(names)<={t["id"] for t in frozen},"Unknown/repeated helper task")
    require(not wanted or len(wanted)<=limit,"Explicit tasks exceed shard bound")
    selected=[t for t in frozen if t["id"] in names][:limit]
    definitions={"upstream_"+r["name"]:r for r in cases()}
    rows=[]
    for task in selected:
        r=definitions[task["id"]]
        require(digest(r["model"])==task["model_sha256"] and r["required_helpers"]==task["required_helpers"],"Helper model/binding changed")
        cfg=cpu_configuration(override or ("poseidon-cpu-eva-w40-v1" if task["compiler_configuration"].endswith("w40-v1") else "poseidon-cpu-eva-w45-v1"))
        row=dict(task=task,source_sha256=hashlib.sha256(r["source"].encode()).hexdigest(),compiler_configuration=cfg,
                 status="ready",expected_calls=r.get("expected_calls",1))
        try:
            request=prepare(r["model"],cfg["ckks_config_sha256"],cfg,helper_profile=task["profile"],
                            helper_exercise=task["required_helpers"],chunk_period=task.get("chunk_period"))
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=r["source"]),request)
            row["request_id"]=request["request_id"]
        except (ValueError,TypeError) as error:row.update(status="blocked_preflight",diagnostic=str(error))
        rows.append(row)
    return rows,definitions,index

def verify_helper_result(row,definition,binding,*,recorded=False):
    report_path=Path(row["evidence"])/"report.json"
    require(sha(report_path)==row["report_sha256"] and row["binding"]==binding,"Helper result report/binding changed")
    report=strict_file(report_path,8*1024**2)
    verify_numerical_report(report,row["compiler_configuration"])
    if row["status"]=="passed":
        require(report["status"]=="passed" and row["exit_code"]==0,"Helper success status mismatch")
        checked=verify_candidate(Path(row["evidence"]))
        require(checked["request_id"]==row["request_id"] and checked["model_sha256"]==row["model_sha256"]==digest(definition["model"]),"Helper request/model identity")
        attempt=next(a for a in report["attempts"] if a["status"]=="passed")
        source=Path(row["evidence"])/("attempt-%02d"%attempt["index"])/"candidate.py"
        require(sha(source)==row["source_sha256"]==hashlib.sha256(definition["source"].encode()).hexdigest(),"Helper source identity")
        coverage=checked["helper_coverage"];trace=checked["upstream_helper_trace"]
        require(trace["actual_upstream_calls"]==definition.get("expected_calls",1),"Actual helper invocation count")
        require(coverage["actual_frontend_checked"] and coverage["finite_return_influence_checked"] and
                set(coverage["witnesses"])==set(definition["required_helpers"]),"Missing actual helper contribution")
        if recorded:
            require(row.get("audited") is True and row.get("comparison")==checked["comparison"] and
                    row.get("actual_calls")==trace["actual_upstream_calls"] and row.get("contribution")==coverage,
                    "Recorded helper comparison/contribution differs from audit")
        return checked
    require(report["status"]!="passed" or row["exit_code"]!=0,"Failed row contradicts candidate report")
    return None

def helper_main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["helper-directed"])
    p.add_argument("--suite",type=Path,default=DEFAULT);p.add_argument("--frozen-suite-index-sha256")
    p.add_argument("--task-id",action="append");p.add_argument("--limit",type=int,default=48)
    p.add_argument("--compiler-configuration",choices=CPU_CONFIGURATIONS)
    p.add_argument("--execute",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--output",type=Path);p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args(argv);require(1<=a.limit<=48,"Helper shard limit")
    if not pinned():require(not a.inside,"Locked Nix environment required");return enter(argv,1850)
    bindings,definitions,index=helper_selection(a.suite,a.frozen_suite_index_sha256,a.task_id,a.limit,a.compiler_configuration)
    frozen=sources();backend=identity()
    body=dict(format="poseidon-helper-benchmark-v1",backend="poseidon_cpu",suite=str(a.suite.resolve()),
         suite_index_sha256=sha(a.suite/"index.json"),bindings=bindings,sources=frozen,backend_identity=backend,
         helper_sources=verify_sources(backend="poseidon_cpu"),dependency=helper_dependency(),
         max_wall_seconds=1800,max_case_seconds=300,max_result_bytes=512*1024**2,
         global_result_bytes=32*1024**3,min_free_bytes=4*1024**3,native_concurrency=1,
         compiler_jobs=2,link_jobs=1,agent_calls=0,evaluation="manual_candidate_not_agent")
    plan=dict(body,binding=digest(body))
    if not a.execute:
        print(json.dumps(dict(mode="plan",backend="poseidon_cpu",binding=plan["binding"],tasks=[r["task"]["id"] for r in bindings],
             statuses={r["task"]["id"]:r["status"] for r in bindings},native_concurrency=1,
             max_wall_seconds=1800,max_result_mib=512,agent_calls=0),indent=2));return 0
    require(a.output is not None,"New result directory required");out=a.output;result_path(out)
    previous=[];prior_seconds=0.
    if out.exists():
        require(a.resume and strict_file(out/"plan.json",8*1024**2)==plan,"Changed task/source/environment requires new evidence")
        progress=strict_file(out/"progress.json",4*1024**2)
        require(progress["binding"]==plan["binding"] and 0<=progress["elapsed_seconds"]<=1800,"Resume binding/time")
        previous=progress["rows"];prior_seconds=progress["elapsed_seconds"]
        require(len({r["id"] for r in previous})==len(previous) and {r["id"] for r in previous}<={r["task"]["id"] for r in bindings},"Resume row identity")
        for row in previous:
            b=next(x for x in bindings if x["task"]["id"]==row["id"])
            require(row["binding"]==plan["binding"] and row["original_task_sha256"]==b["task"]["task_sha256"] and
                    row["source_sha256"]==b["source_sha256"] and row["request_id"]==b.get("request_id") and
                    row["compiler_configuration"]==b["compiler_configuration"]["name"],"Resume frozen task identity")
            if row.get("evidence"):verify_helper_result(row,definitions[row["id"]],plan["binding"],recorded=True)
        if len(previous)==len(bindings):
            print(json.dumps(dict(mode="verified_completed_resume",backend="poseidon_cpu",binding=plan["binding"],
                 completed=len(previous),new_encrypted_executions=0,agent_calls=0)),flush=True)
            return int(any(r["status"]!="passed" for r in previous))
    else:out.mkdir(parents=True);dump(out/"plan.json",plan)
    started=time.monotonic();rows=[]
    def completed_rows():
        found={r["id"]:r for r in [*previous,*rows]}
        return [found[b["task"]["id"]] for b in bindings if b["task"]["id"] in found]
    def save():
        known=completed_rows()
        report=dict(format=plan["format"],backend="poseidon_cpu",binding=plan["binding"],rows=known,planned=len(bindings),
            passed=sum(r["status"]=="passed" for r in known),failed=sum(r["status"]=="failed" for r in known),
            blocked=sum(r["status"]=="blocked_preflight" for r in known),not_run=len(bindings)-len(known),skipped=0,
            elapsed_seconds=prior_seconds+time.monotonic()-started,agent_calls=0,paid_calls=0)
        dump(out/"progress.json",report);dump(out/"report.json",report)
    save()
    for bound in bindings:
        require(sources()==frozen and identity()==backend,"Source/backend changed at boundary")
        old=next((r for r in previous if r["id"]==bound["task"]["id"]),None)
        if old is not None:rows.append(old);save();continue
        if prior_seconds+time.monotonic()-started>=1800:break
        storage(out,completed_rows())
        task=bound["task"];row=dict(id=task["id"],original_task_sha256=task["task_sha256"],binding=plan["binding"],
             model_sha256=task["model_sha256"],compiler_configuration=bound["compiler_configuration"]["name"],
             source_sha256=bound["source_sha256"],request_id=bound.get("request_id"),status=bound["status"])
        if bound["status"]!="ready":row["diagnostic"]=bound["diagnostic"];rows.append(row);save();continue
        r=definitions[task["id"]];folder=out/task["id"]
        require(not folder.exists(),"Uncheckpointed prior case preserved; use fresh output after review")
        folder.mkdir();dump(folder/"model.json",r["model"]);(folder/"candidate.py").write_text(r["source"])
        row["reference_check"]=reference_check(r["model"])
        cmd=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/run_candidate.py"),"--inside","--case",str(folder/"model.json"),
             "--golden-file",str(folder/"candidate.py"),"--max-repairs","0","--compiler-configuration",row["compiler_configuration"],
             "--unified-helpers",task["profile"]]
        if task.get("chunk_period") is not None:cmd+=["--unified-chunk-period",str(task["chunk_period"])]
        for helper in task["required_helpers"]:cmd+=["--unified-helper-exercise",helper]
        log=folder/"run.log";timed_out=False
        with log.open("x") as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(1,min(300,1800-prior_seconds-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        row.update(exit_code=code,timed_out=timed_out,status="failed",log_sha256=sha(log),failure_layer="harness_launch")
        matches=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
        if matches:
            evidence=Path(matches[-1]);result_path(evidence);report=strict_file(evidence/"report.json",8*1024**2)
            verify_numerical_report(report,row["compiler_configuration"])
            require(report["agent_calls"]==0 and not report["llm_generation_validated"],"Manual helper identity")
            require(report["request_id"]==row["request_id"],"Request drift")
            row.update(evidence=str(evidence),report_sha256=sha(evidence/"report.json"),
                 status="passed" if code==0 and report["status"]=="passed" else "failed",
                 failure_layer=report.get("failure_layer") or next((x.get("failure_layer") for x in report["attempts"] if x.get("failure_layer")),None),
                 evidence_bytes=sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file()))
            if row["status"]=="passed":
                checked=verify_helper_result(row,r,plan["binding"])
                row.update(audited=True,comparison=checked["comparison"],actual_calls=checked["upstream_helper_trace"]["actual_upstream_calls"],
                     contribution=checked["helper_coverage"])
        rows.append(row);save();print(task["id"],row["status"],row["failure_layer"],flush=True)
        if timed_out or row["failure_layer"] in ("harness_launch","environment","sandbox","integrity"):break
    require(sources()==frozen and identity()==backend,"Final source/backend binding");storage(out,rows);save()
    return int(len(rows)!=len(bindings) or any(r["status"]!="passed" for r in rows))

def audit_poseidon(folder):
    """Audit verified RuntimePlan instructions, not per-instruction native traces."""
    result_path(folder)
    report=strict_file(folder/"report.json",8*1024**2)
    require(report.get("backend")=="poseidon_cpu","Historical/non-Poseidon reports require their original read-only auditor")
    checked=verify_candidate(folder)
    successful=[a for a in report["attempts"] if a.get("status")=="passed"]
    require(len(successful)==1,"One successful numerical attempt")
    attempt=successful[0];out=folder/("attempt-%02d"%attempt["index"])/"output"
    plan=strict_file(out/"bound/plan.json",16*1024**2)
    meta=strict_file(out/"execution-metadata.json",32768)
    from poseidon_artifact_gate import PRIMES,inspect
    from poseidon_runtime_paths import SPEC
    request=strict_file(folder/"request.json",4*1024**2)
    gate=inspect(out/"bound/plan.json",SPEC,request,bundle=out/"bound/bundle")
    require(gate==attempt["artifact_gate"]["bound"],"Audited bound gate differs")
    require(meta==attempt["execution"] and meta["security"]=="tc128" and meta["degree"]==32768 and
            meta["q"]==list(PRIMES[:13]) and meta["p"]==list(PRIMES[13:]) and
            meta["executor_secret_key"] is False,"Actual fixed Poseidon/key boundary")
    values={v["id"]:v for v in plan["values"]}
    events=[dict(ordinal=o["ordinal"],operation=o["op"],inputs=[values[x] for x in o["inputs"]],output=values[o["output"]],
                 attributes=o.get("attrs",{})) for o in plan["execution"] if o["kind"]=="compute"]
    counts={name:sum(e["operation"]==name for e in events) for name in ("rescale","mod_switch","relinearize","rotate")}
    compiler=dict(rescale=dict(artifact_operations=counts["rescale"],exact_prime_rounding=gate["rescale_rounding"]),
         modswitch=dict(artifact_operations=counts["mod_switch"]),
         relinearization=dict(artifact_operations=counts["relinearize"],explicit_standalone_opcode=True),
         security_parameters=dict(actual_generated_parameters=meta,profile_sha256=report["compiler_profile_sha256"]),
         rotation_keys=dict(required_steps=sorted({e["attributes"]["steps"] for e in events if e["operation"]=="rotate"}),
              request_and_plan_checked=True,external_key_file=False,per_step_runtime_usage_observed=False))
    return dict(format="poseidon-runtime-plan-evidence-v1",backend="poseidon_cpu",evidence=str(folder),
         report_sha256=sha(folder/"report.json"),request_id=checked["request_id"],model_sha256=checked["model_sha256"],
         comparison=checked["comparison"],numerical_passed=True,events=events,compiler_evidence=compiler,
         plan_sha256=sha(out/"bound/plan.json"),operator_spec_sha256=sha(SPEC),
         execution_metadata_sha256=sha(out/"execution-metadata.json"),backend_identity=report["backend_identity"],
         level_convention="zero-based data modulus index; data count = level + 1",
         per_instruction_native_trace=False,new_encrypted_executions=0,agent_calls=0)

def compiler_main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["compiler-evidence"])
    p.add_argument("--candidate",action="append",type=Path,default=[]);p.add_argument("--batch-report",action="append",type=Path,default=[])
    p.add_argument("--suite",type=Path,default=DEFAULT);p.add_argument("--frozen-suite-index-sha256")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args(argv);result_path(a.output);require(not a.output.exists() and a.output.parent.is_dir(),"New direct audit file required")
    require(len(a.candidate)<=48 and len(a.batch_report)<=8,"Audit manifest bounds")
    if not pinned():require(not a.inside,"Locked Nix environment required");return enter(argv,400)
    load_execution_suite(a.suite,a.frozen_suite_index_sha256)
    ledger=strict_file(a.suite/"coverage.json",4*1024**2);ids={r["id"] for r in ledger["requirements"] if r["layer"]=="compiler"}
    require(ids=={"compiler.rescale","compiler.modswitch","compiler.relinearization","compiler.security_parameters","compiler.rotation_keys"},"Compiler ledger drift")
    candidates=list(a.candidate);parents={}
    for path in a.batch_report:
        result_path(path);batch=strict_file(path,16*1024**2);parents[str(path)]=sha(path)
        for row in [*batch.get("rows",[]),*batch.get("records",[]),*batch.get("reports",[])]:
            if row.get("evidence"):candidates.append(Path(row["evidence"]))
    candidates=list(dict.fromkeys(candidates));require(1<=len(candidates)<=48,"Distinct audit candidate bound")
    records=[]
    try:
        for folder in candidates:records.append(audit_poseidon(folder))
    except Exception as error:
        dump(a.output,dict(status="incomplete_audit_failure",backend="poseidon_cpu",error=str(error),records=records,
            planned=len(candidates),completed=len(records),new_encrypted_executions=0,agent_calls=0));raise
    observed={name:[] for name in ids}
    for r in records:
        e=r["compiler_evidence"]
        for name,key in (("rescale","rescale"),("modswitch","modswitch"),("relinearization","relinearization")):
            if e[key]["artifact_operations"]:observed["compiler."+name].append(r["evidence"])
        if e["rotation_keys"]["required_steps"]:observed["compiler.rotation_keys"].append(r["evidence"])
        observed["compiler.security_parameters"].append(r["evidence"])
    report=dict(format="poseidon-compiler-audit-v1",backend="poseidon_cpu",status="audited",records=records,
        planned=len(candidates),audited=len(records),observed_requirements=observed,
        observed_execution_counts={k:len(v) for k,v in observed.items()},numerical_passed=len(records),
        suite=str(a.suite),suite_index_sha256=sha(a.suite/"index.json"),coverage_sha256=sha(a.suite/"coverage.json"),
        batch_report_hashes=parents,checker_sources=sources(),new_encrypted_executions=0,agent_calls=0,
        per_instruction_native_trace=False,scope="verified compiled instructions and actual fixed-parameter numerical executions")
    dump(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("records","observed_requirements","checker_sources","batch_report_hashes")},indent=2))
    return 0
