"""Poseidon static rejection adapter for immutable Benchmark tasks.

Only static validators run. Historical fixture identities remain separate from
current Poseidon request and RuntimePlan identities.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest,require,validate,signature
from benchmark_runner import DEFAULT,dump,strict_file
from rejection_benchmark import VERSION,tasks,fixture,identity,model
from unified_graph_contract import prepare,validate_candidate,validate_request
from semantic_benchmark_execution import CPU_CONFIGURATIONS,cpu_configuration,load_execution_suite,runtime_sources
from poseidon_benchmark_entries import pinned,enter,result_path,audit_poseidon
from hecate_python_env import WORK
from poseidon_artifact_gate import inspect
from poseidon_runtime_paths import SPEC
from poseidon_backend_identity import identity as backend_identity

FORMAT="poseidon-static-rejections-v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    result=runtime_sources()
    for name in ("poseidon_rejection_entries.py","poseidon_benchmark_entries.py"):
        p=Path(__file__).with_name(name)
        result[str(p.relative_to(ROOT))]=sha(p)
    result["scripts/benchmark.py"]=sha(ROOT/"scripts/benchmark.py")
    return result

def selection(suite,index_sha,offset,limit):
    require(type(offset) is int and offset>=0 and type(limit) is int and 1<=limit<=48,"Rejection shard bounds")
    load_execution_suite(suite,index_sha)
    catalog=strict_file(suite/"coverage.json",4*1024**2)["rejection_tasks"]
    require(catalog==tasks(),"Frozen rejection definitions changed")
    for t in catalog:
        require(digest({k:v for k,v in t.items() if k!="task_sha256"})==t["task_sha256"],"Frozen rejection task hash")
    selected=catalog[offset:offset+limit]
    require(selected,"Empty rejection shard")
    return selected,catalog

def controls(paths):
    require(len(paths) in (0,3) and len(paths)==len(set(paths)),"Provide zero or three distinct artifact controls")
    result=[];topologies=set()
    for path in paths:
        result_path(path)
        audited=audit_poseidon(path)
        g=strict_file(path/"model.json",4*1024**2)
        topology=signature(g,True)
        require(topology not in topologies,"Artifact controls require three different topologies")
        topologies.add(topology)
        report=strict_file(path/"report.json",8*1024**2)
        success=next(a for a in report["attempts"] if a.get("status")=="passed")
        folder=path/("attempt-%02d"%success["index"])/"output/bound"
        result.append(dict(evidence=str(path),topology=topology,
            report_sha256=sha(path/"report.json"),request_sha256=sha(path/"request.json"),
            model_sha256=sha(path/"model.json"),plan_sha256=sha(folder/"plan.json"),
            operator_spec_sha256=sha(SPEC),folder=str(folder),
            request_id=audited["request_id"],numerical_passed=audited["numerical_passed"],
            positive_execution_scope="previously independently executed Poseidon control; no new execution"))
    return result

def validate_controls_unchanged(rows):
    for r in rows:
        path=Path(r["evidence"]);folder=Path(r["folder"])
        for f,key in ((path/"report.json","report_sha256"),(path/"request.json","request_sha256"),
                      (path/"model.json","model_sha256"),(folder/"plan.json","plan_sha256"),(SPEC,"operator_spec_sha256")):
            require(sha(f)==r[key],"Artifact control identity changed")

def mapped_fixture(task,requests,control_rows,directory):
    name=task["requirement"].removeprefix("reject.");context=task["context"]
    historical=fixture(name,context)
    require(identity(historical["positive"])==task["positive_sha256"] and
            identity(historical["negative"])==task["negative_sha256"],"Historical rejection fixture changed")
    if task["layer"]=="model_validation":
        return historical["positive"],historical["negative"],validate,historical["expected_reason"],"exact_original_model_fixture",{}
    if task["layer"]=="candidate_ast_contract":
        request=requests[context]
        positive=dict(historical["positive"][1],request_id=request["request_id"])
        negative=dict(historical["negative"][1],request_id=request["request_id"])
        return positive,negative,lambda v:validate_candidate(v,request),historical["expected_reason"],"same_candidate_source_new_poseidon_request",dict(request_id=request["request_id"])
    if task["layer"]=="request_binding":
        positive=requests[context];negative=copy.deepcopy(positive)
        if name=="compiler_configuration":
            key,value=(("waterline",46),("pipeline","other"),("backend","plaintext"))[context]
            negative["compiler_configuration"][key]=value;expected="Changed compiler configuration"
        elif name=="request_integrity":
            negative["request_id"]="0"*64;expected="request hash mismatch"
        else:
            negative["layout"]["output_selectors"][0]=[0,1];expected="Changed unified physical layout"
        if name!="request_integrity":
            negative["request_id"]=digest({k:v for k,v in negative.items() if k!="request_id"})
        return positive,negative,validate_request,expected,"same_mutation_new_poseidon_request",dict(request_id=positive["request_id"])
    require(task["layer"]=="artifact_preflight","Unknown frozen rejection layer")
    require(len(control_rows)==3,"Poseidon artifact controls are required for artifact rejection tasks")
    r=control_rows[context];folder=Path(r["folder"])
    request=strict_file(Path(r["evidence"])/"request.json",4*1024**2)
    positive=strict_file(folder/"plan.json",16*1024**2);negative=copy.deepcopy(positive)
    operation=next((o for o in negative["execution"] if o["kind"]=="compute"),None)
    require(operation is not None,"Positive control has no compute instruction")
    operation["op"]="boot" if name=="unsupported_bootstrap" else "upscale"
    positive_path=directory/(task["id"]+".positive.json");negative_path=directory/(task["id"]+".negative.json")
    require(not positive_path.exists() and not negative_path.exists(),"Fresh static fixtures required")
    dump(positive_path,positive);dump(negative_path,negative)
    return positive_path,negative_path,lambda v:inspect(v,SPEC,request,bundle=folder/"bundle"),"unsupported capability","equivalent_forbidden_capability_new_runtime_plan",dict(
        positive_control_evidence=r["evidence"],positive_control_topology=r["topology"],
        mutated_ordinal=operation["ordinal"],operator_spec_sha256=r["operator_spec_sha256"])

def run_task(task,requests,control_rows,directory):
    row=dict(id=task["id"],original_task_sha256=task["task_sha256"],requirement=task["requirement"],
        context=task["context"],layer=task["layer"],status="failed",
        actual_compile=False,actual_sandbox_execution=False,actual_ciphertext_execution=False)
    if task["layer"]=="artifact_preflight" and not control_rows:
        row.update(status="blocked",diagnostic="Three independently audited Poseidon artifact controls are required")
        return row
    positive,negative,check,expected,relation,extra=mapped_fixture(task,requests,control_rows,directory)
    row.update(relation=relation,**extra,
        positive_identity=sha(positive) if isinstance(positive,Path) else identity(positive),
        negative_identity=sha(negative) if isinstance(negative,Path) else identity(negative))
    check(positive);row["positive_control_accepted"]=True
    try:check(negative)
    except ValueError as error:
        reason=str(error);require(expected in reason,"Unexpected rejection reason: "+reason)
        row.update(status="passed",negative_rejected=True,reason=reason)
        return row
    raise ValueError("Negative fixture unexpectedly accepted")

def output_path(path):
    result_path(path)
    require(not path.exists() and path.parent.is_dir(),"New platform results file required")
    base=(WORK/"results").resolve()
    p=path.parent
    while p!=base:
        require(not p.is_symlink(),"Symlinked output parent")
        require(p!=p.parent,"Output path outside platform results")
        p=p.parent
    require(not path.is_symlink(),"Symlinked output")

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["rejections"]);p.add_argument("--execute",action="store_true")
    p.add_argument("--output",type=Path);p.add_argument("--offset",type=int,default=0)
    p.add_argument("--limit",type=int,default=48);p.add_argument("--suite",type=Path,default=DEFAULT)
    p.add_argument("--frozen-suite-index-sha256");p.add_argument("--artifact-control",action="append",type=Path,default=[])
    p.add_argument("--compiler-configuration",choices=CPU_CONFIGURATIONS,default="poseidon-cpu-eva-w45-v1")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    require(a.offset>=0 and 1<=a.limit<=48,"Rejection shard bounds")
    require(len(a.artifact_control) in (0,3) and len(a.artifact_control)==len(set(a.artifact_control)),"Provide zero or three distinct artifact controls")
    require(a.execute or a.output is None,"Use --execute to write a report")
    if a.execute:
        require(a.output is not None,"New output report is required");output_path(a.output)
    if not pinned():
        require(not a.inside,"Locked Nix environment required")
        return enter(argv,300)
    selected,catalog=selection(a.suite,a.frozen_suite_index_sha256,a.offset,a.limit)
    frozen=sources();backend=backend_identity();cfg=cpu_configuration(a.compiler_configuration)
    requests={c:prepare(model(c),cfg["ckks_config_sha256"],cfg) for c in sorted({t["context"] for t in selected})}
    positive_controls=controls(a.artifact_control)
    plan=dict(format=FORMAT,contract=VERSION,backend="poseidon_cpu",suite=str(a.suite),
        original_index_sha256=sha(a.suite/"index.json"),coverage_sha256=sha(a.suite/"coverage.json"),
        planned=len(selected),total_tasks=len(catalog),offset=a.offset,tasks=selected,
        compiler_configuration=cfg,source_hashes=frozen,backend_identity=backend,
        requests={str(c):r["request_id"] for c,r in requests.items()},artifact_controls=positive_controls,
        artifact_controls_required=any(t["layer"]=="artifact_preflight" for t in selected) and not positive_controls,
        agent_calls=0,paid_calls=0,new_encrypted_executions=0,
        scope="static validators only; previous numerical controls remain separately bound")
    plan["binding"]=digest(plan)
    if not a.execute:
        print(json.dumps(plan,indent=2));return 0
    directory=a.output.with_name(a.output.stem+".fixtures")
    require(not directory.exists(),"Fresh static fixture directory required")
    directory.mkdir();start=time.monotonic();records=[];terminal_error=None;unchanged=True
    def save(final=False):
        passed=sum(r["status"]=="passed" for r in records)
        failed=sum(r["status"]=="failed" for r in records)
        blocked=sum(r["status"]=="blocked" for r in records)
        status="running"
        if final:
            status="stopped" if terminal_error else "passed" if passed==len(selected) else "blocked" if blocked and not failed else "failed"
        result=dict(plan,records=records,passed=passed,failed=failed,blocked=blocked,skipped=0,
            not_run=len(selected)-len(records),unselected=len(catalog)-len(selected),
            sources_unchanged=unchanged,status=status,seconds=time.monotonic()-start,
            actual_compile=False,actual_sandbox_execution=False,actual_ciphertext_execution=False)
        if terminal_error:result["terminal_error"]=terminal_error
        dump(a.output,result)
        return result
    save()
    try:
        for task in selected:
            require(time.monotonic()-start<240,"Static rejection time budget")
            require(sources()==frozen,"Checker/source changed at boundary")
            validate_controls_unchanged(positive_controls)
            try:record=run_task(task,requests,positive_controls,directory)
            except Exception as error:
                record=dict(id=task["id"],original_task_sha256=task["task_sha256"],
                    requirement=task["requirement"],context=task["context"],layer=task["layer"],status="failed",
                    error_type=type(error).__name__,diagnostic=str(error),
                    actual_compile=False,actual_sandbox_execution=False,actual_ciphertext_execution=False)
            records.append(record);save()
            require(sum(f.stat().st_size for f in directory.iterdir() if f.is_file())<=16*1024**2,"Static fixture space budget")
        require(sources()==frozen and backend_identity()==backend,"Final checker/backend identity changed")
        validate_controls_unchanged(positive_controls)
    except Exception as error:
        terminal_error=dict(error_type=type(error).__name__,diagnostic=str(error),
            failure_layer="budget" if "budget" in str(error) else "integrity")
        unchanged=sources()==frozen
    result=save(final=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","tasks","source_hashes","backend_identity","artifact_controls","compiler_configuration","requests")},indent=2))
    return int(result["status"]!="passed")
