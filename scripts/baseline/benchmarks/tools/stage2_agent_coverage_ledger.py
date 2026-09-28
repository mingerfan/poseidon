"""Join frozen Stage2 task definitions with audited Agent evidence; no generation."""
import argparse
from collections import Counter
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import load,strict_file,dump
from semantic_benchmark_execution import runtime_sources
SUITE=BASE/"benchmarks/semantic-v2-ledger-r38"
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked(path,parents,bound=False):
    path=Path(path)
    value=strict_file(path,16*1024**2)
    if bound and value.get("binding")!=digest({k:v for k,v in value.items() if k!="binding"}):
        raise ValueError("Invalid report binding: "+str(path))
    parents[str(path)]=sha(path)
    return value

def attach_history(task,spec,audit,source_digest,plan_binding):
    if spec["id"]!=task["id"] or audit["id"]!=task["id"]:
        raise ValueError("Task identity mismatch")
    if not task["model_sha256"]==spec["model_sha256"]==audit["model_sha256"]:
        raise ValueError("Model identity mismatch")
    if audit["request_id"]!=spec["request_id"] or audit["status"] not in ("passed","failed"):
        raise ValueError("Audit identity/status mismatch")
    if task["history"]:
        raise ValueError("Duplicate historical result")
    task["history"].append(dict(status=audit["status"],request_id=audit["request_id"],
        source_digest=source_digest,plan_binding=plan_binding,evidence=audit["evidence"],
        terminal_failure_layer=audit.get("terminal_failure_layer"),
        first_attempt_passed=audit["first_attempt_passed"],generations=audit["generations"],
        http_attempts=audit["http_attempts"]))
    task["historical_agent_status"]=audit["status"]
    # Static compatibility or manual replay never promotes old evidence to a new binding.
    task["current_source_agent_status"]="not_run"

def task_counts(tasks):
    groups=sorted({t["group"] for t in tasks})
    return {g:dict(total=sum(t["group"]==g for t in tasks),
        historical=dict(Counter(t["historical_agent_status"] for t in tasks if t["group"]==g)),
        current_source=dict(Counter(t["current_source_agent_status"] for t in tasks if t["group"]==g)))
        for g in groups}

def build():
    parents={}
    offline=checked(ROOT/"docs/baseline/stage2-offline-acceptance-r94.json",parents,True)
    pilot=checked(ROOT/"docs/baseline/stage2-agent-pilot-r97.json",parents,True)
    audit=checked(ROOT/"docs/baseline/stage2-agent-pilot-results-r98.json",parents,True)
    pending=checked(ROOT/"docs/baseline/stage2-guidance-retest-r100.json",parents,True)
    if offline["binding"]!="530d9c0877ea612ccd476afd948a2887012def3993d16e59eb6bd1d77685f567":
        raise ValueError("Offline acceptance identity changed")
    for key,value in (("pilot",pilot),("audit",audit)):
        expected=pending["parents"][key]
        if expected["binding"]!=value["binding"] or sha(ROOT/expected["path"])!=expected["sha256"]:
            raise ValueError("Frozen historical parent changed")
    if audit["plan_binding"]!=pilot["binding"]:raise ValueError("Pilot audit parent mismatch")
    if runtime_sources()!=pending["source_hashes"]:raise ValueError("Current source differs from prepared r100")
    for name,hsh in pending["proposal_files"].items():
        if sha(ROOT/name)!=hsh:raise ValueError("r100 proposal changed")
    # Verify the retained audit inputs rather than treating prose or status labels as proof.
    for name,hsh in audit["parents"].items():
        path=Path(name)
        if path.name==".env" or not path.resolve().is_relative_to(RESULTS):
            raise ValueError("Unexpected Agent evidence path")
        if sha(path)!=hsh:raise ValueError("Changed Agent audit evidence")
        parents[str(path)]=hsh
    rows,index=load(SUITE,check_sources=False) # Historical generator version is preserved.
    checked(SUITE/"index.json",parents)
    for shard in index["shards"]:parents[str(SUITE/shard["file"])]=shard["sha256"]
    ledger=checked(SUITE/"coverage.json",parents)
    if sha(SUITE/"coverage.json")!=index["coverage_sha256"]:raise ValueError("Ledger integrity")
    if sha(SUITE/"coverage.json")!=offline["definition_sha256"]:raise ValueError("Offline ledger identity")
    math_path=SUITE/ledger["components"]["math_contexts"]["file"]
    math=checked(math_path,parents)
    if sha(math_path)!=ledger["components"]["math_contexts"]["sha256"]:raise ValueError("Math component integrity")
    tasks=[];by_id={};free_by_hash={}
    def add(task_id,group,model_hash,**kw):
        if task_id in by_id:raise ValueError("Duplicate task ID")
        t=dict(id=task_id,group=group,model_sha256=model_hash,
               historical_agent_status="not_run",current_source_agent_status="not_run",
               current_request_prepared=False,history=[],**kw)
        tasks.append(t);by_id[task_id]=t
        return t
    for row in rows:
        t=add("free_"+row["model"]["id"],"free_primary",row["model_sha256"],
              model_id=row["model"]["id"],topology=row["topology"],split=row["split"],
              definition=dict(kind="frozen_corpus",index=str(SUITE/"index.json")))
        if row["model_sha256"] in free_by_hash:raise ValueError("Duplicate primary model")
        free_by_hash[row["model_sha256"]]=t["id"]
    for row in math["supplemental_models"]:
        if row["model_sha256"] in free_by_hash:raise ValueError("Supplement duplicates primary")
        t=add("free_supplement_"+row["model"]["id"],"free_supplement",row["model_sha256"],
              model_id=row["model"]["id"],topology=row["topology"],split=row["split"],
              definition=dict(kind="frozen_math_supplement",file=str(math_path)))
        free_by_hash[row["model_sha256"]]=t["id"]
    for q in offline["requirements"]:
        if q["layer"]!="model":continue
        for c in q["contexts"]:
            if c["model_sha256"] in free_by_hash:continue
            model_path=Path(c["evidence"])/"model.json"
            if not model_path.resolve().is_relative_to(RESULTS):raise ValueError("Unexpected model evidence path")
            model=checked(model_path,parents)
            if digest(model)!=c["model_sha256"]:raise ValueError("Context model changed")
            # Bind a source-checkout copy if available; do not turn results into the sole source.
            from model_decomposition import decompose
            source=BASE/"cases/stage2-gap-contexts-v1"/(c["id"]+".json")
            original=checked(source,parents)
            lowered,decomposition=decompose(original)
            if digest(lowered)!=c["model_sha256"]:raise ValueError("Source context decomposition drift")
            parents[str(BASE/"model_decomposition.py")]=sha(BASE/"model_decomposition.py")
            t=add("free_gap_"+model["id"],"free_gap_context",c["model_sha256"],
                  model_id=model["id"],topology=c["topology"],split="development_supplement",
                  definition=dict(kind="source_context_model",file=str(source),decomposition_binding=decomposition["binding_sha256"]))
            free_by_hash[c["model_sha256"]]=t["id"]
    for row in ledger["directed_tasks"]:
        add(row["id"],"directed_construction",row["model_sha256"],
            model_id=row["model_id"],requirement=row["requirement"],profile=row["profile"],
            exercise=row["exercise"],acceptance_kind=row["acceptance_kind"],
            frozen_task_sha256=row["task_sha256"],definition=dict(file=str(SUITE/"coverage.json")))
    for row in ledger["helper_directed_tasks"]:
        add(row["id"],"directed_helper",row["model_sha256"],model_id=row["model"]["id"],
            profile=row["profile"],required_helpers=row["required_helpers"],topology=row["topology"],
            frozen_task_sha256=row["task_sha256"],definition=dict(file=str(SUITE/"coverage.json")))
    from fixed_polynomial_cases import cases
    fixture_path=Path(__file__).with_name("fixed_polynomial_cases.py")
    parents[str(fixture_path)]=sha(fixture_path)
    for row in cases():
        # Only public mathematical model/profile identities enter the task index.
        add(row["id"],"directed_fixed_polynomial",row["model_sha256"],model_id=row["model"]["id"],
            profile=row["profile"],required_helpers=[row["helper"]],context=row["context"],
            definition=dict(file=str(fixture_path),fixture_id=row["id"]))
    specs={c["id"]:c for c in pilot["cases"]}
    if set(specs)!={r["id"] for r in audit["rows"]}:raise ValueError("Pilot coverage mismatch")
    for row in audit["rows"]:
        t=by_id[row["id"]]
        attach_history(t,specs[row["id"]],row,audit["source_sha256"],pilot["binding"])
    for spec in pending["cases"]:
        t=by_id[spec["id"]]
        if t["model_sha256"]!=spec["model_sha256"]:raise ValueError("Pending model drift")
        t["current_request_prepared"]=True
        t["pending_proposal"]=dict(binding=pending["binding"],request_id=spec["request_id"],
                                   status="awaiting_separate_paid_approval")
    partitions=[]
    for q in offline["requirements"]:
        contexts=[]
        for c in q["contexts"]:
            task_id=c.get("task_id")
            if q["layer"]=="model":task_id=free_by_hash[c["model_sha256"]]
            t=by_id.get(task_id)
            contexts.append(dict(task_id=task_id,topology=c.get("topology"),
                historical_agent_status=t["historical_agent_status"] if t else "artifact_agent_audit_missing",
                current_source_agent_status=t["current_source_agent_status"] if t else "artifact_agent_audit_missing",
                historical_offline_evidence=c.get("evidence") or c.get("evidence_relative_to_results"),
                historical_offline_status=q["status"]))
        passed_ids={c["task_id"] for c in contexts if c["historical_agent_status"]=="passed"}
        if q["status"]=="backend_blocked":state="backend_blocked"
        elif q["layer"]=="rejection":state="static_controls_not_agent_generation"
        elif q["layer"]=="compiler":state="agent_artifact_coverage_unverified"
        else:state="incomplete_agent_contexts" # Never infer all-context success from a single pilot.
        partitions.append(dict(id=q["id"],layer=q["layer"],scope=q["claim_scope"],
            offline_status=q["status"],agent_status=state,backend_blocker=q["blocker"],
            required_distinct_contexts=0 if q["layer"]=="rejection" or q["status"]=="backend_blocked" else 3,
            historically_passed_designated_tasks=sorted(passed_ids),contexts=contexts,
            current_source_three_contexts_verified=False))
    if len(partitions)!=401 or len(tasks)!=2030:raise ValueError("Unexpected denominator")
    if Counter(t["historical_agent_status"] for t in tasks)!=Counter(passed=6,failed=6,not_run=2018):
        raise ValueError("Agent accounting mismatch")
    result=dict(format="poseidon-stage2-agent-coverage-ledger-r101-v1",
        source_digest=digest(runtime_sources()),parents=parents,
        historical_plan_binding=pilot["binding"],historical_audit_binding=audit["binding"],
        pending_plan_binding=pending["binding"],primary_unique_models=1200,
        task_inventory_count=len(tasks),group_counts=task_counts(tasks),
        task_inventory_scope="1200 primary + 92 math supplement + 30 existing gap-context models; 627 construction + 60 helpers + 21 fixed-polynomial tasks. These are evaluation tasks, not 2030 unique new models.",
        tasks=tasks,partitions=partitions,
        partition_states=dict(Counter(p["agent_status"] for p in partitions)),
        original_rule_baseline_status=offline["original_primary_status"],
        api_classification_count=len(offline["upstream_api"]),
        agent_api_all_parameters_proven=False,
        no_production_source_changes=True,paid_calls=0,new_encrypted_executions=0,stage2_complete=False,
        limitations=[
            "This inventory does not promise every task can currently prepare, compile or execute; it is not a paid dispatch plan.",
            "Old results remain historical; r99 static compatibility/manual replay is not new Agent evaluation.",
            "Compiler artifact semantics need Agent artifact audits; numerical success alone does not prove rescale/modswitch/relinearization evidence.",
            "Bootstrap-blocked partitions and static rejection controls are separate; no skipped task is a pass.",
            "Each future executable request, contract, compiler profile and source revision requires its own frozen binding.",
            "All-context Agent acceptance remains unproven, including any additional tasks discovered by future API/partition audits."],
        runner_sha256=sha(Path(__file__)))
    result["binding"]=digest(result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve previous ledger; use a fresh path")
    report=build();dump(a.output,report)
    print(json.dumps({k:report[k] for k in ("binding","task_inventory_count","group_counts","partition_states","paid_calls")},indent=2))
