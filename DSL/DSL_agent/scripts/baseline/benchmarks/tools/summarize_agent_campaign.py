"""Read-only current-version campaign accounting. No provider calls or pass inference."""
import argparse,json,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from stage2_agent_campaign import ROOT,identity
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_graph import digest
from benchmark_runner import strict_file,dump


def aggregate(rows):
    counts=dict(Counter(r["status"] for r in rows));audited=counts.get("passed",0)+counts.get("failed",0)
    return dict(planned=len(rows),statuses=counts,audited=audited,
                audited_task_success_rate=counts.get("passed",0)/audited if audited else None,
                all_planned_task_success_rate=counts.get("passed",0)/len(rows) if rows else None,
                all_tasks_audited=bool(rows) and audited==len(rows))


def merge_audited(states,row,batch_binding,audit_binding):
    task=states[row["id"]]
    if row["status"] not in ("passed","failed"):return
    old=task.get("audited_result")
    value=dict(batch_binding=batch_binding,report_sha256=row["report_sha256"],status=row["status"],
               evidence=row["evidence"],request_id=row["request_id"])
    if old is not None and old!=value:raise ValueError("Conflicting repeated evaluation; no best-result selection")
    task["audited_result"]=value;task["status"]=row["status"]
    task.setdefault("supporting_audits",[])
    if audit_binding not in task["supporting_audits"]:task["supporting_audits"].append(audit_binding)
    task["first_attempt_passed"]=row["first_attempt_passed"]
    task["generations"]=row["generations"];task["http_attempts"]=row["http_attempts"]
    task["terminal_failure_layer"]=row.get("terminal_failure_layer")
    task["ever_compiled"]=any(x["compiled"] for x in row.get("attempts",[]))
    task["ever_encrypted"]=any(x["executed"] for x in row.get("attempts",[]))
    if row["status"]=="passed":
        task["comparison"]=row["comparison"]
        task["construction_coverage"]=row.get("construction_coverage")
        task["helper_coverage"]=row.get("helper_coverage")


def partition_status(partition,states):
    p=dict(partition)
    if p["current_status"]=="backend_blocked":p["agent_status"]="backend_blocked";return p
    if p["layer"]=="rejection":p["agent_status"]="static_controls_separate";return p
    if p["layer"]=="compiler":p["agent_status"]="current_agent_artifact_audit_required";return p
    designated={c["task_id"] for c in p["contexts"] if c["task_id"] is not None}
    passed=sorted(t for t in designated if states.get(t,{}).get("status")=="passed")
    p["passed_designated_contexts"]=passed
    p["missing_or_failed_designated_contexts"]=sorted(designated-set(passed))
    p["agent_status"]="designated_contexts_passed" if len(passed)>=p["required_distinct_contexts"] and not p["missing_or_failed_designated_contexts"] else "incomplete_agent_contexts"
    return p


def summarize(campaign,audits,batches,compiler_audits=()):
    from workspace_paths import RESULTS
    from semantic_benchmark_execution import runtime_sources
    if campaign.is_symlink() or campaign.resolve().parent.parent!=RESULTS.resolve():raise ValueError("Direct campaign result index required")
    index=strict_file(campaign,16*1024**2);check_binding(index)
    sources=runtime_sources()
    if digest(sources)!=index["runtime_source_sha256"]:raise ValueError("Current source differs from campaign")
    parents={str(campaign):sha(campaign)};specs={}
    for ref in index["shards"]:
        f=campaign.parent/ref["file"]
        if Path(ref["file"]).name!=ref["file"] or sha(f)!=ref["sha256"]:raise ValueError("Shard changed")
        plan=strict_file(f,8*1024**2);check_binding(plan)
        if plan["binding"]!=ref["binding"] or plan["source_hashes"]!=sources:raise ValueError("Shard identity")
        parents[str(f)]=sha(f)
        for s in plan["cases"]:
            if s["id"] in specs:raise ValueError("Duplicate campaign task")
            specs[s["id"]]=s
    states={s["id"]:dict(s,status="not_run" if s["status"]=="prepared_not_run" else s["status"]) for s in index["states"]}
    if len(states)!=index["planned"]:raise ValueError("Index denominator changed")
    usage=Counter();seen_batches={};out_of_scope=[]
    for folder in dict.fromkeys(audits):
        if folder.is_symlink() or folder.resolve().parent!=RESULTS.resolve():raise ValueError("Direct audit directory required")
        f=folder/"report.json";a=strict_file(f,16*1024**2);check_binding(a);parents[str(f)]=sha(f)
        if a["source_sha256"]!=digest(sources):
            out_of_scope.append(dict(audit=str(folder),reason="historical_runtime_version",binding=a["binding"]));continue
        for name,hsh in a["parents"].items():
            q=Path(name)
            if q.is_symlink() or q.name==".env" or not q.resolve().is_relative_to(RESULTS.resolve()) or sha(q)!=hsh:
                raise ValueError("Audit input changed/unsafe")
            parents[str(q)]=hsh
        for row in a["rows"]:
            if row["id"] not in specs:raise ValueError("Audit task outside frozen campaign")
            spec=specs[row["id"]]
            if row["request_id"]!=spec["request_id"] or row["model_sha256"]!=spec["model_sha256"]:
                out_of_scope.append(dict(id=row["id"],audit=str(folder),reason="different_request_or_model"));continue
            if row["status"]=="passed":
                q=folder/(row["id"]+".audit.json")
                if sha(q)!=row["audit_sha256"]:raise ValueError("Per-case pass audit changed")
                check_binding(strict_file(q,16*1024**2));parents[str(q)]=sha(q)
            merge_audited(states,row,a["plan_binding"],a["binding"])
        accounting={k:a[k] for k in ("generations","http_attempts","usage")}
        if a["plan_binding"] in seen_batches:
            if seen_batches[a["plan_binding"]]!=accounting:raise ValueError("Duplicate audit accounting differs")
        else:seen_batches[a["plan_binding"]]=accounting;usage.update(a["usage"])
    observations=[]
    for folder in dict.fromkeys(batches):
        if folder.is_symlink() or folder.resolve().parent!=RESULTS.resolve():raise ValueError("Direct batch directory required")
        f=folder/"plan.json";plan=strict_file(f,8*1024**2);check_binding(plan)
        if plan["source_hashes"]!=sources:
            out_of_scope.append(dict(batch=str(folder),reason="historical_runtime_version"));continue
        parents[str(f)]=sha(f)
        terminal=(folder/"report.json").exists()
        # During resume the prior terminal report remains until the next checkpoint.
        # Prefer current progress for observations; neither file awards a pass.
        progress=strict_file(folder/"progress.json",8*1024**2) if (folder/"progress.json").exists() else strict_file(folder/"report.json",8*1024**2)["rows"] if terminal else []
        observed={r["id"]:r for r in progress};live=[]
        for spec in plan["cases"]:
            task=states.get(spec["id"]);expected=specs.get(spec["id"])
            if task is None or expected is None or identity(spec,sources)!=expected["evaluation_identity"]:
                out_of_scope.append(dict(id=spec["id"],batch=str(folder),reason="different_evaluation"));continue
            if task.get("audited_result"):continue
            if spec["id"] in observed:
                task["status"]="reserved_elsewhere" if observed[spec["id"]]["status"]=="reserved_elsewhere" else "terminal_pending_audit"
                task["batch"]=str(folder);continue
            process=folder/spec["id"]/"process.json";launch=folder/spec["id"]/"launch.json"
            if not launch.exists():continue
            running=False
            if process.exists():
                pid=strict_file(process,131072)["pid"];expected_cmd=strict_file(launch,131072)["command"]
                try:actual=Path("/proc")/str(pid)/"cmdline";argv=actual.read_bytes().rstrip(b"\0").split(b"\0");running=[s.decode() for s in argv]==expected_cmd
                except (FileNotFoundError,ProcessLookupError):pass
                if running:live.append(dict(id=spec["id"],pid=pid))
            task["status"]="running_verified" if running else "launched_unconfirmed";task["batch"]=str(folder)
        observations.append(dict(batch=str(folder),terminal_report_present=terminal,currently_running=bool(live),live_processes=live))
    rows=list(states.values());partitions=[partition_status(p,states) for p in index["semantic_partitions"]]
    compiler_records={}
    from benchmark_graph import signature
    from audit_current_agent_compiler import BUILD
    for f in dict.fromkeys(compiler_audits):
        if f.is_symlink() or not f.resolve().is_relative_to(RESULTS.resolve()):raise ValueError("Unsafe compiler audit")
        evidence=strict_file(f,16*1024**2);check_binding(evidence);parents[str(f)]=sha(f)
        if evidence["evaluated_runtime_digest"]!=digest(sources):raise ValueError("Historical compiler evidence")
        if evidence["runner_sha256"]!=sha(Path(__file__).with_name("audit_current_agent_compiler.py")):raise ValueError("Compiler auditor version changed")
        for name,hsh in evidence["parents"].items():
            q=Path(name)
            if q.name==".env" or not any(q.resolve().is_relative_to(base) for base in (ROOT.resolve(),RESULTS.resolve())) or sha(q)!=hsh:
                raise ValueError("Compiler evidence parent changed")
            parents[str(q)]=hsh
        for record in evidence["records"]:
            task=states.get(record["id"]);spec=specs.get(record["id"])
            if task is None or spec is None or task["status"]!="passed":raise ValueError("Compiler task needs a supplied successful Agent audit")
            if (record["request_id"]!=spec["request_id"] or record["model_sha256"]!=spec["model_sha256"] or
                record["topology"]!=signature(spec["model"],True) or record["runtime_sha256"]!=sha(BUILD/"lib/libSEAL_HEVM.so")):
                raise ValueError("Compiler task identity/binary mismatch")
            item=dict(id=record["id"],topology=record["topology"],evidence=record["compiler_evidence"])
            if item["id"] in compiler_records and compiler_records[item["id"]]!=item:raise ValueError("Conflicting compiler evidence")
            compiler_records[item["id"]]=item
    for part in partitions:
        if part["layer"]!="compiler":continue
        key=part["id"].split(".",1)[1]
        relevant=[r for r in compiler_records.values() if r["evidence"][key]]
        topologies=sorted({r["topology"] for r in relevant})
        part.update(compiler_case_ids=sorted(r["id"] for r in relevant),compiler_distinct_topologies=len(topologies),
                    agent_status="current_compiler_contexts_passed" if len(topologies)>=3 else "current_agent_artifact_audit_required")
    for name,hsh in parents.items():
        if sha(Path(name))!=hsh:raise ValueError("Immutable evidence changed during summary")
    result=dict(format="poseidon-agent-campaign-summary-v1",observed_at_utc=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(Path(__file__)),campaign_binding=index["binding"],
                runtime_source_sha256=digest(sources),parents=parents,rows=rows,summary=aggregate(rows),
                groups={g:aggregate([r for r in rows if r["group"]==g]) for g in sorted({r["group"] for r in rows})},
                semantic_partitions=partitions,partition_states=dict(Counter(p["agent_status"] for p in partitions)),
                audited_generation_count=sum(v["generations"] for v in seen_batches.values()),
                audited_http_attempt_count=sum(v["http_attempts"] for v in seen_batches.values()),audited_usage=dict(usage),
                observations=observations,out_of_scope=out_of_scope,paid_calls=0,
                current_artifact_audit_complete=all(p["agent_status"]=="current_compiler_contexts_passed" for p in partitions if p["layer"]=="compiler"),stage2_complete=False,
                limitation="Audits count success; running and preliminary reports never do. Usage excludes unfinished/unaudited batches and is not a billing receipt.")
    result["binding"]=digest(result);return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-index",type=Path,required=True)
    p.add_argument("--audit",type=Path,action="append",default=[]);p.add_argument("--batch",type=Path,action="append",default=[])
    p.add_argument("--compiler-audit",type=Path,action="append",default=[])
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error("Fresh summary file required")
    result=summarize(a.campaign_index,a.audit,a.batch,a.compiler_audit);dump(a.output,result)
    print(json.dumps({k:result[k] for k in ("binding","summary","partition_states","audited_generation_count","paid_calls")},indent=2))
