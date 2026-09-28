"""Register independently audited current-version tasks without rerunning or rebinding."""
import argparse,json,sys
from pathlib import Path
from stage2_agent_campaign import ROOT,identity
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_runner import strict_file
from campaign_live_state import sealed,exclusive_json,check


def import_audit(batch,audit,index_path):
    from workspace_paths import RESULTS
    from semantic_benchmark_execution import runtime_sources
    from benchmark_graph import digest
    for p in (batch,audit,index_path.parent):
        if p.is_symlink() or p.resolve().parent!=RESULTS.resolve():raise ValueError("Direct result roots required")
    plan=strict_file(batch/"plan.json",8*1024**2);check_binding(plan)
    result=strict_file(audit/"report.json",16*1024**2);check_binding(result)
    index=strict_file(index_path,16*1024**2);check_binding(index)
    sources=runtime_sources()
    if plan["source_hashes"]!=sources or result["source_sha256"]!=digest(sources) or index["runtime_source_sha256"]!=digest(sources):
        raise ValueError("Historical result cannot become a current claim")
    if result["plan_binding"]!=plan["binding"]:raise ValueError("Audit/plan mismatch")
    for name,hsh in result["parents"].items():
        p=Path(name)
        if p.is_symlink() or not p.resolve().is_relative_to(RESULTS.resolve()) or p.name==".env" or sha(p)!=hsh:
            raise ValueError("Audit evidence changed or unsafe")
    planned={s["id"]:s for s in plan["cases"]};current={s["id"]:s for s in index["states"]}
    registry=RESULTS/"stage2-agent-evaluation-claims"
    if registry.is_symlink():raise ValueError("Unsafe registry")
    registry.mkdir(exist_ok=True);records=[]
    for row in result["rows"]:
        if row["status"] not in ("passed","failed"):continue
        spec=planned[row["id"]];state=current[row["id"]]
        token=identity(spec,sources)
        if state.get("evaluation_identity")!=token or row["model_sha256"]!=spec["model_sha256"] or row["request_id"]!=spec["request_id"]:
            raise ValueError("Different task/model/request configuration")
        if row["status"]=="passed":
            p=audit/(row["id"]+".audit.json")
            if sha(p)!=row["audit_sha256"]:raise ValueError("Per-case pass audit changed")
            check_binding(strict_file(p,16*1024**2))
        value=sealed(dict(format="poseidon-paid-task-claim-v1",evaluation_identity=token,
             task_id=spec["id"],model_sha256=spec["model_sha256"],request_id=spec["request_id"],owner=str(batch.resolve()),
             plan_binding=plan["binding"],state="terminal_audited",automatic_retry_allowed=False,
             audited_status=row["status"],audit=str(audit/"report.json"),audit_sha256=sha(audit/"report.json")))
        p=registry/(token+".json")
        try:exclusive_json(p,value);state_="imported"
        except FileExistsError:
            old=check(strict_file(p,1024**2))
            if any(old.get(k)!=value[k] for k in ("evaluation_identity","owner","plan_binding","task_id","model_sha256","request_id")):
                raise ValueError("Conflicting paid claim; do not overwrite")
            state_="already_claimed"
        records.append(dict(id=row["id"],status=state_,audited_status=row["status"],evaluation_identity=token))
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--batch",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True);p.add_argument("--campaign-index",type=Path,required=True)
    a=p.parse_args();print(json.dumps(import_audit(a.batch,a.audit,a.campaign_index),indent=2))
