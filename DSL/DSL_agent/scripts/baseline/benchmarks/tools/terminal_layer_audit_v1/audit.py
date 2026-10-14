"""Classify closed task terminal layers without rewriting historical labels."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
HISTORY=("agent-approved-pending-continuation-r1099","agent-approved-unstarted-continuation-r1113",
         "agent-approved-unstarted-continuation-r1141","agent-approved-unstarted-continuation-r1174")
GROUPS=("functional15","matrix_instance60","attention1")
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def classify(state,loop):
    require(state.get("phase")=="complete" and state.get("status") in
            ("budget_exhausted","service_error","validation_failed"),"Closed failed task")
    calls=[e for e in state.get("ledger",[]) if e.get("event")=="provider_checkpoint"]
    outcomes=calls[-1]["outcomes"] if calls else []
    require(type(state.get("http_attempts")) is int and len(outcomes)==state["http_attempts"] and
            all(e.get("status") in ("response_received","failed") for e in outcomes),"Resolved transport ledger")
    original=state.get("failure") or {}
    require(type(original) is dict,"Historical structured failure")
    terminal=loop.get("status") if type(loop) is dict else None
    if terminal in ("provider_failed","provider_exhausted"):
        layer="provider";code="provider_failure"
    elif terminal=="integrity_failed":
        layer="integrity";code="integrity_rejected"
    elif terminal=="infrastructure_failed":
        layer="environment";code="environment_failure"
    else:
        layer=original.get("layer","unknown");code=original.get("code","unknown")
    return dict(terminal_loop_status=terminal,terminal_layer=layer,terminal_code=code,
                historical_failure=dict(original),historical_label_differs=layer!=original.get("layer"),
                reason="Terminal loop and resolved transport ledger take precedence over previous candidate failure"
                    if layer!=original.get("layer") else "Retained final candidate or service layer",
                provider_failures=sum(e["status"]=="failed" for e in outcomes))
def inspect(output):
    output=Path(output)
    require(output.is_absolute() and output.resolve()==output and output.is_relative_to(RESULTS)
            and not output.exists(),"Fresh developer report")
    original=read(RESULTS/HISTORY[0]/"plan.json")
    approved={r["case"]["id"]:r["case"]["request_id"] for g in original["groups"].values() for r in g["cases"]}
    require(len(approved)==76,"Exact original approval scope")
    protected={}
    for label,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        for name,h in original[label].items():
            path=base/name;require(sha(path)==h,"Original frozen source/evidence")
            protected[str(path)]=h
    rows=[];hashes={};seen=set()
    for label in HISTORY:
        root=RESULTS/label
        for group in GROUPS:
            report=root/group/"report.json"
            if not report.exists():continue
            for entry in read(report)["rows"]:
                if entry["status"]!="failed":continue
                require(entry["id"] in approved and entry["id"] not in seen and
                        entry["request_id"]==approved[entry["id"]],"Exact closed approval identity")
                seen.add(entry["id"])
                task=root/group/entry["id"]/"tasks"/entry["task_id"]
                require(task==task.resolve() and not task.is_symlink(),"Closed task path")
                state=read(task/"state.json");loop_path=task/"loop.json"
                loop=read(loop_path) if loop_path.exists() else None
                require(state["generations"]==entry["result"]["generations"] and
                        state["http_attempts"]==entry["result"]["http_attempts"],"Closed task budget accounting")
                item=dict(id=entry["id"],task_id=entry["task_id"],task=str(task),status=state["status"],
                    request_id=entry["request_id"],generations=state["generations"],http_attempts=state["http_attempts"],
                    **classify(state,loop))
                layers=[]
                for index in range(state["generations"]):
                    validation=task/("validation-%d.json"%index)
                    if validation.exists():
                        value=read(validation);failure=value.get("failure") or {}
                        layers.append(dict(generation=index,status=value.get("status"),
                                           layer=failure.get("layer"),code=failure.get("code")))
                        hashes[str(validation)]=sha(validation)
                    else:layers.append(dict(generation=index,status="no_validation"))
                item["generation_layers"]=layers;rows.append(item)
                for p in [task/"state.json",loop_path]:
                    if p.exists():hashes[str(p)]=sha(p)
    require(all(sha(p)==h for p,h in hashes.items()),"Closed task evidence changed")
    require(all(sha(p)==h for p,h in protected.items()),"Original frozen evidence changed")
    counts={}
    for row in rows:counts[row["terminal_layer"]]=counts.get(row["terminal_layer"],0)+1
    report=dict(format="poseidon-closed-terminal-layer-audit-v1",original_planned=76,
        closed_failed=len(rows),terminal_layer_counts=counts,rows=rows,evidence_hashes=hashes,
        frozen_preservation_checks=len(protected),historical_labels_rewritten=False,
        provider_calls=0,compilations=0,encrypted_executions=0,
        scope="Closed failed tasks in the original approval only; no new pass or repair claim.",
        implementation_sha256=sha(__file__))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    d=inspect(p.parse_args().output)
    print(json.dumps({k:d[k] for k in ("closed_failed","terminal_layer_counts","historical_labels_rewritten",
                                     "provider_calls","compilations","encrypted_executions")}))
