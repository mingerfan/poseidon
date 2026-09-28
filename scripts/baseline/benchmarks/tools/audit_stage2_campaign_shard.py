"""Independent frozen campaign-shard audit; reservations are never passes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    from hecate_python_env import VENV,enter_nix
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
    from workspace_paths import RESULTS
    from stage2_agent_pilot_plan import check_binding
    from semantic_benchmark_execution import runtime_sources
    from stage2_agent_provenance import provider_accounting
    from audit_stage2_live_candidate import verify_candidate,verify_files
    if a.batch.resolve().parent!=RESULTS.resolve() or a.batch.is_symlink():p.error("Direct result batch required")
    if not a.output.resolve().is_relative_to(RESULTS.resolve()) or a.output.exists():p.error("Fresh result directory required")
    plan=strict_file(a.batch/"plan.json",8*1024**2);check_binding(plan)
    batch=strict_file(a.batch/"report.json",8*1024**2);check_binding(batch)
    if batch["plan_binding"]!=plan["binding"]:raise ValueError("Batch plan mismatch")
    if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime drift")
    for name,hsh in plan["proposal_files"].items():
        if sha(ROOT/name)!=hsh:raise ValueError("Proposal source drift")
    planned={c["id"]:c for c in plan["cases"]}
    runs={r["id"]:r for r in batch["rows"]}
    if len(runs)!=len(batch["rows"]) or not set(runs)<=set(planned):raise ValueError("Duplicate/unknown tasks")
    a.output.mkdir()
    outcomes=[];usage=Counter();attempt_layers=Counter()
    generations=http=compiled=executed=numeric=0
    paths={str(a.batch/"plan.json"):sha(a.batch/"plan.json"),str(a.batch/"report.json"):sha(a.batch/"report.json")}
    for case_id,spec in planned.items():
        row=dict(id=case_id,track=spec["track"],status="not_run",first_attempt_passed=False,
                 model_sha256=spec["model_sha256"],request_id=spec["request_id"])
        run=runs.get(case_id)
        if run is None:
            row["status"]="interrupted_or_not_run" if (a.batch/case_id/"launch.json").exists() else "not_run"
            outcomes.append(row);continue
        if "evidence" not in run:
            if run.get("status")=="reserved_elsewhere":
                row.update(status="reserved_elsewhere",claim_owner=run["claim_owner"],claim_binding=run["claim_binding"])
            else:row.update(status="infrastructure_failed",diagnostic="No candidate evidence")
            outcomes.append(row);continue
        folder=Path(run["evidence"])
        if folder.is_symlink() or folder.resolve().parent!=RESULTS.resolve():raise ValueError("Unsafe evidence")
        report_path=folder/"report.json"
        if sha(report_path)!=run["report_sha256"]:raise ValueError("Changed candidate report")
        report=strict_file(report_path,8*1024**2)
        if report["status"]=="running":raise ValueError("Candidate not terminal")
        if strict_file(folder/"request.json",131072)!=spec["request"]:raise ValueError("Request changed")
        if digest(strict_file(folder/"model.json",131072))!=spec["model_sha256"]:raise ValueError("Model changed")
        verify_files(BASE,report["source_hashes"]);verify_files(folder,report["frozen_hashes"])
        accounting=provider_accounting(report,plan["paid_configuration"])
        attempts=report.get("attempts",[])
        # Completed ordinary cases must have one retained response per evaluated generation.
        if report["status"] in ("passed","repair_budget_exhausted"):
            if len(attempts)!=len(accounting["received"]):raise ValueError("Response/attempt count")
            for attempt,call in zip(attempts,accounting["received"]):
                idx=attempt["index"];directory=folder/("attempt-%02d"%idx)
                raw=(directory/"response.txt").read_bytes().decode("utf-8")
                if idx!=call["generation_index"] or len(raw)!=call.get("content_characters"):raise ValueError("Retained response mismatch")
                feedback=strict_file(directory/"feedback.json",131072)
                if feedback!=report["loop"]["feedback_history"][idx]:raise ValueError("Feedback mismatch")
                paths[str(directory/"response.txt")]=sha(directory/"response.txt")
                paths[str(directory/"feedback.json")]=sha(directory/"feedback.json")
        for call in report["provider_metrics"]["calls"]:
            usage.update({k:v for k,v in call.get("usage",{}).items() if type(v)is int})
        generations+=accounting["generations"];http+=accounting["http_attempts"]
        diagnostics=[]
        for attempt in attempts:
            if attempt.get("artifact_hashes"):
                verify_files(folder/("attempt-%02d"%attempt["index"])/"output",attempt["artifact_hashes"])
            layer=attempt.get("failure_layer") or ("complete" if attempt.get("status")=="passed" else "unknown")
            attempt_layers[layer]+=1
            compiled+=int(bool(attempt.get("compiled")));executed+=int(bool(attempt.get("executed")))
            numeric+=int(bool(attempt.get("numerically_correct")))
            diagnostics.append(dict(index=attempt["index"],status=attempt.get("status"),
                                    failure_layer=layer,diagnostic=attempt.get("diagnostic"),
                                    compiled=bool(attempt.get("compiled")),executed=bool(attempt.get("executed"))))
        row.update(evidence=str(folder),provider_status=report["status"],generations=accounting["generations"],
                   http_attempts=accounting["http_attempts"],attempts=diagnostics,
                   terminal_failure_layer=diagnostics[-1]["failure_layer"] if report["status"]!="passed" and diagnostics else None,
                   status="failed",report_sha256=sha(report_path))
        paths[str(report_path)]=sha(report_path)
        if report["status"]=="passed":
            try:
                audit=verify_candidate(folder,plan["paid_configuration"])
                audit.update(plan_binding=plan["binding"],case_id=case_id,
                             audit_source_sha256=sha(Path(__file__).with_name("audit_stage2_live_candidate.py")),
                             provenance_source_sha256=sha(Path(__file__).with_name("stage2_agent_provenance.py")))
                audit["binding"]=digest(audit);dump(a.output/(case_id+".audit.json"),audit)
                row.update(status="passed",audit_sha256=sha(a.output/(case_id+".audit.json")),
                           first_attempt_passed=audit["provenance"]["first_attempt_passed"],
                           comparison={k:audit["comparison"][k] for k in ("compared_values","mae","max_absolute_error","atol","rtol")},
                           construction_coverage=audit["coverage"],helper_coverage=audit["helper_coverage"])
            except (ValueError,KeyError,AssertionError) as error:
                row.update(status="audit_failed",audit_diagnostic=type(error).__name__+": "+str(error))
        outcomes.append(row)
        dump(a.output/"progress.json",outcomes)
    if generations>plan["limits"]["maximum_generations"] or http>plan["limits"]["maximum_http_attempts"]:
        raise ValueError("Approved API budget exceeded")
    for name,hsh in paths.items():
        if sha(Path(name))!=hsh:raise ValueError("Evidence changed during audit")
    if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime changed during audit")
    result=dict(format="poseidon-stage2-agent-pilot-independent-audit-v1",plan_binding=plan["binding"],
                source_sha256=digest(plan["source_hashes"]),runner_sha256=sha(Path(__file__)),
                parents=paths,rows=outcomes,planned=len(planned),
                statuses=dict(Counter(r["status"] for r in outcomes)),
                tracks={t:dict(Counter(r["status"] for r in outcomes if r["track"]==t)) for t in sorted({r["track"] for r in outcomes})},
                first_attempt_passed=sum(r["first_attempt_passed"] for r in outcomes),
                successful_after_repair=sum(r["status"]=="passed" and not r["first_attempt_passed"] for r in outcomes),
                generations=generations,http_attempts=http,transport_retries=http-generations,
                usage=dict(usage),attempt_failure_layers=dict(attempt_layers),
                real_compiled_attempts=compiled,real_executed_attempts=executed,numerically_correct_attempts=numeric,
                new_paid_calls=0,new_encrypted_executions=0,stage2_complete=False,
                scope="One frozen campaign shard; reservations/interruptions/failed tasks are not passes",
                limitations=["Retained API ledger is not a billing receipt or provider-signed proof",
                             "Earlier attempt failure layers remain separate from terminal failure layers"])
    result["binding"]=digest(result);dump(a.output/"report.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("parents","rows")},indent=2))
    return int(result["statuses"].get("audit_failed",0)>0)

if __name__=="__main__":
    raise SystemExit(main())
