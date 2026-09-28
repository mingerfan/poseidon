"""Join explicitly versioned mathematical context evidence, never rewrite old states.

This is a coverage supplement, not a replacement frozen corpus or a full DSL pass.
Rechecks encrypted artifacts and independent numerics without compiling again.
"""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump
from benchmark_semantics import features
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(report):
    if digest({k:v for k,v in report.items() if k!="binding"})!=report["binding"]:
        raise ValueError("Report binding mismatch")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--closure",type=Path,required=True)
    p.add_argument("--batch",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve previous audit")
    from hecate_python_env import enter_nix,VENV
    if not a.inside:
        command=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(command),seconds=300)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
    from audit_unified_candidate import verify_candidate,verify_files
    from historical_context_audit import verify_saved
    old=strict_file(a.closure,4*1024**2);bound(old)
    contexts={r["id"]:[dict(**t,source_binding=old["binding"]) for t in r["tasks"] if t["state"]=="verified"]
              for r in old["math_partitions"]}
    supplements=[];history=[];bindings=[];seen=set();maximum=0.;values=0
    for folder in a.batch:
        report=strict_file(folder/"report.json",4*1024**2)
        plan=strict_file(folder/"plan.json",8*1024**2);bound(plan)
        if report["plan_binding"]!=plan["binding"] or report["paid_calls"]!=0:
            raise ValueError("Batch provenance")
        bindings.append(dict(batch=str(folder),plan_sha256=sha(folder/"plan.json"),
            report_sha256=sha(folder/"report.json"),runtime_sha256=digest(plan["runtime_sources"])))
        for row in report["rows"]:
            history.append(dict(batch=str(folder),id=row["id"],state=row["status"],
                                evidence=row.get("evidence"),diagnostic=row.get("diagnostic")))
            if row["status"]!="passed":continue
            if row["id"] in seen:raise ValueError("Duplicate passed context")
            seen.add(row["id"]);evidence=Path(row["evidence"])
            if sha(evidence/"report.json")!=row["report_sha256"]:raise ValueError("Evidence report drift")
            saved=strict_file(folder/row["id"]/"fhe-audit.json",8*1024**2)
            archived=strict_file(evidence/"report.json",8*1024**2)
            for name,h in archived["source_hashes"].items():
                if plan["runtime_sources"].get("scripts/baseline/"+name)!=h:
                    raise ValueError("Historical runtime binding mismatch")
            current=all(sha(BASE/name)==h for name,h in archived["source_hashes"].items())
            if current:
                verified=verify_candidate(evidence)
                if verified!=saved:raise ValueError("Independent audit drift")
                audit_kind="current_runtime_independent_audit"
            else:
                verified=verify_saved(evidence,saved,plan["runtime_sources"])
                audit_kind="historical_record_integrity_and_independent_numerical_recheck"
            model=strict_file(folder/row["id"]/"model.json",131072)
            if verified["model_sha256"]!=digest(model):raise ValueError("Model drift")
            fs=sorted(set(features(model))&contexts.keys());topology=signature(model,True)
            reference=folder/row["id"]/"reference-report.json"
            item=dict(id=row["id"],model_sha256=digest(model),topology=topology,features=fs,
                source_binding=plan["binding"],runtime_sha256=digest(plan["runtime_sources"]),audit_kind=audit_kind,
                evidence=str(evidence),audit_sha256=sha(folder/row["id"]/"fhe-audit.json"),
                reference_report_sha256=sha(reference),state="verified",
                compared_values=verified["comparison"]["compared_values"],
                max_absolute_error=verified["comparison"]["max_absolute_error"])
            supplements.append(item);values+=item["compared_values"];maximum=max(maximum,item["max_absolute_error"])
            for key in fs:contexts[key].append(item)
    rows=[dict(id=key,verified_contexts=len({r["topology"] for r in records}),contexts=records)
          for key,records in contexts.items()]
    result=dict(format="poseidon-stage2-context-supplement-v1",
        historical_closure_sha256=sha(a.closure),historical_binding=old["binding"],
        historical_runtime_sha256=old["historical_runtime_sha256"],batches=bindings,
        mathematical_partitions=len(rows),three_context_partitions=sum(r["verified_contexts"]>=3 for r in rows),
        partitions=rows,new_contexts=supplements,execution_history=history,
        supplemental_models_counted_in_original=False,original_frozen_tasks_replaced=False,
        original_failed_and_blocked_tasks_preserved=True,original_task_status_report=str(a.closure),
        context_selection="targeted semantic gap diagnostics, not unbiased benchmark success sampling",
        compared_values=values,max_absolute_error=maximum,paid_calls=0,new_encrypted_executions=0,
        agent_generated=False,offline_stage_complete=False,agent_stage_complete=False,
        remaining=[r["id"] for r in rows if r["verified_contexts"]<3],
        runner_sha256=sha(Path(__file__)))
    result["binding"]=digest(result);a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("partitions","new_contexts","execution_history","batches")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
