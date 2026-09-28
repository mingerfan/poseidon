"""Read-only audit of completed shards, with direct artifact and numeric checks.

Does not execute NumPy/Torch, candidates or FHE. The full independent-reference
auditor remains a separate check. Active shard files are never read.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import DEFAULT,strict_file,dump
from semantic_benchmark_execution import runtime_sources,summary
from audit_semantic_benchmark import audit
from workspace_paths import RESULTS

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def files(folder,expected):
    count=0
    for name,h in expected.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe evidence filename")
        path=folder/relative
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Indirect evidence file")
        if sha(path)!=h:raise ValueError("Evidence file hash mismatch")
        count+=1
    return count
def compare(c):
    if (c["atol"],c["rtol"],c["passed"])!=(1e-5,1e-4,True):
        raise ValueError("Numerical contract")
    if len(c["actual"])!=4 or len(c["reference"])!=4:raise ValueError("Four input groups required")
    error=[];relative=[];actual=[];reference=[]
    for av,rv,recorded in zip(c["actual"],c["reference"],c["elementwise_pass"]):
        if not 1<=len(av)<=256 or len(av)!=len(rv) or len(recorded)!=len(av):
            raise ValueError("Comparison shape")
        for a,r,ok in zip(av,rv,recorded):
            if not math.isfinite(a) or not math.isfinite(r):raise ValueError("Nonfinite output")
            e=abs(a-r)
            if e>1e-5+1e-4*abs(r) or ok is not True:raise ValueError("Elementwise failure")
            error.append(e);actual.append(a);reference.append(r)
            if r!=0:relative.append(e/abs(r))
    if len(c["elementwise_pass"])!=4 or c["compared_values"]!=len(error):
        raise ValueError("Incomplete comparison")
    if c["nonzero_reference_count"]!=len(relative):raise ValueError("Relative denominator")
    for key,value in (("mae",math.fsum(error)/len(error)),("max_absolute_error",max(error))):
        if not math.isclose(c[key],value,rel_tol=1e-12,abs_tol=1e-15):raise ValueError("Numeric metric drift")
    maximum=max(relative) if relative else None
    if maximum is None:
        if c["max_nonzero_reference_relative_error"] is not None:raise ValueError("Undefined relative error")
    elif not math.isclose(maximum,c["max_nonzero_reference_relative_error"],rel_tol=1e-12,abs_tol=1e-15):
        raise ValueError("Relative metric drift")
    return dict(compared_values=len(error),max_absolute_error=max(error))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,required=True);p.add_argument("--plaintext",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error("Preserve existing audit")
    if not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform results required")
    source=runtime_sources();stages=strict_file(a.batch/"stages.json",1024**2)
    completed=[s for s in stages if s["not_run"]==0]
    batches=[a.batch/("shard-"+str(s["shard"]).zfill(3)) for s in completed]
    score=audit(DEFAULT,a.plaintext,batches)
    records=[];file_count=0;values=0;maximum=0.
    for stage,folder in zip(completed,batches):
        plan=strict_file(folder/"plan.json",8*1024**2)
        if summary(folder)["binding"]!=stage["binding"]:raise ValueError("Completed shard drift")
        for name in plan["selected_ids"]:
            result=strict_file(folder/(name+".result.json"),1024**2)
            if result["status"]!="passed":continue
            evidence=Path(result["evidence"])
            if not evidence.resolve().is_relative_to(RESULTS.resolve()):raise ValueError("Evidence outside platform")
            report=strict_file(evidence/"report.json",8*1024**2)
            model=strict_file(evidence/"model.json",131072)
            request=strict_file(evidence/"request.json",131072)
            if model!=strict_file(folder/(name+".model.json"),131072) or request["model"]!=model:
                raise ValueError("Candidate evidence belongs to another model")
            if report["request_id"]!=request["request_id"]:raise ValueError("Candidate request identity")
            for module,h in report["source_hashes"].items():
                if source.get("scripts/baseline/"+module)!=h:raise ValueError("Candidate runtime module drift")
            if not strict_file(evidence/"key-cleanup-outcome.json",1024**2)["complete"]:
                raise ValueError("Incomplete key cleanup")
            count=files(evidence,report["frozen_hashes"])
            attempts=[x for x in report["attempts"] if x["status"]=="passed"]
            if len(attempts)!=1 or report["agent_calls"]!=0 or report["llm_generation_validated"]:
                raise ValueError("Wrong evaluation kind")
            attempt=attempts[0];out=evidence/("attempt-%02d"%attempt["index"])/"output"
            count+=files(out,attempt["artifact_hashes"])
            execution=strict_file(out/"execution.json",4*1024**2)
            if execution!=attempt["execution"] or not execution["encrypted_execution"]:
                raise ValueError("Execution record drift")
            checked=compare(attempt["comparison"]);file_count+=count
            values+=checked["compared_values"];maximum=max(maximum,checked["max_absolute_error"])
            records.append(dict(id=name,evidence=str(evidence),report_sha256=result["report_sha256"],
                                files_checked=count,comparison=checked))
    if runtime_sources()!=source:raise ValueError("Source drift during read-only audit")
    a.output.mkdir(parents=True)
    dump(a.output/"score.json",score);dump(a.output/"artifact-records.json",records)
    report=dict(schema=1,completed_shards=len(completed),counts=score["counts"],
        directed_counts=score["directed_counts"],passed_artifacts_checked=len(records),
        bound_files_checked=file_count,compared_values=values,max_absolute_error=maximum,
        terminal_report_present=(a.batch/"report.json").is_file(),
        runtime_source_sha256=digest(source),new_encrypted_executions=0,paid_calls=0,
        scope="Completed shards only; direct hashes and report numeric recomputation, not independent reference rerun",
        score_sha256=sha(a.output/"score.json"),artifact_records_sha256=sha(a.output/"artifact-records.json"),
        auditor_sha256=sha(Path(__file__)))
    dump(a.output/"report.json",report);print(json.dumps(report))
    return 0
if __name__=="__main__":raise SystemExit(main())
