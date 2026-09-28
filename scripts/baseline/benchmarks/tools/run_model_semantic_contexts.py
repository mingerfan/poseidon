"""Frozen supplementary mathematical-context baseline; no paid mode.

Uses the existing run_candidate.py and artifact auditor. The original 1200
corpus is unchanged. Default is a static plan; each execute shard has <=48
models, 1800s wall budget and 256MiB retained-evidence budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from semantic_benchmark_execution import runtime_sources,summary,preflight,CONFIG
from verify_model_semantic_contexts import checked_bundle

def reference_gate(folder,bundle,sources):
    """Verify the entire frozen reference batch before selecting FHE cases."""
    run=strict_file(folder/"run.json",8*1024**2)
    report=strict_file(folder/"report.json",2*1024**2)
    if run["source_hashes"]!=sources or run["bundle_sha256"]!=bundle["bundle_sha256"]:
        raise ValueError("Reference source/bundle binding")
    runner=HERE/"verify_model_semantic_contexts.py"
    runner_sha=hashlib.sha256(runner.read_bytes()).hexdigest()
    expected=digest(dict(bundle_sha256=bundle["bundle_sha256"],runtime_sources=sources,
                         validator_sha256=runner_sha,probes=16,atol=1e-12,rtol=1e-12))
    if report["binding"]!=expected or run["binding"]!=expected or run["runner_sha256"]!=runner_sha:
        raise ValueError("Reference runner/budget binding")
    if (run["reference_atol"],run["reference_rtol"],run["probes_per_model"])!=(1e-12,1e-12,16):
        raise ValueError("Reference comparison contract")
    models=bundle["supplemental_models"]
    if set(report["result_hashes"])!={r["model"]["id"] for r in models}:
        raise ValueError("Reference result denominator")
    states={}
    for row in models:
        name=row["model"]["id"];path=folder/(name+".json")
        record=strict_file(path,1024**2)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report["result_hashes"][name]:
            raise ValueError("Reference result hash")
        if record["model_sha256"]!=row["model_sha256"] or record["provenance"]!=row["provenance"]:
            raise ValueError("Reference model identity")
        if record["state"]=="passed" and record["probes"]!=16:raise ValueError("Partial reference pass")
        states[name]=record["state"]
    return states

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--plaintext",type=Path);p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--execute",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--output",type=Path);p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args();bundle,_=checked_bundle(a.bundle,a.suite)
    models=bundle["supplemental_models"]
    if not 0<=a.shard_index<(len(models)+47)//48:p.error("Invalid supplementary shard")
    window=models[a.shard_index*48:(a.shard_index+1)*48]
    sources=runtime_sources()
    static=dict(mode="supplemental_baseline",bundle_sha256=bundle["bundle_sha256"],
        source_hashes=sources,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_ids=[r["model"]["id"] for r in window],shard_index=a.shard_index,
        original_corpus_models=1200,supplemental_models=len(models),
        max_wall_seconds=1800,max_result_mib=256,compiler_configuration=CONFIG,
        construction_profile="native",concurrency=1,paid_calls=0)
    if not a.execute:
        print(json.dumps({k:v for k,v in static.items() if k not in ("source_hashes","model_ids")}|
              dict(selected=len(window),binding=digest(static),state="static_plan_not_preflight")))
        return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output is None or not a.output.resolve().is_relative_to(RESULTS.resolve()):
        p.error("Platform result output required")
    if a.plaintext is None:p.error("Frozen dual-reference evidence required")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=1830)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    refs=reference_gate(a.plaintext,bundle,sources)
    items=preflight([dict(r,category="supplemental") for r in models])
    for item in items:
        if refs[item["id"]]!="passed":item.update(status="blocked_reference",reason=refs[item["id"]])
    wanted={r["model"]["id"] for r in window}
    chosen=[r for r in items if r["id"] in wanted and r["status"]=="ready"]
    plan=dict(static,preflight=items,selected_ids=[r["id"] for r in chosen],
              reference_report_sha256=hashlib.sha256((a.plaintext/"report.json").read_bytes()).hexdigest(),
              live=False)
    plan["binding"]=digest(plan);out=a.output
    if out.exists():
        if not a.resume or strict_file(out/"plan.json",8*1024**2)!=plan:
            p.error("Preserve evidence; exact frozen --resume required")
    else:
        out.mkdir(parents=True);dump(out/"plan.json",plan)
        (out/"runner.py").write_bytes(Path(__file__).read_bytes())
    prior=summary(out);used=prior["evidence_bytes"];prior_seconds=prior["seconds"]
    hashes=strict_file(out/"checkpoint.json",1024**2) if (out/"checkpoint.json").exists() else {}
    by_id={r["model"]["id"]:r for r in models};start=time.monotonic()
    from audit_unified_candidate import verify_candidate,verify_files
    for item in chosen:
        if runtime_sources()!=sources:raise ValueError("Runtime drift")
        name=item["id"];result_path=out/(name+".result.json")
        if result_path.exists():
            summary(out)
            previous=strict_file(result_path,1024**2)
            if previous["status"]=="passed":
                ap=out/(name+".audit.json")
                if hashlib.sha256(ap.read_bytes()).hexdigest()!=previous["audit_sha256"]:
                    raise ValueError("Resumed audit changed")
                saved=strict_file(ap,8*1024**2)
                verify_files(Path(previous["evidence"]),saved["files"])
            continue
        remaining=1800-prior_seconds-(time.monotonic()-start)
        if remaining<=0 or used>=256*1024**2:break
        case=out/(name+".model.json");dump(case,by_id[name]["model"])
        log=out/(name+".log")
        command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside",
                 "--case",str(case),"--self-test","--max-repairs","0","--compiler-configuration",CONFIG]
        metrics=out/(name+".resources.json")
        wrapper=("import runpy,sys,resource,json; sys.argv=sys.argv[2:]; "
                 "target=sys.argv[0]; sys.path.insert(0,"+repr(str(BASE))+")\ntry: runpy.run_path(target,run_name='__main__')\n"
                 "finally:\n r=resource.getrusage(resource.RUSAGE_SELF); "
                 "c=resource.getrusage(resource.RUSAGE_CHILDREN); "
                 "open("+repr(str(metrics))+",'w').write(json.dumps(dict(max_individual_process_rss_kib=max(r.ru_maxrss,c.ru_maxrss))))")
        command=[command[0],"-B","-c",wrapper,"--",*command[2:]]
        begin=time.monotonic();timed_out=False
        with log.open("w") as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(1,min(900,remaining)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        result=dict(id=name,binding=plan["binding"],status="failed",failure_layer="harness_launch",
            exit_code=code,seconds=time.monotonic()-begin,agent_calls=0,timed_out=timed_out,
            model_sha256=hashlib.sha256(case.read_bytes()).hexdigest(),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        if metrics.exists():result.update(strict_file(metrics,4096))
        matches=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
        if matches and (Path(matches[-1])/"report.json").is_file():
            evidence=Path(matches[-1]).resolve()
            if not evidence.is_relative_to(RESULTS.resolve()):raise ValueError("Unexpected evidence path")
            report=strict_file(evidence/"report.json",8*1024**2)
            if report.get("agent_calls")!=0:raise ValueError("Unexpected provider evidence")
            result.update(evidence=str(evidence),report_sha256=hashlib.sha256((evidence/"report.json").read_bytes()).hexdigest(),
                failure_layer=report.get("failure_layer") or next((x.get("failure_layer") for x in report.get("attempts",[]) if x.get("failure_layer")),None),
                encrypted_execution=any(x.get("executed") for x in report.get("attempts",[])),
                evidence_bytes=sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file()))
            used+=result["evidence_bytes"]
            if code==0 and report["status"]=="passed":
                try:
                    audit=verify_candidate(evidence)
                    if audit["model_sha256"]!=by_id[name]["model_sha256"]:raise ValueError("Executed model changed")
                    dump(out/(name+".audit.json"),audit)
                    result.update(status="passed",failure_layer=None,
                        audit_sha256=hashlib.sha256((out/(name+".audit.json")).read_bytes()).hexdigest(),
                        max_abs_error=audit["comparison"]["max_absolute_error"])
                except (ValueError,AssertionError,KeyError,TypeError) as error:
                    result.update(failure_layer="independent_artifact_audit",error=str(error))
        dump(result_path,result);hashes[result_path.name]=hashlib.sha256(result_path.read_bytes()).hexdigest()
        dump(out/"checkpoint.json",hashes)
        if timed_out or result["failure_layer"]=="harness_launch":break
    if runtime_sources()!=sources:raise ValueError("Runtime drift")
    report=summary(out)
    report.update(original_corpus_models=1200,supplemental_models=len(models),counts_as_new_original_models=False,
        scoped_window=len(window),blocked_in_window=[r for r in items if r["id"] in wanted and r["status"]!="ready"])
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("reports","blocked_in_window")}))
    return int(report["statuses"].get("failed",0)>0 or report["not_run_selected"]>0)
if __name__=="__main__":raise SystemExit(main())
