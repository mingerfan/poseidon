"""Serial execution/resume of one frozen campaign shard with durable paid claims."""
from retained_artifact_usage import size_bytes

import argparse,fcntl,json,os,shlex,subprocess,sys,time
from pathlib import Path
from stage2_agent_campaign import (BASE,ROOT,PAID,LIMITS,definitions,task_arguments as legacy_task_arguments,identity)
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_graph import digest,canonical
from benchmark_runner import strict_file,dump
from campaign_candidate_failures import transparent_failure
from campaign_request_files import read_request,verify_recovery
from campaign_live_state import claim,resume_rows,sealed,exclusive_json
from run_stage2_guidance_retest import outcome,terminal_failure_layer,stop_child,evidence_path


def task_arguments(spec):
    args=legacy_task_arguments(spec)
    i=args.index("--unified-guidance")
    if args[i+1]!="explicit-v2":raise ValueError("Legacy argument baseline changed")
    args[i+1]="explicit-v3"
    return args

def validate_plan(plan,approval):
    check_binding(plan)
    if not approval or approval!=plan["binding"]:raise ValueError("Exact approved shard binding required")
    if plan.get("format")!="poseidon-stage2-agent-campaign-shard-v1":raise ValueError("Campaign shard required")
    if plan.get("generation_guidance")!="explicit-v3":raise ValueError("Explicit mathematical guidance required")
    if plan.get("shared_scheduling")!={"api_workers":10,"native_workers":2,"version":1}:
        raise ValueError("Bounded parallel scheduler policy required")
    n=len(plan["cases"])
    if not 1<=n<=48 or len({s["id"] for s in plan["cases"]})!=n:raise ValueError("Shard task boundary")
    if plan["paid_configuration"]!=PAID:raise ValueError("Paid configuration changed")
    if plan["limits"]!=dict(LIMITS,max_wall_seconds=7200,maximum_generations=4*n,maximum_http_attempts=16*n):raise ValueError("Budget changed")
    from approved_budget_extension import verify_extension
    verify_extension(plan)
    from platform_config import identity as platform_identity,require_python_packages
    import numpy as np,torch
    require_python_packages(torch,np)
    if plan["platform"]!=platform_identity():raise ValueError("Platform identity changed")
    from semantic_benchmark_execution import runtime_sources
    if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime drift")
    for name,hsh in plan["proposal_files"].items():
        path=ROOT/name
        if not path.resolve().is_relative_to(ROOT) or sha(path)!=hsh:raise ValueError("Proposal drift")
    specs,parents,binding=definitions()
    if binding!=plan["inventory_binding"] or parents!=plan["definition_parents"]:raise ValueError("Task definitions changed")
    expected={s["id"]:s for s in specs}
    from unified_graph_contract import prepare,validate_request
    from compiler_configuration import PROFILE_SHA256,configuration
    from hecate_python_env import VENV
    if plan["executable"]!=str(VENV/"bin/python") or plan["entrypoint"]!=str(BASE/"run_candidate.py"):
        raise ValueError("Entrypoint changed")
    for spec in plan["cases"]:
        original=expected.get(spec["id"])
        if original is None or any(spec.get(k)!=v for k,v in original.items()):raise ValueError("Task specification changed")
        public=spec.get("construction_profile")=="hecate-unified-public-v1"
        r=prepare(spec["model"],PROFILE_SHA256,configuration(spec["compiler_configuration"]),spec.get("exercise"),
                  construction_profile=spec["construction_profile"] if public else None,
                  helper_profile=spec.get("helper_profile"),helper_exercise=spec.get("required_helpers"),
                  chunk_period=spec.get("chunk_period"),generation_guidance="explicit-v3")
        validate_request(r)
        if spec["request"]!=r or spec["request_id"]!=r["request_id"]:raise ValueError("Request changed")
        args=task_arguments(spec)
        if spec.get("chunk_period") is not None:args+=["--unified-chunk-period",str(spec["chunk_period"])]
        if spec["candidate_arguments"]!=args or spec["evaluation_identity"]!=identity(spec,plan["source_hashes"]):
            raise ValueError("Execution arguments/identity changed")
    verify_recovery(plan)
    for name in ("run_stage2_budget_supplement_v2.py","audit_stage2_campaign_shard_v2.py","campaign_request_files.py","campaign_candidate_failures.py","approved_budget_extension.py","retained_artifact_usage.py"):
        p=Path(__file__).with_name(name)
        if plan["proposal_files"].get(str(p.relative_to(ROOT)))!=sha(p):raise ValueError("V2 implementation not bound")
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True);p.add_argument("--approve-live-binding")
    p.add_argument("--parallel-queue-plan",type=Path)
    p.add_argument("--check",action="store_true",help="Read-only plan validation; no claims or provider calls")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    plan=strict_file(a.plan,8*1024**2);check_binding(plan)
    if not a.check and a.approve_live_binding!=plan["binding"]:p.error("Exact binding required")
    from hecate_python_env import VENV,enter_nix
    from workspace_paths import RESULTS
    if a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():p.error("Direct result directory required")
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),
                         seconds=LIMITS["max_wall_seconds"]+90)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
    validate_plan(plan,plan["binding"] if a.check else a.approve_live_binding)
    if a.check:
        print(json.dumps(dict(binding=plan["binding"],cases=len(plan["cases"]),validated=True,paid_calls=0)))
        return 0
    if a.parallel_queue_plan is None:raise ValueError("Parallel parent required")
    from budget_supplement_queue_v2 import verify as verify_parallel
    parent=strict_file(a.parallel_queue_plan,1024**2);verify_parallel(parent)
    parent_args=(Path("/proc")/str(os.getppid())/"cmdline").read_bytes().split(bytes([0]))
    if not any(x.endswith(b"/budget_supplement_queue_v2.py") for x in parent_args) or b"--inside" not in parent_args:
        raise ValueError("Verified single parallel Nix parent required")
    if not any(r["binding"]==plan["binding"] and r["output"]==str(a.output) for r in parent["shards"]):
        raise ValueError("Shard outside parallel parent")
    if a.output.exists():
        if not a.resume:p.error("Existing output requires explicit resume; no automatic restart")
    else:
        if a.resume:p.error("Cannot resume nonexistent output")
        a.output.mkdir();dump(a.output/"plan.json",plan)
    global_fd=os.open(RESULTS/"stage2-agent-campaign-execution.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fd=os.open(a.output/"run.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(global_fd,"a+") as global_lock, os.fdopen(fd,"a+") as lock:
        try:
            fcntl.flock(global_lock.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error("Another campaign process owns serial execution or this batch")
        if strict_file(a.output/"plan.json",8*1024**2)!=plan:p.error("Resume plan changed")
        from approved_budget_extension import verify_extension,retained_paths
        extension=verify_extension(plan)
        rows=[];prior=plan["budget_extension"]["seconds_already_used"]
        if a.resume:
            rows,prior=resume_rows(plan,strict_file(a.output/"report.json",8*1024**2),a.output)
            if prior<plan["budget_extension"]["seconds_already_used"]:raise ValueError("Original elapsed time reset")
        recovery=verify_recovery(plan)
        inherited_paths=retained_paths(extension)
        if recovery:
            if not a.resume or prior<recovery["seconds"]:raise ValueError("Recovery wall budget reset")
            inherited={r["id"]:r for r in recovery["rows"]}
            actual={r["id"]:r for r in rows}
            if any(actual.get(k)!=v for k,v in inherited.items()):raise ValueError("Recovered rows changed")
            inherited_paths=[Path(n).parent for n in recovery["parents"] if Path(n).name=="plan.json"]
        started=time.monotonic();child=None;failure=None
        folders={Path(r["evidence"]) for r in rows if "evidence" in r}
        from semantic_benchmark_execution import runtime_sources
        def boundary():
            if prior+time.monotonic()-started>=plan["limits"]["max_wall_seconds"]:raise TimeoutError("Cumulative wall budget exhausted")
            if size_bytes([a.output,*folders,*inherited_paths])>=plan["limits"]["max_retained_mib"]*1024**2:raise ValueError("Retained artifact budget exhausted")
            if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime drift")
            for name,hsh in plan["proposal_files"].items():
                if sha(ROOT/name)!=hsh:raise ValueError("Proposal drift")
        try:
            for spec in plan["cases"]:
                if spec["id"] in {r["id"] for r in rows}:continue
                boundary()
                acquired,record=claim(RESULTS/"stage2-agent-evaluation-claims",spec,plan["source_hashes"],a.output,plan["binding"])
                if not acquired:
                    if record["owner"]==str(a.output.resolve()):raise ValueError("Uncertain own claim: audit before resuming")
                    rows.append(dict(id=spec["id"],track=spec["track"],evaluation_identity=spec["evaluation_identity"],
                                     status="reserved_elsewhere",claim_owner=record["owner"],claim_binding=record["binding"]))
                    dump(a.output/"progress.json",rows);continue
                case=a.output/spec["id"];case.mkdir()
                dump(case/"model.json",spec["model"]);dump(case/"request.expected.json",spec["request"])
                command=[plan["executable"],"-B",plan["entrypoint"],"--case",str(case/"model.json"),*spec["candidate_arguments"]]
                exclusive_json(case/"launch.json",sealed(dict(plan_binding=plan["binding"],evaluation_identity=spec["evaluation_identity"],
                      command=command,automatic_restart_allowed=False,status="launching")))
                log=case/"run.log"
                with log.open("x") as f:
                    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    dump(case/"process.json",dict(pid=child.pid,state="started"))
                    while child.poll() is None:
                        folder=evidence_path(log,RESULTS)
                        if folder:folders.add(folder)
                        boundary()
                        try:child.wait(timeout=1)
                        except subprocess.TimeoutExpired:pass
                    code=child.returncode
                folder=evidence_path(log,RESULTS)
                if folder is None:raise ValueError("No candidate evidence; preserve uncertain claim")
                folders.add(folder)
                report=strict_file(folder/"report.json",8*1024**2)
                if report["status"]=="running":raise ValueError("Nonterminal candidate evidence")
                read_request(folder/"request.json",spec["request"])
                if digest(strict_file(folder/"model.json",131072))!=spec["model_sha256"]:raise ValueError("Executed model changed")
                row=dict(id=spec["id"],track=spec["track"],evaluation_identity=spec["evaluation_identity"],
                         status=outcome(report,code),exit_code=code,evidence=str(folder),report_sha256=sha(folder/"report.json"),
                         agent_calls=report["agent_calls"],generations=report.get("provider_metrics",{}).get("generation_attempts",0),
                         failure_layer=terminal_failure_layer(report))
                rejection=transparent_failure(folder,report,plan["paid_configuration"])
                if rejection:row["candidate_rejection"]=rejection
                rows.append(row);dump(a.output/"progress.json",rows)
                boundary()
                if row["failure_layer"] in ("environment","harness_launch","key_setup","integrity") or (row["failure_layer"]=="seal_runtime" and not rejection):
                    raise ValueError("Environment/integrity failure; stop related batch")
        except BaseException as error:
            failure=type(error).__name__+": "+str(error)
            if child:stop_child(child)
            raise
        finally:
            if child:stop_child(child)
            result=sealed(dict(format="poseidon-stage2-agent-campaign-run-v1",plan_binding=plan["binding"],rows=rows,
                not_run_or_interrupted=[s["id"] for s in plan["cases"] if s["id"] not in {r["id"] for r in rows}],
                seconds=prior+time.monotonic()-started,failure=failure,
                http_attempts_in_completed_reports=sum(r.get("agent_calls",0) for r in rows),
                generations_in_completed_reports=sum(r.get("generations",0) for r in rows),
                artifact_bytes=size_bytes([a.output,*folders,*inherited_paths]),interrupted_call_count_may_be_unknown=bool(failure),
                passed=0,independent_audit_complete=False,stage2_complete=False))
            dump(a.output/"report.json",result)
        print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
        return int(any(r["status"]!="runner_reported_pass_pending_audit" for r in rows))
if __name__=="__main__":raise SystemExit(main())
