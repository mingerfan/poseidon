"""Frozen, bounded benchmark execution using the existing isolated candidate pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time
from benchmark_graph import digest, canonical, samples
from benchmark_runner import DEFAULT, ROOT, load, dump, strict_file
from benchmark_semantics import features
from compiler_configuration import PROFILE_SHA256, configuration

CONFIG="seal-cpu-eva-w45-v1"


def runtime_sources():
    paths=list((ROOT/"scripts/baseline").glob("*.py"))+[ROOT/"scripts/benchmark.py"]
    paths+=list((ROOT/"scripts/baseline/upstream_adapters").glob("*.py"))
    for pattern in ("*.json","*.sh","patches/*.patch"):
        paths+=list((ROOT/"scripts/baseline").glob(pattern))
    paths+=list((ROOT/"src/poseidon/tools/dacapo").glob("*.json"))
    paths+=list((ROOT/"src/poseidon/tools/dacapo").glob("*.nix"))
    for package in ("hecate","poly"):
        paths+=list((ROOT/"third_party/dacapo/python"/package).rglob("*.py"))
    # No environment or credentials; the fixed compiler binaries are separately recorded per case.
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def preflight(rows,need_rule=True,construction_profile=None,chunk_period=None):
    from unified_graph_contract import prepare,validate_candidate
    from unified_graph_lowering import candidate_source
    items=[]
    for row in rows:
        item=dict(id=row["model"]["id"],model_sha256=row["model_sha256"],category=row["category"],
                  features=features(row["model"]),status="ready",work=None)
        try:
            request=prepare(row["model"],PROFILE_SHA256,configuration(CONFIG),construction_profile=construction_profile,chunk_period=chunk_period)
            if need_rule:
                source,lowering=candidate_source(request)
                check=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
                work=check["functions"]["golden"]["expanded_cost"]
            else:
                work=len(row["model"]["nodes"])
            item.update(work=work,request_id=request["request_id"])
        except (ValueError,TypeError,IndexError) as e:
            item.update(status="blocked_rule_preflight" if need_rule else "blocked_request_preflight",reason=str(e))
        items.append(item)
    return items


def representative(items,limit):
    ready=[x for x in items if x["status"]=="ready"]
    selected=[];covered=set()
    # Greedy coverage/cost on registered model facts, never observed FHE success.
    while ready and len(selected)<limit:
        def rank(x):
            core={f for f in x["features"] if f.startswith(("op.","input.count.","output.count.","input.rank.","input.size_bucket."))}
            gain=len(core-covered)
            return (gain,-x["work"],x["id"])
        best=max(ready,key=rank);ready.remove(best);selected.append(best)
        covered.update(f for f in best["features"] if f.startswith(("op.","input.count.","output.count.","input.rank.","input.size_bucket.")))
    return selected


def summary(out):
    plan=strict_file(out/"plan.json",8*1024**2)
    items=[];expected=set(plan["selected_ids"])
    checkpoint=out/"checkpoint.json"
    hashes=strict_file(checkpoint,1024**2) if checkpoint.exists() else {}
    for name in plan["selected_ids"]:
        path=out/(name+".result.json")
        if path.exists():
            if hashes.get(path.name)!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Batch result hash mismatch or uncheckpointed result")
            item=strict_file(path,1024**2)
            for suffix,key in ((".log","log_sha256"),(".model.json","model_sha256")):
                if hashlib.sha256((out/(name+suffix)).read_bytes()).hexdigest()!=item[key]:
                    raise ValueError("Batch log/model changed")
            if item["binding"]!=plan["binding"]:raise ValueError("Result binding changed")
            report_path=Path(item["evidence"])/"report.json" if item.get("evidence") else None
            if report_path and hashlib.sha256(report_path.read_bytes()).hexdigest()!=item["report_sha256"]:
                raise ValueError("Candidate report changed")
            items.append(item)
    from collections import Counter
    counts=Counter(x["status"] for x in items)
    return dict(mode=plan["mode"],binding=plan["binding"],corpus_models=1200,selected=len(expected),
                planned_tasks=len(plan["preflight"]),evaluation_source="live_agent" if plan.get("live") else "scripted_baseline_not_agent",
                completed=len(items),not_run_selected=len(expected)-len(items),
                not_selected=len(plan["preflight"])-len(expected),statuses=dict(counts),
                preflight_blocked=sum(x["status"] not in ("ready","other_profile") for x in plan["preflight"]),
                other_profile=sum(x["status"]=="other_profile" for x in plan["preflight"]),
                agent_calls=sum(x.get("agent_calls",0) for x in items),
                seconds=sum(x.get("seconds",0) for x in items),
                max_individual_process_rss_kib=max((x.get("max_individual_process_rss_kib",0) for x in items),default=0),
                rss_method="maximum process RSS (self or children), not simultaneous process-tree total",
                evidence_bytes=sum(x.get("evidence_bytes",0) for x in items),
                all_dsl_semantics_covered=False,actual_upstream_helper_execution=False,
                reports=[dict(id=x["id"],evidence=x.get("evidence"),sha256=x.get("report_sha256")) for x in items])


def choose_tasks(items,requested,limit):
    if not requested:return None
    if not 1<=len(requested)<=48 or len(set(requested))!=len(requested):raise ValueError("Task selection must contain 1..48 distinct IDs")
    by_id={i["id"]:i for i in items}
    if any(name not in by_id for name in requested):raise ValueError("Unknown frozen task ID")
    if len(requested)>limit:raise ValueError("Task selection exceeds shard limit")
    return [by_id[name] for name in requested if by_id[name]["status"]=="ready"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("baseline","free","directed","summary"))
    parser.add_argument("--suite",type=Path,default=DEFAULT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--limit",type=int,default=48)
    parser.add_argument("--unified-chunk-period",type=int,choices=(4,8,16,32,64,128,256))
    parser.add_argument("--unified-profile",choices=("native","public-v1"),default="native")
    parser.add_argument("--task-id",action="append",default=[],help="Explicit frozen directed task IDs, at most 48; no implicit success for blocked tasks")
    parser.add_argument("--shard-index",type=int,help="Frozen contiguous shard, at most 48 tasks; omit for coverage pilot")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--live",action="store_true",help="Directed Agent plan; default directed execution is a scripted golden")
    parser.add_argument("--approve-live-binding",help="Explicit separately approved paid plan hash; never implied by --execute")
    parser.add_argument("--model",default="deepseek-flash")
    parser.add_argument("--max-repairs",type=int,choices=range(4),default=3)
    parser.add_argument("--provider-retries",type=int,choices=range(4),default=0)
    parser.add_argument("--api-timeout",type=int,default=1200)
    parser.add_argument("--max-tokens",type=int)
    parser.add_argument("--reasoning-effort",choices=("low","high","max"),default="high")
    parser.add_argument("--stream",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--max-wall-seconds",type=int,default=3600)
    parser.add_argument("--max-result-mib",type=int,default=1024)
    args=parser.parse_args()
    if args.mode=="summary":
        if args.output is None:parser.error("summary requires --output")
        print(json.dumps(summary(args.output),indent=2));return 0
    if not 1<=args.limit<=48 or not 1<=args.max_wall_seconds<=43200 or not 1<=args.max_result_mib<=40960:
        parser.error("Invalid bounded execution budget")
    if not args.inside:
        from hecate_python_env import enter_nix,VENV
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/benchmark.py"),*sys.argv[1:],"--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(command),
                         seconds=args.max_wall_seconds+900)
    from hecate_python_env import VENV
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:parser.error("Locked Nix Python required")
    live=args.mode=="free" or args.live
    if args.live and args.mode!="directed":parser.error("--live is only needed for directed mode")
    from deepseek_provider import MODEL_OUTPUT_LIMITS,Config
    if args.model not in MODEL_OUTPUT_LIMITS:parser.error("Unsupported model name")
    tokens=MODEL_OUTPUT_LIMITS[args.model] if args.max_tokens is None else args.max_tokens
    paid_config=dict(model=args.model,max_repairs=args.max_repairs,provider_retries=args.provider_retries,
                     api_timeout=args.api_timeout,max_tokens=tokens,reasoning_effort=args.reasoning_effort,stream=args.stream)
    if live:
        Config(model=args.model,service_provider="deepseek",reasoning_effort=args.reasoning_effort,
               max_calls=args.max_repairs+1,max_tokens=tokens,timeout_seconds=args.api_timeout,
               stream=args.stream,provider_retries=args.provider_retries)
    rows,index=load(args.suite)
    from unified_public_contract import CONTRACT as PUBLIC_CONTRACT
    selected_profile=PUBLIC_CONTRACT if args.unified_profile=="public-v1" else None
    items=[] if args.mode=="directed" else preflight(rows,need_rule=not live,construction_profile=selected_profile,chunk_period=args.unified_chunk_period)
    chosen=representative(items,args.limit)
    if args.mode=="directed":
        from unified_graph_contract import prepare,validate_candidate
        from unified_graph_lowering import candidate_source
        models={r["model"]["id"]:r["model"] for r in rows}
        tasks=strict_file(args.suite/"coverage.json",4*1024**2)["directed_tasks"]
        items=[]
        for task in tasks:
            item=dict(id=task["id"],model_id=task["model_id"],exercise=task.get("exercise"),
                      task_sha256=task["task_sha256"],status="blocked_contract",reason="Unbound historical contract")
            if item["exercise"] and task["profile"]!=(selected_profile or "hecate-unified-native-v1"):
                item.update(status="other_profile",reason="Select the explicit task profile")
            elif item["exercise"]:
                try:
                    request=prepare(models[item["model_id"]],PROFILE_SHA256,configuration(CONFIG),item["exercise"],construction_profile=selected_profile,chunk_period=args.unified_chunk_period)
                    work=len(models[item["model_id"]]["nodes"])
                    if not live:
                        source,lowering=candidate_source(request)
                        check=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
                        work=check["functions"]["golden"]["expanded_cost"]
                    item.update(status="ready",reason=None,work=work)
                except (ValueError,TypeError,IndexError) as e:item.update(status="blocked_rule_preflight",reason=str(e))
            items.append(item)
        chosen=[x for x in items if x["status"]=="ready"][:args.limit]
    if args.shard_index is not None:
        total=len(items)
        if not 0<=args.shard_index<(total+47)//48:parser.error("Invalid frozen shard")
        window=items[args.shard_index*48:(args.shard_index+1)*48]
        chosen=[x for x in window if x["status"]=="ready"][:args.limit]
    if args.task_id:
        if args.mode!="directed" or args.shard_index is not None:parser.error("Task IDs require directed mode without shard-index")
        try:chosen=choose_tasks(items,args.task_id,args.limit)
        except ValueError as error:parser.error(str(error))
    frozen=runtime_sources()
    binding=digest(dict(index=index,source_hashes=frozen,configuration=CONFIG,mode=args.mode,construction_profile=args.unified_profile,chunk_period=args.unified_chunk_period,
                        requested_tasks=args.task_id,selected=[x["id"] for x in chosen],candidate_repair_limit=args.max_repairs if live else 0,
                        paid_configuration=paid_config if live else None,live=live,
                        max_wall_seconds=args.max_wall_seconds,max_result_mib=args.max_result_mib))
    plan=dict(mode=args.mode,binding=binding,compiler_configuration=CONFIG,source_hashes=frozen,construction_profile=args.unified_profile,chunk_period=args.unified_chunk_period,
              requested_tasks=args.task_id,selected_ids=[x["id"] for x in chosen],preflight=items,
              max_wall_seconds=args.max_wall_seconds,max_result_mib=args.max_result_mib,
              concurrency=1,agent_calls=0,live=live,paid_configuration=paid_config if live else None,
              paid_budget=dict(maximum_generations=len(chosen)*(1+args.max_repairs),
                  maximum_http_attempts=len(chosen)*(1+args.max_repairs)*(1+args.provider_retries),
                  monetary_cap=None,timeout_retries_may_duplicate_billing=True) if live else None,
              outbound_content="public logical graph, public constants, frozen layout/rules, candidate and restricted diagnostics")
    if not args.execute:
        print(json.dumps(dict(mode=args.mode,selected=plan["selected_ids"],binding=binding,construction_profile=args.unified_profile,chunk_period=args.unified_chunk_period,
            ready=sum(x["status"]=="ready" for x in items),blocked=sum(x["status"] not in ("ready","other_profile") for x in items),
            other_profile=sum(x["status"]=="other_profile" for x in items),
            compiler_configuration=CONFIG,paid_calls=0,paid_configuration=plan["paid_configuration"],
            paid_budget=plan["paid_budget"],outbound_content=plan["outbound_content"]),indent=2))
        return 0
    if live and args.approve_live_binding!=binding:
        parser.error("Paid execution requires separate approval of this exact plan and --approve-live-binding "+binding)
    if not live and args.approve_live_binding:parser.error("Paid approval supplied to an offline task")
    from workspace_paths import RESULTS
    if args.output is None or not args.output.resolve().is_relative_to(RESULTS.resolve()):
        parser.error("--output must be within selected platform results")
    out=args.output
    if out.exists():
        if not args.resume:parser.error("Preserve existing evidence; --resume requires unchanged task")
        previous=strict_file(out/"plan.json",8*1024**2)
        if previous["binding"]!=binding:parser.error("Frozen batch differs; choose a new output directory")
    else:
        out.mkdir(parents=True);dump(out/"plan.json",plan)
    by_id={r["model"]["id"]:r for r in rows};start=time.monotonic()
    previous_summary=summary(out)
    prior_seconds=previous_summary["seconds"]
    bytes_used=previous_summary["evidence_bytes"]
    checkpoint=out/"checkpoint.json"
    hashes=strict_file(checkpoint,1024**2) if checkpoint.exists() else {}
    from hecate_python_env import VENV
    for item in chosen:
        if runtime_sources()!=frozen:raise ValueError("Source changed during batch; stopped at case boundary")
        result_path=out/(item["id"]+".result.json")
        if result_path.exists():
            summary(out)  # Verify existing result/report hashes before reuse.
            continue
        if prior_seconds+time.monotonic()-start>=args.max_wall_seconds or bytes_used>=args.max_result_mib*1024**2:
            print("Batch budget boundary reached; remaining cases preserved as not_run",flush=True);break
        case=out/(item["id"]+".model.json");dump(case,by_id[item.get("model_id",item["id"])]["model"])
        log=out/(item["id"]+".log");begin=time.monotonic()
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/run_candidate.py"),"--inside",
                 "--case",str(case),"--self-test","--max-repairs","0","--compiler-configuration",CONFIG]
        if args.unified_chunk_period is not None:command += ["--unified-chunk-period",str(args.unified_chunk_period)]
        if args.unified_profile!="native":command += ["--unified-profile",args.unified_profile]
        if live:
            # Existing public launcher owns credential loading and pure-shell isolation.
            command.remove("--inside");command.remove("--self-test")
            command[command.index("--max-repairs")+1]=str(args.max_repairs)
            command += ["--live","--provider","deepseek","--model",args.model,"--max-tokens",str(tokens),
                        "--api-timeout",str(args.api_timeout),"--reasoning-effort",args.reasoning_effort,
                        "--provider-retries",str(args.provider_retries)]
            if args.stream:command += ["--stream"]
        if item.get("exercise"):command += ["--unified-exercise",item["exercise"]]
        metrics=out/(item["id"]+".resources.json")
        wrapper=("import runpy,sys,resource,json; sys.argv=sys.argv[2:]; "
                 "target=sys.argv[0]; sys.path.insert(0,"+repr(str(ROOT/"scripts/baseline"))+")\ntry: runpy.run_path(target,run_name='__main__')\n"
                 "finally:\n r=resource.getrusage(resource.RUSAGE_SELF); "
                 "c=resource.getrusage(resource.RUSAGE_CHILDREN); "
                 "open("+repr(str(metrics))+",'w').write(json.dumps(dict(max_individual_process_rss_kib=max(r.ru_maxrss,c.ru_maxrss))))")
        command=[command[0],"-B","-c",wrapper,"--",*command[2:]]
        timed_out=False
        with log.open("w") as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                from deepseek_provider import generation_deadline
                deadline=1800+generation_deadline(args.api_timeout,args.max_repairs+1,args.provider_retries) if live else 900
                code=child.wait(timeout=max(1,min(deadline,args.max_wall_seconds-prior_seconds-(time.monotonic()-start))))
            except subprocess.TimeoutExpired:
                timed_out=True
                os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        text=log.read_text()
        matches=re.findall(r"^Candidate evidence: (.+)$",text,re.M)
        result=dict(id=item["id"],binding=binding,status="failed",failure_layer="harness_launch",exit_code=code,
                    seconds=time.monotonic()-begin,agent_calls=0,timed_out=timed_out,
                    model_sha256=hashlib.sha256(case.read_bytes()).hexdigest(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        if metrics.exists():result.update(strict_file(metrics,4096))
        if matches and (Path(matches[-1])/"report.json").is_file():
            evidence=Path(matches[-1]).resolve()
            if not evidence.is_relative_to(RESULTS.resolve()):raise ValueError("Unexpected evidence path")
            path=evidence/"report.json";report=strict_file(path,8*1024**2)
            passed=code==0 and report.get("status")=="passed"
            result.update(status="passed" if passed else "failed",evidence=str(evidence),
                          report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          failure_layer=report.get("failure_layer") or next((x.get("failure_layer") for x in report.get("attempts",[]) if x.get("failure_layer")),None),
                          agent_calls=report.get("agent_calls",0),
                          evidence_bytes=sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file()),
                          encrypted_execution=any(x.get("executed") for x in report.get("attempts",[])))
            bytes_used+=result["evidence_bytes"]
        dump(result_path,result)
        hashes[result_path.name]=hashlib.sha256(result_path.read_bytes()).hexdigest()
        dump(checkpoint,hashes)
        if result.get("failure_layer")=="harness_launch" or timed_out:
            print("Infrastructure boundary; retaining remaining tasks as not_run",flush=True);break
    report=summary(out);dump(out/"report.json",report)
    print(json.dumps(report,indent=2))
    return int(report["statuses"].get("failed",0)>0 or report["not_run_selected"]>0)
