"""Run a bounded static rejection shard. No provider or candidate execution."""
import argparse,hashlib,json,platform,time
from pathlib import Path
from benchmark_graph import digest,require
from benchmark_runner import DEFAULT,ROOT,load,strict_file,dump
from rejection_benchmark import VERSION,tasks,run_task

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["rejections"])
    p.add_argument("--execute",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--offset",type=int,default=0)
    p.add_argument("--limit",type=int,default=48)
    a=p.parse_args()
    require(a.offset>=0 and 1<=a.limit<=48,"Rejection shard bounds")
    load(DEFAULT)
    ledger=strict_file(DEFAULT/"coverage.json",4*1024**2)
    catalog=tasks();require(ledger["rejection_tasks"]==catalog,"Frozen rejection catalog changed")
    selected=catalog[a.offset:a.offset+a.limit];require(selected,"Empty rejection shard")
    plan=dict(schema=1,contract=VERSION,suite=str(DEFAULT),planned=len(selected),
        total_tasks=len(catalog),offset=a.offset,tasks=selected,agent_calls=0,
        new_encrypted_executions=0,scope="static validators only; no sandbox or runtime execution")
    if not a.execute:
        require(a.output is None,"Use --execute to write a report")
        print(json.dumps(plan,indent=2));return 0
    from hecate_python_env import WORK
    from semantic_benchmark_execution import runtime_sources
    require(a.output is not None,"New output report is required")
    out=a.output
    require(out.resolve().is_relative_to(WORK/"results") and not out.is_symlink()
            and not out.exists() and out.parent.is_dir(),"New platform results file required")
    sources=runtime_sources()
    binding=dict(runtime_source_sha256=digest(sources),source_hashes=sources,
        coverage_sha256=hashlib.sha256((DEFAULT/"coverage.json").read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256((DEFAULT/"index.json").read_bytes()).hexdigest(),
        task_hashes=[t["task_sha256"] for t in selected])
    records=[];start=time.monotonic()
    for task in selected:
        try:record=run_task(task,catalog)
        except Exception as error:
            record=dict(id=task["id"],task_sha256=task["task_sha256"],status="failed",
                        error_type=type(error).__name__,reason=str(error))
        records.append(record)
    unchanged=runtime_sources()==sources
    passed=sum(r["status"]=="passed" for r in records)
    result=dict(plan,binding=binding,records=records,passed=passed,failed=len(records)-passed,
        skipped=0,not_run=len(catalog)-len(selected),sources_unchanged=unchanged,
        status="passed" if passed==len(selected) and unchanged else "failed",
        seconds=time.monotonic()-start,python=platform.python_version(),machine=platform.machine(),
        system=platform.system(),actual_compile=False,actual_sandbox_execution=False,
        actual_ciphertext_execution=False)
    dump(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("tasks","binding","records")},indent=2))
    return 0 if result["status"]=="passed" else 1
