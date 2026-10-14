"""Bounded offline migration of existing directed helper cases; no provider calls."""
import argparse,copy,json,time,sys
from pathlib import Path
from hecate_python_env import ROOT,WORK,digest
from benchmark_graph import canonical,require
from upstream_helper_directed_cases import cases
from upstream_candidate_helpers import verify_sources,PROFILE,BN_PROFILE
from compiler_configuration import configuration
from unified_graph_contract import prepare
from run_candidate import parse_args,inside
from audit_unified_candidate import verify_candidate
def population():
    rows=[dict(r) for r in cases()]
    for r in rows[:3]:rows.append(dict(r,name=r["name"]+"_profile_v1",profile=PROFILE))
    sys.path.insert(0,str(ROOT/"scripts/baseline/benchmarks/tools"))
    from fixed_polynomial_cases import cases as poly
    for r in poly():rows.append(dict(r,name=r["id"],required_helpers=[r["helper"]]))
    return rows
def snapshot():
    paths=list((ROOT/"scripts/baseline").glob("*.py"))+list((ROOT/"scripts/baseline").glob("*.json"))
    paths+=list((ROOT/"scripts/baseline/upstream_adapters").glob("*.py"))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path);p.add_argument("--start",type=int,default=0)
    p.add_argument("--limit",type=int,default=48);p.add_argument("--case",action="append");p.add_argument("--plan",action="store_true")
    p.add_argument("--compiler-configuration",choices=("poseidon-cpu-eva-w40-v1","poseidon-cpu-eva-w45-v1"))
    a=p.parse_args();require(1<=a.limit<=48 and a.start>=0,"Shard bound")
    rows=population()
    if a.case:
        require(set(a.case)<={r["name"] for r in rows},"Unknown case")
        rows=[r for r in rows if r["name"] in a.case]
    selected=rows[a.start:a.start+a.limit]
    configurations={r["name"]:configuration(a.compiler_configuration or ("poseidon-cpu-eva-w40-v1" if r.get("configuration","").endswith("w40-v1") else "poseidon-cpu-eva-w45-v1")) for r in selected}
    requests={r["name"]:prepare(r["model"],configurations[r["name"]]["ckks_config_sha256"],configurations[r["name"]],
        helper_profile=r.get("profile",BN_PROFILE),helper_exercise=r["required_helpers"],
        chunk_period=r.get("chunk_period")) for r in selected}
    plan=dict(backend="poseidon_cpu",case_ids=[r["name"] for r in selected],
              request_ids={k:v["request_id"] for k,v in requests.items()},paid_calls=0,
              max_wall_seconds=1800,max_artifact_bytes=512*1024**2,native_concurrency=1,
              sources=snapshot(),helper_sources=verify_sources(backend="poseidon_cpu"))
    if a.plan:print(json.dumps(plan,indent=2));return 0
    out=a.output
    require(out and out.resolve().is_relative_to(WORK/"results") and not out.exists(),"Fresh result directory")
    out.mkdir();(out/"plan.json").write_bytes(canonical(plan));start=time.monotonic();records=[]
    for row in selected:
        require(time.monotonic()-start<1800 and snapshot()==plan["sources"],"Time/source boundary")
        folder=out/row["name"];folder.mkdir();(folder/"model.json").write_bytes(canonical(row["model"]))
        (folder/"candidate.py").write_text(row["source"])
        request=requests[row["name"]]
        args=["--inside","--case",str(folder/"model.json"),"--golden-file",str(folder/"candidate.py"),
              "--unified-helpers",row.get("profile",BN_PROFILE),"--max-repairs","0"]
        args+=["--compiler-configuration",configurations[row["name"]]["name"]]
        if "chunk_period" in row:args+=["--unified-chunk-period",str(row["chunk_period"])]
        for helper in row["required_helpers"]:args+=["--unified-helper-exercise",helper]
        terminal={};code=inside(parse_args(args),expected_request=request,
            report_sink=lambda r,p:terminal.update(report=copy.deepcopy(r),evidence=str(p)))
        report=terminal["report"];last=report["attempts"][-1] if report["attempts"] else report
        record=dict(name=row["name"],profile=row.get("profile",BN_PROFILE),helpers=row["required_helpers"],
            request_id=request["request_id"],exit_code=code,status=report["status"],evidence=terminal["evidence"],
            failure_layer=last.get("failure_layer"),diagnostic=last.get("diagnostic"),
            comparison=last.get("comparison"),trace=last.get("upstream_helper_trace"),
            contribution=last.get("upstream_helper_coverage"))
        if code==0:
            audited=verify_candidate(Path(terminal["evidence"]))
            record["audited"]=True;record["audited_compared_values"]=audited["comparison"]["compared_values"]
        records.append(record)
        result=dict(plan_sha256=digest(out/"plan.json"),rows=records,planned=len(selected),
            passed=sum(r["status"]=="passed" and r.get("audited",False) for r in records),
            failed=sum(r["status"]!="passed" for r in records),unrun=len(selected)-len(records),
            seconds=time.monotonic()-start,paid_calls=0)
        (out/"report.json").write_text(json.dumps(result,indent=2))
        print(row["name"],record["status"],record["failure_layer"],flush=True)
        require(last.get("failure_layer")!="integrity","Integrity fault stops shard")
        used=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        used+=sum(p.stat().st_size for r in records for p in Path(r["evidence"]).rglob("*") if p.is_file())
        require(used<=plan["max_artifact_bytes"],"Artifact budget")
    return int(any(r["status"]!="passed" for r in records))
if __name__=="__main__":raise SystemExit(main())
