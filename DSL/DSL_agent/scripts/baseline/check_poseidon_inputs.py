"""Representative existing input regressions for Poseidon; scripted, no provider."""
import argparse,copy,json,time
from pathlib import Path
from hecate_python_env import ROOT,WORK,digest
from benchmark_graph import canonical,require,FORMAT
from run_candidate import parse_args,inside
from check_poseidon_helpers import snapshot
def models():
    root=ROOT/"scripts/baseline/cases";rows=[]
    for name in ("linear-example.json","custom-subneg-linear.json","custom-dual-linear.json",
                 "unified-two-input-two-output.json"):
        rows.append(json.loads((root/name).read_text()))
    for name,indices in (("chunked-input-10-manifest.json",(0,5)),
                         ("packed-input-12-manifest.json",(0,5,11)),
                         ("packed-spatial-16-manifest.json",(1,3,4,10))):
        items=json.loads((root/name).read_text())["cases"]
        rows.extend(items[i] for i in indices)
    for version,indices in (("v3",(0,3,8,9,12,15)),("v4",(0,3,6,9,12,15,18,21,24))):
        folder=root/("operator-extension-"+version)
        items=json.loads((folder/"manifest.json").read_text())["cases"]
        for i in indices:
            row=items[i];m=json.loads((folder/row["model"]).read_text())
            from benchmark_graph import digest as object_digest
            require(object_digest(m)==row["model_sha256"],"Frozen model hash")
            rows.append(m)
    require(len(rows)<=48 and len({m["id"] for m in rows})==len(rows),"Regression population")
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--plan",action="store_true")
    a=p.parse_args();rows=models();plan=dict(models=rows,sources=snapshot(),max_wall_seconds=1800,
        max_artifact_bytes=1024**3,paid_calls=0,native_concurrency=1,backend="poseidon_cpu")
    if a.plan:print(json.dumps(plan,indent=2));return 0
    require(a.output.resolve().is_relative_to(WORK/"results") and not a.output.exists(),"Fresh result directory")
    a.output.mkdir();(a.output/"plan.json").write_bytes(canonical(plan));records=[];start=time.monotonic()
    for model in rows:
        require(snapshot()==plan["sources"] and time.monotonic()-start<1800,"Source/time boundary")
        folder=a.output/model["id"];folder.mkdir();(folder/"model.json").write_bytes(canonical(model))
        args=["--inside","--case",str(folder/"model.json"),"--max-repairs","0"]
        if model.get("format")==FORMAT:args+=["--self-test"]
        else:
            from run_model_batch import prepare_case
            from fx_to_hecate import translate
            obj,shape=prepare_case(model,folder)
            source=translate(obj,shape)["hecate_source"]
            (folder/"candidate.py").write_text(source)
            args+=["--golden-file",str(folder/"candidate.py")]
        terminal={}
        code=inside(parse_args(args),report_sink=lambda r,p:terminal.update(report=copy.deepcopy(r),evidence=str(p)))
        r=terminal["report"];last=r["attempts"][-1] if r["attempts"] else r
        record=dict(id=model["id"],schema=model.get("schema",model.get("format")),status=r["status"],
            evidence=terminal["evidence"],exit_code=code,failure_layer=last.get("failure_layer"),
            diagnostic=last.get("diagnostic"),comparison=last.get("comparison"))
        if code==0 and model.get("format")==FORMAT:
            from audit_unified_candidate import verify_candidate
            verify_candidate(Path(record["evidence"]));record["audited"]=True
        records.append(record)
        result=dict(rows=records,planned=len(rows),passed=sum(x["status"]=="passed" for x in records),
            failed=sum(x["status"]!="passed" for x in records),unrun=len(rows)-len(records),
            paid_calls=0,agent_generated=False,seconds=time.monotonic()-start,plan_sha256=digest(a.output/"plan.json"))
        (a.output/"report.json").write_text(json.dumps(result,indent=2))
        print(model["id"],record["status"],record["failure_layer"],flush=True)
        require(record["failure_layer"]!="integrity","Integrity fault")
        used=sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file())
        used+=sum(p.stat().st_size for x in records for p in Path(x["evidence"]).rglob("*") if p.is_file())
        require(used<=plan["max_artifact_bytes"],"Artifact budget")
    return int(any(x["status"]!="passed" for x in records))
if __name__=="__main__":raise SystemExit(main())
