"""Bounded application/package regression; scripted fixtures are not Agent results."""
import argparse,contextlib,json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from application_component import Application
from application_backend import PoseidonBackend
from component_contract import request_options
from check_poseidon_application import Replay
from candidate_bundle import check_bundle,replay_bundle
from benchmark_graph import canonical,require,digest
from workspace_paths import WORK
from check_poseidon_helpers import snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists() and a.output.resolve().parent==(WORK/"results").resolve(),"Fresh bounded output")
    a.output.mkdir();start=time.monotonic();sources=snapshot()
    old=WORK/"results/poseidon-entrypoint-application-r253"
    fixtures=[]
    for row in json.loads((old/"report.json").read_text())["rows"]:
        package=Path(row["package"]);request=json.loads((package/"request.json").read_text())
        options=request_options(request)
        fixtures.append(dict(id=row["id"],model=request["model"],options=options,level=row["validation_level"],
                             source=(package/"candidate.py").read_text(),prior_package=str(package),
                             prior_manifest_sha256=__import__("hashlib").sha256((package/"manifest.json").read_bytes()).hexdigest()))
    shard=WORK/"results/poseidon-math-shard01-r263/bench_boundary_0011"
    fixtures.append(dict(id="constant_free",model=json.loads((shard/"model.json").read_text()),
                         options={},level="numerical",source=(shard/"candidate.py").read_text()))
    plan=dict(format="poseidon-bundle-application-regression-v1",planned=len(fixtures),sources=sources,
              paid_calls=0,agent_generated=False,max_wall_seconds=1800,max_artifact_bytes=1024**3,
              native_concurrency=1,compile_jobs=2,link_jobs=1,
              fixtures=[{k:v for k,v in f.items() if k!="source"} for f in fixtures])
    (a.output/"plan.json").write_bytes(canonical(plan))
    app=Application(a.output/"tasks",PoseidonBackend());records=[]
    for f in fixtures:
        require(snapshot()==sources and time.monotonic()-start<1800,"Source/time boundary")
        task=app.submit(f["model"],validation_level=f["level"],options=f["options"])
        require(app.status(task)["status"]=="queued","Application preparation")
        provider=Replay(f["source"])
        with (a.output/(f["id"]+".log")).open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            result=app.run(task,provider)
        record=dict(id=f["id"],validation_level=f["level"],task_id=task,result=result,
                    evaluation_source="manual_fixture_not_agent",scripted_generations=provider.calls)
        records.append(record)
        require(result["status"]=="succeeded","Application failed: "+str(result))
        package=a.output/"tasks"/result["package"]["task_id"]/"program";manifest=check_bundle(package)
        record.update(package=str(package),package_binding=manifest["binding"])
        if f["id"]=="constant_free":
            require("plaintext_bundle" not in json.loads((package/"artifacts/bound/plan.json").read_text()),
                    "Constant-free package must omit unused bundle reference")
        before=set((WORK/"results").glob("poseidon-package-replay-*"))
        with (a.output/(f["id"]+"-reload.log")).open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            require(replay_bundle(package)==0,"Package reload")
        after=set((WORK/"results").glob("poseidon-package-replay-*"))
        require(len(after-before)==1,"Fresh reload evidence");evidence=(after-before).pop()
        reload=json.loads((evidence/"report.json").read_text())
        require(reload["agent_calls"]==reload["compiler_calls"]==0,"Reload must reuse compiled artifacts")
        record.update(reload=reload,reload_evidence=str(evidence))
        again=app.submit(f["model"],validation_level=f["level"],options=f["options"]);unused=Replay(f["source"])
        reused=app.run(again,unused)
        require(reused["status"]=="succeeded" and reused["package"]["reused"] and unused.calls==0,"Cache reuse")
        record.update(cache=reused,cache_provider_calls=unused.calls)
        report=dict(**plan,rows=records,passed=len(records),failed=0,not_run=len(fixtures)-len(records),seconds=time.monotonic()-start)
        (a.output/"report.json").write_bytes(canonical(report))
        print(f["id"],result["status"],"reload",reload["status"],"cache",unused.calls,flush=True)
        paths=[a.output]+[Path(r["reload_evidence"]) for r in records]
        require(sum(p.stat().st_size for path in paths for p in path.rglob("*") if p.is_file())<=1024**3,"Artifact boundary")
    require(snapshot()==sources,"Final source binding")
    return 0
if __name__=="__main__":raise SystemExit(main())
