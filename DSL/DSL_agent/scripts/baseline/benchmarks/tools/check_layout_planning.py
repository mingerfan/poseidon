"""Developer acceptance of layout preparation through the existing Application."""
import argparse,json,hashlib,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def run(output):
    from hecate_python_env import WORK
    from benchmark_graph import canonical,digest,require
    from application_component import Application
    from application_backend import PoseidonBackend
    from application_planning.layout import plan_model,submit_planned,source_authority
    from candidate_bundle import check_bundle,replay_bundle,load_model_provenance
    from unified_graph_lowering import candidate_source
    from run_application_matrix import freeze,items_from_plan
    results=WORK/"results"
    frozen=json.loads((ROOT/"docs/baseline/agent-pytorch-state-live-proposal-r635.json").read_text())
    require(freeze(items_from_plan(frozen))==frozen,"Pending paid proposal binding changed")
    require(not output.exists() and output.parent.resolve()==results.resolve(),"Fresh offline evidence directory")
    output.mkdir()
    directory=ROOT/"scripts/baseline/application_planning/cases"
    items=[dict(id="square32",model=directory/"square32.json"),
           dict(id="tail65",model=directory/"square65.json")]
    imported=results/"agent-state-weights-numerical-r631/imports"
    items+=[dict(id="state_"+case,model=imported/case/"model.json",imported=imported/case)
            for case in ("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear")]
    backend=PoseidonBackend();authority=source_authority()
    plan=dict(format="poseidon-layout-offline-plan-v1",cases=[
          dict(id=c["id"],model_path=str(c["model"]),model_sha256=hashlib.sha256(c["model"].read_bytes()).hexdigest())
          for c in items],planner=authority,backend=backend.identity(),
          pending_proposal_binding=frozen["binding"],tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          source="rule baseline",paid_calls=0,max_seconds=3600,case_bytes=4*1024**3,
          global_bytes=32*1024**3,free_bytes=4*1024**3,atol=1e-5,rtol=1e-4)
    plan["binding"]=digest(plan)
    (output/"plan.json").write_bytes(canonical(plan))
    rows=[];paths={output};start=time.monotonic()
    def boundary():
        require(source_authority()==authority and backend.identity()==plan["backend"],"Layout qualification code drift")
        require(freeze(items_from_plan(frozen))==frozen,"Pending paid binding drift")
        require(time.monotonic()-start<3600 and shutil.disk_usage(WORK).free>=plan["free_bytes"],"Offline wall/disk budget")
        total=int(subprocess.check_output(["du","-sb",str(results)],text=True,timeout=30).split()[0])
        require(total<=plan["global_bytes"],"Offline global space budget")
        require(sum(p.stat().st_size for d in paths for p in d.rglob("*") if p.is_file())<=plan["case_bytes"],
                "Offline artifact space budget")
    def save(status):
        data=dict(format="poseidon-layout-offline-report-v1",binding=plan["binding"],status=status,
            rows=rows,planned=8,passed=sum(r["status"]=="passed" for r in rows),
            failed=sum(r["status"]=="failed" for r in rows),not_run=8-len(rows),seconds=time.monotonic()-start,
            agent_generated=False,paid_calls=0,source="rule baseline")
        (output/"report.json").write_bytes(canonical(data))
    class Offline:
        kind="scripted_replay";agent_calls=0
        def generate(self,request,history):
            return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=candidate_source(request)[0]))
    app=Application(output/"tasks",backend);save("running")
    try:
        for item in items:
            boundary()
            model=json.loads(item["model"].read_text())
            options=dict(generation_guidance="explicit-v10",public_weight_policy="descriptors-v1")
            decision=plan_model(model,options=options)
            (output/(item["id"]+"-layout.json")).write_bytes(canonical(decision))
            require(decision["status"]=="prepared","Layout preparation failed")
            provenance=load_model_provenance(item["imported"],model) if "imported" in item else None
            submitted=submit_planned(app,model,decision,model_provenance=provenance,
                limits=dict(generations=1,http_attempts=1,wall_seconds=3600,api_timeout_seconds=1200,max_tokens=384000))
            before=set(results.iterdir())
            result=app.run(submitted["task_id"],Offline())
            paths.update(set(results.iterdir())-before)
            row=dict(id=item["id"],status="running",submission=submitted,
                    layout_binding=decision["binding"],selection=decision["selection"],result=result)
            rows.append(row);save("running")
            if result["status"]!="succeeded":
                row["status"]="failed";save("running")
                print(json.dumps(dict(id=item["id"],status="failed",failure=result.get("failure"))),flush=True)
                continue
            payload=json.loads((output/"tasks"/submitted["task_id"]/"task.json").read_text())
            require(payload["request"]["request_id"]==submitted["request_id"],"Application selection mismatch")
            package=output/"tasks"/submitted["task_id"]/"program"
            manifest=check_bundle(package)
            require(not manifest["agent_generated"],"Rule source classification")
            before=set(results.iterdir())
            require(replay_bundle(package)==0,"Layout package numerical replay")
            new=set(results.iterdir())-before;paths.update(new)
            evidence=[p for p in new if p.name.startswith("poseidon-package-replay-")]
            require(len(evidence)==1,"Unique replay")
            row.update(status="passed",package=str(package),package_binding=manifest["binding"],
                       reload=json.loads((evidence[0]/"report.json").read_text()))
            save("running")
            print(json.dumps(dict(id=item["id"],status="passed",period=payload["request"]["layout"]["input_slot_period"],
                max_absolute_error=result["acceptance"]["maximum_absolute_error"])),flush=True)
        boundary();save("complete")
    except BaseException:
        save("stopped");raise
    return 0 if all(r["status"]=="passed" for r in rows) else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.output))
