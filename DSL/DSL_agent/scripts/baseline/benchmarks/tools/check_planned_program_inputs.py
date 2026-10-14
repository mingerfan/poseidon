"""Replay planned program packages on new inputs without generation or compilation."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def run(output,qualification,case_ids=None):
    import numpy as np
    from hecate_python_env import WORK
    from benchmark_graph import canonical,digest,require
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from numerical_results import compare
    from application_component import Application
    from application_backend import PoseidonBackend
    from application_planning.layout import source_authority,check_plan
    from candidate_bundle import check_bundle,check_execution_dependencies
    from run_application_matrix import freeze,items_from_plan
    frozen=json.loads((ROOT/"docs/baseline/agent-pytorch-state-live-proposal-r635.json").read_text())
    require(freeze(items_from_plan(frozen))==frozen,"Pending paid binding changed")
    source=json.loads((qualification/"report.json").read_text())
    require(source["status"]=="complete" and source["passed"]==source["planned"]==8,"Finished qualification")
    require(not output.exists() and output.parent.resolve()==(WORK/"results").resolve(),"Fresh evidence directory")
    output.mkdir()
    all_ids=("tail65","state_batchnorm","state_matrix_fusion")
    ids=all_ids if case_ids is None else tuple(case_ids)
    require(1<=len(ids)<=3 and len(set(ids))==len(ids) and set(ids)<=set(all_ids),"Bounded case selection")
    chosen=[r for r in source["rows"] if r["id"] in ids]
    require(len(chosen)==len(ids),"Selected planned programs")
    backend=PoseidonBackend();authority=source_authority()
    plan=dict(format="poseidon-layout-new-input-plan-v1",
        qualification_binding=source["binding"],planner=authority,backend=backend.identity(),
        case_ids=list(ids),samples_per_case=3,repetitions=2,random_seed=20261005,
        source="rule baseline",paid_calls=0,compiler_calls=0,max_seconds=1800,
        case_bytes=4*1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,
        pending_paid_binding=frozen["binding"],
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan["binding"]=digest(plan);(output/"plan.json").write_bytes(canonical(plan))
    rows=[];start=time.monotonic();retained={output}
    def boundary():
        require(source_authority()==authority and backend.identity()==plan["backend"],"Execution dependency drift")
        require(freeze(items_from_plan(frozen))==frozen,"Pending paid binding drift")
        require(time.monotonic()-start<plan["max_seconds"],"Execution wall budget")
        require(shutil.disk_usage(WORK).free>=plan["free_bytes"],"Free space budget")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
        require(total<=plan["global_bytes"],"Global artifact budget")
        require(sum(p.stat().st_size for d in retained for p in d.rglob("*") if p.is_file())<=plan["case_bytes"],
                "New evidence size budget")
    def save(status):
        result=dict(format="poseidon-layout-new-input-report-v1",binding=plan["binding"],status=status,
                    rows=rows,planned=len(ids),passed=sum(r["status"]=="passed" for r in rows),
                    failed=sum(r["status"]=="failed" for r in rows),not_run=len(ids)-len(rows),
                    seconds=time.monotonic()-start,paid_calls=0,compiler_calls=0,agent_generated=False)
        (output/"report.json").write_bytes(canonical(result))
    app=Application(qualification/"tasks",backend);save("running")
    try:
        for index,prior in enumerate(chosen):
            boundary()
            package=Path(prior["package"]);manifest=check_bundle(package)
            require(manifest["binding"]==prior["package_binding"] and not manifest["agent_generated"],"Original rule package")
            checked_manifest,request=check_execution_dependencies(package)
            require(checked_manifest==manifest,"Checked package request identity")
            model=request["model"];index=all_ids.index(prior["id"])
            decision=json.loads((qualification/(prior["id"]+"-layout.json")).read_text())
            require(check_plan(decision,model)==request,"Original layout decision")
            task=prior["submission"]["task_id"]
            require(app.result(task)["status"]=="succeeded","Retained Application task")
            samples=[]
            for probe in range(3):
                rng=np.random.default_rng(plan["random_seed"]+index*100+probe)
                samples.append({s["name"]:rng.uniform(-.25,.25,s["shape"]).tolist() for s in model["inputs"]})
            expected=[]
            for sample in samples:
                args={k:np.asarray(v,dtype=np.float64) for k,v in sample.items()}
                math_values=mathematical(model,args);torch_values=tensor_reference(model,args)
                for name,value in math_values.items():
                    np.testing.assert_allclose(value,torch_values[name],atol=1e-12,rtol=1e-12)
                expected.append(np.concatenate([np.asarray(math_values[s["name"]]).reshape(-1) for s in model["outputs"]]))
            (output/(prior["id"]+"-inputs.json")).write_bytes(canonical(samples))
            row=dict(id=prior["id"],task_id=task,package_binding=manifest["binding"],
                     input_sha256=digest(samples),samples=3,status="running",runs=[])
            rows.append(row);save("running")
            for repetition in range(2):
                boundary();before=set((WORK/"results").iterdir())
                execution=app.execute(task,samples)
                retained.update(set((WORK/"results").iterdir())-before)
                require(execution["status"]=="executed" and execution["encrypted_execution"],"Real package execution")
                require(execution["agent_calls"]==execution["compiler_calls"]==0 and not execution["executor_secret_key"],
                        "No generation, compilation or secret key in executor")
                actual=np.asarray([np.concatenate([np.asarray(v[s["name"]]).reshape(-1) for s in model["outputs"]])
                                   for v in execution["outputs"]])
                compared=compare(actual,np.asarray(expected),1e-5,1e-4)
                require(compared["passed"],"New-input numerical check")
                row["runs"].append(dict(repetition=repetition,evidence=execution["evidence"],
                    comparison=compared,output_structure=execution.get("output_structure"),agent_calls=0,compiler_calls=0))
                save("running")
            row["status"]="passed";save("running")
            print(json.dumps(dict(id=row["id"],status="passed",
                maximum_absolute_error=max(r["comparison"]["max_absolute_error"] for r in row["runs"]))),flush=True)
        boundary();save("complete")
    except BaseException:
        if rows and rows[-1]["status"]=="running":rows[-1]["status"]="failed"
        save("stopped");raise
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--case",action="append",choices=("tail65","state_batchnorm","state_matrix_fusion"),help="Run only these original cases")
    args=parser.parse_args();raise SystemExit(run(args.output,args.qualification,args.case))
