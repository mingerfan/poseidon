"""Offline admission, Poseidon qualification and reload of external PyTorch states."""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def run(output, reuse_qualification=None):
    import numpy as np
    from hecate_python_env import WORK
    from benchmark_graph import canonical,digest,require
    from semantic_benchmark_execution import runtime_sources
    from trusted_pytorch_import import run as capture
    from candidate_bundle import check_bundle,replay_bundle
    from application_component import Application
    from application_backend import PoseidonBackend
    from unified_graph_lowering import candidate_source
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from numerical_results import compare
    output=Path(output)
    require(not output.exists() and output.parent.resolve()==(WORK/"results").resolve(),
            "Fresh evidence directory under results required")
    output.mkdir();(output/"imports").mkdir()
    cases=("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear")
    retained={}
    old_plan=None
    if reuse_qualification is not None:
        reuse_qualification=Path(reuse_qualification)
        old_plan=json.loads((reuse_qualification/"plan.json").read_text())
        old_report=json.loads((reuse_qualification/"report.json").read_text())
        require(old_report["status"]=="complete" and old_plan["source_hashes"]==runtime_sources()
                and old_plan["backend_identity"]==PoseidonBackend().identity(),
                "Retained qualification code/dependency binding")
        require(old_plan["binding"]==digest({k:v for k,v in old_plan.items() if k!="binding"}),
                "Retained qualification plan hash")
        retained={row["id"]:row for row in old_report["rows"]}
        require(set(retained)==set(cases),"Complete retained case inventory")
    class RetainedBackend(PoseidonBackend):
        def qualify(self,request,candidate,level):
            require(level=="numerical","Retained qualification grade")
            old=retained[current_case]
            task=reuse_qualification/"tasks"/old["task_id"]
            previous=json.loads((task/"task.json").read_text())
            require(previous["binding"]==digest({k:v for k,v in previous.items() if k!="binding"})
                    and previous["request"]==request,"Retained request binding")
            response=json.loads((task/"response-0.json").read_text())
            require(json.loads(response["raw"])==candidate,"Retained exact candidate source")
            raw=(task/"validation-0.json").read_bytes()
            qualified=json.loads(raw)
            require(qualified["status"]=="passed" and qualified["encrypted_execution"] is True
                    and qualified["numerically_validated"] is True
                    and qualified["request_id"]==request["request_id"],"Retained numerical qualification")
            # export_bundle verifies the retained evidence and current identities.
            rows[-1]["qualification_reused_from"]=dict(task_id=old["task_id"],
                batch_binding=old_plan["binding"],validation_sha256=hashlib.sha256(raw).hexdigest())
            return qualified
    backend=RetainedBackend() if retained else PoseidonBackend()
    plan=dict(format="poseidon-state-weights-offline-plan-v1",cases=list(cases),
        source_hashes=runtime_sources(),backend_identity=backend.identity(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source="rule baseline",paid_calls=0,max_seconds=3600,
        reused_qualification_binding=None if old_plan is None else old_plan["binding"],
        case_bytes=4*1024**3,total_bytes=32*1024**3,free_bytes=4*1024**3,
        inputs_per_capture=16,input_groups_per_qualification=4,
        new_input_cases=["mlp","batchnorm","matrix_fusion"],new_inputs=3,repetitions=2,
        atol=1e-5,rtol=1e-4)
    plan["binding"]=digest(plan)
    (output/"plan.json").write_bytes(canonical(plan))
    started=time.monotonic();rows=[];artifacts=set()
    def boundary():
        require(runtime_sources()==plan["source_hashes"] and backend.identity()==plan["backend_identity"],
                "Offline source/dependency drift")
        require(time.monotonic()-started<plan["max_seconds"],"Offline wall budget")
        require(shutil.disk_usage(WORK).free>=plan["free_bytes"],"Offline disk reserve")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
        require(total<=plan["total_bytes"],"Offline global evidence budget")
        require(sum(f.stat().st_size for d in {output,*artifacts} for f in d.rglob("*") if f.is_file())
                <=plan["case_bytes"],"Offline evidence space budget")
    def save(status):
        value=dict(format="poseidon-state-weights-offline-report-v1",status=status,
            binding=plan["binding"],planned=len(cases),rows=rows,
            passed=sum(r["status"]=="passed" for r in rows),
            failed=sum(r["status"]=="failed" for r in rows),not_run=len(cases)-len(rows),
            paid_calls=0,agent_generated=False,source="rule baseline",
            seconds=time.monotonic()-started)
        (output/"report.json").write_bytes(canonical(value))
    class OfflineProvider:
        kind="scripted_replay"
        agent_calls=0
        def generate(self,request,history):
            return json.dumps(dict(schema=1,request_id=request["request_id"],
                                   hecate_source=candidate_source(request)[0]))
    app=Application(output/"tasks",backend)
    save("running")
    try:
        for index,case in enumerate(cases):
            boundary()
            row=dict(id=case,status="running",new_input_runs=[]);rows.append(row);save("running")
            current_case=case
            folder=output/"imports"/case
            if retained:
                from candidate_bundle import load_model_provenance
                folder=reuse_qualification/"imports"/case
                original=json.loads((folder/"model.json").read_text())
                load_model_provenance(folder,original)
                row["capture"]=retained[case]["capture"]
            else:
                row["capture"]=capture(ROOT/"scripts/baseline/cases/trusted-pytorch-state-v3"/case/"manifest.json",
                                      folder,resource_profile="general-tensors-v1",trusted=True)
            row["model_import"]=str(folder)
            task=app.submit_import(folder,options=dict(chunk_period=4,generation_guidance="explicit-v10",
                                   public_weight_policy="descriptors-v1"),
                limits=dict(generations=1,http_attempts=1,wall_seconds=3600,
                            api_timeout_seconds=1200,max_tokens=384000))
            row["task_id"]=task
            before=set((WORK/"results").iterdir())
            result=app.run(task,OfflineProvider())
            artifacts.update(set((WORK/"results").iterdir())-before)
            row["result"]=result;save("running");boundary()
            if result["status"]!="succeeded":
                row["status"]="failed";save("running")
                print(json.dumps(dict(id=case,status="failed",failure=result.get("failure"))),flush=True)
                continue
            package=output/"tasks"/task/"program"
            manifest=check_bundle(package)
            require(manifest["agent_generated"] is False,"Rule-source classification")
            row.update(package=str(package),package_binding=manifest["binding"],
                       candidate_sha256=hashlib.sha256((package/"candidate.py").read_bytes()).hexdigest())
            before=set((WORK/"results").iterdir())
            require(replay_bundle(package)==0,"Package replay numerical failure")
            fresh=set((WORK/"results").iterdir())-before;artifacts.update(fresh)
            replays=[p for p in fresh if p.name.startswith("poseidon-package-replay-")]
            require(len(replays)==1,"Unique retained replay evidence")
            row["reload"]=json.loads((replays[0]/"report.json").read_text())
            require(row["reload"]["agent_calls"]==row["reload"]["compiler_calls"]==0,"No regeneration/recompilation")
            if case in plan["new_input_cases"]:
                model=json.loads((folder/"model.json").read_text())
                samples=[]
                for probe in range(3):
                    generator=np.random.default_rng(20261004+100*index+probe)
                    samples.append({spec["name"]:generator.uniform(-.3,.3,spec["shape"]).tolist()
                                    for spec in model["inputs"]})
                expected=[]
                for sample in samples:
                    arguments={k:np.asarray(v,dtype=np.float64) for k,v in sample.items()}
                    math_values=mathematical(model,arguments);torch_values=tensor_reference(model,arguments)
                    for name,value in math_values.items():
                        np.testing.assert_allclose(value,torch_values[name],atol=1e-12,rtol=1e-12)
                    expected.append(np.concatenate([np.asarray(math_values[o["name"]]).reshape(-1)
                                                  for o in model["outputs"]]))
                (output/(case+"-new-inputs.json")).write_bytes(canonical(samples))
                for repetition in range(2):
                    boundary();before=set((WORK/"results").iterdir())
                    executed=app.execute(task,samples)
                    artifacts.update(set((WORK/"results").iterdir())-before)
                    actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1)
                                        for o in model["outputs"]]) for s in executed["outputs"]])
                    compared=compare(actual,np.asarray(expected),1e-5,1e-4)
                    require(compared["passed"],"New-input numerical failure")
                    require(executed["compiler_calls"]==executed["agent_calls"]==0 and
                            executed["executor_secret_key"] is False,"Local execution scope")
                    row["new_input_runs"].append(dict(repetition=repetition,
                        evidence=executed["evidence"],comparison=compared,
                        input_sha256=digest(samples),samples=3,compiler_calls=0,agent_calls=0,
                        output_structure=executed.get("output_structure")))
                    save("running")
            row["status"]="passed";save("running")
            print(json.dumps(dict(id=case,status="passed",maximum_absolute_error=
                  result["acceptance"]["maximum_absolute_error"])),flush=True)
        boundary();save("complete")
    except BaseException:
        save("stopped")
        raise
    return 0 if all(r["status"]=="passed" for r in rows) else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-qualification",type=Path,
                        help="Reuse only exact, checked rule candidates and source/dependency bindings")
    args=parser.parse_args();raise SystemExit(run(args.output,args.reuse_qualification))
