"""Bounded offline qualification of the existing 10,000-weight Linear model."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def run(output):
    import numpy as np
    from application_execution.context import WorkerContext,canonical,sha
    from application_execution.backend import application
    from application_planning.layout import plan_model,check_plan
    from application_planning.resources import assess_plan,submit_planned,source_authority
    from benchmark_graph import require,digest,samples
    from public_weight_cases import models
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from numerical_results import compare
    from unified_graph_lowering import candidate_source
    from candidate_bundle import check_bundle,check_execution_dependencies,replay_bundle
    from benchmarks.tools.run_application_matrix import freeze,items_from_plan
    from hecate_python_env import WORK
    results=WORK/"results"
    require(not output.exists() and output.parent.resolve()==results.resolve(),"Fresh large-weight output")
    model=next(m for m in models() if m["id"]=="weights_linear_10k")
    historical=json.loads((ROOT/"docs/baseline/agent-generalization-acceptance-r622.json").read_text())["large_weight_request"]
    require(digest(model)==historical["model_sha256"],"Preserve the original 10k model and weights")
    weights=np.asarray(next(v for v in model["constants"].values() if np.asarray(v).shape==(100,100)))
    require(np.unique(weights).size==10000,"Nonrepeating large weights")
    pending=json.loads((ROOT/"docs/baseline/agent-pytorch-state-live-proposal-r635.json").read_text())
    require(freeze(items_from_plan(pending))==pending,"Pending paid proposal drift")
    output.mkdir(mode=0o700)
    context=WorkerContext.capture(output/"tasks");app=application(context)
    planned=plan_model(model,options=dict(chunk_period=128,generation_guidance="explicit-v10",
                                         public_weight_policy="descriptors-v1"))
    require(planned["status"]=="prepared","Large request layout")
    request=check_plan(planned,model)
    require(request["request_id"]==historical["request_id"],"Preserve original request")
    analysis=assess_plan(planned,model)
    require(analysis["status"]=="assessed" and analysis["execution_readiness"]=="unknown","Preflight scope")
    authority=source_authority()
    plan=dict(format="poseidon-large-weight-offline-plan-v1",model_sha256=digest(model),
        historical_request_id=historical["request_id"],request_id=request["request_id"],
        input_elements=100,output_elements=100,weight_elements=10000,nonrepeating_weights=10000,
        source_authority=authority,worker_binding=context.binding,pending_paid_binding=pending["binding"],
        tool_sha256=sha(Path(__file__).read_bytes()),layout_plan_binding=planned["binding"],
        resource_analysis_binding=analysis["binding"],source="rule baseline",paid_calls=0,
        max_seconds=3600,artifact_bytes=4*1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,
        sample_count=4,repeat_count=2,atol=1e-5,rtol=1e-4)
    plan["binding"]=digest(plan)
    (output/"plan.json").write_bytes(canonical(plan))
    (output/"model.json").write_bytes(canonical(model))
    (output/"request.json").write_bytes(canonical(request))
    (output/"layout.json").write_bytes(canonical(planned))
    (output/"resources.json").write_bytes(canonical(analysis))
    (output/"worker.json").write_bytes(context.payload)
    (output/"worker.json").chmod(0o600)
    original_samples=list(samples(model,count=4))
    for i,args in enumerate(original_samples):
        expected=mathematical(model,args);actual=tensor_reference(model,args)
        for name in expected:np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
    started=time.monotonic();retained={output}
    report=dict(format="poseidon-large-weight-offline-report-v1",binding=plan["binding"],status="running",
        model_sha256=digest(model),request_id=request["request_id"],source="rule baseline",
        agent_generated=False,paid_calls=0,reference_samples=4,reference_crosscheck_passed=True,
        qualification=None,reload=None,repeated_new_inputs=[],cache_reuse=None)
    def save(status):
        report.update(status=status,seconds=time.monotonic()-started)
        (output/"report.json").write_bytes(canonical(report))
    def boundary():
        context.check_current()
        require(source_authority()==authority,"Resource source drift")
        require(time.monotonic()-started<plan["max_seconds"],"Large qualification time budget")
        require(shutil.disk_usage(WORK).free>=plan["free_bytes"],"Large qualification disk reserve")
        total=int(subprocess.check_output(["du","-sb",str(results)],text=True,timeout=30).split()[0])
        require(total<=plan["global_bytes"],"Large qualification global artifact budget")
        require(sum(p.stat().st_size for d in retained for p in d.rglob("*") if p.is_file())
                <=plan["artifact_bytes"],"Large qualification artifact budget")
    def collect(before):retained.update(set(results.iterdir())-before)
    class Rule:
        kind="scripted_replay";agent_calls=0
        def generate(self,request,history):
            source=candidate_source(request)[0]
            (output/"rule-candidate.py").write_text(source)
            return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
    save("running")
    try:
        boundary();before=set(results.iterdir())
        submitted=submit_planned(app,model,planned,analysis,limits=dict(generations=1,http_attempts=1,
            wall_seconds=3600,api_timeout_seconds=1200,max_tokens=384000))
        report["submission"]=submitted;save("running")
        result=app.run(submitted["task_id"],Rule());collect(before)
        report["qualification"]=result;save("running")
        if result["status"]!="succeeded":
            save("failed");print(json.dumps(dict(status="failed",failure=result.get("failure"))),flush=True)
            return 1
        package=output/"tasks"/submitted["task_id"]/"program"
        manifest,checked_request=check_execution_dependencies(package)
        require(checked_request==request and manifest["agent_generated"] is False and result["http_attempts"]==0,
                "Exact rule-source and package request")
        report.update(package=str(package),package_binding=manifest["binding"])
        boundary();before=set(results.iterdir())
        require(replay_bundle(package)==0,"Large-weight package reload")
        new=set(results.iterdir())-before;collect(before)
        paths=[p for p in new if p.name.startswith("poseidon-package-replay-")]
        require(len(paths)==1,"Unique large package replay")
        report["reload"]=json.loads((paths[0]/"report.json").read_text());save("running")
        # Independent deterministic inputs different from the fixed qualification set.
        logical=[];expected=[]
        for index in range(3):
            rng=np.random.default_rng(20261007+index)
            args={v["name"]:rng.uniform(-.25,.25,v["shape"]) for v in model["inputs"]}
            outputs=mathematical(model,args);torch_outputs=tensor_reference(model,args)
            for name in outputs:np.testing.assert_allclose(outputs[name],torch_outputs[name],atol=1e-12,rtol=1e-12)
            logical.append({name:v.tolist() for name,v in args.items()})
            expected.append(np.concatenate([np.asarray(outputs[v["name"]]).reshape(-1) for v in model["outputs"]]))
        (output/"new-inputs.json").write_bytes(canonical(logical))
        for repetition in range(plan["repeat_count"]):
            boundary();before=set(results.iterdir())
            actual=app.execute(submitted["task_id"],logical);collect(before)
            require(actual["status"]=="executed" and actual["encrypted_execution"]
                and actual["agent_calls"]==actual["compiler_calls"]==0 and actual["executor_secret_key"] is False,
                "Large package real new-input execution")
            got=np.asarray([np.concatenate([np.asarray(row[v["name"]]).reshape(-1) for v in model["outputs"]])
                            for row in actual["outputs"]])
            compared=compare(got,np.asarray(expected),plan["atol"],plan["rtol"])
            require(compared["passed"],"Large new-input numerical comparison")
            report["repeated_new_inputs"].append(dict(repetition=repetition,evidence=actual["evidence"],
                comparison=compared,run_context=actual["run_context"],agent_calls=0,compiler_calls=0))
            save("running")
        before=set(results.iterdir());new_task=app.submit(model,options=planned["selection"]["options"])
        class NoGeneration:
            kind="scripted_replay";agent_calls=0
            def generate(self,*_):raise AssertionError("Bound cache must not generate")
        reused=app.run(new_task,NoGeneration());collect(before)
        require(reused["status"]=="succeeded" and reused["package"]["reused"]
            and reused["generations"]==reused["http_attempts"]==0,"Large bound cache reuse")
        report["cache_reuse"]=reused
        boundary();require(freeze(items_from_plan(pending))==pending,"Pending paid proposal final drift")
        save("passed")
        print(json.dumps(dict(status="passed",binding=plan["binding"],maximum_absolute_error=
            result["acceptance"]["maximum_absolute_error"],seconds=report["seconds"],paid_calls=0)),flush=True)
        return 0
    except BaseException:
        save("stopped");raise
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.output))
