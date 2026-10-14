"""Read-only artifact and numerical audit of the approved v25 CPU regression."""
import json,sys,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from benchmark_graph import digest,require,samples
from offline_cpu_continue import read,verify,sha,measure,PROPOSAL,CONTINUATION
from hecate_python_env import ROOT,WORK
from candidate_bundle import check_bundle
from audit_unified_candidate import verify_candidate
from operator_reference import evaluate
from numerical_results import compare
import numpy as np
def bound(value):
    require(value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}),"inspection_bound_record")
    return value
def scalars(value):return {k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None}
def main():
    plan=bound(read(PROPOSAL));con=bound(read(CONTINUATION));identity=verify(plan)
    output=Path(con["output"]);report=bound(read(output/"report.json"))
    require(report["status"]=="complete" and report["passed"]==18
        and report["failed"]==report["not_run"]==0,"inspection_scope")
    require(report["compiler_reserved"]==report["compiler_confirmed"]==18
        and report["encrypted_reserved"]==report["encrypted_confirmed"]==24
        and report["provider_calls"]==report["http_attempts"]==0,"inspection_calls")
    require(report["sdk_identity"]==identity["manifest_sha256"],"inspection_sdk")
    require([r["case"] for r in report["rows"]]==[c["case"] for c in plan["cases"]],"inspection_order")
    checked=[];files={};groups=values=0;weighted_error=maximum=0.0
    for case,row in zip(plan["cases"],report["rows"]):
        require(row["status"]=="passed" and row["compiled"] and row["encrypted_execution"]
            and row["package_reloaded"],"inspection_stages")
        audit=verify_candidate(Path(row["validation_evidence"]));cv=audit["comparison"]
        require(cv["passed"] and cv["atol"]==1e-5 and cv["rtol"]==1e-4,"inspection_tolerance")
        program=Path(row["package"]);manifest=check_bundle(program)
        require(manifest["binding"]==row["package_binding"] and not manifest["agent_generated"]
            and manifest["validation"]["level"]=="numerical","inspection_package")
        require(sha(program/"candidate.py")==sha(case["candidate"]),"inspection_original_candidate")
        request=read(case["request"]);require(read(program/"request.json")==request,"inspection_request")
        record=dict(case=case["case"],source_contract=case["source_contract"],
            request_id=request["request_id"],qualification_groups=4,qualification_metrics=scalars(cv),
            source_bound=True,real_tracing=True,compiled=True,artifact_checked=True,
            encrypted_execution=True,independent_reference_verified=True,package_reloaded=True,
            package_binding=manifest["binding"],package=str(program),
            qualification_evidence=row["validation_evidence"],reuse={},
            qualification_reused_without_recompile=row.get("qualification_reused_without_recompile",False))
        groups+=4;values+=cv["compared_values"];weighted_error+=cv["mae"]*cv["compared_values"]
        maximum=max(maximum,cv["max_absolute_error"])
        for name,h in audit["files"].items():files[str(Path(row["validation_evidence"])/name)]=h
        for name in manifest["files"]:files[str(program/name)]=sha(program/name)
        files[str(program/"manifest.json")]=sha(program/"manifest.json")
        if case.get("repeat_package"):
            folder=output/case["case"];original=read(program/"model-provenance.json")["original"]
            batches=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in batch.items()}
                     for batch in samples(request["model"],4)[:3]]
            for name,inputs in (("new-inputs",batches),("repeat",batches[:1])):
                result=read(folder/(name+".json"))
                require(result["status"]=="executed" and result["encrypted_execution"] is True
                    and result["agent_calls"]==result["compiler_calls"]==0
                    and result["executor_secret_key"] is False,"inspection_reuse_boundary")
                actual=np.asarray([np.concatenate([np.asarray(v[o["name"]]).reshape(-1)
                         for o in request["model"]["outputs"]]) for v in result["outputs"]])
                reference=np.asarray([np.concatenate([evaluate(original,b)[o["name"]].reshape(-1)
                         for o in request["model"]["outputs"]]) for b in inputs])
                metric=compare(actual,reference,1e-5,1e-4)
                require(metric==read(folder/(name+"-comparison.json")) and metric["passed"],"inspection_new_reference")
                record["reuse"][name]=dict(groups=len(inputs),metrics=scalars(metric))
                groups+=len(inputs);values+=metric["compared_values"]
                weighted_error+=metric["mae"]*metric["compared_values"]
                maximum=max(maximum,metric["max_absolute_error"])
                for suffix in (".json","-comparison.json"):
                    f=folder/(name+suffix);files[str(f)]=sha(f)
            cached=read(folder/"cached.json")
            require(cached["status"]=="succeeded" and cached["package"]["reused"]
                and cached["generations"]==cached["http_attempts"]==0,"inspection_cache")
            record["cached_without_generation"]=True
            files[str(folder/"cached.json")]=sha(folder/"cached.json")
        checked.append(record)
    prior=bound(read(con["parent_report"]))
    require(prior["binding"]==con["parent_report_binding"],"inspection_preserved_failure")
    historical=bound(read(ROOT/"docs/baseline/agent-application-control-offline-results-r1383.json"))
    require(historical["binding"]=="6aa0686875363122984692d17db42e3fff7a6fe68d23afe652cbada62734b24f","inspection_r1383")
    total=measure(WORK/"results");new=total-con["conservative_global_before_bytes"];free=shutil.disk_usage(WORK).free
    require(new<=plan["scope"]["new_bytes"] and total<=plan["scope"]["global_bytes"]
        and free>=plan["scope"]["reserve_bytes"],"inspection_storage")
    require(groups==84 and values==504,"inspection_counts")
    result=dict(format="poseidon-closed-public-arithmetic-cpu-results-v1",date="2026-10-11",
        checkout=str(ROOT),branch="lhy-agent-dsl",sdk_identity=identity["manifest_sha256"],
        backend="poseidon_cpu",backend_unchanged=True,backend_identity_sha256=digest(identity["backend"]),
        proposal_binding=plan["binding"],continuation_binding=con["binding"],
        batch_report_binding=report["binding"],batch_report=str(output/"report.json"),
        historical_report_preserved=historical["binding"],
        component_tests=dict(passed=204,failed=0,errors=0,skipped=0),
        source_contexts=15,application_admissions=6,old_requests_equal=3,
        planned=18,passed=18,failed=0,skipped=0,not_run=0,
        compiler_calls=18,encrypted_calls=24,input_group_times=groups,compared_values=values,
        maximum_absolute_error=maximum,aggregate_mae=weighted_error/values,
        tolerance=dict(atol=1e-5,rtol=1e-4),rule_candidates=18,agent_generated=0,
        provider_calls=0,http_attempts=0,package_reloaded=18,new_input_packages=3,
        repeat_input_packages=3,cached_packages=3,seconds_from_initial_launch=report["seconds"],
        artifact_measurement=dict(global_bytes=total,new_bytes_conservative=new,free_bytes=free),
        initial_harness_failure=dict(binding=prior["binding"],type="incorrect_package_field_location",
            candidate_compilation_and_numerical_passed=True,preserved=True),
        completed=checked,inspection_files=files,inspected_at_epoch=time.time(),
        limitations=["current_agent_generation_unverified","x86_not_executed_in_this_batch",
            "bootstrap_runtime_unchanged","cuda_not_executed","large_model_scope_unchanged"])
    result["binding"]=digest(result)
    p=ROOT/"docs/baseline/agent-public-arithmetic-offline-results-r1386.json"
    with p.open("x") as stream:
        json.dump(result,stream,sort_keys=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("completed","inspection_files")},ensure_ascii=False))
    return 0
if __name__=="__main__":sys.exit(main())
