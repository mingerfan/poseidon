#!/usr/bin/env python3
"""Developer-only mapping of frozen semantics to Poseidon evidence and rejections."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import sys
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from workspace_paths import WORK
from benchmark_graph import canonical,digest,require,validate
from compiler_configuration import configuration
from check_poseidon_helpers import snapshot
from unified_graph_contract import prepare,validate_candidate,validate_request
from rejection_benchmark import fixture,identity,model
from audit_unified_candidate import verify_candidate
from poseidon_artifact_gate import inspect
from poseidon_runtime_paths import SPEC
from poseidon_backend_identity import identity as backend_identity

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_bytes(canonical(d))
def artifact_sources():
    reports=[WORK/"results/poseidon-public-pilot-r255/report.json"]
    reports+=sorted((WORK/"results").glob("poseidon-public-shard*-r258/report.json"))
    rows=[]
    for p in reports:
        for r in read(p)["rows"]:
            if r["status"]=="passed" and r.get("audited"):
                rows.append(dict(r,parent_report=str(p),parent_report_sha256=sha(p)))
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();out=a.output
    require(out.resolve().parent==(WORK/"results").resolve() and not out.exists(),"Fresh platform result directory")
    sources=snapshot();back=backend_identity();start=time.monotonic()
    suite=ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
    ledger=read(suite/"coverage.json");original={t["id"]:t for t in ledger["rejection_tasks"]}
    controls=artifact_sources()
    require(len(controls)>=48,"Verified current public controls required")
    out.mkdir();cache={}
    def audit(r):
        name=r["evidence"]
        if name not in cache:cache[name]=verify_candidate(Path(name))
        return cache[name]
    selected=[];tops=set()
    for r in controls:
        if r["topology"] not in tops:
            audit(r);selected.append(r);tops.add(r["topology"])
        if len(selected)==3:break
    require(len(selected)==3,"Three distinct real Poseidon controls")
    cfg=configuration("poseidon-cpu-eva-w45-v1")
    requests={i:prepare(model(i),cfg["ckks_config_sha256"],cfg) for i in range(3)}
    rejects=[]
    for t in ledger["rejection_tasks"]:
        name=t["requirement"].removeprefix("reject.");ctx=t["context"]
        f=fixture(name,ctx)
        require(identity(f["positive"])==t["positive_sha256"] and identity(f["negative"])==t["negative_sha256"],
                "Historical rejection fixture changed")
        row=dict(id=t["id"],requirement=t["requirement"],context=ctx,original_task_sha256=t["task_sha256"],
                 original_layer=t["layer"],status="failed",encrypted_execution=False,
                 actual_compile=False,scope="static gates only; positive artifact controls already independently executed")
        try:
            if f["layer"]=="model_validation":
                positive=f["positive"];negative=f["negative"];check=validate;expected=f["expected_reason"]
                row["relation"]="exact_original_model_fixture"
            elif f["layer"]=="candidate_ast_contract":
                r=requests[ctx]
                positive=dict(f["positive"][1],request_id=r["request_id"])
                negative=dict(f["negative"][1],request_id=r["request_id"])
                check=lambda value,r=r:validate_candidate(value,r);expected=f["expected_reason"]
                row["relation"]="same_candidate_source_new_poseidon_request"
            elif f["layer"]=="request_binding":
                positive=requests[ctx];negative=copy.deepcopy(positive);check=validate_request
                if name=="compiler_configuration":
                    key,value=(("waterline",46),("pipeline","other"),("backend","plaintext"))[ctx]
                    negative["compiler_configuration"][key]=value;expected="Changed compiler configuration"
                elif name=="request_integrity":negative["request_id"]="0"*64;expected="request hash mismatch"
                else:
                    negative["layout"]["output_selectors"][0]=[0,1];expected="Changed unified physical layout"
                if name!="request_integrity":negative["request_id"]=digest({k:v for k,v in negative.items() if k!="request_id"})
                row["relation"]="same_mutation_new_poseidon_request"
            else:
                r=selected[ctx];e=Path(r["evidence"]);request=read(e/"request.json")
                folder=e/"attempt-00/output/bound"
                positive=read(folder/"plan.json");negative=copy.deepcopy(positive)
                kind="boot" if name=="unsupported_bootstrap" else "upscale"
                negative["execution"][0]["op"]=kind;expected="unsupported capability"
                positive_file=out/(t["id"]+".positive.json");negative_file=out/(t["id"]+".negative.json")
                dump(positive_file,positive);dump(negative_file,negative)
                check=lambda value:inspect(value,SPEC,request,bundle=folder/"bundle")
                row.update(relation="equivalent_forbidden_capability_new_runtime_plan_fixture",
                    positive_control_evidence=r["evidence"],positive_control_topology=r["topology"])
                positive,negative=positive_file,negative_file
            row.update(positive_identity=sha(positive) if isinstance(positive,Path) else identity(positive),
                       negative_identity=sha(negative) if isinstance(negative,Path) else identity(negative))
            check(positive);row["positive_control_accepted"]=True
            try:check(negative)
            except ValueError as error:
                reason=str(error);require(expected in reason,"Unexpected rejection reason: "+reason)
                row.update(status="passed",negative_rejected=True,reason=reason)
            else:raise ValueError("Negative fixture unexpectedly accepted")
        except Exception as error:row.update(diagnostic=str(error),error_type=type(error).__name__)
        rejects.append(row)
    compiler=[]
    kinds={"compiler.rescale":"rescale","compiler.modswitch":"mod_switch","compiler.relinearization":"relinearize",
           "compiler.rotation_keys":"rotate","compiler.security_parameters":None}
    for requirement,kind in kinds.items():
        chosen=[];tops=set()
        for r in controls:
            if r["topology"] in tops:continue
            e=Path(r["evidence"]);bound=e/"attempt-00/output/bound";plan=read(bound/"plan.json")
            ops=[op for op in plan["execution"] if op["kind"]=="compute" and (kind is None or op["op"]==kind)]
            if not ops:continue
            audited=audit(r);meta=read(e/"attempt-00/output/execution-metadata.json")
            require(meta["backend"]=="poseidon_cpu" and meta["encrypted_execution"] and meta["executor_secret_key"] is False,
                    "Actual Poseidon execution metadata required")
            require(meta["security"]=="tc128" and meta["degree"]==32768 and len(meta["q"])==13 and len(meta["p"])==1,
                    "Actual fixed parameter identity")
            chosen.append(dict(task_id=r["task"]["id"],topology=r["topology"],evidence=str(e),
                parent_report=r["parent_report"],parent_report_sha256=r["parent_report_sha256"],
                request_id=audited["request_id"],model_sha256=audited["model_sha256"],
                plan_sha256=sha(bound/"plan.json"),execution_metadata_sha256=sha(e/"attempt-00/output/execution-metadata.json"),
                matching_ordinals=[op["ordinal"] for op in ops],actual_encrypted_comparison=audited["comparison"],
                scope="artifact instructions, actual parameter/final-output evidence and pinned runtime source; no per-instruction runtime trace"))
            tops.add(r["topology"])
            if len(chosen)==3:break
        compiler.append(dict(id=requirement,status="passed" if len(chosen)==3 else "not_run",contexts=chosen,
            protocol="poseidon-runtime-plan-evidence-v1",real_agent_generated=False))
    historical=[]
    for name in ["poseidon-helpers-shard0-r233","poseidon-helpers-shard1-r233"]:
        path=WORK/"results"/name/"report.json"
        for r in read(path)["rows"]:
            if r["status"]=="passed" and r.get("audited"):
                historical.append(dict(r,parent_report=str(path),parent_report_sha256=sha(path)))
    helper_rows=[]
    for t in ledger["helper_directed_tasks"]:
        matched=[]
        for r in historical:
            e=Path(r["evidence"])
            if r["name"]!=t["model"]["id"] or r["profile"]!=t["profile"] or r["helpers"]!=t["required_helpers"]:continue
            if digest(read(e/"model.json"))!=t["model_sha256"]:continue
            report=read(e/"report.json");request=read(e/"request.json");attempt=report["attempts"][-1]
            old_backend=report["backend_identity"]
            if old_backend.get("format")=="poseidon-backend-identity-v1":
                require(set(old_backend)==set(back)-{"build_configuration"},"Historical backend identity fields changed")
                require(all(old_backend[k]==back[k] for k in old_backend if k!="format"),
                        "Historical Poseidon source or binary identity changed")
            else:require(old_backend==back,"Historical Poseidon backend identity changed")
            require(report["backend"]=="poseidon_cpu" and report["agent_calls"]==0,
                    "Historical real Poseidon manual evidence required")
            require(attempt["compiled"] and attempt["executed"] and attempt["numerically_correct"] and
                    attempt["comparison"]["passed"],"Historical numerical record incomplete")
            for n,h in report["frozen_hashes"].items():require(sha(e/n)==h,"Historical frozen artifact changed")
            for n,h in attempt["artifact_hashes"].items():require(sha(e/"attempt-00/output"/n)==h,"Historical output artifact changed")
            coverage=r["contribution"]
            require(coverage["actual_frontend_checked"] and coverage["finite_return_influence_checked"] and
                    set(coverage["witnesses"])==set(t["required_helpers"]),"Historical actual helper contribution missing")
            matched.append(dict(evidence=str(e),request_id=request["request_id"],configuration_schema=request["compiler_configuration"]["schema"],
                parent_report=r["parent_report"],parent_report_sha256=r["parent_report_sha256"],
                report_sha256=sha(e/"report.json"),model_sha256=t["model_sha256"],topology=t["topology"],
                original_source_hashes=report["source_hashes"],historical_backend_identity=old_backend,
                missing_historical_build_configuration=old_backend["format"]=="poseidon-backend-identity-v1",
                current_binary_and_source_tree_hashes_match=True,historical_audited_when_executed=True,
                current_source_audit_claim=False,comparison=attempt["comparison"]))
        helper_rows.append(dict(id=t["id"],original_task_sha256=t["task_sha256"],
            status="historical_poseidon_binding_verified" if matched else "not_mapped",contexts=matched))
    # Frozen counterexamples are replayed as static tests, separately from FHE controls.
    corpus={}
    for file in suite.glob("shard-*.json"):
        for row in read(file)["rows"]:corpus[row["model"]["id"]]=row["model"]
    model_negatives=[]
    for task in read(suite/"semantic-component-model_negatives.json")["tasks"]:
        row=dict(id=task["id"],requirement=task["requirement"],task_sha256=task["task_sha256"],
                 status="failed",actual_frontend_executed=False,encrypted_execution=False)
        try:
            positive=corpus[task["positive_model_id"]];negative=task["negative_model"]
            require(digest(positive)==task["positive_model_sha256"] and digest(negative)==task["negative_model_sha256"],
                    "Frozen mathematical counterexample changed")
            validate(positive)
            try:validate(negative)
            except ValueError as error:
                require(str(error)==task["observed_rejection"],"Changed mathematical rejection: "+str(error))
                row.update(status="passed",positive_control_accepted=True,negative_rejected=True,reason=str(error))
            else:raise ValueError("Mathematical negative accepted")
        except Exception as error:row["diagnostic"]=str(error)
        model_negatives.append(row)
    sys.path.insert(0,str(Path(__file__).parent))
    from verify_construction_trace_counterexamples import verify as verify_trace_negative
    construction_negatives=[]
    for case in read(suite/"semantic-component-trace_negatives.json")["fixtures"]:
        row=dict(id=case["id"],requirement=case["requirement"],fixture_sha256=case["fixture_sha256"],status="failed",
                 actual_frontend_executed=False,encrypted_execution=False)
        try:
            require(digest({k:v for k,v in case.items() if k!="fixture_sha256"})==case["fixture_sha256"],
                    "Frozen synthetic trace counterexample changed")
            row.update(verify_trace_negative(case),status="passed")
        except Exception as error:row["diagnostic"]=str(error)
        construction_negatives.append(row)
    from upstream_helper_directed_cases import cases as helper_cases
    from upstream_candidate_helpers import BN_PROFILE
    from verify_helper_call_counterexamples import omit_calls
    definitions={"upstream_"+r["name"]:r for r in helper_cases()}
    helper_negatives=[]
    for case in read(suite/"semantic-component-helper_negatives.json")["fixtures"]:
        original=next(t for t in ledger["helper_directed_tasks"] if t["id"]==case["positive_task_id"])
        definition=definitions[original["id"]]
        row=dict(id=case["id"],original_positive_task_sha256=case["positive_task_sha256"],status="failed",
                 actual_frontend_executed=False,encrypted_execution=False,
                 relation="same_frozen_model_source_and_call_omission_new_poseidon_request")
        try:
            require(original["task_sha256"]==case["positive_task_sha256"] and digest(definition["model"])==original["model_sha256"],
                    "Frozen helper omission parent changed")
            request=prepare(definition["model"],cfg["ckks_config_sha256"],cfg,
                helper_profile=definition.get("profile",BN_PROFILE),helper_exercise=definition["required_helpers"],
                chunk_period=definition.get("chunk_period"))
            positive=dict(schema=1,request_id=request["request_id"],hecate_source=definition["source"])
            validate_candidate(positive,request)
            negative,count=omit_calls(definition["source"],case["omitted_helper"])
            try:validate_candidate(dict(positive,hecate_source=negative),request)
            except ValueError as error:
                require(str(error)==case["expected_rejection"],"Changed helper omission reason: "+str(error))
                row.update(status="passed",positive_control_accepted=True,negative_rejected=True,reason=str(error),
                    request_id=request["request_id"],removed_calls=count,
                    positive_source_sha256=hashlib.sha256(definition["source"].encode()).hexdigest(),
                    negative_source_sha256=hashlib.sha256(negative.encode()).hexdigest())
            else:raise ValueError("Helper omission accepted")
        except Exception as error:row["diagnostic"]=str(error)
        helper_negatives.append(row)
    require(snapshot()==sources,"Source binding changed")
    result=dict(format="poseidon-semantic-contexts-v1",sources=sources,backend_identity=back,
        runner_sha256=sha(Path(__file__)),ledger_sha256=sha(suite/"coverage.json"),
        rejections=rejects,rejection_passed=sum(r["status"]=="passed" for r in rejects),
        rejection_failed=sum(r["status"]!="passed" for r in rejects),compiler=compiler,
        compiler_partitions_passed=sum(r["status"]=="passed" for r in compiler),
        historical_helpers=helper_rows,historical_helper_tasks_mapped=sum(r["status"]!="not_mapped" for r in helper_rows),
        mathematical_partition_negatives=model_negatives,
        construction_trace_negatives=construction_negatives,helper_omission_negatives=helper_negatives,
        static_extra_passed=sum(r["status"]=="passed" for rows in [model_negatives,construction_negatives,helper_negatives] for r in rows),
        static_extra_failed=sum(r["status"]!="passed" for rows in [model_negatives,construction_negatives,helper_negatives] for r in rows),
        paid_calls=0,new_compilations=0,new_encrypted_executions=0,agent_generated_successes=0,seconds=time.monotonic()-start,
        scope="Current static checks, existing independently audited compiler/FHE controls, and explicitly historical helper source bindings")
    dump(out/"report.json",result)
    print(json.dumps({k:result[k] for k in ["rejection_passed","rejection_failed","compiler_partitions_passed","historical_helper_tasks_mapped",
                                         "static_extra_passed","static_extra_failed","paid_calls","new_compilations","new_encrypted_executions","seconds"]}),flush=True)
    return int(result["rejection_failed"] or result["static_extra_failed"] or result["compiler_partitions_passed"]!=5 or result["historical_helper_tasks_mapped"]!=60)
if __name__=="__main__":raise SystemExit(main())
