"""Continue unchanged r1390 models within its original budget after an Agent adapter fix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark_graph import digest, require
from hecate_python_env import ROOT, WORK, VENV, enter_nix
TOOL=Path(__file__).resolve().parent
PROPOSAL=ROOT/"docs/baseline/agent-input-binding-offline-proposal-r1390.json"
AUTH=ROOT/"docs/baseline/agent-input-binding-offline-authorization-r1390.json"
RECEIPT=ROOT/"docs/baseline/agent-input-binding-offline-continuation-r1391.json"

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,sort_keys=False,indent=2,allow_nan=False)
        stream.write("\n")
def measure(folder):
    return max(int(subprocess.check_output(["du",flag,"-s","-B1",str(folder)],
                    text=True,timeout=20).split()[0])
               for flag in ("--apparent-size","--count-links"))

def verify(plan):
    require(plan["binding"]==digest({k:v for k,v in plan.items() if k!="binding"}),"control_proposal_binding")
    require(plan["format"]=="poseidon-structured-input-offline-ready-v1"
            and plan["approved"] is False and plan["backend"]=="poseidon_cpu"
            and plan["candidate_origin"]=="frozen_deterministic_rule"
            and plan["provider_calls"]==plan["http_attempts"]==0,"control_scope")
    from application_identity import current_identity_manifest
    actual=current_identity_manifest()
    require(actual["manifest_sha256"]==plan["sdk_identity"],"control_current_sdk")
    require(actual["backend"]==plan["backend_identity"],"control_backend_identity")
    for name,expected in plan["tools"].items():
        require(sha(name)==expected,"control_tool_changed")
    for case in plan["cases"]:
        for name,expected in case["files"].items():
            require(sha(name)==expected,"control_fixture_changed")
        if "historical_task_metadata" in case:
            require(sha(case["historical_task_metadata"])==case["historical_task_metadata_sha256"],
                    "control_historical_metadata_changed")
    receipt=read(RECEIPT)
    require(receipt["binding"]==digest({k:v for k,v in receipt.items() if k!="binding"}),"control_continuation_binding")
    require(receipt["original_proposal_binding"]==plan["binding"] and
            receipt["maximum_compiler_calls"]==plan["scope"]["maximum_compiler_calls"] and
            receipt["maximum_encrypted_calls"]==plan["scope"]["maximum_encrypted_calls"],
            "control_no_scope_expansion")
    require(sha(AUTH)==receipt["authorization_sha256"] and sha(__file__)==receipt["runner_sha256"],
            "control_continuation_authority")
    for path,expected in receipt["protected_reports"].items():
        require(sha(path)==expected,"control_historical_evidence_changed")
    for path,expected in receipt["controller_files"].items():
        require(sha(path)==expected,"control_adapter_binding")
    return actual

def inside():
    import signal
    import numpy as np
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from candidate_bundle import check_bundle
    from benchmark_graph import samples
    from operator_reference import evaluate
    from numerical_results import compare
    plan=read(PROPOSAL);verify(plan)
    authorization=read(AUTH)
    require(authorization["proposal_binding"]==plan["binding"]
            and authorization.get("approved") is True,"control_human_authorization")
    receipt=read(RECEIPT)
    output=Path(plan["output"])/"continuation-r1391"
    require(not output.exists(),"control_preserve_existing_output")
    output.mkdir(mode=0o700)
    started=time.monotonic();global_before=measure(WORK/"results")
    scope=plan["scope"];rows=[]
    deadline=receipt["original_worker_deadline_unix"]
    require(time.time()<deadline,"control_original_deadline")
    previous=read(receipt["previous_report"])
    retained={r["case"]:r for r in previous["rows"]}
    report=dict(format="poseidon-structured-input-offline-report-v1",
        proposal_binding=plan["binding"],sdk_identity=plan["sdk_identity"],
        rows=rows,provider_calls=0,http_attempts=0,agent_generated=False,
        candidate_origin=plan["candidate_origin"],compiler_reserved=3,encrypted_reserved=3,
        compiler_confirmed=3,encrypted_confirmed=3,status="running",
        continuation_binding=receipt["binding"],retained_qualifications=3,
        original_worker_deadline_unix=deadline,worker_source_authority_changed=True,
        additional_compiler_calls=0,additional_encrypted_calls=0)
    def save():
        report["seconds"]=time.monotonic()-started
        report["additional_compiler_calls"]=report["compiler_confirmed"]-3
        report["additional_encrypted_calls"]=report["encrypted_confirmed"]-3
        report["cumulative_authorization_seconds"]=time.time()-receipt["authorization_unix"]
        report["passed"]=sum(r["status"]=="passed" for r in rows)
        report["failed"]=sum(r["status"]=="failed" for r in rows)
        report["not_run"]=len(plan["cases"])-len(rows)
        report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
        tmp=output/"report.tmp";tmp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        tmp.replace(output/"report.json")
    def boundary(*_):
        # Baseline physical/logical measurements include all fresh qualification
        # and package execution directories outside the host task store.
        if time.time()>=deadline:raise TimeoutError("control_original_wall_budget")
        current=measure(WORK/"results")
        require(current-global_before<=scope["new_bytes"],"control_new_artifact_budget")
        require(current<=scope["global_bytes"],"control_global_artifact_budget")
        require(shutil.disk_usage(WORK).free>=scope["reserve_bytes"],"control_disk_reserve")
        require(report["compiler_reserved"]<=scope["maximum_compiler_calls"] and
                report["encrypted_reserved"]<=scope["maximum_encrypted_calls"],"control_call_budget")
    save()
    try:
        for case in plan["cases"]:
            boundary();row=dict(case=case["case"],status="running",compiled=False,
                                encrypted_execution=False,package_reloaded=False)
            rows.append(row);save()
            folder=output/case["case"];folder.mkdir()
            if case["case"] in retained:
                previous_row=retained[case["case"]]
                source_task=Path(plan["output"])/case["case"]/"tasks"/previous_row["task_id"]
                require(source_task.is_dir() and not any(p.is_symlink() for p in source_task.rglob("*")),
                        "control_retained_regular_task")
                task_snapshot={str(p.relative_to(source_task)):sha(p)for p in source_task.rglob("*")if p.is_file()}
                shutil.copytree(source_task,folder/"tasks"/previous_row["task_id"])
                require(all(sha(folder/"tasks"/previous_row["task_id"]/n)==h for n,h in task_snapshot.items()),
                        "control_exact_task_copy")
            app=application(WorkerContext.capture(folder/"tasks"))
            row["worker_context_binding"]=app.backend.context.binding
            source=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
                        manifest=case["source_manifest"],trust_local_code=True)
            if case["case"] in retained:
                task=retained[case["case"]]["task_id"]
                require(app.result(task)["status"]=="succeeded","control_retained_success")
            else:
                task=app.submit_source(source,resource_profile=case["resource_profile"],
                    options=plan["options"],admission_policy="structured-v1",
                    operational_limits=plan["request_operational_limits"],
                    limits=plan["generation_limits"])
                status=app.prepare_source(task)
                require(status["status"]=="queued" and status["intake"]["reference_verified"] is True,
                        "control_source_admission")
            row["task_id"]=task
            meta=app._checked_task_metadata(task);request=meta["request"]
            expected=read(case["request"])
            require(request==expected and request["request_id"]==case["request_id"],"control_unchanged_request")
            expected_original=(read(case["provenance"])["original"] if "provenance" in case else
                               read(case["historical_task_metadata"])["model_provenance"]["original"])
            require(meta["model_provenance"]["original"]==expected_original,
                    "control_unchanged_reference_model")
            candidate_source=Path(case["candidate"]).read_text()
            class FrozenRule:
                kind="scripted_replay";agent_calls=0
                def generate(self,actual,feedback=None):
                    require(actual==expected and feedback==[],"control_one_frozen_candidate")
                    return json.dumps(dict(schema=1,request_id=actual["request_id"],hecate_source=candidate_source))
            if case["case"] in retained:
                qualified=retained[case["case"]]["application_result"]
                row["qualification_reused_from_original_binding"]=True
            else:
                report["compiler_reserved"]+=1;report["encrypted_reserved"]+=1;save();boundary()
                qualified=app.run(task,FrozenRule())
            write(folder/"qualification.json",qualified)
            if (app._dir(task)/"validation-0.json").exists():
                observed=app._load(app._dir(task)/"validation-0.json")
                row["observed_stages"]=observed.get("observed_stages",{})
                row["validation_evidence"]=observed.get("evidence")
                row["compiled"]=row["observed_stages"].get("compiled") is True
                row["encrypted_execution"]=observed.get("encrypted_execution") is True
                if case["case"] not in retained:
                    report["compiler_confirmed"]+=int(row["compiled"])
                    report["encrypted_confirmed"]+=int(row["encrypted_execution"])
            row["application_result"]=qualified
            if qualified["status"] in ("validation_failed","unsupported","budget_exhausted","timed_out"):
                failure=qualified.get("failure") or {}
                if failure.get("layer") in ("integrity","pipeline","environment"):
                    raise RuntimeError("control_global_environment_or_integrity_failure")
                row.update(status="failed",failure=failure)
                boundary();save()
                continue
            require(qualified["status"]=="succeeded" and qualified["http_attempts"]==0
                    and not qualified["generation"]["agent_generated"]
                    and not qualified["package"]["reused"],"control_numerical_qualification")
            require(qualified["acceptance"]["numerically_validated"] is True
                    and qualified["acceptance"]["input_groups"]==4,"control_numerical_scope")
            require(qualified["operational_usage"]["evidence_directories"]==1
                    and qualified["operational_usage"]["hard_filesystem_quota"] is False,"control_qualification_monitor")
            program=app._dir(task)/"program";manifest=check_bundle(program)
            require(manifest["validation"]["level"]=="numerical"
                    and manifest.get("model_provenance")=="included"
                    and "model_source" in manifest["origin"],"control_source_bundle")
            frozen={name:sha(program/name) for name in manifest["files"]}
            row.update(package_reloaded=True,package=str(program),package_binding=manifest["binding"])
            row["reuse"]={}
            if case.get("repeat_package"):
                probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in batch.items()}
                        for batch in samples(request["model"],4)[:3]]
                row["reuse"]={}
                for name,batches in (("new-inputs",probes),("repeat",probes[:1]),("named-equivalence",probes)):
                    boundary();report["encrypted_reserved"]+=1;save();boundary()
                    execution_samples=[{k:v.tolist()for k,v in batch.items()}for batch in batches]
                    execute_options={}
                    if name!="named-equivalence":
                        from pytorch_input_binding import materialize
                        call=meta["model_provenance"]["trusted_pytorch"]["manifest"]["call"]
                        execution_samples=[dict(args=[materialize(v,b)for v in call["args"]],
                            kwargs={k:materialize(v,b)for k,v in call["kwargs"].items()})for b in execution_samples]
                        execute_options["sample_format"]="forward_call"
                    result=app.execute(task,execution_samples,
                                       operational_limits=plan["execution_operational_limits"],**execute_options)
                    write(folder/(name+".json"),result)
                    report["encrypted_confirmed"]+=int(result.get("encrypted_execution") is True)
                    require(result["status"]=="executed" and result["encrypted_execution"] is True
                            and result["agent_calls"]==result["compiler_calls"]==0
                            and not result["executor_secret_key"],"control_package_execution")
                    require(result["operational_usage"]["evidence_directories"]==1,"control_execution_monitor")
                    actual=np.asarray([np.concatenate([np.asarray(v[o["name"]]).reshape(-1)
                              for o in request["model"]["outputs"]]) for v in result["outputs"]])
                    reference=np.asarray([np.concatenate([evaluate(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1)
                                 for o in request["model"]["outputs"]]) for b in batches])
                    comparison=compare(actual,reference,1e-5,1e-4)
                    write(folder/(name+"-comparison.json"),comparison)
                    require(comparison["passed"],"control_independent_reference")
                    from pytorch_source_contract import restore_outputs
                    expected_tree=meta["model_provenance"]["reference_report"]["capture_metadata"]["output_structure"]
                    expected_structured=[restore_outputs(v,expected_tree,allow_none=True)for v in result["outputs"]]
                    require(result["structured_outputs"]==expected_structured,"control_output_structure")
                    row["reuse"][name]=dict(groups=len(batches),values=comparison["compared_values"],
                              maximum_absolute_error=comparison["max_absolute_error"])
            require(check_bundle(program)==manifest and all(sha(program/n)==h for n,h in frozen.items()),
                    "control_immutable_program")
            # Check that the newly optional request policy does not change the
            # semantic package cache key or create a second validation call.
            if case.get("repeat_package"):
                cached=app.submit_source(source,resource_profile=case["resource_profile"],
                      options=plan["options"],admission_policy="structured-v1",
                      operational_limits=plan["request_operational_limits"],limits=plan["generation_limits"])
                class NoGeneration:
                    kind="scripted_replay"
                    def generate(self,*a,**k):raise AssertionError("Cache must not generate")
                reused=app.run(cached,NoGeneration())
                write(folder/"cached.json",reused)
                require(reused["status"]=="succeeded" and reused["package"]["reused"] is True
                        and reused["generations"]==0 and reused["http_attempts"]==0,"control_cache_reuse")
                row["cached_without_generation"]=True
            row["status"]="passed";boundary();save()
        verify(plan);report["status"]="complete";save();return 0
    except BaseException as error:
        # No retries. Keep the current task and all backend evidence.
        if rows and rows[-1]["status"]=="running":
            rows[-1].update(status="failed",error_type=type(error).__name__)
        report["status"]="failed";report["stop_reason"]=type(error).__name__;save()
        import traceback
        (output/"failure-private.txt").write_text(traceback.format_exc())
        return 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--inside",action="store_true")
    parser.add_argument("--run-approved",action="store_true")
    args=parser.parse_args()
    if not args.run_approved:
        plan=read(PROPOSAL);verify(plan)
        print(json.dumps(dict(binding=plan["binding"],approved=False,
                 cases=[x["case"] for x in plan["cases"]],provider_calls=0),indent=2))
        return 0
    require(AUTH.is_file(),"control_explicit_authorization_required")
    if args.inside:return inside()
    import shlex
    command='PYTHONPATH='+shlex.quote(str(ROOT/"scripts/baseline"))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
    command+=shlex.quote(str(VENV/"bin/python"))+" -B "+shlex.quote(str(Path(__file__).resolve()))+" --inside --run-approved"
    receipt=read(RECEIPT)
    remaining=max(1,int(receipt["original_launcher_deadline_unix"]-time.time()))
    return enter_nix(command,seconds=min(2430,remaining),environment="poseidon-cpu")

if __name__=="__main__":sys.exit(main())
