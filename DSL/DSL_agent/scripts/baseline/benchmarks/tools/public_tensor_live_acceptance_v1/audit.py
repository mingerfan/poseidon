"""Read-only acceptance audit for the completed human-approved five-case campaign."""
import hashlib,json,math,os,shlex,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE.parent/"public_tensor_live_v1"))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=60,keep_env=(),environment="poseidon-cpu"))
import numpy as np
import scope,run
from candidate_bundle import check_bundle
from application_transport_receipt import snapshot
require=scope.require;read=scope.read;sha=scope.sha;digest=scope.digest
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"audit_pinned_environment")
proposal=scope.verify();plan=read(scope.RUN/"plan.json");report=read(scope.RUN/"report.json")
run.no_active();require(not run.same_process(plan["owner"]),"audit_live_owner")
require(plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"})and
        plan["proposal_binding"]==proposal["binding"]and plan["deadline_wall"]==plan["created"]+43200,"audit_plan_identity")
scope.authority(read(plan["approval_record"]),proposal,sha(scope.PROPOSAL))
require(sha(plan["approval_record"])==plan["approval_sha256"],"audit_approval")
calls=scope.totals(run.states(scope.RUN))
require(report["status"]=="complete"and report["passed"]==5 and report["failed"]==report["not_run"]==0 and
        calls["calls_resolved"]and calls["generations"]==report["generations"]==6 and
        calls["http_attempts"]==report["http_attempts"]==6,"audit_terminal_ledger")
require([r["case"]for r in report["rows"]]==list(scope.CASES),"audit_case_denominator")
observations={str(scope.RUN/"report.json"):sha(scope.RUN/"report.json"),str(scope.RUN/"plan.json"):sha(scope.RUN/"plan.json")}
rows=[];groups=values=0;maximum=0.;initial_maximum=0.;total_abs=0.;relative=[];cosines=[]

def comparison(value,expected_groups):
    actual=np.asarray(value["actual"],dtype=np.float64);reference=np.asarray(value["reference"],dtype=np.float64)
    require(actual.shape==reference.shape and actual.ndim==2 and actual.shape[0]==expected_groups and
            actual.size>0 and np.isfinite(actual).all()and np.isfinite(reference).all(),"audit_numeric_shape")
    error=np.abs(actual-reference);passed=bool(np.all(error<=1e-5+1e-4*np.abs(reference)))
    require(value["passed"]is True and passed and value["atol"]==1e-5 and value["rtol"]==1e-4 and
            value["compared_values"]==actual.size,"audit_frozen_tolerance")
    require(float(error.max())==value["max_absolute_error"],"audit_saved_max_error")
    nonzero=reference!=0
    rel=float((error[nonzero]/np.abs(reference[nonzero])).max())if nonzero.any()else None
    denom=float(np.linalg.norm(actual)*np.linalg.norm(reference))
    cosine=float(np.dot(actual.reshape(-1),reference.reshape(-1))/denom)if denom>0 else None
    return dict(input_groups=expected_groups,compared_values=int(actual.size),mae=float(error.mean()),
                maximum_absolute_error=float(error.max()),maximum_nonzero_reference_relative_error=rel,
                cosine_similarity=cosine,numerically_validated=True),float(error.sum())

for frozen,row in zip(proposal["cases"],report["rows"]):
    require(row["status"]=="passed"and row["new_agent_generation"]is True and row["exit_code"]==0,"audit_case_success")
    folder=scope.RUN/row["case"];task=folder/"tasks"/row["task_id"]
    receipt=run.checked_receipt(folder,row["case"],plan,frozen)
    require(all(receipt.get(k)==v for k,v in row.items()if k!="exit_code"),"audit_case_receipt")
    state=read(task/"state.json");result=row["result"];generation=result["generation"]
    require(state["status"]==result["status"]=="succeeded"and result["failure"]is None and
            result["package"]["reused"]is False and generation["agent_generated"]is True,"audit_genuine_generation")
    package=Path(row["package"]);manifest=check_bundle(package)
    require(manifest["agent_generated"]and manifest["binding"]==row["package_binding"]and
            manifest["validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True),"audit_package")
    index=state["generations"]-1
    response=read(task/("response-"+str(index)+".json"));body=json.loads(response["raw"])
    require(set(body)=={"schema","request_id","hecate_source"}and body["request_id"]==row["request_id"]==frozen["request_id"]and
            body["hecate_source"]==(package/"candidate.py").read_text(),"audit_exact_agent_source")
    transport=read(row["transport_receipt"])
    require(transport==snapshot(transport)and sha(row["transport_receipt"])==row["transport_receipt_sha256"],"audit_transport_binding")
    validation=read(task/("validation-"+str(index)+".json"))
    evidence=Path(validation["evidence"]);replay=read(evidence/"report.json");attempt=read(evidence/"attempt-00/report.json")
    require(replay["status"]=="passed"and replay["backend"]=="poseidon_cpu"and
            replay["validation_level"]=="numerical"and replay["tolerance"]==dict(atol=1e-5,rtol=1e-4)and
            replay["artifact_replay_only"]is True and replay["agent_calls"]==0,"audit_validation_replay_boundary")
    require(attempt["status"]=="passed"and attempt["compiled"]and attempt["executed"]and
            attempt["numerically_correct"]and attempt["execution"]["encrypted_execution"]and
            attempt["execution"]["executor_secret_key"]is False and attempt["execution"]["backend"]=="poseidon_cpu","audit_poseidon_execution")
    require(all(replay["sandbox_probe"].get(k)is True for k in("frontend_loaded","private_sentinel_hidden","workspace_hidden",
            "keys_hidden","environment_cleared","only_loopback_interface","request_read_only","resource_limits_applied")),"audit_sandbox")
    for name,expected in replay["frozen_hashes"].items():
        require(sha(evidence/name)==expected,"audit_model_reference_binding")
    for name,expected in attempt["artifact_hashes"].items():
        require(sha(evidence/"attempt-00/output"/name)==expected,"audit_compiled_artifact_binding")
    for name in("artifacts/candidate_trace.mlir","artifacts/lowered.earth.mlir","artifacts/lowered.ckks.mlir",
                "artifacts/operator-spec.json","artifacts/bound/plan.json","artifacts/bound/bundle/manifest.json"):
        require(name in manifest["files"],"audit_complete_artifact_set")
    phases={};initial,total=comparison(attempt["comparison"],4);phases["initial"]=initial
    require(result["acceptance"]["maximum_absolute_error"]==initial["maximum_absolute_error"]and
            result["acceptance"]["compared_values"]==initial["compared_values"],"audit_initial_metrics")
    initial_maximum=max(initial_maximum,initial["maximum_absolute_error"])
    for key,expected_groups in (("new_inputs",3),("repeat",1)):
        phase=row[key];execution=phase["execution"]
        require(phase["status"]=="passed"and phase["package_binding"]==manifest["binding"]and phase["package_reloaded"]and
                phase["encrypted_execution"]and not phase["compiled"]and phase["provider_calls"]==0 and
                execution["agent_calls"]==execution["compiler_calls"]==0 and execution["executor_secret_key"]is False and
                execution["backend"]=="poseidon_cpu","audit_package_reexecution")
        phases[key],subtotal=comparison(phase["comparison"],expected_groups);total+=subtotal
    cache=row["cache"]
    require(cache["status"]=="passed"and cache["package_reused"]and cache["package_binding"]==manifest["binding"]and
            cache["provider_calls"]==0 and not cache["compiled"]and not cache["encrypted_execution"],"audit_cache_reuse")
    for metric in phases.values():
        groups+=metric["input_groups"];values+=metric["compared_values"];maximum=max(maximum,metric["maximum_absolute_error"])
        if metric["maximum_nonzero_reference_relative_error"]is not None:relative.append(metric["maximum_nonzero_reference_relative_error"])
        if metric["cosine_similarity"]is not None:cosines.append(metric["cosine_similarity"])
    total_abs+=total
    for file in(folder/"case-report.json",task/"state.json",task/("response-"+str(index)+".json"),
                 task/("validation-"+str(index)+".json"),package/"manifest.json",evidence/"report.json",
                 evidence/"attempt-00/report.json"):
        observations[str(file)]=sha(file)
    first=read(task/"validation-0.json")
    rows.append(dict(case=row["case"],status="passed",first_attempt_passed=generation["first_attempt_passed"],
        repairs_used=generation["repairs_used"],generations=state["generations"],http_attempts=state["http_attempts"],
        request_id=row["request_id"],package=str(package),package_binding=manifest["binding"],
        candidate_sha256=sha(package/"candidate.py"),initial_failure=first if row["case"]=="divided_gelu_linear"else None,
        numeric_phases=phases,package_reloaded=True,cache_reused_without_calls=True,
        backend_identity=replay["backend_identity"],parameter_identity=replay["compiler_configuration"]["parameter_identity"]))
require(groups==40 and values==304,"audit_sample_denominator")
added=[p for p in run.RESULTS.iterdir()if p.name not in plan["initial_result_directories"]]
new_bytes=sum(run.measure(p)for p in added);global_bytes=run.measure(run.RESULTS);free=shutil.disk_usage(run.WORK).free
require(new_bytes<=proposal["bounds"]["new_batch_bytes"]and global_bytes<=proposal["bounds"]["global_result_bytes"]and
        free>=proposal["bounds"]["minimum_free_bytes"],"audit_resources")
data=dict(format="poseidon-public-tensor-live-acceptance-v1",status="passed",source_root=str(ROOT),
    branch=proposal["branch"],head=proposal["head"],proposal_binding=proposal["binding"],plan_binding=plan["binding"],
    human_authorization_record=plan["approval_record"],human_authorization_sha256=plan["approval_sha256"],
    planned=5,passed=5,failed=0,skipped=0,not_run=0,first_attempt_passed=4,repaired_passed=1,
    generations=calls["generations"],http_attempts=calls["http_attempts"],calls_resolved=True,
    compiler_qualifications=5,encrypted_execution_calls=15,input_groups=groups,compared_values=values,
    maximum_absolute_error=maximum,initial_maximum_absolute_error=initial_maximum,mae=total_abs/values,
    maximum_nonzero_reference_relative_error=max(relative),minimum_defined_cosine_similarity=min(cosines),
    numeric_tolerance=dict(atol=1e-5,rtol=1e-4),rows=rows,evidence_hashes=observations,
    source_stage_identity=proposal["source_stage_identity"],source_stage_counts=proposal["source_stage_counts"],
    resource_observation=dict(new_bytes=new_bytes,global_result_bytes=global_bytes,free_bytes=free,
        elapsed_seconds=time.time()-plan["created"],elapsed_scope="launch to final read-only audit"),
    new_api_calls_from_audit=0,new_compiler_calls_from_audit=0,new_encrypted_calls_from_audit=0,
    backend_changes=False,parameter_changes=False,network_configuration_changes=False,downloads=0,installation=0,
    commits=0,pushes=0,production_default_changed=False,cuda_validated=False,bootstrap_validated=False,
    original_application_accuracy_certified=False,scope="five explicit-v22 public-tensor source contexts; finite independent numerical checks")
data["binding"]=digest(data)
out=ROOT/"docs/baseline/agent-public-tensor-live-acceptance-r1366.json"
with out.open("x")as stream:stream.write(json.dumps(data,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:data[k]for k in("status","planned","passed","generations","http_attempts","input_groups","compared_values","maximum_absolute_error","mae","binding")}))
