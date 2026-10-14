"""Read-only audit of genuine generation for 3 broadcast-source requests."""
import argparse,copy,hashlib,json,math,os,shlex,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
CAMPAIGNS={"normalized":dict(tool="static_broadcast_live_v1",output="agent-static-broadcast-live-acceptance-r1429.json",
 format="poseidon-static-broadcast-live-acceptance-v1",
 scope="3 genuine Agent requests selected from 6 versioned broadcast source contexts; 3 aliases require unchanged-source replay")}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--campaign",choices=tuple(CAMPAIGNS),default="normalized")
parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
args=parser.parse_args();campaign=CAMPAIGNS[args.campaign]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE.parent/"spatial_source_live_acceptance_v1"));sys.path.insert(0,str(HERE.parent/campaign["tool"]))
from hecate_python_env import VENV,enter_nix
import scope,run
from acceptance_outcomes import terminal_counts,receipt_coverage
import acceptance_outcomes
if not args.inside:
    run.no_active()
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--campaign",args.campaign,"--inside"])
    raise SystemExit(enter_nix(command,seconds=300,keep_env=(),environment="poseidon-cpu"))
import numpy as np
import scope,run
from candidate_bundle import check_bundle
from application_transport_receipt import snapshot
from audit_unified_candidate import verify_candidate
from benchmark_math import evaluate as mathematical_reference
from benchmark_torch import evaluate as independent_torch_reference
from unified_request_storage import compact_request,restore_public_constants
from program_execution import package_execution_metadata
from pytorch_source_contract import restore_outputs
require=scope.require;read=scope.read;sha=scope.sha;digest=scope.digest
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"audit_pinned_environment")
proposal=scope.verify(read(scope.PROPOSAL));plan=read(scope.RUN/"plan.json");report=read(scope.RUN/"report.json")
run.no_active();require(not run.same_process(plan["owner"]),"audit_live_owner")
require(plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"})and
        plan["proposal_binding"]==proposal["binding"]and plan["deadline_wall"]==plan["created"]+proposal["bounds"]["batch_wall_seconds"],"audit_plan_identity")
scope.authority(read(plan["approval_record"]),proposal,sha(scope.PROPOSAL))
require(sha(plan["approval_record"])==plan["approval_sha256"],"audit_approval")
calls=scope.totals(run.states(scope.RUN))
counts=terminal_counts(report,calls,scope.CASES)
require(report["binding"]==plan["binding"]and report["proposal_binding"]==proposal["binding"],"audit_report_identity")
observations={str(scope.RUN/"report.json"):sha(scope.RUN/"report.json"),str(scope.RUN/"plan.json"):sha(scope.RUN/"plan.json"),
 str(scope.PROPOSAL):sha(scope.PROPOSAL),str(scope.MANIFEST):sha(scope.MANIFEST),
 str(scope.AUTHORIZATION):sha(scope.AUTHORIZATION),str(Path(__file__)):sha(Path(__file__)),
 str(Path(acceptance_outcomes.__file__)):sha(Path(acceptance_outcomes.__file__))}
audited_files=0
controls_path=HERE.parent/campaign["tool"]/"runner-controls-acceptance-v2.json"
controls=read(controls_path)
require(controls["binding"]==digest({k:v for k,v in controls.items()if k!="binding"})and
 controls["proposal_binding"]==proposal["binding"]and controls["status"]=="passed"and
 controls["failed"]==controls["errors"]==controls["skipped"]==0 and
 controls["tool_hashes"]==scope.tools(),"audit_real_static_controls")
require(sha(controls["log"])==controls["log_sha256"],"audit_static_controls_log")
observations[str(controls_path)]=sha(controls_path);observations[controls["log"]]=controls["log_sha256"]
usage_totals={k:0 for k in ("completion_tokens","prompt_cache_hit_tokens","prompt_cache_miss_tokens","prompt_tokens","reasoning_tokens","total_tokens","responses_with_usage","responses_without_usage","unresolved_http_attempts")}
rows=[];groups=values=0;maximum=0.;initial_maximum=0.;total_abs=0.;relative=[];cosines=[]
first_passed=repaired_passed=0
usage_coverage=dict(terminal_http_outcomes=0,received_responses=0,terminal_failed_http_attempts=0,
 received_responses_without_usage=0,received_responses_with_partial_usage=0,unmeasured_http_attempts=0,usage_complete=True)
missing_usage_fields=[];returned_candidates=0
all_candidate_attempts=[];compiled_attempts=encrypted_candidate_calls=0

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

reported={row["case"]:row for row in report["rows"]}
for frozen in proposal["cases"]:
    if frozen["id"]not in reported:
        rows.append(dict(case=frozen["id"],status="not_run",request_id=frozen["request_id"],
            compilations=0,encrypted_calls=0,generations=0,http_attempts=0))
        continue
    row=reported[frozen["id"]]
    require(row["status"]in("passed","failed")and row["new_agent_generation"]is True,"audit_case_outcome")
    folder=scope.RUN/row["case"];task=folder/"tasks"/row["task_id"]
    receipt=run.checked_receipt(folder,row["case"],plan,frozen)
    require(all(receipt.get(k)==v for k,v in row.items()if k!="exit_code"),"audit_case_receipt")
    state=read(task/"state.json");result=row["result"]
    require(state["status"]==result["status"]and state["generations"]==result["generations"]and
            state["http_attempts"]==result["http_attempts"],"audit_task_result_accounting")
    frozen_request=read(frozen["request_file"])
    source_meta=read(Path(frozen["qualification"]["source_task_directory"])/"task.json")
    actual_meta=read(task/"task.json")
    require(actual_meta["binding"]==digest({k:v for k,v in actual_meta.items()if k!="binding"}),"audit_task_metadata_binding")
    require(actual_meta["request"]==source_meta["request"]==frozen_request and
            actual_meta["identity"]==source_meta["identity"]==proposal["source_stage_identity"] and
            actual_meta["model_provenance"]==source_meta["model_provenance"],"audit_original_source_and_reference")
    transport=read(row["transport_receipt"])
    require(row["transport_receipt"]==str(task/"provider-transport.json")and transport==snapshot(transport)and
            sha(row["transport_receipt"])==row["transport_receipt_sha256"],"audit_transport_binding")
    from generation_receipt import safe_metrics,usage_summary
    raw_metrics=read(task/"provider-metrics.json");metrics=safe_metrics(raw_metrics)
    require(metrics["provider"]=="deepseek_api"and metrics["model"]==proposal["provider"]["model"]and
            metrics["reasoning_effort"]==proposal["provider"]["reasoning_effort"]and
            metrics["max_tokens"]==proposal["provider"]["max_tokens"]and
            metrics["generation_attempts"]==state["generations"]and
            metrics["request_attempts"]==state["http_attempts"],"audit_provider_configuration_and_calls")
    require(len(metrics["calls"])==len(transport["calls"])==state["http_attempts"],"audit_transport_case_calls")
    for a,b in zip(metrics["calls"],transport["calls"]):
        for key in("index","generation_index","retry_index","status","error","retry_scheduled"):
            if key in a:require(b.get(key)==a[key],"audit_transport_outcome_mismatch")
    coverage=receipt_coverage(metrics)
    for name,value in coverage.items():
        if name=="usage_complete":usage_coverage[name]=usage_coverage[name]and value
        else:usage_coverage[name]+=value
    summary=usage_summary(metrics)
    for name in usage_totals:
        value=summary.get(name)
        if value is None:
            missing_usage_fields.append(dict(case=row["case"],field=name))
            continue
        require(type(value)is int and value>=0,"audit_usage_count")
        usage_totals[name]+=value
    for file in(folder/"case-report.json",task/"state.json",task/"task.json",task/"provider-metrics.json",Path(row["transport_receipt"])):
        observations[str(file)]=sha(file)
    for attempt_index in range(state["generations"]):
        raw=task/("response-"+str(attempt_index)+".json")
        if raw.exists():
            observations[str(raw)]=sha(raw);returned_candidates+=1
            recorded=read(raw)
            original=task/("generation-"+str(attempt_index)+".json")
            require(original.exists()and read(original)["raw"]==recorded["raw"],"audit_original_returned_response")
            observations[str(original)]=sha(original)
        check=task/("validation-"+str(attempt_index)+".json")
        if check.exists():
            observations[str(check)]=sha(check);checked=read(check)
            evidence_path=Path(checked.get("evidence","/nonexistent"))
            for trial in sorted(evidence_path.glob("attempt-*/report.json")):
                saved=read(trial)
                compiled_attempts+=int(saved.get("compiled") is True)
                encrypted_candidate_calls+=int(saved.get("executed") is True)
                record=dict(case=row["case"],generation_index=attempt_index,evidence=str(trial.parent),
                            report_sha256=sha(trial),status=saved["status"],
                            compiled=saved.get("compiled") is True,encrypted_execution=saved.get("executed") is True,
                            failure_layer=saved.get("failure_layer"))
                for filename,want in saved.get("artifact_hashes",{}).items():
                    observed=trial.parent/"output"/filename
                    require(sha(observed)==want,"audit_all_attempt_artifacts")
                    observations[str(observed)]=want
                if saved.get("comparison"):
                    c=saved["comparison"];a=np.asarray(c["actual"]);b=np.asarray(c["reference"])
                    require(a.shape==b.shape and a.size>0 and np.isfinite(a).all()and np.isfinite(b).all(),"audit_attempt_numeric_shape")
                    delta=np.abs(a-b)
                    require(bool(np.all(delta<=1e-5+1e-4*np.abs(b)))==c["passed"] and
                            c["atol"]==1e-5 and c["rtol"]==1e-4,"audit_failed_attempt_tolerance")
                    record.update(maximum_absolute_error=float(delta.max()),compared_values=int(a.size),
                                  numerically_validated=c["passed"])
                observations[str(trial)]=sha(trial);all_candidate_attempts.append(record)
    if row["status"]=="failed":
        failure=result.get("failure")
        require(failure is not None or row.get("failure_layer")is not None,"audit_failure_diagnostic")
        loop=read(task/"loop.json")
        observations[str(task/"loop.json")]=sha(task/"loop.json")
        rows.append(dict(case=row["case"],status="failed",request_id=frozen["request_id"],
            task_status=state["status"],failure_layer=row.get("failure_layer",(failure or{}).get("layer")),
            failure=failure,generations=state["generations"],http_attempts=state["http_attempts"],
            returned_candidate_responses=sum((task/("response-"+str(i)+".json")).exists()for i in range(state["generations"])),
            generation_loop_status=loop["status"],terminal_transport_calls=transport["calls"],
            source_and_reference_bound=True,package_qualified=False,
            host_collected_evidence_not_provider_signed=True))
        continue
    generation=result["generation"]
    require(row["exit_code"]==0 and state["status"]=="succeeded"and result["failure"]is None and
            result["package"]["reused"]is False and generation["agent_generated"]is True,"audit_genuine_generation")
    package=Path(row["package"]);manifest=check_bundle(package)
    frozen_request=read(frozen["request_file"])
    stored=read(package/"request.json")
    require(stored==compact_request(frozen_request),"audit_exact_frozen_request_storage")
    restored=copy.deepcopy(stored);restore_public_constants(restored)
    require(restored==frozen_request,"audit_lossless_weight_and_request_binding")
    require(manifest["agent_generated"]and manifest["binding"]==row["package_binding"]and
            manifest["validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True),"audit_package")
    source_meta=read(Path(frozen["qualification"]["source_task_directory"])/"task.json")
    require(read(package/"model-provenance.json")==source_meta["model_provenance"],
            "audit_original_source_provenance")
    require(source_meta["request"]==frozen_request and source_meta["identity"]==proposal["source_stage_identity"],
            "audit_source_reference_identity")
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
    qualification=verify_candidate(evidence)
    require(qualification["request_id"]==row["request_id"] and
            qualification["comparison"]==attempt["comparison"],"audit_independent_qualification")
    for relative_path,expected_sha in qualification["files"].items():
        observations[str(evidence/relative_path)]=expected_sha
    audited_files+=len(qualification["files"])
    for name in("artifacts/candidate_trace.mlir","artifacts/lowered.earth.mlir","artifacts/lowered.ckks.mlir",
                "artifacts/operator-spec.json","artifacts/bound/plan.json","artifacts/bound/bundle/manifest.json"):
        require(name in manifest["files"],"audit_complete_artifact_set")
    phases={};initial,total=comparison(attempt["comparison"],4);phases["qualification"]=initial
    require(result["acceptance"]["maximum_absolute_error"]==initial["maximum_absolute_error"]and
            result["acceptance"]["compared_values"]==initial["compared_values"],"audit_initial_metrics")
    initial_maximum=max(initial_maximum,initial["maximum_absolute_error"])
    for key,expected_groups in (("new_inputs",3),("repeat",1)):
        phase=row[key];execution=phase["execution"]
        require(phase["status"]=="passed"and phase["package_binding"]==manifest["binding"]and phase["package_reloaded"]and
                phase["encrypted_execution"]and not phase["compiled"]and phase["provider_calls"]==0 and
                execution["agent_calls"]==execution["compiler_calls"]==0 and execution["executor_secret_key"]is False and
                execution["backend"]=="poseidon_cpu","audit_package_reexecution")
        require(phase["sample_format"]=="named",
                "audit_structured_forward_execution")
        frozen_model=frozen_request["model"]
        tree=package_execution_metadata(package,manifest,frozen_model)["output_structure"]
        require(execution.get("output_structure")==tree,"audit_output_structure")
        if tree is not None:
            # The frozen live checker validated Python tuple/list types before JSON storage.
            # Canonical JSON stores tuples as arrays; the separately bound tree retains their kinds.
            require(scope.canonical(execution["structured_outputs"])==
                    scope.canonical([restore_outputs({o["name"]:item[o["name"]] for o in frozen_model["outputs"]},tree,allow_none=True) for item in execution["outputs"]]),
                    "audit_reconstructed_output_tree_storage")
        expected_rows=[]
        for sample_index in range(expected_groups):
            rng=np.random.default_rng(20261008+sample_index)
            inputs={s["name"]:rng.uniform(-.2,.2,s["shape"]).astype(np.float64) for s in frozen_model["inputs"]}
            math_outputs=mathematical_reference(frozen_model,inputs)
            torch_outputs=independent_torch_reference(frozen_model,inputs)
            require(set(math_outputs)==set(torch_outputs)=={o["name"] for o in frozen_model["outputs"]},"audit_new_input_names")
            for name in math_outputs:np.testing.assert_allclose(math_outputs[name],torch_outputs[name],atol=2e-12,rtol=2e-12)
            expected_rows.append(np.concatenate([math_outputs[o["name"]].reshape(-1) for o in frozen_model["outputs"]]))
        require(np.array_equal(np.asarray(phase["comparison"]["reference"],dtype=np.float64),np.asarray(expected_rows)),
                "audit_new_independent_reference")
        named=execution["outputs"]
        decoded=np.asarray([np.concatenate([np.asarray(sample[o["name"]],dtype=np.float64).reshape(-1)
                        for o in frozen_model["outputs"]]) for sample in named],dtype=np.float64)
        require(np.array_equal(decoded,np.asarray(phase["comparison"]["actual"],dtype=np.float64)),"audit_new_output_decoding")
        raw=Path(execution["evidence"])/"output"
        require(read(raw/"execution-metadata.json")==execution["execution"] and
                read(raw/"resource-precheck.json")==execution["resource_precheck"],"audit_reexecution_evidence_binding")
        require(phase.get("resource_precheck_verified") is True and phase["resource_precheck_verified"],
                "audit_reexecution_resources")
        for file in raw.rglob("*"):
            if file.is_file():
                require(not file.is_symlink(),"audit_reexecution_regular_files")
                observations[str(file)]=sha(file);audited_files+=1
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
    require(type(generation["first_attempt_passed"]) is bool and
            generation["repairs_used"]==state["generations"]-1 and
            generation["first_attempt_passed"]==(first["status"]=="passed"),"audit_generation_success_accounting")
    first_passed+=int(generation["first_attempt_passed"])
    repaired_passed+=int(not generation["first_attempt_passed"])
    rows.append(dict(case=row["case"],status="passed",first_attempt_passed=generation["first_attempt_passed"],
        repairs_used=generation["repairs_used"],generations=state["generations"],http_attempts=state["http_attempts"],
        request_id=row["request_id"],package=str(package),package_binding=manifest["binding"],
        candidate_sha256=sha(package/"candidate.py"),initial_failure=None if generation["first_attempt_passed"] else first,
        numeric_phases=phases,package_reloaded=True,cache_reused_without_calls=True,
        backend_identity=replay["backend_identity"],parameter_identity=replay["compiler_configuration"]["parameter_identity"]))
require(groups==counts["passed"]*8 and values==sum(r["result"]["acceptance"]["compared_values"]*2 for r in report["rows"]if r["status"]=="passed"),
        "audit_sample_denominator")
require(usage_coverage["terminal_http_outcomes"]==calls["http_attempts"],"audit_terminal_http_denominator")
require(returned_candidates<=calls["generations"],"audit_returned_response_denominator")
added=[p for p in run.RESULTS.iterdir()if p.name not in plan["initial_result_directories"]]
new_bytes=sum(run.measure(p)for p in added);global_bytes=run.measure(run.RESULTS);free=shutil.disk_usage(run.WORK).free
require(new_bytes<=proposal["bounds"]["new_batch_bytes"]and global_bytes<=proposal["bounds"]["global_result_bytes"]and
        free>=proposal["bounds"]["minimum_free_bytes"],"audit_resources")
data=dict(format=campaign["format"],status="passed",audit_status="passed",benchmark_status=counts["benchmark_status"],source_root=str(ROOT),
    branch=proposal["branch"],head=proposal["head"],proposal_binding=proposal["binding"],plan_binding=plan["binding"],
    human_authorization_record=plan["approval_record"],human_authorization_sha256=plan["approval_sha256"],
    planned=len(scope.CASES),passed=counts["passed"],failed=counts["failed"],skipped=0,not_run=counts["not_run"],first_attempt_passed=first_passed,repaired_passed=repaired_passed,
    generations=calls["generations"],http_attempts=calls["http_attempts"],calls_resolved=True,
    successful_compiler_qualifications=counts["passed"],successful_encrypted_execution_calls=3*counts["passed"],
    candidate_attempts=calls["generations"],compiled_candidate_attempts=compiled_attempts,
    encrypted_execution_calls=encrypted_candidate_calls+2*counts["passed"],returned_candidate_responses=returned_candidates,
    failed_candidate_attempts=sum(r["status"]!="passed" for r in all_candidate_attempts),
    candidate_attempt_evidence=all_candidate_attempts,input_groups=groups,compared_values=values,
    maximum_absolute_error=maximum if values else None,qualification_maximum_absolute_error=initial_maximum if values else None,mae=total_abs/values if values else None,
    maximum_nonzero_reference_relative_error=max(relative)if relative else None,minimum_defined_cosine_similarity=min(cosines)if cosines else None,
    numeric_tolerance=dict(atol=1e-5,rtol=1e-4),rows=rows,evidence_hashes=observations,
    audited_files=len(observations),usage=usage_totals,usage_coverage=usage_coverage,missing_usage_fields=missing_usage_fields,
    usage_scope="Observed received-response token counts only. Legacy unresolved_http_attempts includes terminal failures; see usage_coverage. It is not an in-flight or billing counter.",
    numeric_summary_scope="Only fully qualified cases; failed candidate numerical results remain in candidate_attempt_evidence.",
    cost_usd=None,monetary_cap=None,
    rule_candidate_substitution=False,manual_candidate_substitution=False,
    runner_static_controls={k:controls[k]for k in ("passed","failed","errors","skipped")},
    source_formats=sorted({c["source_format"]for c in proposal["cases"]}),guidance="explicit-v13",period=4,
    source_contexts=len(proposal["source_contexts"]),independent_reference_qualified_contexts=6,
    genuine_generation_requests=3,additional_original_source_replays=3,additional_source_replays_status="not_run",
    source_stage_identity=proposal["source_stage_identity"],source_stage_counts=proposal["source_stage_counts"],
    resource_observation=dict(new_bytes=new_bytes,global_result_bytes=global_bytes,free_bytes=free,
        elapsed_seconds=time.time()-plan["created"],elapsed_scope="launch to final read-only audit"),
    new_api_calls_from_audit=0,new_compiler_calls_from_audit=0,new_encrypted_calls_from_audit=0,
    backend_changes=False,parameter_changes=False,network_configuration_changes=False,downloads=0,installation=0,
    commits=0,pushes=0,production_default_changed=False,cuda_validated=False,bootstrap_validated=False,
    original_application_accuracy_certified=False,scope=campaign["scope"])
data["binding"]=digest(data)
out=ROOT/"docs/baseline"/campaign["output"]
with out.open("x")as stream:stream.write(json.dumps(data,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:data[k]for k in("audit_status","benchmark_status","planned","passed","failed","not_run","generations","http_attempts","input_groups","compared_values","maximum_absolute_error","mae","binding")}))
