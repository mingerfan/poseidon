"""Read-only audit of completed r1382 controls; does not compile or execute."""
import sys,json,hashlib,time,os,subprocess
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from offline_cpu_continuation import verify
from benchmark_graph import digest,require,samples
from candidate_bundle import check_bundle
from audit_unified_candidate import verify_candidate
from operator_reference import evaluate
from numerical_results import compare
import numpy as np
plan,continuation=verify();output=Path(plan["output"])
raw=json.loads((output/"continuation-report.json").read_text())
require(raw["binding"]==digest({k:v for k,v in raw.items() if k!="binding"}),"final_report_binding")
require(raw["status"]=="complete" and (raw["passed"],raw["failed"],raw["not_run"])==(3,0,0),"final_completed")
require(raw["compiler_confirmed"]==raw["compiler_reserved"]==3 and
        raw["encrypted_confirmed"]==raw["encrypted_reserved"]==9 and
        raw["provider_calls"]==raw["http_attempts"]==0,"final_call_counts")
checkpoint=json.loads((ROOT/"docs/baseline/agent-application-admission-budget-checkpoint-r1381.json").read_text())
require(checkpoint["binding"]==digest({k:v for k,v in checkpoint.items() if k!="binding"}),"final_static_checkpoint")
require(len(set(checkpoint["unique_tests"]))==199 and checkpoint["passed"]==199 and
        checkpoint["failed"]==checkpoint["skipped"]==0,"final_static_test_count")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in checkpoint["source_files"].items():require(sha(ROOT/name)==h,"final_application_source")
for r in checkpoint["test_runs"]:require(sha(ROOT/r["file"])==r["sha256"],"final_static_log")
files={};rows=[]
def record(path):
    path=Path(path);files[str(path)]=sha(path)
for i,(row,case) in enumerate(zip(raw["rows"],plan["cases"])):
    require(row["case"]==case["case"] and row["status"]=="passed","final_case_order")
    folder=output/case["case"];task=folder/"tasks"/row["task_id"]
    meta=json.loads((task/"task.json").read_text());qualified=json.loads((folder/"qualification.json").read_text())
    require(meta["request"]==json.loads(Path(case["request"]).read_text()),"final_request")
    require(meta["model_provenance"]["original"]==json.loads(Path(case["historical_task_metadata"]).read_text())["model_provenance"]["original"],"final_reference_model")
    evidence=Path(row["validation_evidence"]);audit=verify_candidate(evidence,validation_level="numerical")
    report=json.loads((evidence/"report.json").read_text());attempt=report["attempts"][0]
    require(report["backend"]=="poseidon_cpu" and attempt["trace"]["frontend"]=="real_Hecate"
          and attempt["compiled"] and attempt["executed"] and attempt["numerically_correct"],"final_backend")
    require(report["tolerance"]==dict(atol=1e-5,rtol=1e-4),"final_tolerance")
    request=meta["request"];program=Path(row["package"]);manifest=check_bundle(program)
    require(manifest["binding"]==row["package_binding"] and manifest["origin"]["model_source"]["file"]=="artifacts/model-source.json","final_bundle")
    for n in manifest["files"]:record(program/n)
    record(program/"manifest.json");record(task/"task.json");record(task/"validation-0.json");record(evidence/"report.json")
    for n in attempt["artifact_hashes"]:record(evidence/"attempt-00/output"/n)
    probes=[{k:np.asarray(v,dtype=np.float64)*.5+.015 for k,v in b.items()} for b in samples(request["model"],4)[:3]]
    acceptance=qualified["acceptance"];groups=acceptance["input_groups"];values=acceptance["compared_values"]
    maximum=acceptance["maximum_absolute_error"];peak=[qualified["operational_usage"]["observed_peak_bytes"]]
    reuses={}
    for name,batches in (("new-inputs",probes),("repeat",probes[:1])):
        result=json.loads((folder/(name+".json")).read_text())
        require(result["status"]=="executed" and result["encrypted_execution"] and
                result["backend"]=="poseidon_cpu" and not result["executor_secret_key"] and
                result["agent_calls"]==result["compiler_calls"]==0,"final_reloaded_execution")
        actual=np.asarray([np.concatenate([np.asarray(v[o["name"]]).reshape(-1) for o in request["model"]["outputs"]]) for v in result["outputs"]])
        reference=np.asarray([np.concatenate([evaluate(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1) for o in request["model"]["outputs"]]) for b in batches])
        numeric=compare(actual,reference,1e-5,1e-4)
        require(numeric==json.loads((folder/(name+"-comparison.json")).read_text()) and numeric["passed"],"final_independent_numeric")
        if case["case"]=="compat_v23_named":
            require(all(list(v)==["primary","saved"] and v["saved"][1] is None for v in result["structured_outputs"]),"final_output_structure")
        reuses[name]=dict(groups=len(batches),values=numeric["compared_values"],maximum_absolute_error=numeric["max_absolute_error"])
        groups+=len(batches);values+=numeric["compared_values"];maximum=max(maximum,numeric["max_absolute_error"])
        peak.append(result["operational_usage"]["observed_peak_bytes"])
        record(folder/(name+".json"));record(folder/(name+"-comparison.json"))
    cached=json.loads((folder/"cached.json").read_text())
    require(cached["status"]=="succeeded" and cached["package"]["reused"] is True and
            cached["generations"]==cached["http_attempts"]==0,"final_cache")
    record(folder/"cached.json");record(folder/"qualification.json")
    rows.append(dict(case=case["case"],request_id=case["request_id"],source_contract=case["source_contract"],status="passed",
       validation_level="numerical",frontend="real_Hecate",compiled=True,encrypted_execution=True,
       qualification_groups=acceptance["input_groups"],qualification_values=acceptance["compared_values"],
       qualification_error=acceptance["maximum_absolute_error"],package_reloaded=True,cache_reused=True,
       total_group_times=groups,total_values=values,maximum_absolute_error=maximum,
       new_and_repeat=reuses,operational_peak_bytes=peak,package_binding=manifest["binding"],
       candidate_sha256=sha(case["candidate"]),historical_candidate_unmodified=True,
       new_agent_generated=False))
verify()
launch=json.loads((Path(__file__).parent/"offline-cpu-launch-r1382.json").read_text())
terminal=(output/"continuation-report.json").stat().st_mtime
require(terminal<=continuation["inherited_deadline_wall"],"final_original_deadline")
from offline_cpu import measure
global_bytes=measure(output.parent);reserve=__import__("shutil").disk_usage(output).free
new_directories=[output]+[Path(r["validation_evidence"]) for r in raw["rows"]]
for row in raw["rows"]:
 for name in ("new-inputs","repeat"):
  result=json.loads((output/row["case"]/(name+".json")).read_text())
  new_directories.append(Path(result["evidence"]))
owned=sum(measure(p) for p in new_directories)
require(owned<=plan["scope"]["new_bytes"] and global_bytes<=plan["scope"]["global_bytes"] and
        reserve>=plan["scope"]["reserve_bytes"],"final_resource_limits")
head=subprocess.check_output(["git","rev-parse","HEAD"],text=True,timeout=10).strip()
branch=subprocess.check_output(["git","branch","--show-current"],text=True,timeout=10).strip()
require(branch=="lhy-agent-dsl" and head=="7588e48093631eaf8385506e3db058e7ca3cfe7e","final_branch")
result=dict(format="poseidon-application-control-final-verification-v1",
    backend="poseidon_cpu",platform="aarch64-linux",checkout=str(ROOT),branch=branch,head=head,
    proposal_binding=plan["binding"],continuation_binding=continuation["binding"],raw_report_binding=raw["binding"],
    sdk_identity=plan["sdk_identity"],backend_unchanged_from_r1376=True,
    candidate_origin="frozen_deterministic_rule",agent_generated=False,passed=3,failed=0,skipped=0,not_run=0,
    compiler_calls=3,encrypted_calls=9,provider_calls=0,http_attempts=0,installations=0,downloads=0,
    network_changes=False,backend_library_changes=False,parameter_changes=False,git_commit=False,git_push=False,
    static_unique_passed=199,static_failed=0,static_skipped=0,
    rows=rows,total_group_times=sum(r["total_group_times"] for r in rows),
    total_values=sum(r["total_values"] for r in rows),maximum_absolute_error=max(r["maximum_absolute_error"] for r in rows),
    tolerance=dict(atol=1e-5,rtol=1e-4),application_accuracy_certified=False,
    resources=dict(batch_owned_bytes=owned,global_result_bytes=global_bytes,free_disk_bytes=reserve,
                   execution_elapsed_from_first_launcher_seconds=terminal-launch["started_wall"],
                   continued_worker_seconds=raw["seconds"],original_deadline_met=True),
    execution_record_numerical_fields_unchanged=True,
    verification_scope="initial four fixed groups and external independent comparisons for three new and one repeat inputs per model",
    preserved_errors=["launcher_import_failure_before_execution","collector_model_source_field_failure_after_first_numerical_pass"],
    evidence_files=files,static_checkpoint_binding=checkpoint["binding"],source_files=checkpoint["source_files"])
result["binding"]=digest(result)
target=Path(__file__).parent/"verified-evidence-r1383.json"
with target.open("x") as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("evidence_files","source_files","rows")},indent=2))
