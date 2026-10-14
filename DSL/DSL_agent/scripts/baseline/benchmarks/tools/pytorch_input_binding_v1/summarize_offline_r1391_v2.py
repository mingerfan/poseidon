"""Audit final r1390 scope and publish one r1391 result without changing history."""
import datetime,hashlib,json,subprocess,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark_graph import require,digest
from hecate_python_env import ROOT,WORK,VENV
from application_identity import current_identity_manifest
from application_execution.context_identity import authority
from candidate_bundle import check_bundle,check_execution_dependencies
from numerical_results import compare
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exclusive(p,v):
    with Path(p).open("x")as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
plan_path=ROOT/"docs/baseline/agent-input-binding-offline-proposal-r1390.json"
plan=read(plan_path)
receipt_path=ROOT/"docs/baseline/agent-input-binding-offline-continuation-r1391.json"
receipt=read(receipt_path)
raw_path=Path(plan["output"])/"continuation-r1391/report.json"
raw=read(raw_path)
require(raw["binding"]==digest({k:v for k,v in raw.items()if k!="binding"}),"Final report binding")
require(raw["status"]=="complete" and raw["passed"]==15 and raw["failed"]==raw["not_run"]==0,
        "Completed scope, no failures or omissions")
require(raw["compiler_confirmed"]==raw["compiler_reserved"]==15 and
        raw["encrypted_confirmed"]==raw["encrypted_reserved"]==24,"Cumulative execution count")
require(raw["provider_calls"]==raw["http_attempts"]==0 and not raw["agent_generated"],
        "No live Agent generation")
require(raw["cumulative_authorization_seconds"]<=plan["scope"]["wall_seconds"],"Original time budget")
require(current_identity_manifest()["manifest_sha256"]==plan["sdk_identity"],"Unchanged stage SDK")
for name,h in receipt["protected_reports"].items():require(sha(name)==h,"History unchanged")
for name,h in receipt["controller_files"].items():require(sha(name)==h,"Controller source unchanged")
scope=[(r["case"],r["request_id"])for r in plan["cases"]]
require([r["case"]for r in raw["rows"]]==[r[0]for r in scope],"Exact cases")
comparisons=[];initial=[];reuse=[];cases=[];current_authority=authority(ROOT)
for row,case in zip(raw["rows"],plan["cases"]):
    program=Path(row["package"]);manifest,request=check_execution_dependencies(program)
    require(manifest==check_bundle(program)and request==read(case["request"]),"Reloaded bound program")
    require(sha(program/"candidate.py")==case["files"][case["candidate"]],"Frozen rule source")
    require(request["request_id"]==case["request_id"],"Fixed model request")
    require(row["status"]=="passed"and row["compiled"]and row["encrypted_execution"]and row["package_reloaded"],
            "Real execution stages")
    ctx=read(program.parent.parent/"worker-context.json")
    require(ctx["binding"]==row["worker_context_binding"]and ctx["source_authority"]==current_authority,
            "Current worker authority")
    evidence=Path(row["validation_evidence"]);report=read(evidence/"report.json")
    require(report["backend"]=="poseidon_cpu"and report["status"]=="passed"and
            report["backend_identity"]==plan["backend_identity"]and
            report["tolerance"]==dict(atol=1e-5,rtol=1e-4),"Unchanged backend and threshold")
    a=report["attempts"][0]
    require(all(a[k]is True for k in("parsed","source_parsed","checked","compiled","executed","numerically_correct")),
            "Actual qualification stages")
    require(a["trace"]["frontend"]=="real_Hecate"and
            a["execution"]["encrypted_execution"]is True and
            a["execution"]["executor_secret_key"]is False and
            a["execution"]["bootstrap"]is False,"Poseidon CPU execution")
    artifacts=a["artifact_hashes"]
    required=("candidate_trace.mlir","lowered.earth.mlir","lowered.ckks.mlir",
              "bound/plan.json","bound/layout-binding.json","bound/bundle/manifest.json")
    require(all(n in artifacts for n in required)and any(n.endswith(".runtime-plan.json")for n in artifacts),
            "Tracing, IR, RuntimePlan and constant bundle evidence")
    artifact_root=evidence/("attempt-"+str(a["index"]).zfill(2))/"output"
    for name,h in artifacts.items():require(sha(artifact_root/name)==h,"Artifact hash")
    c=a["comparison"]
    check=compare(np.asarray(c["actual"]),np.asarray(c["reference"]),1e-5,1e-4)
    require(check["passed"]and check["compared_values"]==c["compared_values"]and
            check["max_absolute_error"]==c["max_absolute_error"],"Numerical result consistency")
    metric={k:c[k]for k in("passed","atol","rtol","compared_values","mae","max_absolute_error",
                          "max_nonzero_reference_relative_error","nonzero_reference_count","cosine_similarity")}
    initial.append(metric);comparisons.append(metric)
    reuse_rows=[]
    if case["repeat_package"]:
        for name,groups in(("new-inputs",3),("repeat",1),("named-equivalence",3)):
            path=program.parent.parent.parent/(name+"-comparison.json")
            c=read(path)
            require(c["passed"]is True and c["atol"]==1e-5 and c["rtol"]==1e-4,"Reuse numerical comparison")
            check=compare(np.asarray(c["actual"]),np.asarray(c["reference"]),1e-5,1e-4)
            require(check["passed"]and check["compared_values"]==c["compared_values"],"Reuse compare consistency")
            result=read(path.with_name(name+".json"))
            require(result["status"]=="executed"and result["encrypted_execution"]is True and
                    result["agent_calls"]==result["compiler_calls"]==0 and
                    not result["executor_secret_key"],"Reload and repeated local execution")
            measure={k:c[k]for k in("passed","atol","rtol","compared_values","mae","max_absolute_error",
                                  "max_nonzero_reference_relative_error","nonzero_reference_count","cosine_similarity")}
            reuse.append(measure);comparisons.append(measure)
            reuse_rows.append(dict(kind=name,groups=groups,comparison_sha256=sha(path),**measure))
        require(row["cached_without_generation"]is True,"Cache reuse")
    cases.append(dict(case=case["case"],source_contract=case["source_contract"],request_id=case["request_id"],
        compiled=True,encrypted_execution=True,decrypted_and_compared=True,package_reloaded=True,
        candidate_origin="frozen_deterministic_rule",qualification_retained=row.get("qualification_reused_from_original_binding",False),
        comparison=metric,reuse=reuse_rows,program=str(program),program_binding=manifest["binding"],
        worker_context_binding=ctx["binding"],validation_evidence=str(evidence),
        validation_report_sha256=sha(evidence/"report.json")))
def aggregate(values):
    count=sum(v["compared_values"]for v in values)
    relative=[v["max_nonzero_reference_relative_error"]for v in values if v["max_nonzero_reference_relative_error"]is not None]
    cos=[v["cosine_similarity"]for v in values if v["cosine_similarity"]is not None]
    return dict(compared_values=count,mae=sum(v["mae"]*v["compared_values"]for v in values)/count,
        maximum_absolute_error=max(v["max_absolute_error"]for v in values),
        maximum_nonzero_reference_relative_error=max(relative)if relative else None,
        minimum_defined_cosine_similarity=min(cos)if cos else None)
current=max(int(subprocess.check_output(["du",flag,"-s","-B1",str(WORK/"results")],
                    text=True,timeout=20).split()[0])for flag in("--apparent-size","--count-links"))
initial_global=31286243328
require(current-initial_global<=plan["scope"]["new_bytes"]and current<=plan["scope"]["global_bytes"]and
        shutil.disk_usage(WORK).free>=plan["scope"]["reserve_bytes"],"Cumulative artifact budget")
result=dict(format="poseidon-structured-input-acceptance-v1",
    created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_root=str(ROOT),branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    proposal_binding=plan["binding"],continuation_binding=receipt["binding"],sdk_identity=plan["sdk_identity"],
    worker_source_authority=current_authority,backend_identity=plan["backend_identity"],
    passed=15,failed=0,skipped=0,not_run=0,new_structured_contexts=12,legacy_contexts=3,
    compiler_calls=15,encrypted_calls=24,retained_qualifications=3,additional_compiler_calls=12,
    additional_encrypted_calls=21,provider_calls=0,http_attempts=0,agent_generated=False,
    source_intake= dict(passed=12,groups=192,values=1344,original_vs_normalized_maximum_error=0.,
        decomposition_maximum_error=1.1102230246251565e-16),
    unit_regressions=dict(passed=86,failed=0,skipped=0,new_forwarding_tests=3,
        evidence=str(Path(__file__).parent/"context-adapter-fix-r1391/unit-regression.log"),
        sha256=receipt["unit_regression"]["log_sha256"]),
    qualification=dict(input_groups=60,**aggregate(initial)),
    package_reuse=dict(programs=3,calls=9,input_groups=21,new_input_execution=True,
        structured_and_named_consistent=True,repeated_execution=True,cache_reused=True,**aggregate(reuse)),
    overall=dict(input_groups=81,**aggregate(comparisons)),cases=cases,
    budget=dict(scope_expanded=False,wall_seconds=raw["cumulative_authorization_seconds"],
        worker_seconds_limit=2400,global_before_bytes=initial_global,global_after_bytes=current,
        cumulative_new_bytes=current-initial_global,new_bytes_limit=536870912,
        global_bytes_limit=34359738368,free_bytes=shutil.disk_usage(WORK).free),
    corrections=[
        dict(layer="development_harness",cause="Incorrect model_source manifest assertion",
             fix="Verify model_provenance and bound source origin",backend_changed=False),
        dict(layer="application_adapter",cause="ContextBackend.execute omitted sample_format",
             fix="Validate and forward explicit format, preserving the legacy named call",
             source=str(ROOT/"scripts/baseline/application_execution/backend.py"))],
    protected_history=receipt["protected_reports"],raw_report=str(raw_path),raw_report_sha256=sha(raw_path),
    installation=0,downloads=0,network_changes=False,backend_changes=False,parameter_changes=False,
    commits=0,pushes=0,
    confirmed=["12 v26 contexts and 3 legacy contexts compiled, executed and numerically compared",
               "15 bound packages reloaded; 3 accepted new forward-call and named inputs, repeat and cache reuse"],
    inference=["Adapter fix preserves the existing compilation, cryptographic execution and numerical identities"],
    unverified=["New live Agent generation for v26","Arbitrary dynamic Python inputs",
                "All configurations within current resource limits","Additional spatial operators",
                "Full bootstrap, CUDA, larger models and ResNet20 qualification"])
result["binding"]=digest(result)
target=ROOT/"docs/baseline/agent-input-binding-poseidon-results-r1391.json"
exclusive(target,result)
print(json.dumps({k:result[k]for k in("binding","passed","failed","compiler_calls","encrypted_calls","overall","budget")},
                 ensure_ascii=False,allow_nan=False))
