"""Consolidated phase closure; historical failures stay under their own bindings."""
import sys,shlex,json,subprocess,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=q.read(HERE/"identity-snapshot.json");assert PoseidonBackend().identity_manifest()==identity
old_path=ROOT/"docs/baseline/agent-public-mask-acceptance-r1472.json"
old=q.read(old_path);q.bound(old);assert old["status"]=="partial"and old["qualified_cases"]==dict(offline=21,live=16)
old_runtime=ROOT/"scripts/baseline/benchmarks/tools/public_mask_intake_v1"
assert identity["backend"]==q.read(old_runtime/"identity-snapshot.json")["backend"]
controls=q.read(HERE/"controls.json");guards=q.read(HERE/"controls-scope.json");probe=q.read(HERE/"sandbox-probe.json")
assert controls["status"]==guards["status"]==probe["status"]=="passed"
assert controls["tests"]==162 and controls["failures"]==controls["errors"]==controls["skipped"]==0
assert guards["tests"]==12 and probe["cases"]==6
cap_plan=q.read(HERE/"capture-plan.json");q.bound(cap_plan)
capture=q.read(Path(cap_plan["campaign"])/"report.json");q.bound(capture)
assert capture["passed"]==6 and capture["failed"]==capture["not_run"]==capture["skipped"]==0
reports={};audits={};numerics={};plans={}
for mode in("offline","live"):
    plan=q.verify(mode);out=Path(plan["campaign"])
    report=q.read(out/"report.json");audit=q.read(out/"audit.json");num=q.read(out/"numerical-audit.json")
    for value in(report,audit,num):q.bound(value)
    assert report["status"]=="complete"and report["passed"]==len(plan["cases"])
    assert report["failed"]==report["skipped"]==report["not_run"]==0 and report["calls_resolved"]
    assert q.read(out/"terminal.json")["exit_code"]==0
    assert audit["status"]==num["status"]=="passed"and audit["report_binding"]==num["report_binding"]==report["binding"]
    assert num["qualification_audit_binding"]==audit["binding"]
    reports[mode]=report;audits[mode]=audit;numerics[mode]=num;plans[mode]=plan
old_report=q.read(Path(old_runtime/"live-plan.json"))
old_report=q.read(Path(old_report["campaign"])/"report.json")
qualified_old={c["case"]for c in old_report["cases"]if c["status"]=="passed"}
qualified_new={c["case"]for c in reports["live"]["cases"]}
assert qualified_old|qualified_new=={c["case"]for c in old_report["cases"]} and len(qualified_old|qualified_new)==18
usage=[c["result"]["generation"]["usage"]for c in reports["live"]["cases"]]
keys=("prompt_tokens","completion_tokens","total_tokens","reasoning_tokens","prompt_cache_hit_tokens",
      "prompt_cache_miss_tokens","responses_with_usage","responses_without_usage","unresolved_http_attempts")
usage_total={k:sum(u.get(k,0)or 0 for u in usage)for k in keys};usage_total["cost_usd"]=None
branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert branch=="lhy-agent-dsl"and head=="7588e48093631eaf8385506e3db058e7ca3cfe7e"
record=dict(format="poseidon-public-mask-output-binding-acceptance-v1",status="complete",
 scope="bounded v33 input and final-output binding work; broader plan remains in progress",
 checkout=str(ROOT),platform="aarch64-linux",branch=branch,head=head,sdk=identity["manifest_sha256"],
 source_format="poseidon-trusted-pytorch-v33",generation_guidance="explicit-v14",
 controls=controls,execution_guards=guards,real_sandbox_probe=probe,
 source_capture=dict(binding=capture["binding"],contexts=6,input_groups=sum(c["groups"]for c in capture["cases"]),
 output_values=sum(c["values"]for c in capture["cases"]),compiler_calls=0,encrypted_calls=0,http_attempts=0),
 offline=audits["offline"],live=audits["live"],independent_numerical_audits=numerics,
 first_attempt_passed=sum(c["result"]["generation"]["first_attempt_passed"]is True for c in reports["live"]["cases"]),
 repairs=sum(c["result"]["generation"]["repairs_used"]for c in reports["live"]["cases"]),
 provider_usage=usage_total,provider_usage_is_billing_proof=False,
 all_original_contexts_qualified=True,original_contexts=18,
 original_contexts_coverage_binding_note="16 original passes use r1471 SDK and explicit-v13; three targeted current passes use this SDK and explicit-v14, with one overlapping context",
 original_failed_cases_resolved=["public_where_2","public_fill_2"],
 original_closed_report=dict(path=str(old_path),binding=old["binding"],status="partial",passed=16,failed=2,
  generations=27,http_attempts=27,usage_missing_responses=1,old_records_rewritten=False),
 original_rule_contexts=21,original_rule_binding=old["offline"]["binding"],
 original_trace_and_static_rejections=old["retained_failed_validations"],
 original_sandbox_failure=q.read(ROOT/"docs/baseline/agent-output-binding-sandbox-fix-r1475.json"),
 current_final_output_check=dict(format="poseidon-artifact-gate-v2",policy="distinct-value-ids-v1",
  code="compiled_duplicate_final_output",public_location="output index",
  repeated_positions_are_not_expression_identity=True),
 current_modules=["poseidon_artifact_gate.py","application_diagnostics.py","validation_adapter.py",
 "unified_graph_contract.py","unified_output_binding_guidance.py","poseidon_sandbox_policy.py","candidate_bundle.py","run_candidate.py"],
 tests=["test_unified_output_binding.py"],
 backend_core_changed=False,parameters_changed=False,numerical_tolerance=dict(atol=1e-5,rtol=1e-4),
 network_changed=False,installations=0,downloads=0,git_actions=[],old_request_hashes_unchanged=True,
 default_source="poseidon-trusted-pytorch-v18",latest_source="poseidon-trusted-pytorch-v28",
 package_reload_and_cache_contexts={m:r["passed"]for m,r in reports.items()},
 new_input_and_repeat_contexts=list(q.REUSE_CASES),
 scope_bindings={m:p["binding"]for m,p in plans.items()},source_freeze_released=True,all_http_resolved=True,
 resources=dict(approved_result_bytes=48*1024**3,case_and_batch_bytes=4*1024**3,reserve_bytes=4*1024**3,
 actual_result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(ROOT).free,
 api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1),
 confirmed=["Source v33 public masks and mixed selection normalize to unchanged arithmetic",
 "All current numerical outputs independently checked; source and weights retained",
 "Current Agent source equals retained Provider response bytes; no manual replacements",
 "Packages reloaded and executed on fresh and repeated inputs"],
 evidence_based_inference=["Specific output-binding feedback and versioned guidance address the observed alias representation failure"],
 unverified=["arbitrary Python and dynamic shapes","encrypted boolean selection","maximum-rank or maximum-capacity encrypted cases",
 "original-model application accuracy","real CPU bootstrap integration","CUDA hardware","full ResNet20"],
 next_work=["mixed public/encrypted concat and stack","stable common source profile",
 "larger weights and physical layout capacity under unchanged backend interfaces"])
record["binding"]=q.digest(record)
target=ROOT/"docs/baseline/agent-public-mask-output-binding-acceptance-r1476.json"
q.write(target,record)
print(json.dumps({k:record[k]for k in("status","binding","sdk","first_attempt_passed","repairs","provider_usage","resources")}))
