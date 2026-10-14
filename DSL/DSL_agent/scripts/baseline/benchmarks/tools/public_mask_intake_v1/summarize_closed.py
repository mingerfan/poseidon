"""Stage acceptance from independently checked source, numerical and transport receipts."""
import sys,json,shlex,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as s
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=s.read(HERE/"identity-snapshot.json")
assert PoseidonBackend().identity_manifest()==identity
assert s.read(HERE/"prechange-identity.json")["backend"]==identity["backend"]
controls=s.read(HERE/"controls.json");guards=s.read(HERE/"controls-execution.json")
assert controls["status"]==guards["status"]=="passed"
assert controls["tests"]==289 and controls["failures"]==controls["errors"]==controls["skipped"]==0
assert guards["passed"]==12 and guards["failed"]==guards["skipped"]==0
assert s.sha(HERE/"controls-qualified.log")==guards["log_sha256"]
assert s.sha(HERE/"test_qualification_scope.py")==guards["test_source_sha256"]
prepared=s.read(HERE/"prepared-requests.json");s.bound(prepared)
cap_plan=s.read(HERE/"capture-plan.json");s.bound(cap_plan)
cap_path=Path(cap_plan["campaign"])/"report.json";capture=s.read(cap_path);s.bound(capture)
assert capture["passed"]==21 and capture["failed"]==capture["not_run"]==capture["skipped"]==0
reports={};audits={};numerics={};plans={}
for mode in ("offline","live"):
 plan=s.verify(mode);out=Path(plan["campaign"])
 report=s.read(out/"report.json")
 audit=s.read(out/("audit-closed.json"if mode=="live"else"audit.json"))
 numeric=s.read(out/("numerical-audit-closed.json"if mode=="live"else"numerical-audit.json"))
 for record in (report,audit,numeric):s.bound(record)
 assert report["status"]=="complete" and report["passed"]+report["failed"]==len(plan["cases"]) and report["calls_resolved"]
 assert report["skipped"]==report["not_run"]==0 and s.read(out/"terminal.json")["exit_code"]==(0 if report["failed"]==0 else 1)
 assert audit["status"]==numeric["status"]=="passed" and audit["report_binding"]==numeric["report_binding"]==report["binding"]
 assert numeric["qualification_audit_binding"]==audit["binding"]
 assert audit["sdk"]==numeric["sdk"]==identity["manifest_sha256"]
 reports[mode]=report;audits[mode]=audit;numerics[mode]=numeric;plans[mode]=plan
prior_path=ROOT/"docs/baseline/agent-public-constants-acceptance-r1463.json"
prior=s.read(prior_path);s.bound(prior)
assert prior["binding"]=="e86ecf6bfd54fa2481d11dabdf9a064815e2acca72e27e4d0acb1083bf70718b"
initial=s.read(HERE/"controls-capture-initial-failures.json")
assert initial["failed_subtests"]==7 and initial["compiler_calls"]==initial["encrypted_calls"]==initial["api_calls"]==0
for rel,want in initial["source_files"].items():
 assert s.sha(HERE/rel)==want
 if not rel.endswith("manifest.json"):
  current=HERE/rel.replace("cases-initial/","cases/")
  assert s.sha(current)==want
branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert branch=="lhy-agent-dsl" and head=="7588e48093631eaf8385506e3db058e7ca3cfe7e"
repairs_path=HERE/"supplemental-audit-repairs.json"
supplemental=s.read(repairs_path)
assert supplemental["accepted_tool"]=="numerical_audit_v3.py"
for failure in supplemental["failures"]:
 assert s.sha(HERE/failure["log"])==failure["log_sha256"]
 assert s.sha(HERE/failure["tool"])==failure["tool_sha256"]
gap_path=ROOT/"docs/baseline/agent-output-alias-runtime-gap-r1473.json"
gap=s.read(gap_path);original=Path(gap["original_evidence"])/"attempt-00"
assert gap["sdk"]==identity["manifest_sha256"] and gap["backend_changed"] is False
assert s.sha(original/"execute.log")==gap["error_log_sha256"]
rejected=s.read(original/"report.json")
assert rejected["compiled"] is True and rejected["executed"] is False and rejected["status"]=="failed"
assert "duplicate final output" in (original/"execute.log").read_text()
usage=[c["result"]["generation"]["usage"]for c in reports["live"]["cases"]if c["status"]=="passed"]
for case in reports["live"]["cases"]:
 if case["status"]=="passed":continue
 task=Path(plans["live"]["campaign"])/case["case"]/"tasks"/case["task_id"]
 metrics=s.read(task/"provider-metrics.json");calls=metrics["calls"]
 fields=("prompt_tokens","completion_tokens","total_tokens","reasoning_tokens","prompt_cache_hit_tokens","prompt_cache_miss_tokens")
 partial={k:sum(c.get("usage",{}).get(k,0)or 0 for c in calls)for k in fields}
 partial.update(responses_with_usage=sum("usage"in c for c in calls),
                responses_without_usage=sum("usage"not in c for c in calls),
                unresolved_http_attempts=0,cost_usd=None)
 usage.append(partial)
retained_failures=[]
for row in reports["live"]["cases"]:
 task=Path(plans["live"]["campaign"])/row["case"]/"tasks"/row["task_id"]
 for path in sorted(task.glob("validation-*.json")):
  value=s.read(path)
  if value["status"]=="passed":continue
  if "evidence"in value:
   ev=Path(value["evidence"]);a=s.read(ev/"attempt-00/report.json")
   log=ev/"attempt-00/execute.log"
   retained_failures.append(dict(case=row["case"],validation=path.name,
    layer=a["failure_layer"],compiled=a["compiled"],encrypted_execution=a["executed"],
    symptom="duplicate_final_output"if log.exists()and "duplicate final output"in log.read_text()else "other",
    original_report_sha256=s.sha(ev/"attempt-00/report.json")))
  else:
   retained_failures.append(dict(case=row["case"],validation=path.name,
    layer=value["failure"]["layer"],compiled=False,encrypted_execution=False,
    symptom="native_storage_method_requires_object_array",original_report_sha256=s.sha(path)))

usage_keys=("prompt_tokens","completion_tokens","total_tokens","reasoning_tokens","prompt_cache_hit_tokens",
 "prompt_cache_miss_tokens","responses_with_usage","responses_without_usage","unresolved_http_attempts")
totals={k:sum(u.get(k,0)or 0 for u in usage)for k in usage_keys};totals["cost_usd"]=None
record=dict(format="poseidon-public-mask-phase-acceptance-v1",status="partial"if reports["live"]["failed"]else"complete",
 qualified_cases={m:a.get("qualified_cases",len(a["cases"]))for m,a in audits.items()},
 failed_cases=audits["live"].get("failed_cases",[]),retained_failed_validations=retained_failures,
 checkout=str(ROOT),branch=branch,head=head,sdk=identity["manifest_sha256"],backend="poseidon_cpu",
 backend_identity=identity["backend"],pytorch_source_format="poseidon-trusted-pytorch-v33",
 public_value_policy="poseidon-public-value-expressions-v6",source_capture="poseidon-public-expression-source-capture-v10",
 source_flow="poseidon-public-expression-flow-v6",selection_policy="poseidon-public-mask-selection-v1",
 calls=["torch.where","Tensor.where","Tensor.masked_fill","Tensor.masked_fill_"],
 semantic_groups=["where","tensor_where","masked_fill","masked_fill_","public_where","public_masked_fill"],
 condition="bound public bool tensor, statically broadcast; encrypted conditions rejected",
 source_contexts=18,compatibility_contexts=3,unique_public_generation_requests=len({c["request_id"]for c in plans["live"]["cases"]}),
 benchmark_structural_count_added=0,controls=controls,execution_guards=guards,
 plaintext=dict(groups=sum(c["plaintext_groups"]for c in prepared["cases"]),values=sum(c["plaintext_values"]for c in prepared["cases"]),
  independent_math="NumPy public selection and closed graph mathematics; original Torch CPU source",
  tolerance=dict(atol=1e-12,rtol=1e-12)),
 source_capture_report=str(cap_path),source_capture_binding=capture["binding"],
 offline=audits["offline"],live=audits["live"],numerical_audits=numerics,
 first_attempt_passed=sum(c["result"]["generation"]["first_attempt_passed"]is True for c in reports["live"]["cases"]if c["status"]=="passed"),
 repairs=sum(c["result"]["generation"]["repairs_used"]for c in reports["live"]["cases"]if c["status"]=="passed"),
 package_reloaded_contexts={m:r["passed"]for m,r in reports.items()},
 new_input_and_repeat_contexts=list(s.REUSE_CASES),source_cache_contexts={m:r["passed"]for m,r in reports.items()},
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,generation_guidance="explicit-v13"),
 provider_usage_records=usage,provider_usage_totals=totals,
 provider_usage_evidence_scope="returned usage and task transport receipts, not provider billing proof",
 provider_usage_complete=all(u["responses_without_usage"]==0 for u in usage),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),core_backend_unchanged=True,parameters_unchanged=True,network_unchanged=True,
 installations=0,downloads=0,git_actions=[],history_rewritten=False,
 default_source="poseidon-trusted-pytorch-v18",latest_source="poseidon-trusted-pytorch-v28",
 plan_bindings={m:p["binding"]for m,p in plans.items()},
 closed_execution_report_bindings={m:r["binding"]for m,r in reports.items()},
 source_freeze_released=True,all_http_resolved=True,agent_source_equals_provider_response=True,
 initial_controls_rejections=dict(report=str(HERE/"controls-capture-initial-failures.json"),report_sha256=s.sha(HERE/"controls-capture-initial-failures.json"),
  failed_subtests=7,original_source_and_weights_preserved=True,
  resolution="v33 public scalar flip, owned input working copy and correct three-leaf output declarations"),
 supplemental_audit_repairs=dict(path=str(repairs_path),sha256=s.sha(repairs_path),record=supplemental),
 retained_runtime_output_alias_rejection=dict(path=str(gap_path),sha256=s.sha(gap_path),record=gap,
  accepted_without_manual_source_change=True,agent_only_checker_followup="not implemented in this frozen SDK"),
 prior_v32_history=dict(path=str(prior_path),binding=prior["binding"],file_sha256=s.sha(prior_path)),
 resources=dict(approved_result_bytes=48*1024**3,per_batch_bytes=4*1024**3,reserve_bytes=4*1024**3,
  observed_result_bytes=live.measure(s.RESULTS),observed_free_bytes=shutil.disk_usage(ROOT).free,
  compiler_jobs=2,link_jobs=1,api_concurrency=1,encrypted_concurrency=1),
 confirmed=["289 component/control tests and 12 execution-budget guards passed",
  "21 isolated source captures, original source versus independent graph reference",
  str(reports["offline"]["passed"])+" rule and "+str(reports["live"]["passed"])+" current Agent Poseidon CPU numerical qualifications",
   "All original planned case statuses retained; failed Provider attempts are not counted as passes",
  "Every bundle reloaded and cache reused; three representatives ran fresh and repeated inputs per track",
  "Decrypted arrays independently decoded and mathematically checked; candidate bytes equal received Provider source"],
 evidence_based_inference=["Public mask source forms compose using the existing encrypted arithmetic and ordered alias semantics"],
 unverified=["encrypted-data mask or control flow","public-state in-place mutation","dynamic shapes and arbitrary Python",
  "maximum-rank and maximum-capacity encrypted cases","original-model application accuracy",
  "real CPU bootstrap","CUDA hardware","full ResNet20"],
 next_work=["mixed public/encrypted concat and stack","stable common source profile and broader legal source-form combinations",
  "larger weights and physical layout capacity under unchanged backend limits"])
record["binding"]=s.digest(record)
target=ROOT/"docs/baseline/agent-public-mask-acceptance-r1472.json"
assert not target.exists()
s.write(target,record)
print(json.dumps({k:record[k]for k in("binding","sdk","source_contexts","first_attempt_passed","repairs","provider_usage_totals","resources")}))
