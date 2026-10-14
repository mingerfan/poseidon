"""Verify closed qualification receipts and write a data-only acceptance summary."""
import json,shlex,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as scope
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=scope.read(HERE/"identity-snapshot-routed.json")
assert PoseidonBackend().identity_manifest()==identity
old=scope.read(HERE/"prechange-identity.json")
assert old["backend"]==identity["backend"]
reports={};audits={};plans={}
for mode in("offline","live"):
 plan=scope.verify(mode);out=Path(plan["campaign"])
 report=scope.read(out/"report.json");audit=scope.read(out/"audit.json");terminal=scope.read(out/"terminal.json")
 scope.bound(report);scope.bound(audit)
 assert report["status"]=="complete"and report["passed"]==len(plan["cases"])
 assert report["failed"]==report["skipped"]==report["not_run"]==0 and report["calls_resolved"]
 assert terminal["exit_code"]==0 and audit["status"]=="passed"and audit["report_binding"]==report["binding"]
 assert audit["plan_binding"]==plan["binding"]and audit["sdk"]==identity["manifest_sha256"]
 reports[mode]=report;audits[mode]=audit;plans[mode]=plan
controls=scope.read(HERE/"controls-routed.json");guards=scope.read(HERE/"controls-execution.json")
assert controls["status"]==guards["status"]=="passed"
assert controls["tests"]==270 and guards["tests"]==12
assert controls["failures"]==controls["errors"]==controls["skipped"]==guards["failed"]==guards["skipped"]==0
prepared=scope.read(HERE/"prepared-requests.json");scope.bound(prepared)
capture_plan=scope.read(HERE/"capture-plan-routed.json");scope.bound(capture_plan)
capture_path=Path(capture_plan["campaign"])/"report.json";capture=scope.read(capture_path);scope.bound(capture)
assert capture["passed"]==21 and capture["failed"]==capture["not_run"]==capture["skipped"]==0
failed_plan=scope.read(HERE/"capture-plan.json");scope.bound(failed_plan)
failed_path=Path(failed_plan["campaign"])/"report.json";failed=scope.read(failed_path);scope.bound(failed)
assert failed["failed"]==1 and failed["not_run"]==20 and failed["provider_calls"]==failed["compiler_calls"]==failed["encrypted_calls"]==0
history_path=ROOT/"docs/baseline/agent-public-tensor-transforms-acceptance-r1458.json"
history=scope.read(history_path);scope.bound(history)
assert history["binding"]=="bc3bea655c8352949e5967a1b2eeaca38738ba3f2f0b0eff3769ad169da5f39d"
branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert branch=="lhy-agent-dsl"and head=="7588e48093631eaf8385506e3db058e7ca3cfe7e"
usage=[c["result"]["generation"]["usage"]for c in reports["live"]["cases"]]
usage_keys=("prompt_tokens","completion_tokens","total_tokens","reasoning_tokens","prompt_cache_hit_tokens",
 "prompt_cache_miss_tokens","responses_with_usage","responses_without_usage","unresolved_http_attempts")
usage_totals={k:sum(u.get(k,0)or 0 for u in usage)for k in usage_keys}
usage_totals["cost_usd"]=None
record=dict(format="poseidon-public-constant-phase-acceptance-v1",status="complete",checkout=str(ROOT),branch=branch,head=head,
 sdk=identity["manifest_sha256"],backend="poseidon_cpu",backend_identity=identity["backend"],
 pytorch_source_format="poseidon-trusted-pytorch-v32",public_value_policy="poseidon-public-value-expressions-v5",
 source_capture="poseidon-public-expression-source-capture-v9",source_flow="poseidon-public-expression-flow-v5",
 calls=["torch.zeros_like","torch.ones_like","torch.full_like","Tensor.new_zeros","Tensor.new_ones","Tensor.new_full"],
 source_contexts=18,unique_public_generation_requests=len({c["request_id"]for c in plans["live"]["cases"]}),
 compatibility_contexts=3,benchmark_structural_count_added=0,
 controls=dict(passed=270,failed=0,skipped=0,report=str(HERE/"controls-routed.json")),
 execution_guards=dict(passed=12,failed=0,skipped=0,report=str(HERE/"controls-execution.json")),
 plaintext=dict(groups=sum(c["plaintext_groups"]for c in prepared["cases"]),values=sum(c["plaintext_values"]for c in prepared["cases"]),
  torch_cpu_dtype="float64",independent_math="numpy_closed_public_shape_constructor_and_existing_graph_math",
  tolerance=dict(atol=1e-12,rtol=1e-12)),
 source_capture_report=str(capture_path),source_capture_binding=capture["binding"],
 offline=audits["offline"],live=audits["live"],
 first_attempt_passed=sum(c["result"]["generation"]["first_attempt_passed"]is True for c in reports["live"]["cases"]),
 repairs=sum(c["result"]["generation"]["repairs_used"]for c in reports["live"]["cases"]),
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,generation_guidance="explicit-v13"),
 provider_usage_records=usage,provider_usage_totals=usage_totals,
 provider_usage_evidence_scope="runner transport receipts and returned usage; not provider billing proof",
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),core_backend_unchanged=True,parameters_unchanged=True,network_unchanged=True,
 installations=0,downloads=0,git_actions=[],history_rewritten=False,
 default_source="poseidon-trusted-pytorch-v18",latest_source="poseidon-trusted-pytorch-v28",
 plan_bindings={m:p["binding"]for m,p in plans.items()},
 closed_execution_report_bindings={m:r["binding"]for m,r in reports.items()},
 source_freeze_released=True,all_http_resolved=True,agent_source_equals_provider_response=True,
 prior_capture_failure=dict(sdk=failed["sdk"],report=str(failed_path),binding=failed["binding"],failed=1,not_run=20,
  compiler_calls=0,encrypted_calls=0,provider_calls=0,
  layer="isolated_input_capture",cause="v32 omitted from production worker dispatch; old capture path rejected constructor",
  resolution="explicit v32 dispatch in capture_request and public_options; new binding and original failed record retained"),
 prior_v31_history=dict(path=str(history_path),binding=history["binding"],file_sha256=scope.sha(history_path)),
 resources=dict(approved_result_bytes=48*1024**3,per_batch_bytes=4*1024**3,reserve_bytes=4*1024**3,
  observed_result_bytes=live.measure(scope.RESULTS),observed_free_bytes=shutil.disk_usage(ROOT).free,
  compiler_jobs=2,link_jobs=1,api_concurrency=1,encrypted_concurrency=1),
 confirmed=[
  "Closed declaration, public shape/dtype/fill validation before allocation and total constant budget check",
  "21 isolated source captures and independent plaintext comparisons",
  "21 deterministic rule and 18 current real Agent Poseidon CPU encrypted numerical and reusable-package qualifications",
  "Frozen current SDK and backend identities, tc128 artifact gates and returned candidate source provenance",
  "Old v31 records and initial v32 isolated-capture failure preserved"],
 evidence_based_inference=[
  "Static shape-only constructors compose with the existing encrypted graph without backend changes"],
 unverified=["arbitrary Python or dynamic size/fill","GPU constructor device","zero-size tensors",
  "all rank32 and maximum-capacity encrypted cases","JSON public constructor interface",
  "original-model application accuracy","real CPU bootstrap","CUDA hardware","full ResNet20"],
 next_work=["public mask where/masked_fill source normalization",
  "mixed public/encrypted tensor assembly and legal source-form composition",
  "larger weights and physical layout capacity under current backend limits"])
record["binding"]=scope.digest(record)
target=ROOT/"docs/baseline/agent-public-constants-acceptance-r1463.json"
assert not target.exists()
scope.write(target,record)
print(json.dumps({k:record[k]for k in("binding","sdk","source_contexts","compatibility_contexts","first_attempt_passed","repairs",
 "provider_usage_totals","resources")},ensure_ascii=False))
