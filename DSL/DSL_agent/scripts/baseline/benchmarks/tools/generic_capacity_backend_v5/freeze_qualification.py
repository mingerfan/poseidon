"""Freeze finite independent offline and live scopes; preserve all old evidence."""
import sys,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=PoseidonBackend().identity_manifest()
assert identity==q.read(HERE/"identity-snapshot.json")
control=q.read(HERE/"controls.json")
assert control["status"]=="passed" and not(control["failures"] or control["errors"] or control["skipped"])
guard=q.read(HERE/"controls-scope.json");assert guard["status"]=="passed" and guard["tests"]==12
prepared=q.read(HERE/"prepared-requests.json");q.bound(prepared)
assert tuple(c["id"]for c in prepared["cases"])==q.OFFLINE_CASES==q.LIVE_CASES
frozen=dict(prepared["frozen_files"])
for stage in ("generation","compilation","execution","validation"):
 for section in ("sources","configuration_files"):
  for name,want in identity[stage][section].items():
   path=BASE/name if section=="sources"else ROOT/name
   assert q.sha(path)==want;rel=str(path.relative_to(ROOT))
   assert rel not in frozen or frozen[rel]==want;frozen[rel]=want
for name in ("qualification.py","qualification_scope.py","test_qualification_scope.py","audit.py","numerical_audit_v3.py","controls.py","controls.json","controls_scope.py","controls-scope.json",
 "prepared-requests.json","freeze_qualification.py","prepare_requests.py","cases.json","identity-snapshot.json","legacy_checks.py","legacy-request-checks.json"):
 frozen[str((HERE/name).relative_to(ROOT))]=q.sha(HERE/name)
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation);assert activation["approved"]is True
for mode,gens,http,compiles,encrypted,seconds in (
 ("offline",13,0,13,19,1800),("live",52,208,52,58,14400)):
 p=dict(format="poseidon-generic-capacity-qualification-plan-v1",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 mode=mode,platform="aarch64-linux",backend="poseidon_cpu",sdk=identity["manifest_sha256"],
 source_root=str(ROOT),campaign=str(q.RESULTS/("agent-generic-capacity-"+mode+"-r1498")),
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=prepared["cases"],package_new_input_cases=list(q.REUSE_CASES),
 source_reference_parent_bindings=prepared["capture_bindings"],prepared_request_binding=prepared["binding"],
 unique_new_model_contexts=12,historical_model_reused=1,original_models_or_references_modified=False,
 backend_core_changed=False,parameters_changed=False,network_changed=False,installations=0,downloads=0,git_actions=[],
 generation_guidance="explicit-v15; single-output v14",public_weight_policy="descriptors-compact-origins-v3",source_format="poseidon-trusted-pytorch-v34",
 budget=dict(global_result_bytes=48*1024**3,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3,
 wall_seconds=43200,case_seconds=seconds,max_generations=gens,max_http=http,max_compiler_calls=compiles,max_encrypted_calls=encrypted,
 compiler_jobs=2,link_jobs=1,encrypted_concurrency=1,api_concurrency=1),
 frozen_files=frozen,storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 provider=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,api_timeout=1200,
 max_tokens=384000,max_repairs=3,provider_retries=3,money_limit=None),
 outward_content=["public graph","fixed public weights or descriptors","layout","DSL rules","controlled diagnostic"],
 transmission_retry_may_duplicate_billing=True,historical_records_rewritten=False)
 p["binding"]=q.digest(p);q.write(HERE/(mode+"-plan.json"),p)
 q.write(ROOT/("docs/baseline/agent-generic-capacity-"+mode+"-scope-r1498.json"),p)
 q.verify(mode)
 print(json.dumps(dict(mode=mode,cases=len(prepared["cases"]),binding=p["binding"],sdk=p["sdk"])),flush=True)
