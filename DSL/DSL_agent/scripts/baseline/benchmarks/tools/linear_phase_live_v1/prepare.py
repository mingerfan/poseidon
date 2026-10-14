"""Freeze one failed original graph for current Agent evaluation; no rule answers."""
from pathlib import Path
import sys,json,shlex,copy
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
audit=q.read(B/"benchmarks/tools/linear_phase_acceptance_audit_v1/report.json");q.bound(audit)
q.require(audit["status"]=="passed"and audit["sdk"]==identity["manifest_sha256"],"current_offline_accepted")
prior=q.read(B/"benchmarks/tools/linear_phase_backend_v1/offline-plan.json");q.bound(prior)
old=prior["cases"][0];request=q.read(old["request_file"]);layout=q.read(old["layout_plan"])
dest=HERE/"prepared/wide_tail_linear";dest.mkdir(parents=True)
for kind,data in(("model",request["model"]),("request",request),("plan",layout),("public-request",public_request(request))):q.write(dest/(kind+".json"),data)
case=dict(id="wide_tail_linear",execution_source_kind="graph",source_intake_failure=None,
 execution_source_graph=str(dest/"model.json"),execution_source_graph_sha256=q.sha(dest/"model.json"),
 request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
 public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
 layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=old["options"])
frozen={str(p.relative_to(ROOT)):q.sha(p)for p in dest.iterdir()}
for rel,want in layout["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-linear-phase-live-v1",mode="live",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],backend="poseidon_cpu",platform="aarch64-linux",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=[case],frozen_files=frozen,package_new_input_cases=["wide_tail_linear"],campaign=str(q.RESULTS/"agent-linear-phase-live-r1566"),
 budget=dict(wall_seconds=14400,case_seconds=14400,max_generations=4,max_http=16,max_compiler_calls=4,max_encrypted_calls=6,
 api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,
 global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,api_timeout_seconds=1200,
 max_tokens=384000,max_repairs=3,provider_retries=3),explicit_currency_limit=None,transmission_retry_may_duplicate_billing=True,
 origin="current_real_agent",execution_source_kind="graph",full_pytorch_source_capture_claimed=False,
 old_model_public_weights_layout_reference_and_params_retained=True,offline_acceptance_binding=audit["binding"],
 excluded_content=["rule baseline source","manual repair source","private inputs","reference arrays","credentials","secret keys"],
 outward_content=["public model","public weights or descriptors","layout","DSL rules","controlled feedback"],
 installations=0,downloads=0,backend_core_changed=False,parameters_changed=False,network_changed=False,
 historical_records_rewritten=False,previous_unknown_api_outcomes_retried=False)
scope["binding"]=q.digest(scope);q.write(HERE/"live-plan.json",scope);q.verify("live")
q.write(ROOT/"docs/baseline/agent-linear-phase-live-proposal-r1566.json",scope)
print("LIVE_FROZEN",scope["binding"],flush=True)
