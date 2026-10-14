"""Freeze one complete trusted source request after current offline acceptance."""
from pathlib import Path
import sys,json,copy,shlex
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from component_contract import reconstruct_request
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest()
q.write(HERE/"identity-snapshot.json",identity)
audit=q.read(B/"benchmarks/tools/binding_frames_acceptance_audit_v1/report.json");q.bound(audit)
q.require(audit["status"]=="passed"and audit["sdk"]==identity["manifest_sha256"],"current_offline_accepted")
capture=q.read(B/"benchmarks/tools/binding_frames_source_refresh_v1/capture-scope.json");q.bound(capture)
capture_report=q.read(Path(capture["campaign"])/"report.json");q.bound(capture_report)
q.require(capture_report["status"]=="passed"and capture_report["passed"]==capture_report["source_intake_accepted"]==1
 and capture_report["failed"]==0 and capture_report["sdk"]==identity["manifest_sha256"],"complete_current_capture")
original=capture["cases"][0];folder=Path(capture["campaign"])/original["id"]
model=q.read(folder/"model.json");provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
prior=q.read(B/"benchmarks/tools/provenance_transport_live_v2/live-plan.json");q.bound(prior)
old=next(c for c in prior["cases"]if c["id"]==original["id"]);previous=q.read(old["request_file"])
base=copy.deepcopy(old["options"]);base["generation_guidance"]="explicit-v19"
options=graph_options(provenance["original"],model,preparation_options(provenance,model,base))
plan=plan_model(model,options=options);q.require(plan["status"]=="prepared","source_layout")
request=check_plan(plan,model);q.require(reconstruct_request(request)==request,"planned_request")
q.require(all(request[k]==v for k,v in previous.items()if k not in("generation_guidance","request_id")),
 "original_model_layout_weights_and_params")
dest=HERE/"prepared"/original["id"];dest.mkdir(parents=True)
for kind,data in(("plan",plan),("request",request),("public-request",public_request(request))):
 q.write(dest/(kind+".json"),data)
case=dict(id=original["id"],execution_source_kind="trusted_pytorch",source_manifest=original["manifest"],
 source_intake_failure=None,source_format="poseidon-trusted-pytorch-v36",capture_parent_binding=capture_report["binding"],
 request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
 public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
 layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"])
frozen={str(p.relative_to(ROOT)):q.sha(p)for p in dest.iterdir()}
for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
for p in folder.rglob("*"):
 if p.is_file():frozen[str(p)]=q.sha(p)
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
for p in (HERE/"identity-snapshot.json",B/"benchmarks/tools/binding_frames_source_refresh_v1/capture-scope.json",
 Path(capture["campaign"])/"report.json"):
 frozen[str(p)]=q.sha(p)
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-binding-frames-source-live-v1",mode="live",
 authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",
 storage_activation_binding=activation["binding"],sdk=identity["manifest_sha256"],
 backend="poseidon_cpu",platform="aarch64-linux",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=[case],frozen_files=frozen,package_new_input_cases=[original["id"]],
 campaign=str(q.RESULTS/"agent-binding-frames-source-live-r1570"),
 budget=dict(wall_seconds=14400,case_seconds=14400,max_generations=4,max_http=16,
 max_compiler_calls=4,max_encrypted_calls=6,api_concurrency=1,encrypted_concurrency=1,
 compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,
 global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
 api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),
 explicit_currency_limit=None,transmission_retry_may_duplicate_billing=True,
 origin="current_real_agent",source_pipeline_required=True,source_format="poseidon-trusted-pytorch-v36",
 complete_source_cases=[original["id"]],retained_graph_cases=[],
 outward_content=["public model","weights or descriptors","layout","DSL rules","controlled feedback"],
 excluded_content=["rule or manual DSL source","private inputs","reference arrays","credentials","secret keys"],
 old_model_public_weights_layout_reference_and_params_retained=True,
 offline_acceptance_binding=audit["binding"],current_capture_binding=capture_report["binding"],
 installations=0,downloads=0,backend_core_changed=False,parameters_changed=False,network_changed=False,
 historical_records_rewritten=False,previous_unknown_api_outcomes_retried=False)
scope["binding"]=q.digest(scope);q.write(HERE/"live-plan.json",scope);q.verify("live")
q.write(ROOT/"docs/baseline/agent-binding-frames-source-live-proposal-r1570.json",scope)
print("LIVE_FROZEN",scope["binding"],"sdk",scope["sdk"],flush=True)
