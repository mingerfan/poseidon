"""Prepare new source-specific live identities after the prior campaign and audit finish."""
import pathlib,sys,shlex,json,copy
HERE=pathlib.Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=B.parents[1]
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 previous=q.read(B/"benchmarks/tools/contract_scope_live_v1/live-plan.json");q.bound(previous)
 q.require((pathlib.Path(previous["campaign"])/"terminal.json").is_file(),"prior_live_not_terminal")
 dispatch=q.read(B/"benchmarks/tools/contract_scope_audit_dispatch_v1/report.json");q.bound(dispatch)
 audit=q.read(B/"benchmarks/tools/contract_scope_live_audit_v1/report.json");q.bound(audit)
 q.require(dispatch["status"]=="passed"and audit["audit_status"]=="passed"and audit["calls_resolved"],"prior_audit_not_complete")
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=600,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from component_contract import prepare_task,reconstruct_request
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
integrity=q.read(ROOT/"docs/baseline/agent-v36-source-integrity-r1551.json");q.bound(integrity)
q.require(integrity["status"]=="passed","public_source_integrity")
for name,want in integrity["evidence_files"].items():q.require(q.sha(name)==want,"source_integrity_hash")
capture=q.read(B/"benchmarks/tools/provenance_transport_component_v1/capture-scope-r1520.json");q.bound(capture)
capture_report=q.read(pathlib.Path(capture["campaign"])/"report.json");q.bound(capture_report)
q.require(capture_report["passed"]==capture_report["source_intake_accepted"]==3 and capture_report["failed"]==0,"historical_complete_capture")
prior=q.read(B/"benchmarks/tools/contract_scope_live_v1/live-plan.json");q.bound(prior)
prior_report=q.read(pathlib.Path(prior["campaign"])/"report.json");q.bound(prior_report)
q.require(prior_report["status"]!="running"and prior_report["calls_resolved"],"prior_live_http_not_settled")
rows=[];frozen={}
for name in q.LIVE_CASES:
 c=next(c for c in capture["cases"]if c["id"]==name)
 folder=pathlib.Path(capture["campaign"])/name;model=q.read(folder/"model.json")
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
 old_options=q.read(B/"benchmarks/tools/provenance_transport_backend_v1/prepared"/name/"plan.json")["selection"]["options"]
 base=copy.deepcopy(old_options);base.update(generation_guidance="explicit-v17",capacity_packing_version="capacity-public-tensors-v10")
 options=graph_options(provenance["original"],model,preparation_options(provenance,model,base))
 plan=plan_model(model,options=options);q.require(plan["status"]=="prepared","source_layout")
 request=check_plan(plan,model);q.require(prepare_task(model,plan["selection"]["options"])==reconstruct_request(request)==request,"same_planned_request")
 dest=HERE/"prepared"/name;dest.mkdir(parents=True)
 for kind,data in(("plan",plan),("request",request),("public-request",public_request(request))):q.write(dest/(kind+".json"),data)
 row=dict(id=name,execution_source_kind="trusted_pytorch",source_manifest=c["manifest"],source_intake_failure=None,
  source_format="poseidon-trusted-pytorch-v36",capture_parent_binding=capture_report["binding"],
  request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"])
 rows.append(row)
 for p in dest.iterdir():frozen[str(p.relative_to(ROOT))]=q.sha(p)
 for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
 for p in folder.rglob("*"):
  if p.is_file():
   q.require(not p.is_symlink(),"captured_evidence_symlink")
   frozen[str(p)]=q.sha(p)
 print(json.dumps(dict(case=name,request_id=request["request_id"],source_kind="trusted_pytorch",public_request_bytes=(dest/"public-request.json").stat().st_size)),flush=True)
# The prior SDK source hashes stay historical. The new application capture is
# bound to the current SDK; preserve old capture records without claiming those
# older code files still have their old contents.
frozen.update(integrity["evidence_files"])
for p in(B/"benchmarks/tools/provenance_transport_component_v1/capture-scope-r1520.json",pathlib.Path(capture["campaign"])/"report.json"):
 frozen[str(p)]=q.sha(p)
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
q.require(identity==PoseidonBackend().identity_manifest(),"sdk_unchanged")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-v36-complete-source-live-v1",mode="live",
 authorization="human_explicit_current_plan_test_approval",user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],platform="aarch64-linux",backend="poseidon_cpu",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=rows,frozen_files=frozen,budget=dict(wall_seconds=43200,case_seconds=14400,max_generations=12,max_http=48,
 max_compiler_calls=12,max_encrypted_calls=18,api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,
 new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 package_new_input_cases=list(q.REUSE_CASES),campaign=str(q.RESULTS/"agent-v36-complete-source-live-r1556"),
 origin="current_real_agent",provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
 api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),explicit_currency_limit=None,
 transmission_retry_may_duplicate_billing=True,outward_content=["public graph","weights or descriptors","layout","DSL rules","controlled feedback"],
 excluded_content=["rule or manual DSL answers","private inputs","reference","credentials","secret keys"],
 network_changed=False,backend_core_changed=False,parameters_changed=False,installations=0,downloads=0,git_actions=[],
 historical_records_rewritten=False,previous_unknown_api_outcomes_retried=False,
 source_format="poseidon-trusted-pytorch-v36",source_pipeline_required=True,source_integrity_binding=integrity["binding"],
 historical_capture_sdk=capture_report.get("sdk"),historical_capture_source_hashes_not_rebound=True,
 current_application_source_capture_required=True,
 previous_scope_binding=prior["binding"],previous_terminal_report=prior_report["binding"])
scope["binding"]=q.digest(scope);q.write(HERE/"live-plan.json",scope);q.verify("live")
q.write(ROOT/"docs/baseline/agent-v36-complete-source-live-proposal-r1556.json",scope)
print("FROZEN",scope["binding"],flush=True)
