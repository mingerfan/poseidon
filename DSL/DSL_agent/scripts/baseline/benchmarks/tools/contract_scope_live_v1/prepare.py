"""Freeze live request-only evidence; no model API or backend execution."""
import pathlib,sys,shlex,copy,json
HERE=pathlib.Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=600,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
first=q.read(B/"benchmarks/tools/capacity_shallow_backend_v2/offline-plan.json");q.bound(first)
audit=q.read(B/"benchmarks/tools/contract_scope_acceptance_audit_v2/report.json");q.bound(audit)
q.require(audit["status"]=="passed"and audit["sdk"]==identity["manifest_sha256"]and
 len(audit["cases"])==3,"offline_guidance_accepted")
prior_live=q.read(B/"benchmarks/tools/capacity_shallow_live_v1/live-plan.json");q.bound(prior_live)
prior_report=q.read(pathlib.Path(prior_live["campaign"])/"report.json");q.bound(prior_report)
q.require(prior_report["calls_resolved"]and prior_report["status"]=="stopped","prior_calls_settled")
rows=[];frozen={}
for name in q.LIVE_CASES:
 prior=next(c for c in first["cases"]if c["id"]==name);model=q.read(prior["request_file"])["model"]
 options=copy.deepcopy(prior["options"]);options["generation_guidance"]="explicit-v16"
 plan=plan_model(model,options=options);q.require(plan["status"]=="prepared","layout")
 request=check_plan(plan,model);previous=q.read(prior["request_file"]);q.require(all(request[k]==v for k,v in previous.items()if k not in("generation_guidance","request_id")),"unchanged_model_layout_constants")
 dest=HERE/"prepared"/name;dest.mkdir(parents=True)
 for kind,data in (("model",model),("plan",plan),("request",request),("public-request",public_request(request))):
  q.write(dest/(kind+".json"),data)
 row=dict(id=name,execution_source_kind="graph",source_intake_failure=None,
  execution_source_graph=str(dest/"model.json"),execution_source_graph_sha256=q.sha(dest/"model.json"),
  request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"])
 rows.append(row)
 for f in dest.iterdir():frozen[str(f.relative_to(ROOT))]=q.sha(f)
 for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
for f in HERE.glob("*.py"):frozen[str(f.relative_to(ROOT))]=q.sha(f)
for name in ("run.py","scope.py"):
 f=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(f.relative_to(ROOT))]=q.sha(f)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
q.require(identity==PoseidonBackend().identity_manifest(),"sdk_unchanged")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-contract-scope-live-v1",mode="live",
 authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],platform="aarch64-linux",backend="poseidon_cpu",
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=rows,frozen_files=frozen,
 budget=dict(wall_seconds=43200,case_seconds=14400,max_generations=36,max_http=144,max_compiler_calls=36,max_encrypted_calls=42,
  api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,
  per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 package_new_input_cases=list(q.REUSE_CASES),campaign=str(q.RESULTS/"agent-contract-scope-live-r1548"),
 origin="current_real_agent",provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
  api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),explicit_currency_limit=None,
 transmission_retry_may_duplicate_billing=True,
 outward_content=["public graph","public weights or descriptors","layout","DSL rules","controlled feedback"],
 excluded_content=["rule or manual DSL answers","private inputs","reference","credentials","secret keys"],
 network_changed=False,backend_core_changed=False,parameters_changed=False,installations=0,downloads=0,git_actions=[],
 historical_records_rewritten=False,previous_unknown_api_outcomes_retried=False,offline_acceptance_binding=audit["binding"],prior_live_scope=prior_live["binding"],prior_terminal_report=prior_report["binding"])
scope["binding"]=q.digest(scope);q.write(HERE/"live-plan.json",scope);q.verify("live")
q.write(ROOT/"docs/baseline/agent-contract-scope-live-proposal-r1548.json",scope)
print("LIVE_FROZEN",scope["binding"],flush=True)
