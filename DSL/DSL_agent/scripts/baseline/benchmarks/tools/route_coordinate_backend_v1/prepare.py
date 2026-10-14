"""Freeze representative offline regression for the new coordinate context."""
from pathlib import Path
import sys,copy,json,shlex
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
prior=q.read(B/"benchmarks/tools/contract_scope_backend_v1/offline-plan.json");q.bound(prior)
controls=q.read(B/"benchmarks/tools/route_coordinate_component_v1/controls-r2.json");q.bound(controls)
q.require(controls["status"]=="passed","controls")
rows=[];frozen={}
for c in prior["cases"]:
 previous=q.read(c["request_file"]);model=previous["model"];options=copy.deepcopy(c["options"]);options["generation_guidance"]="explicit-v17"
 plan=plan_model(model,options=options);q.require(plan["status"]=="prepared","layout")
 request=check_plan(plan,model)
 q.require(all(request[k]==v for k,v in previous.items()if k not in("generation_guidance","request_id")),"only_guidance_changed")
 dest=HERE/"prepared"/c["id"];dest.mkdir(parents=True)
 for kind,data in(("model",model),("plan",plan),("request",request),("public-request",public_request(request))):q.write(dest/(kind+".json"),data)
 row=dict(id=c["id"],origin=c["origin"],candidate=c["candidate"],candidate_sha256=c["candidate_sha256"],
  execution_source_kind="graph",source_intake_failure=None,execution_source_graph=str(dest/"model.json"),
  execution_source_graph_sha256=q.sha(dest/"model.json"),request_file=str(dest/"request.json"),
  request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"])
 q.require(q.sha(row["candidate"])==row["candidate_sha256"],"unchanged_candidate")
 rows.append(row)
 for f in dest.iterdir():frozen[str(f.relative_to(ROOT))]=q.sha(f)
 frozen[str(Path(row["candidate"]))]=row["candidate_sha256"]
 for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-route-coordinate-offline-v1",mode="offline",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],platform="aarch64-linux",backend="poseidon_cpu",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=rows,frozen_files=frozen,package_new_input_cases=list(q.REUSE_CASES),
 campaign=str(q.RESULTS/"agent-route-coordinate-offline-r1554"),origin="rule_or_manual_original_bytes",
 budget=dict(wall_seconds=900,case_seconds=300,max_generations=3,max_http=0,max_compiler_calls=3,max_encrypted_calls=9,
 api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,
 global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 controls_binding=controls["binding"],previous_scope_binding=prior["binding"],new_agent_generation=False,
 provider_calls=0,network_changed=False,backend_core_changed=False,parameters_changed=False,historical_records_rewritten=False)
scope["binding"]=q.digest(scope);q.write(HERE/"offline-plan.json",scope);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-route-coordinate-offline-proposal-r1554.json",scope)
print("OFFLINE_FROZEN",scope["binding"],flush=True)
