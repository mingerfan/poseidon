"""Freeze one original large-input graph with the new phase context."""
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
from application_planning.layout import plan_model,check_plan
from component_contract import request_options
from deepseek_provider import public_request
from unified_graph_lowering import candidate_source
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
controls=q.read(B/"benchmarks/tools/layout_phase_component_v1/controls-r3.json");q.bound(controls);q.require(controls["status"]=="passed","controls")
prior=q.read(B/"benchmarks/tools/provenance_transport_live_v2/live-plan.json");q.bound(prior)
old=next(c for c in prior["cases"]if c["id"]=="wide_tail_linear");previous=q.read(old["request_file"])
options=request_options(previous);options["generation_guidance"]="explicit-v18"
plan=plan_model(previous["model"],options=options);q.require(plan["status"]=="prepared","layout")
request=check_plan(plan,previous["model"]);q.require(all(request[k]==v for k,v in previous.items()if k not in("generation_guidance","request_id")),"unchanged_model_layout_vectors")
source,baseline=candidate_source(request)
dest=HERE/"prepared/wide_tail_linear";dest.mkdir(parents=True)
for kind,data in(("model",request["model"]),("plan",plan),("request",request),("public-request",public_request(request)),("baseline",baseline)):q.write(dest/(kind+".json"),data)
(dest/"rule.py").write_text(source)
case=dict(id="wide_tail_linear",origin="new_rule_baseline",execution_source_kind="graph",source_intake_failure=None,
 execution_source_graph=str(dest/"model.json"),execution_source_graph_sha256=q.sha(dest/"model.json"),
 candidate=str(dest/"rule.py"),candidate_sha256=q.sha(dest/"rule.py"),request_file=str(dest/"request.json"),
 request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],public_request_file=str(dest/"public-request.json"),
 public_request_file_sha256=q.sha(dest/"public-request.json"),layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),
 options=plan["selection"]["options"])
frozen={str(p.relative_to(ROOT)):q.sha(p)for p in dest.iterdir()}
for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
p=dict(format="poseidon-linear-phase-offline-v1",mode="offline",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],backend="poseidon_cpu",platform="aarch64-linux",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=[case],frozen_files=frozen,package_new_input_cases=["wide_tail_linear"],
 campaign=str(q.RESULTS/"agent-linear-phase-offline-r1564"),budget=dict(wall_seconds=900,case_seconds=600,max_generations=1,max_http=0,
 max_compiler_calls=1,max_encrypted_calls=3,api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,
 new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 controls_binding=controls["binding"],old_model_and_parameters_retained=True,new_agent_generation=False,
 backend_core_changed=False,network_changed=False,parameters_changed=False)
p["binding"]=q.digest(p);q.write(HERE/"offline-plan.json",p);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-linear-phase-offline-proposal-r1564.json",p)
print("OFFLINE_FROZEN",p["binding"],"sdk",p["sdk"],flush=True)
