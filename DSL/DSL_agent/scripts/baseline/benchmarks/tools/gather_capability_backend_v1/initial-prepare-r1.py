"""Current component regression using unchanged historical real-Agent source."""
from pathlib import Path
import sys,json,shlex,copy
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=BASE.parents[1]
sys.path[:0]=[str(HERE),str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
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
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
controls=q.read(BASE/"benchmarks/tools/gather_capability_component_v1/controls.json");q.bound(controls)
q.require(controls["status"]=="passed","host_controls")
prior=q.read(BASE/"benchmarks/tools/tensor_gather_live_v1/live-plan.json");q.bound(prior)
audit=q.read(BASE/"benchmarks/tools/tensor_gather_live_audit_v1/report.json");q.bound(audit)
q.require(audit["audit_status"]=="passed"and audit["passed"]==3 and audit["unknown_http"]==0,"historical_real_agent")
old=prior["cases"][0];previous=q.read(old["request_file"])
parent=Path(prior["campaign"])/"vector";task=parent/"tasks"/q.read(parent/"task-id.json")["task_id"]
program=task/"program";candidate=(program/"candidate.py").read_bytes()
pv=load_model_provenance(q.RESULTS/"agent-tensor-gather-source-r1575/vector",previous["model"],max_bytes=4*1024**2)
options=graph_options(pv["original"],previous["model"],preparation_options(pv,previous["model"],old["options"]))
plan=plan_model(previous["model"],options=options);q.require(plan["status"]=="prepared","host_layout")
request=check_plan(plan,previous["model"]);q.require(request==previous,"unchanged_request_and_model")
dest=HERE/"prepared/vector";dest.mkdir(parents=True)
for kind,data in(("request",request),("plan",plan),("public-request",public_request(request))):q.write(dest/(kind+".json"),data)
(dest/"rule.py").write_bytes(candidate)
case=dict(old,origin="historical_real_agent_source_replay",candidate=str(dest/"rule.py"),
 candidate_sha256=q.sha(dest/"rule.py"),request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),
 public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
 layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"])
frozen={str(p.relative_to(ROOT)):q.sha(p)for p in dest.iterdir()}
for rel,want in plan["source_authority"]["files"].items():frozen[str((BASE/rel).relative_to(ROOT))]=want
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=BASE/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-host-capability-current-backend-regression-v1",mode="offline",
 authorization="human_explicit_current_plan_test_approval",storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",
 storage_activation_binding=activation["binding"],sdk=identity["manifest_sha256"],backend="poseidon_cpu",
 platform="aarch64-linux",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=[case],frozen_files=frozen,
 package_new_input_cases=["vector"],campaign=str(q.RESULTS/"agent-gather-host-capability-offline-r1580"),
 budget=dict(wall_seconds=900,case_seconds=600,max_generations=1,max_http=0,max_compiler_calls=1,max_encrypted_calls=3,
 api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,
 per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 historical_generation_binding=audit["binding"],historical_candidate_sha256=q.sha(program/"candidate.py"),
 controls_binding=controls["binding"],new_agent_generation=False,backend_changed=False,parameters_changed=False,
 network_changed=False,downloads=0,installations=0)
scope["binding"]=q.digest(scope);q.write(HERE/"offline-plan.json",scope);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-gather-host-capability-offline-proposal-r1580.json",scope)
print("REGRESSION_FROZEN",scope["binding"],scope["sdk"])
