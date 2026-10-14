"""Freeze three accepted source models and current Poseidon rule qualification."""
from pathlib import Path
import sys,shlex,json,copy
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from application_planning.layout import plan_model,check_plan
from deepseek_provider import public_request
from unified_graph_lowering import candidate_source
parent=BASE/"benchmarks/tools/tensor_gather_input_component_v1"
controls=q.read(parent/"source37-controls-r2.json");q.bound(controls)
q.require(controls["status"]=="passed","compatibility_controls")
source=q.read(parent/"source-acceptance.json");q.bound(source)
q.require(source["passed"]==3 and source["failed"]==0,"source_acceptance")
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
rows=[];frozen={}
for name,period in (("vector",8),("rows",8),("middle",16)):
 folder=q.RESULTS/"agent-tensor-gather-source-r1575"/name;model=q.read(folder/"model.json")
 pv=load_model_provenance(folder,model,max_bytes=4*1024**2)
 base=dict(chunk_period=period,capacity_packing_version="capacity-public-tensors-v10",
 generation_guidance="explicit-v19",compiler_configuration="poseidon-cpu-eva-w45-compact-v1",
 output_binding_policy="exact-values-v1",public_weight_policy="descriptors-compact-origins-v3")
 options=graph_options(pv["original"],model,preparation_options(pv,model,base))
 plan=plan_model(model,options=options);q.require(plan["status"]=="prepared","source_layout")
 request=check_plan(plan,model);candidate,baseline=candidate_source(request)
 dest=HERE/"prepared"/name;dest.mkdir(parents=True)
 for kind,data in(("request",request),("plan",plan),("public-request",public_request(request)),("baseline",baseline)):
  q.write(dest/(kind+".json"),data)
 (dest/"rule.py").write_text(candidate)
 manifest=parent/"source-cases-r2"/name/"manifest.json"
 rows.append(dict(id=name,origin="new_rule_baseline",execution_source_kind="trusted_pytorch",
 source_manifest=str(manifest),source_intake_failure=None,source_format="poseidon-trusted-pytorch-v37",
 candidate=str(dest/"rule.py"),candidate_sha256=q.sha(dest/"rule.py"),
 request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),
 request_id=request["request_id"],public_request_file=str(dest/"public-request.json"),
 public_request_file_sha256=q.sha(dest/"public-request.json"),layout_plan=str(dest/"plan.json"),
 layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"]))
 for p in dest.iterdir():frozen[str(p.relative_to(ROOT))]=q.sha(p)
 for rel,want in plan["source_authority"]["files"].items():frozen[str((BASE/rel).relative_to(ROOT))]=want
 for p in manifest.parent.rglob("*"):
  if p.is_file():frozen[str(p)]=q.sha(p)
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=BASE/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-public-tensor-gather-offline-v1",mode="offline",
 authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",
 storage_activation_binding=activation["binding"],sdk=identity["manifest_sha256"],
 backend="poseidon_cpu",platform="aarch64-linux",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=rows,frozen_files=frozen,package_new_input_cases=list(q.OFFLINE_CASES),
 campaign=str(q.RESULTS/"agent-tensor-gather-offline-r1576"),
 budget=dict(wall_seconds=1800,case_seconds=300,max_generations=3,max_http=0,
 max_compiler_calls=3,max_encrypted_calls=9,api_concurrency=1,encrypted_concurrency=1,
 compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,
 global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 source_acceptance_binding=source["binding"],controls_binding=controls["binding"],
 backend_changed=False,parameters_changed=False,network_changed=False,
 new_agent_generation=False)
scope["binding"]=q.digest(scope);q.write(HERE/"offline-plan.json",scope);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-tensor-gather-offline-proposal-r1576.json",scope)
print("OFFLINE_FROZEN",scope["binding"],scope["sdk"])
