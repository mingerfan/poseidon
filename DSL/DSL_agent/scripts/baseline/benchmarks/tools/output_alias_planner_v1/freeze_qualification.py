"""Freeze planner-selected layouts and unchanged model provenance."""
import sys,json,shlex,copy
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
from application_planning.layout import plan_model,check_plan
from component_contract import prepare_task,reconstruct_request
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from deepseek_provider import public_request
parent=BASE/"benchmarks/tools/output_alias_backend_v2"
identity=PoseidonBackend().identity_manifest()
assert identity==q.read(parent/"identity-snapshot.json")
q.write(HERE/"identity-snapshot.json",identity)
control=q.read(HERE/"controls.json");assert control["status"]=="passed"and not control["failures"]and not control["errors"]and not control["skipped"]
guard=q.read(HERE/"controls-scope.json");assert guard["status"]=="passed"and guard["tests"]==12
prepared=q.read(parent/"prepared-requests.json");q.bound(prepared)
rows=[];frozen=copy.deepcopy(q.read(parent/"offline-plan.json")["frozen_files"])
for name in q.OFFLINE_CASES:
 old=next(c for c in prepared["cases"]if c["id"]==name)
 old_request=q.read(old["request_file"]);model=old_request["model"]
 folder=Path(old["model_import"]);origin=load_model_provenance(folder,model,max_bytes=4*1024**2)
 options=dict(generation_guidance="explicit-v15",output_binding_policy="exact-values-v1",
  public_weight_policy="descriptors-math-roles-v2",capacity_packing_version="capacity-public-tensors-v9")
 options=preparation_options(origin,model,options);options=graph_options(origin["original"],model,options)
 decision=plan_model(model,options=options);assert decision["status"]=="prepared"
 request=check_plan(decision,model);assert request==prepare_task(model,decision["selection"]["options"])and reconstruct_request(request)==request
 target=HERE/"prepared-requests"/name;target.mkdir(parents=True)
 q.write(target/"plan.json",decision);q.write(target/"request.json",request);q.write(target/"public-request.json",public_request(request))
 row=dict(old,options=decision["selection"]["options"],layout_plan=str(target/"plan.json"),layout_plan_sha256=q.sha(target/"plan.json"),
  request_file=str(target/"request.json"),request_file_sha256=q.sha(target/"request.json"),request_id=request["request_id"],
  public_request_file=str(target/"public-request.json"),public_request_file_sha256=q.sha(target/"public-request.json"),
  parent_request_id=old_request["request_id"],parent_request_file=old["request_file"],original_model_and_weights_unchanged=True,
  selected_period=request["layout"]["input_slot_period"])
 rows.append(row)
 for rel,want in decision["source_authority"]["files"].items():
  path=BASE/rel;assert q.sha(path)==want
  frozen[str(path.relative_to(ROOT))]=want
 for key in("layout_plan","request_file","public_request_file"):frozen[str(Path(row[key]).relative_to(ROOT))]=q.sha(row[key])
 print(json.dumps(dict(case=name,selected_period=row["selected_period"],physical_outputs=request["layout"]["output_ciphertexts"],logical_elements=request["layout"]["output_shape"][0])))
result=dict(format="poseidon-planner-alias-prepared-requests-v1",cases=rows,parent_binding=prepared["binding"],paid_calls=0,compiler_calls=0,encrypted_calls=0)
result["binding"]=q.digest(result);q.write(HERE/"prepared-requests.json",result)
for name in("qualification.py","qualification_scope.py","test_qualification_scope.py","audit.py","numerical_audit_v3.py","controls.py","controls.json","controls-scope.json","prepared-requests.json","freeze_qualification.py"):
 frozen[str((HERE/name).relative_to(ROOT))]=q.sha(HERE/name)
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation);assert activation["approved"]is True
for mode,gens,http,compiles,encrypted,wall,seconds in(
 ("offline",3,0,3,9,1200,180),("live",12,48,12,18,43200,14400)):
 p=dict(format="poseidon-planner-alias-qualification-plan-v1",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 mode=mode,platform="aarch64-linux",backend="poseidon_cpu",sdk=identity["manifest_sha256"],
 source_root=str(ROOT),campaign=str(q.RESULTS/("agent-output-alias-planned-"+mode+"-r1487")),
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=rows,package_new_input_cases=list(q.REUSE_CASES),
 source_reference_parent_binding=prepared["binding"],request_values_unchanged=False,only_layout_and_request_binding_changed=True,
 original_models_or_references_modified=False,backend_core_changed=False,parameters_changed=False,network_changed=False,installations=0,downloads=0,git_actions=[],
 generation_guidance="explicit-v15",source_format="poseidon-trusted-pytorch-v34",
 budget=dict(global_result_bytes=48*1024**3,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3,
 wall_seconds=wall,case_seconds=seconds,max_generations=gens,max_http=http,max_compiler_calls=compiles,
 max_encrypted_calls=encrypted,compiler_jobs=2,link_jobs=1,encrypted_concurrency=1,api_concurrency=1),
 frozen_files=frozen,storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 provider=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,api_timeout=1200,
 max_tokens=384000,max_repairs=3,provider_retries=3,money_limit=None),
 outward_content=["public graph","fixed public weights or descriptors","layout","DSL rules","controlled diagnostic"],
 transmission_retry_may_duplicate_billing=True,historical_records_rewritten=False)
 p["binding"]=q.digest(p);q.write(HERE/(mode+"-plan.json"),p);q.write(ROOT/("docs/baseline/agent-output-alias-planned-"+mode+"-scope-r1487.json"),p)
 q.verify(mode);print(json.dumps(dict(mode=mode,cases=len(rows),binding=p["binding"])))
