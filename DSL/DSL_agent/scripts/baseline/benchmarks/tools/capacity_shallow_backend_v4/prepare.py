"""Freeze a finite explicit-packing qualification; no provider/backend calls."""
import pathlib,sys,json,shlex,math,copy
HERE=pathlib.Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=600,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from component_contract import prepare_task,reconstruct_request,request_options
from unified_capacity_constants import SHALLOW_VERSION,INDEX_VERSION
from unified_graph_lowering import candidate_source
from benchmark_graph import samples,validate
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from test_unified_capacity_linear import interpreted_output
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
cases=[]
oldroot=B/"benchmarks/tools/logical_view_backend_v1/prepared"
for name in ("rank_axis_8","rank_linear_16","rank_permute_32"):
 old=q.read(oldroot/name/"request.json")
 opts=dict(request_options(old),capacity_packing_version=INDEX_VERSION if name=="rank_permute_32"else SHALLOW_VERSION)
 cases.append(dict(id="compat_"+name if name=="rank_permute_32"else name,model=old["model"],options=opts,
   prior_request=str(oldroot/name/"request.json"),prior_request_sha256=q.sha(oldroot/name/"request.json"),
   unchanged_model=True,new_model=False))
def graph(name,inputs,constants,nodes,outputs):
 return dict(format="poseidon-model-graph-v2",resource_profile="general-tensors-v1",id=name,
   inputs=[dict(name=k,shape=v)for k,v in inputs],constants=constants,nodes=nodes,
   outputs=[dict(name=k,value=v)for k,v in outputs])
def node(name,op,inputs,output,**attrs):
 return dict(id=name,op=op,inputs=inputs,outputs=[output],attrs=attrs)
def add(name,model,period):
 cases.append(dict(id=name,model=model,options=dict(chunk_period=period,capacity_packing_version=SHALLOW_VERSION,
  compiler_configuration="poseidon-cpu-eva-w45-compact-v1",generation_guidance="explicit-v14",
  public_weight_policy="descriptors-compact-origins-v3"),new_model=True))
for name,shape,yshape,period in (("broadcast_inner",[2,3,4],[1,3,1],16),("broadcast_tail",[3,5],[3,1],8)):
 add(name,graph(name,[("x",shape),("y",yshape)],{},
  [node("add","add",["x","y"],"v"),node("square","square",["v"],"z")],[("out","z")]),period)
for name,shape,period in (("suffix_three",[4,3],16),("suffix_five",[3,5],16),("suffix_cross_chunk",[9,3],16)):
 add(name,graph(name,[("x",shape)],{},
  [node("square","square",["x"],"v"),node("mean","mean",["v"],"z",axes=[-1],keepdims=False)],[("out","z")]),period)
for name,shape,outputs,period in (("linear_rows",[3,2],3,16),("linear_tail",[5,3],2,8)):
 rng=np.random.default_rng(202610130 if name=="linear_rows"else 202610131)
 const=dict(w=rng.uniform(-.2,.2,(outputs,shape[-1])).tolist(),b=rng.uniform(-.05,.05,outputs).tolist())
 add(name,graph(name,[("x",shape)],const,
  [node("square","square",["x"],"v"),node("fc","linear",["v","w","b"],"z")],[("out","z")]),period)
cases=[c for c in cases if c["id"] in q.OFFLINE_CASES]
for c in cases:c["new_model"]=False
cases.sort(key=lambda c:q.OFFLINE_CASES.index(c["id"]))
q.require(tuple(c["id"]for c in cases)==q.OFFLINE_CASES,"exact_case_order")
rows=[];frozen={};groups=values=0
for c in cases:
 name=c["id"];model=c["model"];validate(model)
 plan=plan_model(model,options=c["options"]);q.require(plan["status"]=="prepared","layout:"+name)
 request=check_plan(plan,model);q.require(request==prepare_task(model,plan["selection"]["options"])==reconstruct_request(request),"request")
 source,strategy=candidate_source(request)
 previous=next(v for v in q.read(B/"benchmarks/tools/capacity_shallow_backend_v2/offline-plan.json")["cases"] if v["id"]==name)
 q.require(request==q.read(previous["request_file"]) and source==pathlib.Path(previous["candidate"]).read_text(),
   "unchanged_request_and_rule_after_mount_fix")
 dest=HERE/"prepared"/name;dest.mkdir(parents=True)
 for probe in samples(model,4):
  a,b=math_ref(model,probe),torch_ref(model,probe)
  for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=1e-12)
  expected=np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]])
  np.testing.assert_allclose(interpreted_output(source,request,probe),expected,atol=1e-12,rtol=1e-12)
  groups+=1;values+=expected.size
 for key,data in (("model",model),("plan",plan),("request",request),("public-request",public_request(request)),("strategy",strategy)):
  q.write(dest/(key+".json"),data)
 (dest/"rule.py").write_text(source)
 row=dict(id=name,execution_source_kind="graph",source_intake_failure=None,
  execution_source_graph=str(dest/"model.json"),execution_source_graph_sha256=q.sha(dest/"model.json"),
  request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),
  candidate=str(dest/"rule.py"),candidate_sha256=q.sha(dest/"rule.py"),options=plan["selection"]["options"],
  new_model=c["new_model"],origin="new_rule_baseline")
 if "prior_request"in c:
  row.update(prior_request=c["prior_request"],prior_request_sha256=c["prior_request_sha256"])
  frozen[str(pathlib.Path(c["prior_request"]).relative_to(ROOT))]=c["prior_request_sha256"]
 rows.append(row)
 for file in dest.iterdir():frozen[str(file.relative_to(ROOT))]=q.sha(file)
 for rel,want in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
 print(json.dumps(dict(case=name,request_id=request["request_id"],constants=len(request["public_constants"]),source_bytes=len(source.encode()))),flush=True)
q.require(PoseidonBackend().identity_manifest()==identity,"source_unchanged")
for file in HERE.glob("*.py"):frozen[str(file.relative_to(ROOT))]=q.sha(file)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
control=B/"benchmarks/tools/capacity_shallow_component_v1/controls-r1537b.json"
ct=q.read(control);q.bound(ct);q.require(ct["failed"]==ct["errors"]==ct["skipped"]==0 and ct["sdk"]==identity["manifest_sha256"],"controls")
frozen[str(control.relative_to(ROOT))]=q.sha(control)
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
scope=dict(format="poseidon-capacity-shallow-offline-scope-v1",mode="offline",
 authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],platform="aarch64-linux",backend="poseidon_cpu",
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=rows,frozen_files=frozen,
 budget=dict(wall_seconds=2400,case_seconds=600,max_generations=5,max_http=0,max_compiler_calls=5,max_encrypted_calls=9,
  api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,
  per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 package_new_input_cases=list(q.REUSE_CASES),campaign=str(q.RESULTS/"agent-capacity-shallow-offline-r1538"),
 origin="new_rule_baseline",provider=None,api_calls=0,network_changed=False,backend_core_changed=False,
 parameters_changed=False,installations=0,downloads=0,git_actions=[],historical_records_rewritten=False,
 reference_precheck=dict(groups=groups,values=values,atol=1e-12,rtol=1e-12,interpreted_rule=True),
 distinct_new_models=sum(c["new_model"]for c in cases),old_failed_models_retained=["rank_axis_8","rank_linear_16"])
scope["prior_scope_binding"]="2753f2f9cfa8101cb5529e7947e79de168dc56c265d02ce3b28ab8491335961f"
scope["prior_passed_audit_binding"]="6e1e0d07a2cb2db05ada7aac3eb1228a9cfc6c88a1974d7b3b48ccba2851f139"
scope["reason"]="Same five cases; original Linear compiled and numerical passed in r1536 but manifest pretty size blocked delivery; compact only when needed within unchanged 128KiB"
scope["mount_fix_scope_binding"]="8aeadd538d4374a3313cfff4d797d2a2f0e308887089495a3511e01ba41c902e"
scope["binding"]=q.digest(scope);q.write(HERE/"offline-plan.json",scope);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-capacity-shallow-offline-proposal-r1538.json",scope)
print("FROZEN",scope["binding"],scope["sdk"],flush=True)
