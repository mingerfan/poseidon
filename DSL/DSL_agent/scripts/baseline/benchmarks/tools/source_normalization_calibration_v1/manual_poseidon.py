"""Origin-enabled application and package qualification; no fresh Agent calls."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=900,environment="poseidon-cpu"))
import numpy as np
from application_model_source import FORMAT as SOURCE_REQUEST
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from component_contract import request_options
from unified_graph_lowering import candidate_source
from task_numerical_assessment import mathematical
from numerical_results import compare
from candidate_bundle import check_bundle
OUT=q.RESULTS/"agent-normalization-source-manual-r1691";q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+512*1024**2<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+512*1024**2,"storage_budget")
OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest()
admission=q.read(HERE/"acceptance.json");q.bound(admission)
q.require(admission["passed"]==6 and admission["sdk"]==sdk["manifest_sha256"],"source_component_acceptance")
manifests=[(r["case"],q.RESULTS/"agent-normalization-source-calibration-r1689"/r["case"]/"freeze/source/manifest.json")for r in admission["cases"]]


requests=[]
for name,manifest in manifests:
 requests.append((name,dict(source_manifest=str(manifest),manifest_sha256=q.sha(manifest))))
files={str(p):q.sha(p)for p in HERE.glob("*.py")}
files.update({str(B/n):q.sha(B/n)for n in("generation_model_origin.py","unified_origin_guidance.py","public_math_context.py","unified_suffix_stage_guidance.py","unified_graph_contract.py",
 "unified_contract_scope_guidance.py","application_diagnostics.py","validation_adapter.py","run_candidate.py","source_calibration.py","source_fx_observer.py","source_fx_metadata.py","source_approximation_preparation.py","pytorch_static_trace.py","sandbox_policy.py","poseidon_sandbox_policy.py")})
plan=dict(format="poseidon-normalization-manual-plan-v1",authorization="human_current_plan_all_tests_approved",
 sdk=sdk["manifest_sha256"],requests=[dict(case=n,source=r)for n,r in requests],
 max_compiler_calls=6,max_encrypted_calls=18,wall_seconds=900,new_bytes=512*1024**2,
 total_bytes=48*1024**3,reserve_bytes=4*1024**3,files=files,api_calls=0)
plan["binding"]=q.digest(plan);q.write(HERE/"manual-plan.json",plan);q.write(OUT/"poseidon-plan.json",plan);q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-normalization-source-manual-scope-r1691.json",plan)
start=time.monotonic();baseline=live.measure(q.RESULTS)
def guard():
 q.require(time.monotonic()-start<900,"time_budget")
 total=live.measure(q.RESULTS)
 q.require(total<=48*1024**3 and total-baseline<=512*1024**2 and shutil.disk_usage(q.RESULTS).free>=4*1024**3,"artifact_budget")
 q.require(PoseidonBackend().identity_manifest()==sdk,"frozen_sdk")
 q.require(all(q.sha(Path(p))==h for p,h in files.items()),"frozen_sources")
class RuleProvider:
 kind="scripted_replay";agent_calls=0
 def generate(self,request,feedback_history):
  source,_=candidate_source(request)
  sys.path.insert(0,str(T/"layernorm_calibration_v1"))
  from manual_plain_factors import compact
  from test_unified_capacity_linear import interpreted_output
  from benchmark_graph import samples as probes
  from unified_graph_contract import validate_candidate
  changed,merges=compact(source,request)
  validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=changed),request)
  maximum=0.
  for sample in probes(request["model"],16):
   expected_values=mathematical(request["model"],sample)
   expected=np.concatenate([expected_values[o["name"]].reshape(-1)for o in request["model"]["outputs"]])
   original=interpreted_output(source,request,sample);compact_values=interpreted_output(changed,request,sample)
   q.require(np.all(np.abs(original-compact_values)<=1e-12+1e-12*np.abs(original)) and
     np.all(np.abs(expected-compact_values)<=1e-12+1e-12*np.abs(expected)),"independent_plain_equivalence")
   maximum=max(maximum,float(np.max(np.abs(expected-compact_values))))
  (folder/"original-rule.py").write_text(source);(folder/"manual-equivalent.py").write_text(changed)
  q.write(folder/"manual-expression-witness.json",dict(exact_binary_factor_merges=merges,
   original_source_sha256=q.sha(folder/"original-rule.py"),manual_source_sha256=q.sha(folder/"manual-equivalent.py"),
   maximum_absolute_error=maximum,provider_source_changed=False,agent_generated=False))
  source=changed
  return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
rows=[]
for name,req in requests:
 guard();folder=OUT/name;folder.mkdir(mode=0o700)
 app=application(WorkerContext.capture(folder/"tasks"))
 source_manifest=Path(req["source_manifest"])
 q.require(q.sha(source_manifest)==req["manifest_sha256"],"frozen_source_manifest")
 task=app.submit_source(dict(format=SOURCE_REQUEST,kind="trusted_pytorch",manifest=str(source_manifest),trust_local_code=True),
   resource_profile="general-tensors-v1",options=dict(capacity_packing_version="capacity-public-tensors-v10",generation_guidance="explicit-v21",public_weight_policy="descriptors-math-roles-v2",chunk_period=4),
   limits=dict(generations=1,http_attempts=1,wall_seconds=240,api_timeout_seconds=1200,max_tokens=384000))
 prepared=app.prepare_source(task);q.write(folder/"prepared-state.json",prepared)
 q.require(prepared["status"]=="queued","current_source_admission")
 q.write(folder/"task-id.json",dict(task_id=task))
 metadata=app._checked_task_metadata(task)
 req=metadata["request"]
 from generation_model_origin import verify
 from component_contract import reconstruct_request
 verify(req["generation_model_origin"],metadata["model_provenance"],req)
 q.require(reconstruct_request(req)==req,"origin_request_reconstruction")
 expected=next(r for r in admission["cases"]if r["case"]==name)
 q.require(q.digest(req["model"])==expected["captured_model_sha256"],"same_frozen_captured_model")
 q.write(folder/"request.json",req)
 result=app.run(task,RuleProvider());q.write(folder/"result.json",result)
 row=dict(case=name,status=result["status"],agent_generated=False,api_calls=0,request_id=req["request_id"],task_id=task,capture_workers=1,source_origin_binding=req["generation_model_origin"]["binding"],
  candidate_origin="manual_equivalent")
 if result["status"]=="succeeded":
  q.require(result["acceptance"]["level"]=="numerical","full_acceptance")
  bundle=check_bundle(app._dir(task)/"program")
  rng=np.random.default_rng(2026101650)
  samples=[{i["name"]:rng.uniform(-.2,.2,i["shape"]).tolist()for i in req["model"]["inputs"]}for _ in range(3)]
  expected=[mathematical(req["model"],{n:np.asarray(v,dtype=np.float64)for n,v in sample.items()})for sample in samples]
  q.write(folder/"reuse-inputs.json",samples)
  values=0;maximum=0.
  for label,input_samples,references in(("new",samples,expected),("repeat",samples[:1],expected[:1])):
   guard();executed=app.execute(task,input_samples)
   q.require(executed["status"]=="executed"and executed["encrypted_execution"]is True and
      executed["executor_secret_key"]is False and executed["compiler_calls"]==executed["agent_calls"]==0,"real_package_reuse")
   a=[];b=[]
   for got,want in zip(executed["outputs"],references):
    for item in req["model"]["outputs"]:
     n=item["name"];left=np.asarray(got[n]).reshape(-1);right=np.asarray(want[n]).reshape(-1)
     q.require(left.shape==right.shape,"output_shape");a.extend(left.tolist());b.extend(right.tolist())
   comp=compare(np.asarray(a).reshape(1,-1),np.asarray(b).reshape(1,-1),1e-5,1e-4)
   q.require(comp["passed"],"original_tolerance")
   q.write(folder/(label+"-execution.json"),dict(execution=executed,comparison=comp))
   values+=comp["compared_values"];maximum=max(maximum,comp["max_absolute_error"])
  q.require(check_bundle(app._dir(task)/"program")==bundle,"same_package")
  row.update(compiled_calls=1,encrypted_calls=3,primary_groups=4,
   primary_values=result["acceptance"]["compared_values"],primary_maximum_absolute_error=result["acceptance"]["maximum_absolute_error"],
   reuse_groups=4,reuse_values=values,reuse_maximum_absolute_error=maximum,reuse_verified=True,
   package_binding=bundle["binding"])
 else:row["failure"]=result.get("failure")
 q.write(folder/"case-report.json",row);rows.append(row);print(json.dumps(row),flush=True);guard()
report=dict(format="poseidon-normalization-manual-acceptance-v1",application_accuracy_certified=False,plan_binding=plan["binding"],sdk=sdk["manifest_sha256"],
 cases=rows,planned=6,passed=sum(r["status"]=="succeeded"for r in rows),failed=sum(r["status"]!="succeeded"for r in rows),
 skipped=0,not_run=0,api_calls=0,agent_generated=False,seconds=time.monotonic()-start,
 new_result_bytes=live.measure(q.RESULTS)-baseline,total_result_bytes=live.measure(q.RESULTS))
report["binding"]=q.digest(report);q.write(HERE/"manual-report.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
