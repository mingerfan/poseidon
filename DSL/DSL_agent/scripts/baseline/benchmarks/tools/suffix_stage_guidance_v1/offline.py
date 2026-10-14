"""Finite offline Poseidon qualification of public suffix-stage contracts."""
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
from test_suffix_stage_guidance import request,graph
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from component_contract import request_options
from unified_graph_lowering import candidate_source
from task_numerical_assessment import mathematical
from numerical_results import compare
from candidate_bundle import check_bundle
OUT=q.RESULTS/"agent-suffix-stage-offline-r1650";q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+512*1024**2<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+512*1024**2,"storage_budget")
OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest()
requests=[("original_rsqrt_mean",request()),("nonbinary_window_sum",request(graph((2,3),"sum"),8)),
          ("multichunk_mean",request(graph((3,4),"mean"),4))]
files={str(p):q.sha(p)for p in HERE.glob("*.py")}
files.update({str(B/n):q.sha(B/n)for n in("unified_suffix_stage_guidance.py","unified_graph_contract.py",
 "unified_contract_scope_guidance.py","application_diagnostics.py","validation_adapter.py","run_candidate.py","test_suffix_stage_guidance.py")})
plan=dict(format="poseidon-suffix-stage-offline-plan-v1",authorization="human_current_plan_all_tests_approved",
 sdk=sdk["manifest_sha256"],requests=[dict(case=n,request=r)for n,r in requests],
 max_compiler_calls=3,max_encrypted_calls=9,wall_seconds=900,new_bytes=512*1024**2,
 total_bytes=48*1024**3,reserve_bytes=4*1024**3,files=files,api_calls=0)
plan["binding"]=q.digest(plan);q.write(HERE/"offline-plan.json",plan);q.write(OUT/"offline-plan.json",plan)
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
  return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
rows=[]
for name,req in requests:
 guard();folder=OUT/name;folder.mkdir(mode=0o700)
 app=application(WorkerContext.capture(folder/"tasks"))
 task=app.submit(req["model"],options=request_options(req),limits=dict(generations=1,http_attempts=1,
   wall_seconds=240,api_timeout_seconds=1200,max_tokens=384000))
 q.write(folder/"task-id.json",dict(task_id=task))
 metadata=app._checked_task_metadata(task)
 q.require(metadata["request"]==req,"same_request")
 result=app.run(task,RuleProvider());q.write(folder/"result.json",result)
 row=dict(case=name,status=result["status"],agent_generated=False,api_calls=0,request_id=req["request_id"],task_id=task)
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
report=dict(format="poseidon-suffix-stage-offline-acceptance-v1",plan_binding=plan["binding"],sdk=sdk["manifest_sha256"],
 cases=rows,planned=3,passed=sum(r["status"]=="succeeded"for r in rows),failed=sum(r["status"]!="succeeded"for r in rows),
 skipped=0,not_run=0,api_calls=0,agent_generated=False,seconds=time.monotonic()-start,
 new_result_bytes=live.measure(q.RESULTS)-baseline,total_result_bytes=live.measure(q.RESULTS))
report["binding"]=q.digest(report);q.write(HERE/"offline-report.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
