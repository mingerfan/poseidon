"""Resume only package repeat/cache checks from an accepted unchanged Agent DSL."""
import sys,json,time,shlex,shutil,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2]
sys.path[:0]=[str(B/"benchmarks/tools/generic_capacity_backend_v10"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from application_backend import PoseidonBackend
from hecate_python_env import VENV,enter_nix
def save(path,value):
 value=dict(value);value["binding"]=q.digest(value);q.write(path,value);return value
def verify():
 p=q.read(HERE/"scope.json");q.bound(p)
 for path,want in p["frozen_files"].items():q.require(q.sha(path)==want,"resume_frozen:"+path)
 q.require(PoseidonBackend().identity_manifest()==q.read(HERE/"identity.json"),"resume_sdk")
 return p
def inside():
 import numpy as np
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from candidate_bundle import check_bundle
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from numerical_results import compare
 from program_execution import package_execution_metadata
 from pytorch_source_contract import restore_outputs
 p=verify();out=Path(p["campaign"]);owner=q.read(out/"owner.json");task=p["task_id"]
 def boundary():
  verify();q.require(time.time()<owner["deadline"],"wall_budget")
  q.require(live.measure(live.RESULTS)<=p["budget"]["global_result_bytes"],"storage")
  q.require(max(0,live.measure(live.RESULTS)-owner["initial_bytes"])<=p["budget"]["new_artifact_bytes"],"new_artifacts")
  q.require(shutil.disk_usage(out).free>=p["budget"]["reserve_bytes"],"disk_reserve")
 try:
  boundary();app=application(WorkerContext.capture(p["task_store"]))
  meta=app._checked_task_metadata(task);model=meta["request"]["model"]
  q.require(meta["request"]==q.read(p["request_file"]),"request")
  package=app._dir(task)/"program";before=check_bundle(package)
  rng=np.random.default_rng(20261008)
  values={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
  a,b=math_ref(model,values),torch_ref(model,values)
  for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=1e-12)
  output=app.execute(task,[{k:v.tolist()for k,v in values.items()}])
  actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)for o in model["outputs"]])for v in output["outputs"]])
  expected=np.asarray([np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]])])
  cmp=compare(actual,expected,1e-5,1e-4)
  q.require(output["encrypted_execution"]is True and output["agent_calls"]==output["compiler_calls"]==0
   and output["executor_secret_key"]is False and cmp["passed"],"repeat_encrypted_numerical")
  structure=package_execution_metadata(package,before,model)["output_structure"]
  q.require(output["output_structure"]==structure,"output_structure")
  if structure is not None:q.require(output["structured_outputs"]==[restore_outputs(v,structure,allow_none=True)for v in output["outputs"]],"restored_outputs")
  save(out/"repeat.json",dict(status="passed",execution=output,comparison=cmp))
  boundary();old_source=q.read(app._dir(task)/"source.json")
  src=dict(format=FORMAT,kind="trusted_pytorch",manifest=p["manifest"],trust_local_code=True)
  again=app.submit_source(src,resource_profile="general-tensors-v1",options=old_source["requirements"]["options"],limits=old_source["budget"])
  class Never:
   kind="scripted_replay";agent_calls=0
   def generate(self,*a):raise AssertionError("cache_must_not_generate")
  cached=app.run(again,Never())
  q.require(cached["status"]=="succeeded"and cached["package"]["reused"]is True and
   cached["package"]["binding"]==before["binding"]and cached["generations"]==cached["http_attempts"]==0,"source_package_cache")
  q.require(check_bundle(package)==before,"immutable_package")
  save(out/"cache.json",cached);boundary()
  report=save(out/"report.json",dict(format="poseidon-complete-provenance-package-continuation-v1",status="passed",
   scope_binding=p["binding"],sdk=p["sdk"],case="wide_tail_linear",package_binding=before["binding"],
   package_reloaded=True,package_reused=True,repeat_groups=1,repeat_values=cmp["compared_values"],maximum_absolute_error=cmp["max_absolute_error"],
   original_acceptance_binding=p["original_observation_binding"],new_agent_generation=False,provider_calls=0,compiler_calls=0,encrypted_calls=1))
  print(json.dumps(report));return 0
 except BaseException as e:
  save(out/"report.json",dict(status="failed",scope_binding=p["binding"],sdk=p["sdk"],error_type=type(e).__name__,diagnostic=str(e)[:240],
   provider_calls=0,compiler_calls=0,new_agent_generation=False));raise
def outer():
 p=verify();live.no_active();out=Path(p["campaign"]);q.require(not out.exists(),"fresh_campaign")
 initial=live.measure(live.RESULTS)
 q.require(initial+p["budget"]["new_artifact_bytes"]<=p["budget"]["global_result_bytes"]and
  shutil.disk_usage(live.RESULTS).free>=p["budget"]["reserve_bytes"]+p["budget"]["new_artifact_bytes"],"storage")
 out.mkdir(mode=0o700);save(out/"owner.json",dict(pid=os.getpid(),start=live.process_info(os.getpid())["start"],
  deadline=time.time()+p["budget"]["wall_seconds"],initial_bytes=initial,scope_binding=p["binding"]))
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 tree=live.OwnedTree().start()
 try:code=enter_nix(command,seconds=p["budget"]["wall_seconds"],environment="poseidon-cpu")
 finally:background=tree.finish()
 q.require(not background,"owned_background");verify()
 save(out/"terminal.json",dict(exit_code=code,scope_binding=p["binding"],finished=time.time()))
 return code
if __name__=="__main__":
 if sys.argv[1:]==["--execute"]:raise SystemExit(outer())
 if sys.argv[1:]==["--inside"]:raise SystemExit(inside())
 raise SystemExit("resume.py --execute")
