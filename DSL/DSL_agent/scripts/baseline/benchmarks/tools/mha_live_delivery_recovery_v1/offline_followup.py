"""Finish already-qualified normal component package; zero new generation."""
import json,time,sys
from pathlib import Path
ROOT=Path("/home/lhohy/Code/Poseidon");BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v1")
OLD=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-source-live-v2/boolean_padding_residual__declared_shallow_v1/tasks/38dab79ea7b0419a91be6bd8612569b3")
from benchmark_graph import canonical,digest,require
def read(p):return json.loads(Path(p).read_text())
def write(p,v):require(not Path(p).exists(),"preserve_followup_evidence");Path(p).write_bytes(canonical(v))
def main():
 started=time.monotonic()
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from candidate_bundle import check_bundle
 from candidate_contract import request_file_bytes
 from generation_receipt import verify
 app=application(WorkerContext.capture(OUT/"tasks"));task=read(OUT/"task-id.json")["task_id"]
 result=app.result(task);require(result["status"]=="succeeded","normal_component_package_required")
 meta=app._checked_task_metadata(task);qualification=read(app._dir(task)/"validation-0.json")
 package=app._dir(task)/"program";manifest=check_bundle(package)
 current=app.backend.identity_manifest();require(meta["identity"]==current,"recovery_identity_changed")
 proposal=read(ROOT/"docs/baseline/agent-mha-v16-live-proposal-r1313.json")
 case=next(c for c in proposal["cases"] if c["id"].startswith("boolean"))
 require(request_file_bytes(meta["request"])==Path(case["request_file"]).read_bytes(),"unchanged_request")
 answer=json.loads(read(OLD/"response-1.json")["raw"])
 require((package/"candidate.py").read_text()==answer["hecate_source"],"unchanged_real_agent_source")
 receipt=read(OLD/"generation-provenance.json")
 verify(receipt,meta["request"],answer["hecate_source"],generation_identity_sha256=proposal["source_stage_identity"]["generation"]["sha256"])
 require(manifest["agent_generated"] is False,"offline_replay_package_origin")
 import numpy as np
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from numerical_results import compare
 samples=[];refs=[];model=meta["request"]["model"]
 for index in range(3):
  rng=np.random.default_rng(20261008+index)
  values={s["name"]:rng.uniform(-.2,.2,s["shape"]).astype(np.float64) for s in model["inputs"]}
  a,b=math_ref(model,values),torch_ref(model,values)
  for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
  samples.append({key:value.tolist() for key,value in values.items()})
  refs.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
 comparisons=[];executions=[]
 for phase,inputs,reference in (("new-inputs",samples,refs),("repeat",[samples[0]],refs[:1])):
  execution=app.execute(task,inputs)
  actual=np.asarray([np.concatenate([np.asarray(s[o["name"]],dtype=np.float64).reshape(-1) for o in model["outputs"]]) for s in execution["outputs"]])
  compared=compare(actual,np.asarray(reference),1e-5,1e-4)
  require(compared["passed"] and execution["encrypted_execution"] is True and execution["executor_secret_key"] is False
    and execution["agent_calls"]==execution["compiler_calls"]==0,"reloaded_package_numeric_execution")
  require(check_bundle(package)==manifest,"package_not_modified")
  write(OUT/(phase+".json"),dict(execution=execution,comparison=compared,input_groups=len(inputs)))
  comparisons.append(compared);executions.append(execution["evidence"])
 another=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],trust_local_code=True),
  resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2",chunk_period=4),
  limits=dict(generations=1,http_attempts=1,wall_seconds=600,api_timeout_seconds=1200,max_tokens=384000))
 class NeverGenerate:
  kind="scripted_replay";agent_calls=0
  def generate(self,*args):raise AssertionError("cached_package_required")
 cached=app.run(another,NeverGenerate())
 require(cached["status"]=="succeeded" and cached["package"]["reused"] is True and cached["generations"]==cached["http_attempts"]==0,"normal_component_cache_reuse")
 write(OUT/"cache.json",cached)
 report=dict(format="poseidon-mha-exact-agent-delivery-recovery-v2",status="passed",
  source_kind="same_batch_real_agent_source_offline_replay",new_agent_generation=False,paid_calls=0,http_attempts=0,
  original_generation_receipt_binding=receipt["binding"],original_generation_identity=proposal["source_stage_identity"]["generation"]["sha256"],
  original_generation_receipt_remains_in_old_task=True,package_origin="scripted_replay; verified prior real-Agent source retained separately",
  old_identity_direct_export_rejected=True,old_identity_export_attempt_preserved=str(OUT/"failure.json"),
  current_stage_identity=current,normal_application_result=result,request_id=case["request_id"],candidate_sha256=__import__("hashlib").sha256((package/"candidate.py").read_bytes()).hexdigest(),
  package=str(package),package_binding=manifest["binding"],qualification=qualification,
  execution_evidence=executions,new_inputs_comparison=comparisons[0],repeat_comparison=comparisons[1],cache_reused=True,
  input_groups=8,compared_values=qualification["compared_values"]+sum(c["compared_values"] for c in comparisons),
  max_absolute_error=max(qualification["max_absolute_error"],*(c["max_absolute_error"] for c in comparisons)),
  source_unchanged=True,model_weights_reference_parameters_unchanged=True,
  original_application_accuracy_certified=False,elapsed_followup_seconds=time.monotonic()-started)
 report["binding"]=digest(report);write(OUT/"report.json",report)
 print(json.dumps({k:report[k] for k in ("status","binding","input_groups","compared_values","max_absolute_error","paid_calls")}),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:
  write(OUT/"followup-failure.json",dict(error_type=type(e).__name__,diagnostic=str(e)[:600],paid_calls=0))
  raise
