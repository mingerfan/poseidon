"""Exact real-Agent source replay; no Provider transport or credentials."""
import copy,hashlib,json,os,time,sys
from pathlib import Path
ROOT=Path("/home/lhohy/Code/Poseidon")
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
OUT=WORK/"results/agent-mha-live-delivery-recovery-v1"
OLD=WORK/"results/agent-mha-source-live-v2/boolean_padding_residual__declared_shallow_v1/tasks/38dab79ea7b0419a91be6bd8612569b3"
P=ROOT/"docs/baseline/agent-mha-v16-live-proposal-r1313.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from benchmark_graph import canonical,digest,require
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_bytes(canonical(v))
def main():
 require(not OUT.exists(),"fresh_recovery_root")
 OUT.mkdir(mode=0o700)
 started=time.monotonic()
 proposal=read(P);case=next(c for c in proposal["cases"] if c["id"].startswith("boolean"))
 frozen=read(case["request_file"])
 raw=read(OLD/"response-1.json")["raw"];answer=json.loads(raw)
 original_receipt=read(OLD/"generation-provenance.json")
 from generation_receipt import verify
 verified_origin=verify(original_receipt,frozen,answer["hecate_source"],
  generation_identity_sha256=proposal["source_stage_identity"]["generation"]["sha256"])
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from candidate_contract import request_file_bytes
 from candidate_bundle import check_bundle,export_bundle,check_execution_dependencies
 from application_identity import check_manifest,compatible
 app=application(WorkerContext.capture(OUT/"tasks"))
 current=app.backend.identity_manifest()
 class ExactReplay:
  kind="scripted_replay";agent_calls=0
  def __init__(self):self.calls=0
  def generate(self,request,*args):
   require(request_file_bytes(request)==Path(case["request_file"]).read_bytes(),"unchanged_replay_request")
   require(self.calls==0,"single_exact_replay")
   self.calls+=1
   return raw
 source=dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],trust_local_code=True)
 options=dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
 task=app.submit_source(source,resource_profile="general-tensors-v1",options=options,
  limits=dict(generations=1,http_attempts=1,wall_seconds=600,api_timeout_seconds=1200,max_tokens=384000))
 write(OUT/"task-id.json",dict(task_id=task))
 replay=ExactReplay();result=app.run(task,replay);write(OUT/"application-result.json",result)
 require(result["status"]=="succeeded","normal_component_delivery:"+str(result.get("failure")))
 meta=app._checked_task_metadata(task)
 require(request_file_bytes(meta["request"])==Path(case["request_file"]).read_bytes(),"same_mathematical_request")
 require(app.backend.identity_manifest()==current,"source_changed_during_recovery")
 package=app._dir(task)/"program";manifest=check_bundle(package)
 require((package/"candidate.py").read_text()==answer["hecate_source"],"original_candidate_unchanged")
 require(manifest["agent_generated"] is False and result["http_attempts"]==0,"honest_scripted_replay_label")
 qualification=read(app._dir(task)/"validation-0.json")
 require(qualification["compiled_validated"] and qualification["encrypted_execution"] and qualification["numerically_validated"],"real_numerical_qualification")
 mixed=copy.deepcopy(current);mixed["generation"]=copy.deepcopy(proposal["source_stage_identity"]["generation"])
 mixed["manifest_sha256"]=digest({k:v for k,v in mixed.items() if k!="manifest_sha256"})
 check_manifest(mixed);require(compatible(mixed,current),"historical_generation_current_execution")
 historic=OUT/"historical-agent-program"
 export_bundle(Path(qualification["evidence"]),historic,model_provenance=meta["model_provenance"],
  application_identities=mixed,generation_provenance=original_receipt,model_source=meta["model_source"])
 historic_manifest,historic_request=check_execution_dependencies(historic)
 require(historic_manifest["agent_generated"] is True and (historic/"candidate.py").read_text()==answer["hecate_source"],"retained_verified_agent_origin")
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
 execution=app.execute(task,samples)
 actual=np.asarray([np.concatenate([np.asarray(s[o["name"]],dtype=np.float64).reshape(-1) for o in model["outputs"]]) for s in execution["outputs"]])
 compared=compare(actual,np.asarray(refs),1e-5,1e-4)
 require(compared["passed"] and execution["encrypted_execution"] is True and execution["executor_secret_key"] is False
  and execution["agent_calls"]==execution["compiler_calls"]==0,"new_inputs_on_reloaded_package")
 write(OUT/"new-inputs.json",dict(execution=execution,comparison=compared,input_groups=3))
 from program_execution import execute_program
 repeated=execute_program(historic,[samples[0]])
 actual_repeat=np.asarray([np.concatenate([np.asarray(s[o["name"]],dtype=np.float64).reshape(-1) for o in model["outputs"]]) for s in repeated["outputs"]])
 compared_repeat=compare(actual_repeat,np.asarray(refs[:1]),1e-5,1e-4)
 require(compared_repeat["passed"] and repeated["encrypted_execution"] and repeated["executor_secret_key"] is False
  and repeated["agent_calls"]==repeated["compiler_calls"]==0,"historical_package_new_identity_execution")
 require(check_bundle(package)==manifest and check_bundle(historic)==historic_manifest,"package_integrity_after_reload")
 write(OUT/"repeat.json",dict(execution=repeated,comparison=compared_repeat,input_groups=1))
 another=app.submit_source(source,resource_profile="general-tensors-v1",options=options,
  limits=dict(generations=1,http_attempts=1,wall_seconds=600,api_timeout_seconds=1200,max_tokens=384000))
 class NeverGenerate:
  kind="scripted_replay";agent_calls=0
  def generate(self,*args):raise AssertionError("cached_package_required")
 cached=app.run(another,NeverGenerate())
 require(cached["status"]=="succeeded" and cached["package"]["reused"] is True and cached["generations"]==cached["http_attempts"]==0,"normal_component_cache_reuse")
 write(OUT/"cache.json",cached)
 values=sum(np.asarray(x).size for x in refs)
 report=dict(format="poseidon-mha-exact-agent-delivery-recovery-v1",status="passed",source_kind="same_batch_real_agent_source_offline_replay",
  new_agent_generation=False,paid_calls=0,http_attempts=0,request_id=case["request_id"],candidate_sha256=sha(package/"candidate.py"),
  original_generation_receipt_binding=original_receipt["binding"],original_generation_identity=proposal["source_stage_identity"]["generation"]["sha256"],
  current_stage_identity=current,historical_package_stage_identity=mixed,normal_application_result=result,
  package=str(package),package_binding=manifest["binding"],historical_agent_package=str(historic),historical_package_binding=historic_manifest["binding"],
  numerical_qualification=qualification,new_inputs=compared,repeat=compared_repeat,cache_reused=True,
  input_groups=8,compared_values=qualification["compared_values"]+values+np.asarray(refs[0]).size,
  max_absolute_error=max(qualification["max_absolute_error"],compared["max_absolute_error"],compared_repeat["max_absolute_error"]),
  original_application_accuracy_certified=False,source_unchanged=True,model_weights_reference_parameters_unchanged=True,
  elapsed_seconds=time.monotonic()-started)
 report["binding"]=digest(report);write(OUT/"report.json",report)
 print(json.dumps({k:report[k] for k in ("status","binding","input_groups","compared_values","max_absolute_error","elapsed_seconds","paid_calls")}),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:
  if OUT.exists():write(OUT/"failure.json",dict(error_type=type(e).__name__,diagnostic=str(e)[:600],provider_calls=0))
  raise
