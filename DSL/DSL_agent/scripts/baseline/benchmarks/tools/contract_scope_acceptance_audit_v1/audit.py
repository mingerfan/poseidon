"""Independent frozen-array/source/package audit. No new compile, FHE or Provider calls."""
import sys,json,math,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2]
sys.path[:0]=[str(B/"benchmarks/tools/contract_scope_backend_v1"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from application_backend import PoseidonBackend
def verify():
 p=q.read(HERE/"scope.json");q.bound(p)
 for path,want in p["frozen_files"].items():q.require(q.sha(path)==want,"audit_frozen:"+path)
 q.require(PoseidonBackend().identity_manifest()==q.read(HERE/"identity.json"),"audit_sdk")
 return p
def inside():
 import numpy as np
 from candidate_bundle import check_bundle
 from provenance_json_transport import unpack,raw
 from benchmark_math import evaluate
 from benchmark_torch import evaluate as torch_evaluate
 from program_execution import named_outputs
 from numerical_results import compare
 p=verify();rows=[];totals={True:[0,0,0.],False:[0,0,0.]}
 for c in p["cases"]:
  request=q.read(c["request_file"]);model=request["model"];task=Path(c["task"])
  def flatten(outputs):return np.concatenate([np.asarray(outputs[d["name"]],dtype=np.float64).reshape(-1)for d in model["outputs"]])
  checks=[];package_info=None
  if c["expected_pass"]:
   package=task/"program";bundle=check_bundle(package)
   q.require(bundle["agent_generated"]is False,"replay_not_new_agent")
   q.require((package/"candidate.py").read_bytes()==Path(c["candidate"]).read_bytes(),"candidate_origin")
   package_info=dict(binding=bundle["binding"],origin=c["origin"],agent_generated=False)
  validations=[q.read(v)for v in sorted(task.glob("validation-*.json"))]
  v=next(v for v in validations if v.get("encrypted_execution")is True and v.get("observed_stages",{}).get("compiled")is True)
  ev=Path(v["evidence"]);evidence=q.read(ev/"report.json")
  q.require(evidence["backend"]=="poseidon_cpu"and evidence["tolerance"]==dict(atol=1e-5,rtol=1e-4),"actual_backend_tolerance")
  attempt=next(a for a in evidence["attempts"]if "numerically_correct"in a)
  actual_path=ev/("attempt-%02d"%attempt["index"])/"output/decrypted.npy"
  with np.load(ev/"arrays.npz",allow_pickle=False)as data:
   samples=np.array(data["inputs"],copy=True);frozen=np.array(data["reference"],copy=True)
  layout=request["layout"];q.require(q.read(ev/"logical-bindings.json")["inputs"]==layout["inputs"],"input_binding")
  expected=[]
  for sample in samples:
   vectors=sample[None,:]if sample.ndim==1 else sample
   q.require(vectors.shape==(len(layout["inputs"]),layout["input_slot_period"]),"physical_input_shape")
   flats={d["name"]:np.zeros(math.prod(d["shape"]),dtype=np.float64)for d in model["inputs"]}
   seen={k:np.zeros(len(v),dtype=np.bool_)for k,v in flats.items()}
   for binding,vector in zip(layout["inputs"],vectors):
    name=binding["name"];start=binding.get("offset",0);count=binding["elements"]
    q.require(0<=start and 0<count<=len(vector)and start+count<=len(flats[name])and
      not seen[name][start:start+count].any()and np.all(vector[count:]==0),"tail_binding")
    flats[name][start:start+count]=vector[:count];seen[name][start:start+count]=True
   q.require(all(x.all()for x in seen.values()),"input_coverage")
   inputs={d["name"]:flats[d["name"]].reshape(d["shape"])for d in model["inputs"]}
   a,t=evaluate(model,inputs),torch_evaluate(model,inputs)
   np.testing.assert_allclose(flatten(a),flatten(t),atol=1e-12,rtol=1e-12);expected.append(flatten(a))
  expected=np.asarray(expected);np.testing.assert_allclose(expected,frozen,atol=1e-12,rtol=1e-12)
  actual=named_outputs(request,np.load(actual_path,allow_pickle=False),len(samples))
  cmp=compare(np.asarray([flatten(x)for x in actual]),expected,1e-5,1e-4)
  q.require(cmp["passed"]is c["expected_pass"]and attempt["numerically_correct"]is c["expected_pass"],"expected_numeric_status")
  checks.append(dict(phase="initial",groups=len(samples),values=cmp["compared_values"],passed=cmp["passed"],
   maximum_absolute_error=cmp["max_absolute_error"],actual_sha256=q.sha(actual_path),frozen_sha256=q.sha(ev/"arrays.npz")))
  for phase,path,count in c["reuse_receipts"]:
   receipt=q.read(path)
   q.require(set(receipt)in({"status","execution","comparison"},{"status","execution","comparison","binding"})
    and receipt["status"]=="passed"and q.sha(path)==p["frozen_files"][path],"receipt_schema_and_hash")
   if "binding"in receipt:q.bound(receipt)
   execution=receipt["execution"]
   q.require(execution["encrypted_execution"]is True and execution["agent_calls"]==execution["compiler_calls"]==0 and
    execution["executor_secret_key"]is False,"reuse_real_execution")
   expected=[]
   for i in range(count):
    rng=np.random.default_rng(20261008+i)
    inputs={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
    a,t=evaluate(model,inputs),torch_evaluate(model,inputs)
    np.testing.assert_allclose(flatten(a),flatten(t),atol=1e-12,rtol=1e-12);expected.append(flatten(a))
   cmp=compare(np.asarray([flatten(x)for x in execution["outputs"]]),np.asarray(expected),1e-5,1e-4)
   q.require(cmp["passed"],"reuse_numerical")
   checks.append(dict(phase=phase,groups=count,values=cmp["compared_values"],passed=True,
    maximum_absolute_error=cmp["max_absolute_error"],receipt_sha256=q.sha(path)))
  for x in checks:
   totals[x["passed"]][0]+=x["groups"];totals[x["passed"]][1]+=x["values"];totals[x["passed"]][2]=max(totals[x["passed"]][2],x["maximum_absolute_error"])
  rows.append(dict(case=c["id"],origin=c["origin"],expected_pass=c["expected_pass"],checks=checks,package=package_info))
 verify()
 q.require(totals[True][:2]==p["expected_pass_totals"]and totals[False][:2]==p["expected_fail_totals"],"complete_frozen_scope")
 result=dict(format="poseidon-capacity-shallow-independent-acceptance-audit-v1",status="passed",
  scope_binding=p["binding"],sdk=p["sdk"],cases=rows,positive_input_groups=totals[True][0],positive_output_values=totals[True][1],
  maximum_absolute_error=totals[True][2],retained_failed_input_groups=totals[False][0],retained_failed_output_values=totals[False][1],
  numerical_tolerance=dict(atol=1e-5,rtol=1e-4),independent_reference_tolerance=dict(atol=1e-12,rtol=1e-12),
  new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,backend_batch_binding=p["backend_batch_binding"])
 result["binding"]=q.digest(result);q.write(HERE/"report.json",result)
 print(json.dumps({k:v for k,v in result.items()if k!="cases"}))
 return 0
def outer():
 p=verify();live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=p["budget"]["wall_seconds"],environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
