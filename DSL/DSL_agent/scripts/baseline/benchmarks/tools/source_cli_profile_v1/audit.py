"""Independent evidence/reference audit for CLI source packages; no execution calls."""
import json,math,sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(HERE),str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
import qualify as scope
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle,check_execution_dependencies
from benchmark_math import evaluate
from numerical_results import compare
from program_execution import named_outputs
p=scope.plan();scope.identity(p);C=scope.CAMPAIGN
report=scope.read(C/"report.json");scope.common.bound(report)
terminal=scope.read(C/"terminal.json")
assert terminal["exit_code"]==0 and not terminal["owned_background"]
assert report["status"]=="complete"and report["passed"]==3
assert report["failed"]==report["not_run"]==report["skipped"]==report["api_calls"]==0
owner=scope.read(C/"owner.json");assert live.process_info(owner["pid"])is None
rows=[];groups=values=0;maximum=0.
for row in report["cases"]:
 name=row["case"];case=next(c for c in p["cases"]if c["id"]==name)
 package=Path(row["package"]);manifest,q=check_execution_dependencies(package)
 assert check_bundle(package)==manifest and q==scope.read(C/name/"request.json")
 assert manifest["agent_generated"]is False and manifest["validation"]["level"]=="numerical"
 assert row["result"]["http_attempts"]==0 and row["result"]["generation"]["agent_generated"]is False
 assert (package/"candidate.py").read_bytes()==(C/name/"rule.py").read_bytes()
 assert "model_source"in manifest["origin"]
 artifacts=manifest["files"]
 assert all(any(n.endswith(s)for n in artifacts)for s in
  ("earth.mlir","ckks.mlir","runtime-plan.json","operator-spec.json"))
 assert any("/bound/bundle/manifest.json"in n for n in artifacts)
 task=package.parent;model=q["model"];parent=scope.read(case["parent_request"])["model"]
 def flatten(output):
  return np.concatenate([np.asarray(output[o["name"]],dtype=np.float64).reshape(-1)for o in model["outputs"]])
 validations=[scope.read(f)for f in sorted(task.glob("validation-*.json"))]
 assert len(validations)==1
 passed=validations[0];assert passed["status"]=="passed"and passed["encrypted_execution"]is True
 ev=Path(passed["evidence"]);record=scope.read(ev/"report.json")
 assert record["backend"]=="poseidon_cpu"and record["poseidon_gpu_validated"]is False
 assert record["tolerance"]==dict(atol=1e-5,rtol=1e-4)
 attempts=[a for a in record["attempts"]if a.get("numerically_correct")is True]
 assert len(attempts)==1;a=attempts[0]
 assert all(a.get(k)is True for k in("compiled","executed","numerically_correct"))
 gate=a["artifact_gate"]
 assert gate["raw"]["format"]==gate["bound"]["format"]=="poseidon-artifact-gate-v2"
 assert gate["raw"]["security"]==gate["bound"]["security"]=="tc128"
 assert gate["binding"]["source_plan_sha256"]==gate["raw"]["plan_sha256"]
 assert gate["binding"]["plan_sha256"]==gate["bound"]["plan_sha256"]
 actual_path=ev/("attempt-%02d"%a["index"])/"output/decrypted.npy"
 with np.load(ev/"arrays.npz",allow_pickle=False)as arrays:
  raw=np.array(arrays["inputs"],copy=True);frozen=np.array(arrays["reference"],copy=True)
 layout=q["layout"]
 assert scope.read(ev/"logical-bindings.json")["inputs"]==layout["inputs"]
 expected=[]
 for sample in raw:
  vectors=sample[None,:]if sample.ndim==1 else sample
  assert vectors.shape==(len(layout["inputs"]),layout["input_slot_period"])
  flats={d["name"]:np.zeros(math.prod(d["shape"]),dtype=np.float64)for d in model["inputs"]}
  cover={n:np.zeros(len(v),dtype=np.bool_)for n,v in flats.items()}
  for binding,vector in zip(layout["inputs"],vectors):
   n=binding["name"];start=binding.get("offset",0);count=binding["elements"]
   assert 0<=start and 0<count<=len(vector)and start+count<=len(flats[n])
   assert not cover[n][start:start+count].any()and np.all(vector[count:]==0)
   flats[n][start:start+count]=vector[:count];cover[n][start:start+count]=True
  assert all(v.all()for v in cover.values())
  inputs={d["name"]:flats[d["name"]].reshape(d["shape"])for d in model["inputs"]}
  fresh=flatten(evaluate(model,inputs));old=flatten(evaluate(parent,inputs))
  np.testing.assert_allclose(fresh,old,atol=1e-12,rtol=1e-12)
  expected.append(fresh)
 expected=np.asarray(expected);np.testing.assert_allclose(expected,frozen,atol=1e-12,rtol=1e-12)
 decoded=named_outputs(q,np.load(actual_path,allow_pickle=False),len(raw))
 comparison=compare(np.asarray([flatten(v)for v in decoded]),expected,1e-5,1e-4)
 assert comparison["passed"]and len(raw)==4
 groups+=4;values+=comparison["compared_values"];maximum=max(maximum,comparison["max_absolute_error"])
 checks=[dict(phase="initial",groups=4,values=comparison["compared_values"],
  maximum_absolute_error=comparison["max_absolute_error"],actual_sha256=scope.common.sha(actual_path))]
 for phase,count in(("new_inputs",3),("repeat",1)):
  receipt=scope.read(C/name/(phase+".json"));execution=receipt["execution"];expected=[]
  for i in range(count):
   rng=np.random.default_rng(20261008+i)
   inputs={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
   fresh=flatten(evaluate(model,inputs));old=flatten(evaluate(parent,inputs))
   np.testing.assert_allclose(fresh,old,atol=1e-12,rtol=1e-12);expected.append(fresh)
  assert execution["encrypted_execution"]is True and execution["agent_calls"]==execution["compiler_calls"]==0
  assert execution["executor_secret_key"]is False
  c=compare(np.asarray([flatten(v)for v in execution["outputs"]]),np.asarray(expected),1e-5,1e-4)
  assert c["passed"];groups+=count;values+=c["compared_values"];maximum=max(maximum,c["max_absolute_error"])
  assert receipt["comparison"]==c
  checks.append(dict(phase=phase,groups=count,values=c["compared_values"],
   maximum_absolute_error=c["max_absolute_error"],receipt_sha256=scope.common.sha(C/name/(phase+".json"))))
 cached=scope.read(C/name/"cache.json")
 assert cached["status"]=="succeeded"and cached["package"]["reused"]is True
 assert cached["package"]["binding"]==manifest["binding"]and cached["generations"]==cached["http_attempts"]==0
 assert not list((task.parent/cached["task_id"]).glob("validation-*.json"))
 rows.append(dict(case=name,request_id=q["request_id"],package_binding=manifest["binding"],
  origin="deterministic_rule",source_origin_preserved=True,compiled=True,
  encrypted_execution=True,gate_checked=True,cache_verified=True,parent_reference_equivalent=True,checks=checks))
assert groups==24 and report["compiler_calls_confirmed"]==3 and report["encrypted_calls_confirmed"]==9
result=dict(format="poseidon-source-cli-independent-audit-v1",status="passed",
 plan_binding=p["binding"],report_binding=report["binding"],sdk=p["sdk"],cases=rows,
 input_groups=groups,output_values=values,maximum_absolute_error=maximum,
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),
 new_compiler_calls=0,new_encrypted_calls=0,new_provider_calls=0,tool_sha256=scope.common.sha(Path(__file__)))
result["binding"]=scope.digest(result);scope.write(C/"audit.json",result)
print(json.dumps({k:result[k]for k in("status","input_groups","output_values","maximum_absolute_error","binding")}))
