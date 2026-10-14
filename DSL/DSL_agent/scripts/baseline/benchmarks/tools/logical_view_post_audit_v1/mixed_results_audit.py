"""Audit terminal mixed results; failures never count as numerical passes."""
import sys,json,shlex,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
SCOPE=BASE/"benchmarks/tools/logical_view_backend_v1"
sys.path[:0]=[str(BASE),str(SCOPE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import qualification_scope as s
import run as live
if "--inside" not in sys.argv:
 live.no_active()
 plan=s.verify("live");out=Path(plan["campaign"])
 assert (out/"terminal.json").exists(),"Terminal checkpoint required; no concurrent Nix entry."
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
import numpy as np
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle,check_execution_dependencies
from candidate_contract import strict_json
from program_execution import named_outputs
from numerical_results import compare
from benchmark_math import evaluate
plan=s.verify("live");out=Path(plan["campaign"]);report=s.read(out/"report.json");s.bound(report)
terminal=s.read(out/"terminal.json")
assert terminal["plan_binding"]==report["plan_binding"]==plan["binding"]
assert report["status"]in("complete","failed","stopped") and report["sdk"]==plan["sdk"]
assert PoseidonBackend().identity_manifest()==s.read(SCOPE/"identity-snapshot.json")
calls=s.calls(s.states(out),plan["budget"]["max_generations"],plan["budget"]["max_http"])
assert calls["calls_resolved"],"Unsettled transport must be reviewed before any continuation."
observed_compiler=observed_encrypted=groups=values=0;maximum=0.;rows=[]
by_id={c["case"]:c for c in report["cases"]}
def flatten(model,outputs):
 return np.concatenate([np.asarray(outputs[d["name"]],dtype=np.float64).reshape(-1)for d in model["outputs"]])
for case in plan["cases"]:
 name=case["id"];row=by_id.get(name);folder=out/name
 validations=[]
 for p in sorted(folder.glob("tasks/*/validation-*.json")):
  v=s.read(p);observed=v.get("observed_stages",{})
  observed_compiler+=observed.get("compiled")is True;observed_encrypted+=v.get("encrypted_execution")is True
  validations.append((p,v))
 record=dict(case=name,status="not_run" if row is None else row["status"],
  execution_source_kind=case["execution_source_kind"],source_intake_failure=case["source_intake_failure"],
  numerical_pass=False,package_reloaded=False,cache_verified=False)
 if row is None or row["status"]!="passed":
  record.update(failure_layer=None if row is None else row.get("failure_layer"),
   failure=None if row is None else row.get("result",{}).get("failure"),
   validations=[dict(file=str(p),sha256=s.sha(p),failure=v.get("failure"),
    compiled=v.get("observed_stages",{}).get("compiled")is True,
    encrypted_execution=v.get("encrypted_execution")is True)for p,v in validations])
  rows.append(record);continue
 package=Path(row["package"]);manifest,request=check_execution_dependencies(package)
 assert check_bundle(package)==manifest and request==s.read(case["request_file"])
 assert manifest["agent_generated"]is True and manifest["validation"]["level"]=="numerical"
 files=manifest["files"]
 assert all(any(n.endswith(ext)for n in files)for ext in("earth.mlir","ckks.mlir","runtime-plan.json","operator-spec.json"))
 assert any("/bound/bundle/manifest.json"in n for n in files)
 task=package.parent;receipt=s.read(task/"generation-provenance.json")
 answer=strict_json(receipt["answers"][-1]["raw"]);candidate=(package/"candidate.py").read_bytes()
 assert answer["request_id"]==request["request_id"] and candidate==answer["hecate_source"].encode()
 assert s.sha(package/"candidate.py")==receipt["candidate_sha256"]
 assert receipt["answers"][-1]["feedback"]==dict(layer="complete",status="passed")
 passed=next(v for _,v in validations if v.get("encrypted_execution")is True and v["status"]=="passed")
 ev=Path(passed["evidence"]);evidence=s.read(ev/"report.json")
 assert evidence["backend"]=="poseidon_cpu" and evidence["tolerance"]==dict(atol=1e-5,rtol=1e-4)
 attempt=next(a for a in evidence["attempts"]if a.get("numerically_correct")is True)
 gate=attempt["artifact_gate"]
 assert gate["raw"]["security"]==gate["bound"]["security"]=="tc128"
 assert gate["binding"]["source_plan_sha256"]==gate["raw"]["plan_sha256"]
 assert gate["binding"]["plan_sha256"]==gate["bound"]["plan_sha256"]
 actual_path=ev/("attempt-%02d"%attempt["index"])/"output/decrypted.npy"
 with np.load(ev/"arrays.npz",allow_pickle=False)as arrays:
  raw_inputs=np.array(arrays["inputs"],copy=True);frozen=np.array(arrays["reference"],copy=True)
 model=request["model"];layout=request["layout"];expected=[]
 assert len(raw_inputs)==4
 for samples in raw_inputs:
  vectors=samples[None,:]if samples.ndim==1 else samples
  assert vectors.shape==(len(layout["inputs"]),layout["input_slot_period"])
  flats={d["name"]:np.zeros(math.prod(d["shape"]),dtype=np.float64)for d in model["inputs"]}
  covered={n:np.zeros(len(v),dtype=np.bool_)for n,v in flats.items()}
  for bind,vector in zip(layout["inputs"],vectors):
   n=bind["name"];start=bind.get("offset",0);count=bind["elements"]
   assert start>=0 and 0<count<=len(vector) and start+count<=len(flats[n])
   assert not covered[n][start:start+count].any() and np.all(vector[count:]==0)
   flats[n][start:start+count]=vector[:count];covered[n][start:start+count]=True
  assert all(x.all()for x in covered.values())
  inputs={d["name"]:flats[d["name"]].reshape(d["shape"])for d in model["inputs"]}
  expected.append(flatten(model,evaluate(model,inputs)))
 expected=np.asarray(expected,dtype=np.float64)
 np.testing.assert_allclose(expected,frozen,atol=1e-12,rtol=1e-12)
 actual=np.asarray([flatten(model,v)for v in named_outputs(request,np.load(actual_path,allow_pickle=False),4)])
 comparisons=[(compare(actual,expected,1e-5,1e-4),4)]
 if name in s.reuse_cases("live"):
  for phase,count in(("new_inputs",3),("repeat",1)):
   extra=s.read(folder/(phase+".json"));execution=extra["execution"];expected=[]
   assert execution["agent_calls"]==execution["compiler_calls"]==0 and execution["encrypted_execution"]
   assert execution["executor_secret_key"]is False
   observed_encrypted+=1
   for i in range(count):
    rng=np.random.default_rng(20261008+i)
    inputs={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
    expected.append(flatten(model,evaluate(model,inputs)))
   actual=np.asarray([flatten(model,v)for v in execution["outputs"]])
   comparisons.append((compare(actual,np.asarray(expected),1e-5,1e-4),count))
 cache=s.read(folder/"cache.json")
 assert cache["package"]["reused"] and cache["generations"]==cache["http_attempts"]==0
 for comp,count in comparisons:
  assert comp["passed"] and comp["atol"]==1e-5 and comp["rtol"]==1e-4
  groups+=count;values+=comp["compared_values"];maximum=max(maximum,comp["max_absolute_error"])
 record.update(numerical_pass=True,package_reloaded=True,cache_verified=True,package_binding=manifest["binding"],
  candidate_sha256=receipt["candidate_sha256"],comparison_count=len(comparisons))
 rows.append(record)
assert observed_compiler<=plan["budget"]["max_compiler_calls"] and observed_encrypted<=plan["budget"]["max_encrypted_calls"]
result=dict(format="poseidon-logical-view-mixed-results-audit-v1",audit_status="passed",
 evaluation_status=report["status"],plan_binding=plan["binding"],report_binding=report["binding"],
 sdk=plan["sdk"],planned=len(plan["cases"]),passed=sum(x["numerical_pass"]for x in rows),
 failed=sum(x["status"]=="failed"for x in rows),not_run=sum(x["status"]=="not_run"for x in rows),
 cases=rows,**calls,compiler_calls_confirmed=observed_compiler,encrypted_calls_confirmed=observed_encrypted,
 input_groups=groups,output_values=values,maximum_absolute_error=maximum,tool_sha256=s.sha(Path(__file__)),
 numerical_reference="independent NumPy mathematics; frozen inputs and fixed new-input seeds",
 new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,historical_records_rewritten=False)
result["binding"]=s.digest(result);p=HERE/"live-mixed-results-audit.json"
assert not p.exists(),"Preserve previous audit";s.write(p,result)
print(json.dumps({k:v for k,v in result.items()if k!="cases"},ensure_ascii=False))
