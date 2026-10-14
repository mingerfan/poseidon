"""Recompute frozen decrypted evidence and references; no backend or API calls."""
import sys,json,shlex,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as s
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",sys.argv[-1]])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import numpy as np
from benchmark_math import evaluate
from program_execution import named_outputs
from numerical_results import compare
mode=sys.argv[-1];plan=s.verify(mode);out=Path(plan["campaign"]);report=s.read(out/"report.json");s.bound(report)
assert s.read(out/"terminal.json")["exit_code"]==(0 if report["failed"]==0 else 1) and report["status"]=="complete" and report["not_run"]==report["skipped"]==0
aud=s.read(out/"audit-closed.json");s.bound(aud);assert aud["status"]=="passed"and aud["report_binding"]==report["binding"]
groups=values=0;maximum=0;rows=[]
for row in report["cases"]:
 if row["status"]!="passed":continue
 name=row["case"];case=next(c for c in plan["cases"]if c["id"]==name)
 q=s.read(case["request_file"]);model=q["model"];task=Path(row["package"]).parent
 def flatten(outputs):return np.concatenate([np.asarray(outputs[d["name"]],dtype=np.float64).reshape(-1)for d in model["outputs"]])
 validations=[s.read(p)for p in sorted(task.glob("validation-*.json"))]
 passed=next(v for v in validations if v.get("encrypted_execution")is True and v["status"]=="passed")
 ev=Path(passed["evidence"]);evidence=s.read(ev/"report.json")
 assert evidence["backend"]=="poseidon_cpu"and evidence["tolerance"]==dict(atol=1e-5,rtol=1e-4)
 index=next(a["index"]for a in evidence["attempts"]if a.get("numerically_correct")is True)
 actual_path=ev/("attempt-%02d"%index)/"output/decrypted.npy"
 with np.load(ev/"arrays.npz",allow_pickle=False)as data:
  raw_inputs=np.array(data["inputs"],copy=True);frozen=np.array(data["reference"],copy=True)
 layout=q["layout"]
 recorded_layout=s.read(ev/"logical-bindings.json")
 assert recorded_layout["inputs"]==layout["inputs"]
 expected=[]
 for samples in raw_inputs:
  vectors=samples[None,:] if samples.ndim==1 else samples
  assert vectors.shape==(len(layout["inputs"]),layout["input_slot_period"])
  flats={d["name"]:np.zeros(math.prod(d["shape"]),dtype=np.float64)for d in model["inputs"]}
  covered={n:np.zeros(len(v),dtype=np.bool_)for n,v in flats.items()}
  for binding,vector in zip(layout["inputs"],vectors):
   input_name=binding["name"];start=binding.get("offset",0);count=binding["elements"]
   assert 0<=start and 0<count<=len(vector) and start+count<=len(flats[input_name])
   assert not covered[input_name][start:start+count].any() and np.all(vector[count:]==0)
   flats[input_name][start:start+count]=vector[:count];covered[input_name][start:start+count]=True
  assert all(v.all()for v in covered.values())
  inputs={d["name"]:flats[d["name"]].reshape(d["shape"])for d in model["inputs"]}
  expected.append(flatten(evaluate(model,inputs)))
 expected=np.asarray(expected,dtype=np.float64)
 np.testing.assert_allclose(expected,frozen,atol=1e-12,rtol=1e-12)
 decoded=named_outputs(q,np.load(actual_path,allow_pickle=False),len(raw_inputs))
 compared=compare(np.asarray([flatten(v)for v in decoded]),expected,1e-5,1e-4)
 assert compared["passed"];groups+=len(raw_inputs);values+=compared["compared_values"];maximum=max(maximum,compared["max_absolute_error"])
 checks=[dict(phase="initial",groups=len(raw_inputs),values=compared["compared_values"],
  actual_sha256=s.sha(actual_path),frozen_reference_sha256=s.sha(ev/"arrays.npz"),maximum_absolute_error=compared["max_absolute_error"])]
 if name in s.REUSE_CASES:
  for phase,count in (("new_inputs",3),("repeat",1)):
   receipt=s.read(out/name/(phase+".json"));expected=[]
   for i in range(count):
    rng=np.random.default_rng(20261008+i)
    inputs={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
    expected.append(flatten(evaluate(model,inputs)))
   execution=receipt["execution"]
   assert execution["encrypted_execution"]is True and execution["agent_calls"]==execution["compiler_calls"]==0 and execution["executor_secret_key"]is False
   c=compare(np.asarray([flatten(v)for v in execution["outputs"]]),np.asarray(expected),1e-5,1e-4)
   assert c["passed"];groups+=count;values+=c["compared_values"];maximum=max(maximum,c["max_absolute_error"])
   checks.append(dict(phase=phase,groups=count,values=c["compared_values"],receipt_sha256=s.sha(out/name/(phase+".json")),
     maximum_absolute_error=c["max_absolute_error"]))
 rows.append(dict(case=name,checks=checks))
assert (groups,values)==(aud["input_groups"],aud["output_values"])
result=dict(format="poseidon-public-mask-closed-independent-numerical-audit-v1",status="passed",mode=mode,
 qualification_status=aud["qualification_status"],planned_cases=aud["planned_cases"],
 qualified_cases=aud["qualified_cases"],excluded_failed_cases=aud["failed_cases"],
 plan_binding=plan["binding"],report_binding=report["binding"],qualification_audit_binding=aud["binding"],sdk=plan["sdk"],
 input_groups=groups,output_values=values,maximum_absolute_error=maximum,cases=rows,
 reference="independent NumPy graph mathematics on frozen inputs; fixed new-input seeds",
 independent_tool_sha256=s.sha(Path(__file__)),new_compiler_calls=0,new_encrypted_calls=0,new_provider_calls=0)
result["binding"]=s.digest(result);s.write(out/"numerical-audit-closed.json",result)
print(json.dumps({k:result[k]for k in("status","mode","input_groups","output_values","maximum_absolute_error","binding")}))
