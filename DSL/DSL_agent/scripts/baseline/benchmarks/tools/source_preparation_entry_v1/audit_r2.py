"""Recheck current package repeats against unchanged independent approximate math."""
from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
import numpy as np
from task_numerical_assessment import mathematical
from numerical_results import compare
report=q.read(HERE/"acceptance.json");q.bound(report)
O=q.RESULTS/"agent-source-preparation-cli-poseidon-r1643";rows=[]
for case in report["cases"]:
 folder=O/case["case"];task=case["task_id"]
 result=q.read(folder/"result.json");freeze=q.read(folder/"freeze/freeze.json");q.bound(freeze)
 metadata=q.read(folder/"tasks"/task/"task.json")
 source=q.read(folder/"tasks"/task/"source/manifest.json");model=metadata["request"]["model"]
 q.require(source["approximations"]==freeze["configuration"],"normal_application_frozen_configuration")
 q.require(result["acceptance"]["level"]=="numerical"and result["acceptance"]["input_groups"]==4,"qualified_four_groups")
 inputs=q.read(folder/"observe-request.json")["samples"]
 new=q.read(folder/"new-input-execution.json");repeat=q.read(folder/"repeat-execution.json")
 q.require(new["status"]==repeat["status"]=="executed","package_execution")
 expected=[mathematical(model,{n:np.asarray(v,dtype=np.float64)for n,v in s.items()})for s in inputs]
 def comparison(actual,reference):
  first=[];second=[]
  for got,want in zip(actual,reference):
   for output in model["outputs"]:
    name=output["name"];a=np.asarray(got[name],dtype=np.float64);b=np.asarray(want[name],dtype=np.float64)
    q.require(a.size==b.size,"repeat_output_shape")
    first.extend(a.reshape(-1).tolist());second.extend(b.reshape(-1).tolist())
  return compare(np.asarray(first).reshape(1,-1),np.asarray(second).reshape(1,-1),1e-5,1e-4)
 one=comparison(new["outputs"],expected);two=comparison(repeat["outputs"],expected[:1])
 q.require(one["passed"]and two["passed"],"frozen_numeric_limits_on_new_and_repeat")
 rows.append(dict(case=case["case"],status="passed",input_groups=4,
  compared_values=result["acceptance"]["compared_values"],
  maximum_absolute_error=result["acceptance"]["maximum_absolute_error"],
  new_input_maximum_absolute_error=one["max_absolute_error"],
  repeat_maximum_absolute_error=two["max_absolute_error"],
  frozen_configuration_matches_application=True,repeat_matches_same_independent_reference=True))
audit=dict(format="poseidon-source-preparation-cli-poseidon-audit-v1",status="passed",passed=3,
 failed=0,skipped=0,cases=rows,acceptance_binding=report["binding"],
 new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
 maximum_absolute_error=max(r["maximum_absolute_error"]for r in rows),
 maximum_new_or_repeat_absolute_error=max(max(r["new_input_maximum_absolute_error"],r["repeat_maximum_absolute_error"])for r in rows))
audit["binding"]=q.digest(audit);q.write(HERE/"audit.json",audit)
print(json.dumps(audit),flush=True)
