from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1")]
import qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
import numpy as np
from task_numerical_assessment import application_metric
rows=[]
for original_correct,encrypted_correct in ((96,95),(95,94),(100,99)):
 def logits(correct):
  array=np.zeros((100,2),dtype=np.float64);array[:correct,0]=1.;array[correct:,1]=1.;return [dict(out=array)]
 profile=dict(kind="top1_accuracy_drop",output="out",class_axis=1,labels=[[0]*100],max_drop=.01)
 result=application_metric(profile,logits(original_correct),logits(encrypted_correct),logits(encrypted_correct),{})
 rows.append(dict(original_correct=original_correct,encrypted_correct=encrypted_correct,expected_drop=.01,
  expected_pass=True,observed_pass=result["passed"],observed_drop=result["accuracy_drop"]))
record=dict(format="poseidon-current-application-metric-boundary-audit-v1",cases=rows,
 incorrect_boundary_cases=sum(not r["observed_pass"]for r in rows),api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=q.digest(record);q.write(HERE/"report.json",record);print(json.dumps(record))
