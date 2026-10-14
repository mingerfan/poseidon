from pathlib import Path
import sys,shlex,unittest,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
import qualification_scope as q
import numpy as np
from benchmark_graph import samples
from benchmark_math import evaluate
from unified_capacity_lowering import lower
from test_unified_capacity_linear import interpreted_output
D=q.RESULTS/"agent-normalization-source-poseidon-r1690"
for c in q.read(D/"report.json")["cases"]:
 request=q.read(D/c["case"]/"request.json");model=request["model"]
 expected=[];inputs_set=samples(model,4)
 inputs_set.append({i["name"]:np.random.default_rng(1690).normal(size=i["shape"])for i in model["inputs"]})
 for inp in inputs_set:
  refs=evaluate(model,inp)
  expected.append(np.concatenate([refs[o["name"]].reshape(-1)for o in model["outputs"]]))
 row={"case":c["case"]}
 for flag in(False,True):
  if flag:
   v=q.read(D/c["case"]/"tasks"/c["task_id"]/"validation-0.json")
   source=(Path(v["evidence"])/"attempt-00/candidate.py").read_text()
  else:source,record=lower(request,semantic_pairs=False)
  errors=[]
  for inp,ref in zip(inputs_set,expected):
   actual=interpreted_output(source,request,inp)
   errors.append(float(np.max(np.abs(actual-ref))))
  row["raw"if not flag else"scheduled"]=max(errors)
 print(json.dumps(row),flush=True)
