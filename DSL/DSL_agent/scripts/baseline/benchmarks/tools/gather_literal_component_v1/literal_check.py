"""Check usual public integer literal constructors against the v37 trace path."""
import sys,shlex,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
if sys.argv[1:]!=["--inside"]:
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import torch
from pytorch_public_expression_capture import TRACE_OPTIONS,source_policy
from pytorch_static_trace import trace
from gather_fx_normalization import POLICY,normalize
class Explicit(torch.nn.Module):
 def forward(self,x):
  return torch.gather(x,1,torch.tensor([[3,0],[1,2]],dtype=torch.int64))
class Inferred(torch.nn.Module):
 def forward(self,x):
  return x.gather(1,torch.tensor([[3,0],[1,2]]))
rows=[]
for name,module in(("explicit",Explicit()),("inferred",Inferred())):
 try:
  graph=trace(module,[dict(name="x",shape=[2,4])],resource_profile="general-tensors-v1",
    **TRACE_OPTIONS,public_expression_policy=source_policy(dict(format="poseidon-trusted-pytorch-v37")),
    tensor_gather_source_policy=POLICY)
  result,record=normalize(graph,[dict(name="x",shape=[2,4])],resource_profile="general-tensors-v1")
  x=torch.arange(8,dtype=torch.float64).reshape(2,4)
  torch.testing.assert_close(module(x),result(x),rtol=1e-12,atol=1e-12)
  rows.append(dict(case=name,status="passed"))
 except Exception as e:rows.append(dict(case=name,status="failed",type=type(e).__name__,diagnostic=str(e)[:220]))
record=dict(cases=rows,api_calls=0,compiler_calls=0,encrypted_calls=0)
with(HERE/"literal-check.json").open("x")as f:json.dump(record,f,sort_keys=True,indent=2)
print(json.dumps(record))
