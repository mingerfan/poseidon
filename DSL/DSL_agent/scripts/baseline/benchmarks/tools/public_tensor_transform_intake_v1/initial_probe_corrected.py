"""Read-only probe of the three existing public-state transform rejection paths."""
import sys,json,shlex
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
if "--inside" not in sys.argv:
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=60,environment="poseidon-cpu"))
import torch
from pytorch_static_trace import trace,ALIAS_POLICY
from pytorch_public_value_ops import ARITHMETIC_POLICY
from pytorch_static_tensor_transforms import POLICY
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
class Model(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.register_buffer("public",torch.tensor([[.19,-.31]],dtype=torch.float64))
    def forward(self,x):
        if self.kind=="repeat":return x+self.public.repeat(2,1)
        if self.kind=="flip":return x+self.public.flip(-1)
        return x+self.public.roll(1,-1)
records=[]
for kind in ("repeat","flip","roll"):
    try:
        trace(Model(kind).eval(),[dict(name="x",shape=[2,2] if kind=="repeat" else [1,2])],
            resource_profile="general-tensors-v1",**TRACE_OPTIONS,
            public_expression_policy=ARITHMETIC_POLICY,tensor_transform_source_policy=POLICY)
    except ValueError as error:
        records.append(dict(operation=kind,status="rejected",error=str(error)))
    else:records.append(dict(operation=kind,status="accepted"))
out=dict(format="poseidon-public-transform-prechange-probe-v1",cases=records,api_calls=0,compiler_calls=0,encrypted_calls=0)
Path(__file__).with_name("initial-probe-corrected.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
