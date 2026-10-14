"""Development probe of fixed public tensor joins; no compiler, FHE or API."""
import sys,json,shlex,tempfile,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=120,environment="poseidon-cpu"))
import torch
from pytorch_public_expression_capture import prepare,check_record
from pytorch_public_expression_flow import _arrays
from pytorch_weight_package import export_state,binding
class Model(torch.nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  self.register_buffer("w",torch.tensor([[.25,-.125]],dtype=torch.float64))
  self.register_buffer("mask",torch.tensor([[True,False]],dtype=torch.bool))
 def forward(self,x):
  if self.kind=="zeros_like":return x+torch.zeros_like(x)
  if self.kind=="ones_like":return x+torch.ones_like(x)
  if self.kind=="full_like":return x*torch.full_like(x,.375)
  if self.kind=="new_zeros":return x+x.new_zeros(x.shape)
  if self.kind=="new_ones":return x*x.new_ones(x.shape)
  if self.kind=="new_full":return x+x.new_full(x.shape,-.125)
  if self.kind=="where":return torch.where(self.mask,x,self.w)
  if self.kind=="tensor_where":return x.where(self.mask,self.w)
  if self.kind=="masked_fill":return x.masked_fill(self.mask,-.125)
  if self.kind=="masked_fill_":return x.clone().masked_fill_(self.mask,.25)
  if self.kind=="cat":return torch.cat((x,self.w),dim=0)
  if self.kind=="cat_closed":return x+torch.cat((self.w,self.w*.5),dim=0)
  if self.kind=="stack_closed":return x+torch.stack((self.w[0],self.w[0]*.5),dim=0)
  if self.kind=="cat_boolean":return torch.cat((x,self.mask),dim=0)
  return torch.stack((x,self.w),dim=0)
records=[]
for kind in ("cat","stack"):
 module=Model(kind).eval()
 try:
  with tempfile.TemporaryDirectory()as td:
   p=Path(td);(p/"w").mkdir();(p/"out").mkdir()
   weights=export_state(module,p/"w",resource_profile="general-tensors-v1")
   spec=dict(format="poseidon-trusted-pytorch-v33",files={"model.py":hashlib.sha256(b"def make(): return None").hexdigest()},
    entry="model.make",inputs=[dict(name="x",shape=[2,2] if kind in ("cat_closed","stack_closed") else [1,2])],outputs=["y"],factory=dict(args=[],kwargs={}),
    call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}),weights=weights,
    dtype_policy="strict-float64-v1",approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
   a=_arrays(module);record=prepare(module,spec,p/"out",expected_weight_binding=binding(a,weights))
   check_record(record,spec,a,expected_record_binding=record["binding"])
  records.append(dict(operation=kind,status="reference_and_replay_passed"))
 except (ValueError,TypeError,RuntimeError,KeyError,IndexError)as e:
  import traceback; records.append(dict(operation=kind,status="rejected",error=str(e)[:300],traceback=traceback.format_exc()))
out=dict(format="poseidon-public-join-intake-probe-v1",cases=records,api_calls=0,compiler_calls=0,encrypted_calls=0)
with Path(__file__).with_name("probe-traceback-before.json").open("x") as f:f.write(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
