"""Developer-owned public mask source forms; immutable models and weights."""
import sys,json,hashlib,shlex,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
import numpy as np,torch
from pytorch_weight_package import export_state
from pytorch_public_mask_source import SOURCE_FORMAT
def save(path,data):
 with path.open("x")as f:json.dump(data,f,indent=2,sort_keys=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
expressions={
 "where":["torch.where(self.mask,x,self.w)",
          "torch.where(condition=self.mask.reshape(1,2),input=x.square(),other=self.w)",
          "torch.where(self.mask.flip(0).reshape(2,1),-.125,z+x)"],
 "tensor_where":["x.where(self.mask,self.w)",
                 "self.w.float().double().where(self.mask.reshape(1,2),x.square())",
                 "(x+z).where(condition=self.mask.flip(0).reshape(2,1),other=self.w)"],
 "masked_fill":["x.masked_fill(self.mask,.125)",
                "(x.square()+self.w).masked_fill(self.mask.reshape(1,2),x.new_full((),-.125))",
                "(x+z).masked_fill(mask=self.mask.reshape(2,1),value=-.25)"],
 "masked_fill_":["base.masked_fill_(self.mask,.125)",
                 "base.masked_fill_(self.mask.reshape(1,2),self.fill)",
                 "base.masked_fill_(mask=self.mask.reshape(2,1),value=-.25)"],
 "public_where":["torch.where(self.mask,self.w,.25)",
                 "self.w.float().where(self.mask.reshape(1,2),-.125000017).double()",
                 "torch.where(self.mask.reshape(2,1),self.w.new_full(self.w.shape,2,dtype=torch.int64),False).double()"],
 "public_fill":["self.w.masked_fill(self.mask,.125)",
                "self.w.float().masked_fill(self.mask.reshape(1,2),self.w.new_full((),.125000017,dtype=torch.float32)).double()",
                "torch.ones_like(self.w,dtype=torch.int64).masked_fill(self.mask.reshape(2,1),-2).double()"]}
rows=[]
for number,(op,variants)in enumerate(expressions.items()):
 for context,expression in enumerate(variants):
  name=op+"_"+str(context);folder=HERE/"cases"/name;folder.mkdir(parents=True,exist_ok=False)
  shape=[4]if context==0 else[2,2]
  mask=[True,False,False,True]if context==0 else[True,False]
  rng=np.random.default_rng(2026101230+number*3+context)
  w=rng.uniform(-.4,.4,shape).tolist()
  state='        self.register_buffer("w",torch.tensor('+repr(w)+',dtype=torch.float64))\n'
  state+='        self.register_buffer("mask",torch.tensor('+repr(mask)+',dtype=torch.bool))\n'
  if context==1:
   proj=rng.uniform(-.3,.3,(2,2)).tolist();bias=rng.uniform(-.125,.125,(2,)).tolist()
   state+='        self.register_buffer("proj",torch.tensor('+repr(proj)+',dtype=torch.float64))\n'
   state+='        self.register_buffer("bias",torch.tensor('+repr(bias)+',dtype=torch.float64))\n'
   if op=="masked_fill_":state+='        self.register_buffer("fill",torch.tensor(.1875,dtype=torch.float64))\n'
  signature="x"if context<2 else"x,*,z"
  setup='        snapshot=x.clone()\n'
  if op=="masked_fill_":
   setup+='        base=x.clone()\n'if context==0 else'        base=x.view(2,2)\n'
  if op.startswith("public_"):
   body=("return x*(c+.125)+self.w"if context==0 else
         "return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x"if context==1 else
         "return {'y':x*(c+.125)+z+self.w,'saved':[snapshot,x,None]}")
  else:
   body=("return c*self.w+snapshot"if context==0 else
         "return F.linear(c*self.w,self.proj,self.bias)+snapshot"if context==1 else
         "return {'y':c+z+self.w,'saved':[snapshot,x,None]}")
  header="import torch\nfrom torch.nn import functional as F\n"
  if op=="tensor_where"and context==2:
   (folder/"helpers.py").write_text("def masked_selection(x,z,mask,w):\n    return (x+z).where(condition=mask.flip(0).reshape(2,1),other=w)\n")
   header+="from helpers import masked_selection\n";expression="masked_selection(x,z,self.mask,self.w)"
  source=header+"class Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n"+state+"    def forward(self,"+signature+"):\n"+setup+"        c="+expression+"\n        "+body+"\ndef make():\n    return Model().eval()\n"
  (folder/"model.py").write_text(source)
  env={}
  if(folder/"helpers.py").exists():
   exec(compile((folder/"helpers.py").read_text(),"developer_mask_fixture","exec"),env)
   source=source.replace("from helpers import masked_selection\n","")
  exec(compile(source,"developer_mask_fixture","exec"),env)
  module=env["make"]()
  with tempfile.TemporaryDirectory()as stage:
   weights=export_state(module,Path(stage),resource_profile="general-tensors-v1")
   for item in weights.values():
    with(folder/item["file"]).open("xb")as f:f.write((Path(stage)/item["file"]).read_bytes())
  inputs=[dict(name="x",shape=shape)]
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
  if context==2:inputs.append(dict(name="z",shape=shape));call["kwargs"]["z"]=dict(kind="tensor",name="z")
  manifest=dict(format=SOURCE_FORMAT,files={p.name:sha(p)for p in folder.glob("*.py")},entry="model.make",inputs=inputs,
   outputs=["y","snapshot","current"]if context==2 else["output"],call=call,factory=dict(args=[],kwargs={}),weights=weights,
   dtype_policy="strict-float64-v1",approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
  save(folder/"manifest.json",manifest)
  rows.append(dict(name=name,manifest=str(folder/"manifest.json"),kind="new_public_mask_context",operator=op))
prior=json.loads((BASE/"benchmarks/tools/public_constant_intake_v1/cases.json").read_text())
for name in("zeros_like_1","new_full_2","full_like_0"):
 original=next(c for c in prior if c["name"]==name)
 rows.append(dict(name="compat_v32_"+name,manifest=original["manifest"],kind="legacy_source_compatibility",operator=original["operator"]))
save(HERE/"cases.json",rows)
print(json.dumps(dict(cases=len(rows),new=18,compatibility=3,api_calls=0)))
