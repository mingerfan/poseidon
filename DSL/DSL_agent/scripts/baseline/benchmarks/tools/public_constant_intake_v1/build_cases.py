"""Static public constructors in varied general source graphs."""
import sys,json,hashlib,shlex,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=120,environment="poseidon-cpu"))
import numpy as np,torch
from pytorch_weight_package import export_state
from pytorch_public_constant_source import SOURCE_FORMAT
rows=[]
def save(p,v):
 with p.open("x")as f:json.dump(v,f,sort_keys=True,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expressions={
 "zeros_like":["torch.zeros_like(x)","torch.zeros_like(x.square())","torch.zeros_like(input=x,dtype=torch.float32).double()"],
 "ones_like":["torch.ones_like(x)","torch.ones_like(x,dtype=torch.int64).double()","torch.ones_like(self.w.float(),dtype=torch.float32).double()"],
 "full_like":["torch.full_like(x,.125000017,dtype=torch.float32).double()","torch.full_like(x.roll(1,-1),-.375)","torch.full_like(x,True,dtype=torch.bool).double()"],
 "new_zeros":["x.new_zeros((1,))","x.new_zeros(size=(1,2))","x.new_zeros(())"],
 "new_ones":["x.new_ones((1,4)).reshape(4)","x.new_ones(x.shape,dtype=torch.int64).double()","x.new_ones(())"],
 "new_full":["x.new_full((1,),-.375)","x.new_full((1,2),.125000017,dtype=torch.float32).double()","self.w.new_full(x.shape,.25)"]}
for number,(op,variants)in enumerate(expressions.items()):
 for context,expression in enumerate(variants):
  name=op+"_"+str(context);folder=HERE/"cases"/name;folder.mkdir(parents=True,exist_ok=False)
  shape=[4]if context==0 else[2,2]
  rng=np.random.default_rng(2026101220+number*3+context)
  w=rng.uniform(-.45,.45,shape).tolist()
  state='        self.register_buffer("w",torch.tensor('+repr(w)+',dtype=torch.float64))\n'
  linear=context==1
  if linear:
   proj=rng.uniform(-.3,.3,(2,2)).tolist();bias=rng.uniform(-.15,.15,(2,)).tolist()
   state+='        self.register_buffer("proj",torch.tensor('+repr(proj)+',dtype=torch.float64))\n'
   state+='        self.register_buffer("bias",torch.tensor('+repr(bias)+',dtype=torch.float64))\n'
  signature="x"if context<2 else"x,*,z"
  body=("return x*(c+.125)+self.w"if context==0 else
        "return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x"if context==1 else
        "return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}")
  header="import torch\nfrom torch.nn import functional as F\n"
  if op=="new_full"and context==2:
   (folder/"helpers.py").write_text("def constant_like(x,w):\n    return w.new_full(x.shape,.25)\n")
   header+="from helpers import constant_like\n";expression="constant_like(x,self.w)"
  source=header+"class Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n"+state+"    def forward(self,"+signature+"):\n        c="+expression+"\n        "+body+"\ndef make():\n    return Model().eval()\n"
  with (folder/"model.py").open("x")as f:f.write(source)
  env={}
  if (folder/"helpers.py").exists():
   exec(compile((folder/"helpers.py").read_text(),"trusted_developer_fixture","exec"),env)
   source=source.replace("from helpers import constant_like\n","")
  exec(compile(source,"trusted_developer_fixture","exec"),env)
  module=env["make"]()
  with tempfile.TemporaryDirectory()as stage:
   weights=export_state(module,Path(stage),resource_profile="general-tensors-v1")
   for item in weights.values():
    with (folder/item["file"]).open("xb")as f:f.write((Path(stage)/item["file"]).read_bytes())
  inputs=[dict(name="x",shape=shape)]
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
  if context==2:
   inputs.append(dict(name="z",shape=shape));call["kwargs"]["z"]=dict(kind="tensor",name="z")
  manifest=dict(format=SOURCE_FORMAT,files={p.name:sha(p)for p in folder.glob("*.py")},
   entry="model.make",inputs=inputs,outputs=["y","saved"]if context==2 else["output"],
   call=call,factory=dict(args=[],kwargs={}),weights=weights,dtype_policy="strict-float64-v1",
   approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
  save(folder/"manifest.json",manifest)
  rows.append(dict(name=name,manifest=str(folder/"manifest.json"),kind="new_public_constructor_context",operator=op))
prior=json.loads((BASE/"benchmarks/tools/public_tensor_transform_intake_v1/cases.json").read_text())
for name in ("public_repeat_1","public_tile_2","public_roll_2"):
 original=next(row for row in prior if row["name"]==name)
 rows.append(dict(name="compat_v31_"+name,manifest=original["manifest"],kind="legacy_source_compatibility",operator=original["operator"]))
save(HERE/"cases.json",rows)
print(json.dumps(dict(cases=len(rows),new=18,compatibility=3,paid_calls=0)))
