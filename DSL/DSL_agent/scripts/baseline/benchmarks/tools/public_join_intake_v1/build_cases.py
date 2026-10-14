"""Frozen tensor assembly source contexts; not extra benchmark topologies."""
import sys,json,shlex,hashlib,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
import numpy as np,torch
from pytorch_weight_package import export_state
from pytorch_public_join_source import SOURCE_FORMAT
def save(path,data):
 with path.open("x")as f:json.dump(data,f,indent=2,sort_keys=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
bodies={
 "mixed_cat_0":("x",["output"],"return torch.cat((x,self.w),dim=0)"),
 "mixed_cat_1":("x",["output"],"return torch.concat((self.w.float(),x.square()),dim=1)"),
 "mixed_cat_2":("x",["output"],"return torch.concatenate((self.mask,x+self.w),dim=-1)"),
 "mixed_stack_0":("x",["output"],"return torch.stack((x,self.w),dim=0)"),
 "mixed_stack_1":("x",["output"],"return torch.stack(tensors=[self.i,x.square()],dim=1)"),
 "mixed_stack_2":("x",["output"],"return torch.stack((self.scalar,x.sum(),self.scalar*.5),dim=-1)"),
 "public_cat_0":("x",["output"],"c=torch.cat((self.w,self.w*.5),dim=0)\nreturn x.repeat(2,1)+c"),
 "public_cat_1":("x",["output"],"c=torch.concat((self.w.float(),self.i),dim=-1)\nreturn x.repeat(1,2)*c"),
 "public_cat_2":("x",["output"],"c=torch.concatenate((self.mask,self.mask.flip(0)),dim=1).double()\nreturn x.repeat(1,2)+c"),
 "public_stack_0":("x",["output"],"c=torch.stack((self.w,self.w.flip(1)),dim=0)\nreturn x.unsqueeze(0)+c"),
 "public_stack_1":("x",["output"],"c=torch.stack((self.w.float(),self.i),dim=-1)\nreturn x.unsqueeze(-1)*c"),
 "public_stack_2":("x",["output"],"c=torch.stack((self.mask,self.i),dim=1).double()\nreturn x.unsqueeze(1)+c"),
 "combination_0":("x",["output"],"c=x.masked_fill(self.mask,.125)\nreturn torch.cat((c,self.w),dim=1)"),
 "combination_1":("x,*,z",["output"],"c=torch.stack((x,self.w.flip(0)),dim=-1).flatten(1)\nreturn F.linear(c,self.proj,self.bias)+z"),
 "combination_2":("x",["output"],"a,b=x.chunk(2,dim=1)\nreturn torch.cat((self.w[:,0:1],a,b),dim=-1)"),
 "combination_3":("x",["output"],"c=self.w.repeat(1,2).reshape(2,4)\nreturn torch.cat((x,c),dim=1)"),
 "combination_4":("x,*,z",["output"],"c=torch.cat((x,self.mask.double()),dim=1)\nreturn F.linear(c,self.proj,self.bias)+z.square()"),
 "combination_5":("x",["joined","updated","snapshot"],"snapshot=x.clone()\nj=torch.cat((snapshot,self.w),dim=1)\nx=x.clone()\nx.masked_fill_(self.mask,-.25)\nreturn {'joined':j,'updated':x,'saved':[snapshot,None]}"),
}
rows=[];case_root=HERE/"cases";case_root.mkdir()
for number,(name,(signature,outputs,body))in enumerate(bodies.items()):
 folder=case_root/name;folder.mkdir()
 state='        self.register_buffer("w",torch.zeros((2,2),dtype=torch.float64))\n        self.register_buffer("mask",torch.zeros((2,2),dtype=torch.bool))\n        self.register_buffer("i",torch.zeros((2,2),dtype=torch.int64))\n        self.register_buffer("scalar",torch.tensor(0.,dtype=torch.float64))\n        self.register_buffer("proj",torch.zeros((2,4),dtype=torch.float64))\n        self.register_buffer("bias",torch.zeros(2,dtype=torch.float64))\n'
 source="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n"+state+"    def forward(self,"+signature+"):\n"+"".join("        "+line+"\n"for line in body.splitlines())+"def make():\n    return Model().eval()\n"
 (folder/"model.py").write_text(source)
 env={};exec(compile(source,"developer_public_join_fixture","exec"),env);module=env["make"]()
 rng=np.random.default_rng(2026101240+number)
 state_values={"w":torch.tensor(rng.uniform(-.4,.4,(2,2)),dtype=torch.float64),
  "mask":torch.tensor([[True,False],[False,True]],dtype=torch.bool),
  "i":torch.tensor([[1,-2],[3,0]],dtype=torch.int64),"scalar":torch.tensor(.1875,dtype=torch.float64),
  "proj":torch.tensor(rng.uniform(-.3,.3,(2,4)),dtype=torch.float64),
  "bias":torch.tensor(rng.uniform(-.1,.1,(2,)),dtype=torch.float64)}
 module.load_state_dict(state_values,strict=True)
 with tempfile.TemporaryDirectory()as stage:
  weights=export_state(module,Path(stage),resource_profile="general-tensors-v1")
  for item in weights.values():
   with (folder/item["file"]).open("xb")as f:f.write((Path(stage)/item["file"]).read_bytes())
 inputs=[dict(name="x",shape=[2,2])]
 call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
 if "z"in signature:inputs.append(dict(name="z",shape=[2,2]));call["kwargs"]["z"]=dict(kind="tensor",name="z")
 manifest=dict(format=SOURCE_FORMAT,files={"model.py":sha(folder/"model.py")},entry="model.make",inputs=inputs,
  outputs=outputs,call=call,factory=dict(args=[],kwargs={}),weights=weights,dtype_policy="strict-float64-v1",
  approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
 save(folder/"manifest.json",manifest)
 rows.append(dict(name=name,manifest=str(folder/"manifest.json"),kind="new_public_join_context",
  operator=name.rsplit("_",1)[0]))
prior=json.loads((BASE/"benchmarks/tools/public_mask_intake_v1/cases.json").read_text())
for name in("where_0","masked_fill__1","public_where_1"):
 original=next(c for c in prior if c["name"]==name)
 rows.append(dict(name="compat_v33_"+name,manifest=original["manifest"],kind="legacy_source_compatibility",operator=original["operator"]))
save(HERE/"cases.json",rows)
print(json.dumps(dict(cases=len(rows),new=18,compatibility=3,api_calls=0)))
