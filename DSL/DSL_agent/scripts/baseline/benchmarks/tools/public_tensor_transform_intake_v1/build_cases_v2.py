"""Immutable public-state transform qualification sources; not model templates."""
import sys,json,hashlib,shlex,copy,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
import numpy as np,torch
from pytorch_weight_package import export_state
from pytorch_public_tensor_transform_source import SOURCE_FORMAT
ROWS=[
("repeat",0,[4],[2],torch.float64,"self.public.repeat(2)","return x+w","x",["output"],False),
("repeat",1,[2,2],[2,1],torch.float64,"self.public.repeat(1,2)","return x.square()*w+x","x",["output"],False),
("repeat",2,[2,2],[1,2],torch.float64,"self.public.repeat(2,1)","return F.linear(x.flip((-1,))*w,self.proj,self.bias)","x",["output"],True),
("tile",0,[4],[2],torch.float64,"torch.tile(self.public,(2,))","return x*w","x",["output"],False),
("tile",1,[2,2],[1,2],torch.float64,"self.public.tile((2,1))","return F.linear(x+w,self.proj,self.bias)+x","x",["output"],True),
("tile",2,[2,2],[2,1],torch.bool,"torch.tile(self.public,(1,2)).double()","return {'y':x*w+z,'saved':[x,None]}","x,*,z",["y","saved"],False),
("interleave",0,[4],[2],torch.float64,"torch.repeat_interleave(self.public,2)","return x+w","x",["output"],False),
("interleave",1,[2,4],[2,2],torch.float32,"self.public.float().repeat_interleave(2,dim=-1).double()","return x*w+x","x",["output"],False),
("interleave",2,[2,2],[1,2],torch.int64,"self.public.repeat_interleave(2,dim=0).double()","return {'y':x*w+z,'saved':[x,None]}","x,*,z",["y","saved"],False),
("flip",0,[4],[4],torch.float64,"self.public.flip((0,))","return x*w","x",["output"],False),
("flip",1,[2,2],[2,2],torch.float64,"torch.flip(self.public,(-2,-1))","return x.square()+w","x",["output"],False),
("flip",2,[2,2],[2,2],torch.bool,"self.public.flip((0,)).double()","return {'y':x*w+z,'saved':[x,None]}","x,*,z",["y","saved"],False),
("roll",0,[4],[4],torch.float64,"torch.roll(self.public,shifts=1)","return x+w","x",["output"],False),
("roll",1,[2,2],[2,2],torch.float64,"self.public.roll(shifts=(1,-1),dims=(0,-1))","return F.linear(x*w,self.proj,self.bias)+x","x",["output"],True),
("roll",2,[2,2],[2,2],torch.float64,"reorder(self.public)","return {'y':x.roll(1,-1)*w+z,'saved':[x,None]}","x,*,z",["y","saved"],False)]
def write(p,v):
    with p.open("x")as f:json.dump(v,f,sort_keys=True,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for number,(op,c,shape,rawshape,dtype,expression,body,signature,outputs,linear)in enumerate(ROWS):
    name="public_"+op+"_"+str(c);folder=HERE/"cases-v2"/name;folder.mkdir(parents=True)
    rng=np.random.default_rng(2026101200+number)
    value=rng.uniform(-.5,.5,rawshape).astype(np.float64)
    if dtype is torch.bool:value=np.asarray(np.arange(np.prod(rawshape)).reshape(rawshape)%2==0)
    elif dtype is torch.int64:value=np.asarray([[1,-2]],dtype=np.int64)
    value_literal=repr(value.tolist());dtype_name=str(dtype)
    state='        self.register_buffer("public",torch.tensor('+value_literal+',dtype='+dtype_name+'))\n'
    if linear:
        proj=rng.uniform(-.35,.35,(2,2)).tolist();bias=rng.uniform(-.15,.15,(2,)).tolist()
        state+='        self.register_buffer("proj",torch.tensor('+repr(proj)+',dtype=torch.float64))\n'
        state+='        self.register_buffer("bias",torch.tensor('+repr(bias)+',dtype=torch.float64))\n'
    header="import torch\nfrom torch.nn import functional as F\n"
    if op=="roll"and c==2:
        (folder/"helpers.py").write_text("import torch\ndef reorder(w):\n    return torch.roll(w,shifts=(5,-3),dims=(0,1))\n")
        header+="from helpers import reorder\n"
    source=header+"class Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n"+state+"    def forward(self,"+signature+"):\n        w="+expression+"\n        "+body+"\ndef make():\n    return Model().eval()\n"
    (folder/"model.py").write_text(source)
    # These fixtures are developer-owned trusted code, outside the uploaded source path.
    context={}
    if (folder/"helpers.py").exists():
        exec(compile((folder/"helpers.py").read_text(),"trusted_fixture_helpers","exec"),context)
        source_for_state=source.replace("from helpers import reorder\n","")
    else:source_for_state=source
    exec(compile(source_for_state,"trusted_fixture_model","exec"),context)
    model=context["make"]()
    with tempfile.TemporaryDirectory()as staging:
        weights=export_state(model,Path(staging),resource_profile="general-tensors-v1")
        for item in weights.values():
            with (folder/item["file"]).open("xb")as output:output.write((Path(staging)/item["file"]).read_bytes())
    specs=[dict(name="x",shape=shape)]
    call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
    if "z"in signature:specs.append(dict(name="z",shape=shape));call["kwargs"]["z"]=dict(kind="tensor",name="z")
    files={p.name:sha(p)for p in folder.glob("*.py")}
    spec=dict(format=SOURCE_FORMAT,files=files,entry="model.make",inputs=specs,outputs=outputs,
        call=call,factory=dict(args=[],kwargs={}),weights=weights,
        dtype_policy="convert-float64-v1"if dtype is torch.float32 else"strict-float64-v1",
        approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
    write(folder/"manifest.json",spec)
    rows.append(dict(name=name,manifest=str(folder/"manifest.json"),kind="new_public_source_context",
        operator=op,public_dtype=dtype_name))
prior=json.loads((BASE/"benchmarks/tools/static_tensor_transforms_v1/cases-v2.json").read_text())
for original in ("repeat_0","tile_1","roll_1"):
    item=next(row for row in prior if row["name"]==original)
    rows.append(dict(name="compat_v30_"+original,manifest=item["manifest"],kind="legacy_source_compatibility",operator=item["operator"]))
write(HERE/"cases.json",rows)
print(json.dumps(dict(cases=len(rows),new=15,compatibility=3,paid_calls=0)))
