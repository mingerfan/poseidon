"""Frozen high-rank and wide-layout models; original model/reference rules unchanged."""
import sys,json,hashlib,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(ROOT/"scripts")]
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import numpy as np
from prepare_pytorch_source import build
from benchmark_graph import digest
from application_backend import PoseidonBackend
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2)
SOURCE_AXIS="""import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        z=(x+y).transpose(-1,-2)
        pooled=z.square().mean(dim=(-1,-3),keepdim=True)
        return {'pooled':pooled,'same':pooled,'flat':z.flatten()}
def make(rank):
    return Model().eval()
"""
SOURCE_LINEAR="""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(2,3,dtype=torch.float64)
    def forward(self,x):
        h=self.fc(x.transpose(-2,-1))
        z=h.square()+h*.125
        return {'features':z,'summary':z.mean(dim=-1),'same':z}
def make(rank):
    return Model().eval()
"""
SOURCE_PERMUTE="""import torch
class Model(torch.nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.axes=tuple(range(rank-6))+tuple(reversed(range(rank-6,rank)))
        self.reductions=(rank-6,rank-4,rank-2)
        self.register_buffer('w',torch.zeros((1,)*(rank-6)+(2,)*6,dtype=torch.float64))
    def forward(self,x):
        z=x.permute(self.axes)*self.w+x.square()*.125
        pooled=z.mean(dim=self.reductions,keepdim=True)
        return {'features':z,'summary':pooled,'same':z}
def make(rank):
    return Model(rank).eval()
"""
SOURCE_WIDE_SQUARE="""import torch
class Model(torch.nn.Module):
    def forward(self,x):
        pooled=x.square().mean(dim=0)
        return {'pooled':pooled,'same':pooled,'head':pooled[:8]}
def make():
    return Model().eval()
"""
SOURCE_WIDE_CONCAT="""import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        z=torch.cat((x,y),dim=-1)
        pooled=z.mean(dim=-1)
        total=pooled.sum().reshape(1)
        return {'rows':pooled,'summary':total,'same':pooled}
def make():
    return Model().eval()
"""
SOURCE_WIDE_LINEAR="""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(1025,3,dtype=torch.float64)
    def forward(self,x,y):
        z=self.fc(x+y)
        pooled=z.mean(dim=-1,keepdim=True)
        return {'projection':z,'summary':pooled,'same':z}
def make():
    return Model().eval()
"""
SOURCE_BIG_LINEAR="""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(100,100,dtype=torch.float64)
    def forward(self,x):
        return self.fc(x)
def make():
    return Model().eval()
"""
rows=[]
rng=np.random.default_rng(20261013)
def create(name,source,inputs,outputs,*,kwargs=None,arrays=None,options=None,purpose=None,overlap=None):
 folder=HERE/"cases"/name;folder.mkdir(parents=True,exist_ok=True)
 model_file=folder/"model.py"
 if model_file.exists():assert model_file.read_text()==source
 else:model_file.write_text(source)
 weights=[]
 for i,(field,value)in enumerate((arrays or {}).items()):
  filename="state_"+str(i)+".npy";np.save(folder/filename,np.asarray(value,dtype=np.float64),allow_pickle=False)
  weights.append(field+"="+filename)
 spec=build(folder/"model.py",entry="model.make",inputs=[n+":"+','.join(map(str,s))for n,s in inputs],
  outputs=outputs,weights=weights,factory_kwargs=kwargs or{},source_format="poseidon-trusted-pytorch-v34",
  resource_profile="general-tensors-v1",dtype_policy="strict-float64-v1",
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name=n)for n,_ in inputs],kwargs={}))
 write(folder/"manifest.json",spec)
 rows.append(dict(name=name,manifest=str(folder/"manifest.json"),source_format=spec["format"],
  source_kind="trusted_pytorch",purpose=purpose,inputs=[dict(name=n,shape=s)for n,s in inputs],outputs=outputs,
  total_input_elements=sum(np.prod(s).item()for _,s in inputs),input_ranks=[len(s)for _,s in inputs],
  nonrepeating_public_weights=sum(np.asarray(x).size for x in(arrays or{}).values()),
  registered_weight_entries=len(arrays or{}),options=options or{},prior_mathematical_overlap=overlap))
default=dict(generation_guidance="explicit-v15",output_binding_policy="exact-values-v1",
 public_weight_policy="descriptors-math-roles-v2",capacity_packing_version="capacity-public-tensors-v9",
 compiler_configuration="poseidon-cpu-eva-w45-compact-v1")
for rank in(8,16,32):
 shape=[1]*(rank-6)+[2,3,2,3,2,3]
 create("rank_axis_"+str(rank),SOURCE_AXIS,[("x",shape),("y",[2,3])],["pooled","same","flat"],
  kwargs=dict(rank=rank),options=dict(default),purpose="different-rank encrypted broadcast, noncontiguous transpose and partial reduction")
 create("rank_linear_"+str(rank),SOURCE_LINEAR,[("x",shape)],["features","summary","same"],
  kwargs=dict(rank=rank),arrays={"fc.weight":rng.uniform(-.0625,.0625,(3,2)),"fc.bias":rng.uniform(-.01,.01,(3,))},
  options=dict(default),purpose="high-rank transpose, nonrepeating public Linear, square and reduction")
 shape=[1]*(rank-6)+[2]*6
 create("rank_permute_"+str(rank),SOURCE_PERMUTE,[("x",shape)],["features","summary","same"],
  kwargs=dict(rank=rank),arrays={"w":rng.uniform(-.2,.2,shape)},options=dict(default),
  purpose="six nontrivial axes, public tensor weights, residual fusion and nonadjacent reduction")
create("wide_square_reduce",SOURCE_WIDE_SQUARE,[("x",[8,512])],["pooled","same","head"],
 options=dict(default),purpose="4096 input elements, row-axis square reduction and output slicing")
create("wide_concat_reduce",SOURCE_WIDE_CONCAT,[("x",[32,96]),("y",[32,32])],["rows","summary","same"],
 options=dict(default),purpose="4096 elements in two shapes, interleaved row concat and hierarchical reduction")
create("wide_tail_linear",SOURCE_WIDE_LINEAR,[("x",[1,2,1025]),("y",[2,1])],["projection","summary","same"],
 arrays={"fc.weight":rng.uniform(-.015625,.015625,(3,1025)),"fc.bias":rng.uniform(-.01,.01,(3,))},
 options=dict(default),purpose="1025 row width, encrypted broadcast and nonrepeating public Linear")
old_path=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-resource-large-weights-r661/model.json")
old=read(old_path);linear=next(n for n in old["nodes"]if n["op"]=="linear")
w=np.asarray(old["constants"][linear["inputs"][1]],dtype=np.float64);bias=np.asarray(old["constants"][linear["inputs"][2]],dtype=np.float64)
assert w.shape==(100,100)and bias.shape==(100,)and np.unique(w).size==10000
assert hashlib.sha256(old_path.read_bytes()).hexdigest()
opts=dict(default,chunk_period=128);opts.pop("output_binding_policy")
create("weights_linear_10k",SOURCE_BIG_LINEAR,[("x",[100])],["output0"],arrays={"fc.weight":w,"fc.bias":bias},
 options=opts,purpose="real Agent generation for original independently verified 10000-weight Linear",
 overlap=dict(report="docs/baseline/agent-resource-acceptance-r665.json",model_sha256=digest(old),
  source=str(old_path),source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),weights_unchanged=True,period_unchanged=128))
write(HERE/"cases.json",rows)
identity=PoseidonBackend().identity_manifest()
assert identity==read(BASE/"benchmarks/tools/output_alias_planner_v1/identity-snapshot.json")
write(HERE/"identity-snapshot.json",identity)
print(json.dumps(dict(cases=len(rows),high_rank_cases=9,wide_cases=3,prior_model_agent_gap=1,
 large_input_max=max(r["total_input_elements"]for r in rows),sdk=identity["manifest_sha256"],
 nonrepeating_weight_cases=sum(r["registered_weight_entries"]>0 for r in rows),compiler_calls=0,encrypted_calls=0,provider_calls=0)))
