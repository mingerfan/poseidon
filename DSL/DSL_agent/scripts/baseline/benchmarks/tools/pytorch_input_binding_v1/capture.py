import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE.parent))
from benchmark_graph import digest,canonical
from pytorch_input_binding import CALL_FORMAT,SOURCE_FORMAT
from pytorch_weight_package import export_state
from prepare_pytorch_source import build,save
from pytorch_model_intake import run
from candidate_bundle import check_provenance
from hecate_python_env import WORK
from models import make_model
torch.set_num_threads(1)
TOOLS=Path(__file__).resolve().parent;root=WORK/"results/agent-pytorch-input-binding-source-r1389-final"
assert not root.exists();root.mkdir()
def tensor(n):return dict(kind="tensor",name=n)
def literal(v):return dict(kind="literal",value=v)
def container(kind,*items):return dict(kind=kind,items=list(items))
rows=[];started=time.monotonic()
for group in("dictionary","nested","keywords","defaults"):
 for variant in range(3):
  name=group+str(variant);source=TOOLS/"cases-final"/name;source.mkdir(parents=True)
  model=make_model(group,variant)
  rng=np.random.RandomState(18397+len(rows))
  with torch.no_grad():
   for p in model.parameters():p.copy_(torch.from_numpy(rng.uniform(-.35,.35,tuple(p.shape))))
  weights=export_state(model,source,resource_profile="general-tensors-v1")
  raw=(TOOLS/"models.py").read_bytes();(source/"models.py").write_bytes(raw)
  shape_x=[2,3];shape_y=[2,3];outputs=["output"]
  if group=="dictionary":
   args=[container("dict",["left",tensor("x")],["right",tensor("y")])];kwargs={}
   if variant==1:shape_y=[3];outputs=["sum"]
   if variant==2:shape_x=shape_y=[2,2];outputs=["masked","skip"]
  elif group=="nested":
   if variant==0:
    args=[container("tuple",tensor("x"),container("list",tensor("y"))),container("dict",["gain",literal(.5)])]
    outputs=["sum","square"]
   elif variant==1:
    shape_x=[1,2,2];shape_y=[2]
    args=[container("tuple",container("dict",["x",tensor("x")],["y",tensor("y")])),container("dict")]
   else:args=[container("list",container("dict",["x",tensor("x")],["y",tensor("y")])),
              container("tuple",literal(.5),literal(-.25))]
   kwargs={}
  elif group=="keywords":
   args=[tensor("x")];kwargs=dict(other=tensor("y"))
   if variant==1:shape_x=shape_y=[2,2]
   if variant==2:kwargs.update(mode=literal("subtract"),gain=literal(.5))
  else:
   args=[tensor("x"),tensor("y")];kwargs={}
   if variant==2:kwargs=dict(include_aux=literal(True));outputs=["value","aux"]
  spec,package_root=build(source/"models.py",entry="models.make_model",
    inputs=["y:"+",".join(map(str,shape_y)),"x:"+",".join(map(str,shape_x))],
    outputs=outputs,weights=[n+"="+v["file"]for n,v in weights.items()],
    factory_args=[group,variant],source_format=SOURCE_FORMAT,call=dict(format=CALL_FORMAT,args=args,kwargs=kwargs),
    resource_profile="general-tensors-v1")
  manifest=save(spec,source/"manifest.json",source)
  target=root/name
  try:
   result=run(manifest,target,resource_profile="general-tensors-v1",trusted=True)
   report=json.loads((target/"capture-reference.json").read_text())
   metadata=report["capture_metadata"]["public_value_expressions"]
   high=metadata["high_graph"];lowered=metadata["lowered_graph"]
   flow=metadata["input_call_flow"]
   item=dict(case=name,status="passed",manifest_sha256=digest(spec),
       output=str(target),source_version=SOURCE_FORMAT,flow_binding=flow["binding"],
       high_sha256=digest(high),lowered_sha256=digest(lowered),finite=report["finite_reference"],
       groups=report["finite_reference"]["groups"],
       compared_values=report["finite_reference"]["compared_values"],
       removed_guards=len(flow["removed_guards"]),
       compiler_calls=0,encrypted_calls=0,provider_calls=0)
  except Exception as e:
   item=dict(case=name,status="failed",error=type(e).__name__+": "+str(e),output=str(target))
  rows.append(item);print(json.dumps({k:v for k,v in item.items()if k!="finite"}),flush=True)
  (root/"progress.json").write_text(json.dumps(rows,indent=2))
report=dict(format="poseidon-pytorch-input-binding-source-audit-v1",cases=rows,
 passed=sum(i["status"]=="passed"for i in rows),failed=sum(i["status"]!="passed"for i in rows),
 skipped=0,not_run=0,provider_calls=0,compiler_calls=0,encrypted_calls=0,
 elapsed_seconds=time.monotonic()-started)
report["binding"]=digest(report);(root/"report.json").write_text(json.dumps(report,indent=2))
raise SystemExit(0 if report["failed"]==0 else 1)
