"""Trusted tiny fixtures only; standalone mathematical checks, never Agent/backend execution."""
import copy,json,os,resource,shlex,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,WORK,enter_nix
if "--inside"not in sys.argv:
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(150,150))
import numpy as np,torch
from normalize import normalize,NormalizationError
from benchmark_graph import digest
assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
start=time.monotonic();rows=[];values=groups=0;maxdiff=0.
class Model(torch.nn.Module):
 def __init__(self,method,shape,style,keyword=False,transpose=False):super().__init__();self.method=method;self.shape=shape;self.style=style;self.keyword=keyword;self.transposed=transpose
 def forward(self,x,other):
  a=x.transpose(0,1)if self.transposed else x
  if self.method=="expand_as":y=a.expand_as(other=other)if self.keyword else a.expand_as(other)
  else:y=a.broadcast_to(size=self.shape)if self.keyword else a.broadcast_to(self.shape)
  if self.style==0:return y
  if self.style==1:return y+other
  return {"expanded":y,"base":a,"aux":[y.clone(),y.square()]}
def flattened(value):
 if isinstance(value,dict):return sum([flattened(v)for v in value.values()],[])
 if isinstance(value,(tuple,list)):return sum([flattened(v)for v in value],[])
 return [value]
def evaluate_case(name,method,sx,sy,style,keyword=False,transpose=False,dtype=torch.float64,other_dtype=torch.float64):
 global groups,values,maxdiff
 module=Model(method,sy,style,keyword,transpose).eval();raw=torch.fx.symbolic_trace(module)
 converted,manifest=normalize(raw,dict(x=sx,other=sy));assert len(manifest["records"])==1
 error=0.
 for i in range(16):
  rng=np.random.default_rng(20261012+i)
  x=np.zeros(sx)if i==0 else np.linspace(-.2,.3,mathprod(sx)).reshape(sx)if i==1 else rng.uniform(-.5,.5,sx)
  other=np.zeros(sy)if i==0 else rng.uniform(-.3,.4,sy)
  tx=torch.as_tensor(x,dtype=dtype);ty=torch.as_tensor(other,dtype=other_dtype)
  left,right=raw(tx,ty),converted(tx,ty)
  a=tx.numpy().transpose(1,0)if transpose else tx.numpy();y=np.broadcast_to(a,sy)
  expected=y if style==0 else y+ty.numpy()if style==1 else {"expanded":y,"base":a,"aux":[y.copy(),np.square(y)]}
  for old,new,ref in zip(flattened(left),flattened(right),flattened(expected)):
   assert old.shape==new.shape and old.dtype==new.dtype and old.stride()==new.stride()
   np.testing.assert_array_equal(old.numpy(),new.numpy());np.testing.assert_array_equal(new.numpy(),ref)
   difference=float(np.max(np.abs(new.numpy()-ref)));error=max(error,difference);values+=new.numel()
  if style!=1:
   assert flattened(right)[0].untyped_storage().data_ptr()==tx.untyped_storage().data_ptr()
  groups+=1
 maxdiff=max(maxdiff,error);rows.append(dict(case=name,status="passed",groups=16,maximum_difference=error,manifest=manifest))
def mathprod(s):return __import__("math").prod(s)
for method in("expand_as","broadcast_to"):
 for index,(sx,sy)in enumerate((((1,),(4,)),((2,1),(2,3)),((1,2,1),(3,2,4)))):
  for style in range(3):evaluate_case(method+str(index)+"_"+str(style),method,sx,sy,style)
 evaluate_case(method+"_keyword",method,(1,3),(2,3),0,keyword=True)
 evaluate_case(method+"_float32",method,(1,3),(2,3),1,dtype=torch.float32,other_dtype=torch.float32)
 evaluate_case(method+"_target_dtype",method,(1,3),(2,3),0,other_dtype=torch.float32)
 evaluate_case(method+"_noncontiguous",method,(3,1),(2,3),2,transpose=True)
 evaluate_case(method+"_rank32",method,(1,)*32,(1,)*31+(2,),0)
rejections=[]
for name,model,signature,wanted in(
 ("signature",Model("expand_as",(2,3),0),dict(x=(1,3)),"normalization_signature"),
 ("rank33",Model("expand_as",(1,)*33,0),dict(x=(1,)*33,other=(1,)*33),"normalization_rank"),
 ("zero_dimension",Model("expand_as",(0,),0),dict(x=(1,),other=(0,)),"normalization_dimension"),
 ("capacity",Model("broadcast_to",(65537,),0),dict(x=(1,),other=(1,)),"normalization_capacity"),
 ("incompatible",Model("expand_as",(4,),0),dict(x=(3,),other=(4,)),"normalization_broadcast")):
 try:normalize(torch.fx.symbolic_trace(model.eval()),signature);raise AssertionError("accepted_invalid:"+name)
 except NormalizationError as e:assert e.code==wanted;rejections.append(dict(case=name,code=e.code,status="rejected"))
class Dynamic(torch.nn.Module):
 def forward(self,x,other):return x.broadcast_to(other.nonzero().shape)
try:normalize(torch.fx.symbolic_trace(Dynamic().eval()),dict(x=(1,),other=(3,)));raise AssertionError("accepted_dynamic")
except NormalizationError as e:assert e.code=="normalization_unresolved_static_shape";rejections.append(dict(case="data_shape",code=e.code,status="rejected"))
class StaticShape(torch.nn.Module):
 def forward(self,x,other):return x.broadcast_to((other.shape[0],other.size(1)))
m=torch.fx.symbolic_trace(StaticShape().eval());converted,_=normalize(m,dict(x=(1,3),other=(2,3)))
np.testing.assert_array_equal(converted(torch.ones(1,3,dtype=torch.float64),torch.ones(2,3,dtype=torch.float64)).numpy(),np.ones((2,3)))
result=dict(format="poseidon-static-broadcast-source-draft-check-v1",positive_contexts=len(rows),passed=len(rows),failed=0,
 plaintext_groups=groups,compared_values=values,maximum_difference=maxdiff,rejected=len(rejections),
 static_shape_query="passed",rows=rows,rejections=rejections,torch_version=torch.__version__,
 source_admission_registered=False,new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
 runtime_changes=False,backend_changes=False,production_source_changes=False,seconds=time.monotonic()-start)
result["binding"]=digest(result)
out=WORK/"results"/("agent-static-broadcast-source-draft-"+str(time.time_ns()));out.mkdir()
(out/"report.json").write_text(json.dumps(result,indent=2)+"\\n")
print(json.dumps({k:v for k,v in result.items()if k not in("rows","rejections")},indent=2));print("RESULT",out)
