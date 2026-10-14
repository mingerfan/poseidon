import sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from pytorch_input_binding import *
from pytorch_static_trace import trace
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_public_value_ops import ARITHMETIC_POLICY
class Dictionary(torch.nn.Module):
 def forward(self,batch):return {"kind":batch["x"]+batch["y"]*2,"unused":None}
class Mixed(torch.nn.Module):
 def forward(self,x,*,other,scale=2.0,bias=None,aux=False):
  y=x+other*scale
  if bias is not None:y=y+bias
  return {"y":y,"aux":x}if aux else y
class Nested(torch.nn.Module):
 def forward(self,pair,options):return pair[0]+pair[1][0]*options["scale"]
def t(n):return dict(kind="tensor",name=n)
def lit(v):return dict(kind="literal",value=v)
cases=[
 ("dict",Dictionary(),[dict(kind="dict",items=[["x",t("x")],["y",t("y")]])],{}),
 ("mixed",Mixed(),[t("x")],dict(other=t("y"))),
 ("nested",Nested(),[dict(kind="tuple",items=[t("x"),dict(kind="list",items=[t("y")])]),
 dict(kind="dict",items=[["scale",lit(2.)]])],{})]
for name,m,args,kwargs in cases:
 spec=dict(inputs=[dict(name=n,shape=[2,3])for n in ("y","x")],call=dict(format=CALL_FORMAT,args=args,kwargs=kwargs))
 record=plan(spec,signature_record(m))
 fx=trace(m,spec["inputs"],resource_profile="general-tensors-v1",**TRACE_OPTIONS,
          public_expression_policy=ARITHMETIC_POLICY,input_plan=record)
 normalized,flow=normalize(fx,record);check_flow(flow,spec)
 x=torch.arange(6,dtype=torch.float64).reshape(2,3);y=x*.25-1
 a=invoke(m,record,dict(x=x,y=y));b=normalized(y,x)
 torch.testing.assert_close(a,b,atol=0,rtol=0)
 print(json.dumps(dict(case=name,slots=len(record["placeholders"]),guards=len(flow["removed_guards"]),passed=True,flow=flow["binding"])))
