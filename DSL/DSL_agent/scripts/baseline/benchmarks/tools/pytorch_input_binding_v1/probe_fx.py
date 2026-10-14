"""Inspect fixed Torch FX concrete-argument behavior; no FHE or Provider."""
import inspect,json,sys,torch
from torch.fx import Tracer,GraphModule,PH
torch.set_num_threads(1)
class Dictionary(torch.nn.Module):
    def forward(self,batch):
        return batch["x"]+batch["y"]*2
class Mixed(torch.nn.Module):
    def forward(self,x,*,other,scale=2.0,bias=None,aux=False):
        y=x+other*scale
        if bias is not None:y=y+bias
        return {"y":y,"aux":x} if aux else y
class Nested(torch.nn.Module):
    def forward(self,pair,options):
        return pair[0]+pair[1][0]*options["scale"]
cases=[
 ("dictionary",Dictionary(),{"batch":{"x":PH,"y":PH}}),
 ("mixed",Mixed(),{"x":PH,"other":PH,"scale":2.0,"bias":None,"aux":False}),
 ("nested",Nested(),{"pair":(PH,[PH]),"options":{"scale":2.0}})]
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,python=list(sys.version_info[:3]),tracer_file=inspect.getsourcefile(Tracer))))
for name,module,args in cases:
 tracer=Tracer();graph=tracer.trace(module,concrete_args=args);gm=GraphModule(module,graph)
 print(name)
 print(gm.code)
 print(str(graph))
 print(json.dumps({"placeholders":[dict(name=n.name,target=n.target,args=repr(n.args)) for n in graph.nodes if n.op=="placeholder"],
  "codegen_type":type(graph._codegen).__name__}))
