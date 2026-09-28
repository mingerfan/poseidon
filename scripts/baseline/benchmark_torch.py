"""Independent trusted CPU float64 Torch graph. No DSL/helper implementation reuse."""
import torch
import torch.nn.functional as F
from benchmark_graph import validate


class Model(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph=graph
        validate(graph)
        for i,(name,value) in enumerate(graph["constants"].items()):
            self.register_buffer("constant_"+str(i),torch.tensor(value,dtype=torch.float64))
        self.constant_names=list(graph["constants"])

    def forward(self,*inputs):
        g=self.graph
        values=dict(zip([s["name"] for s in g["inputs"]],inputs))
        values.update({n:getattr(self,"constant_"+str(i)) for i,n in enumerate(self.constant_names)})
        for node in g["nodes"]:
            op=node["op"]; a=node["attrs"]; xs=[values[r] for r in node["inputs"]]; x=xs[0]
            if op=="add": y=torch.add(x,xs[1])
            elif op=="subtract": y=torch.sub(x,xs[1])
            elif op=="multiply": y=torch.mul(x,xs[1])
            elif op=="negate": y=torch.neg(x)
            elif op=="square": y=torch.square(x)
            elif op=="power": y=torch.pow(x,a["exponent"])
            elif op=="linear": y=F.linear(x,xs[1],xs[2] if len(xs)==3 else None)
            elif op=="flatten": y=torch.flatten(x)
            elif op=="reshape": y=torch.reshape(x,a["shape"])
            elif op=="transpose": y=torch.transpose(x,a["dim0"],a["dim1"])
            elif op=="permute": y=x.permute(a["dims"])
            elif op=="rotate": y=torch.roll(x.flatten(),-a["step"]).reshape(x.shape)
            elif op=="concat": y=torch.cat(xs,dim=a["axis"])
            elif op=="stack": y=torch.stack(xs,dim=a["axis"])
            elif op=="slice":
                axis=a["axis"]%x.ndim
                indexes=list(range(*slice(a["start"],a["stop"],a["step"]).indices(x.shape[axis])))
                y=torch.index_select(x,axis,torch.tensor(indexes,dtype=torch.long))
            elif op=="split": y=torch.split(x,a["sections"],dim=a["axis"])
            elif op in ("sum","mean"):
                y=(torch.sum if op=="sum" else torch.mean)(x,dim=tuple(a["axes"]),keepdim=a["keepdims"])
                if y.ndim==0: y=y.reshape(1)
            elif op=="batch_norm":
                y=F.batch_norm(x,xs[1],xs[2],xs[3],xs[4],training=False,eps=a["eps"])
            elif op=="polynomial":
                coeff=xs[1]
                if a["basis"]=="power":
                    y=torch.zeros_like(x)
                    for c in reversed(coeff): y=y*x+c
                else:
                    t0=torch.ones_like(x); y=coeff[0]*t0
                    if len(coeff)>1:
                        t1=x; y=y+coeff[1]*t1
                        for c in coeff[2:]:
                            t2=2*x*t1-t0; y=y+c*t2; t0,t1=t1,t2
            elif op.startswith("conv"):
                fn=F.conv1d if op=="conv1d" else F.conv2d
                y=fn(x,xs[1],xs[2] if len(xs)==3 else None,stride=a["stride"],
                     padding=a["padding"],dilation=a["dilation"],groups=a["groups"])
            elif op.startswith("avg_pool"):
                fn=F.avg_pool1d if op=="avg_pool1d" else F.avg_pool2d
                y=fn(x,kernel_size=a["kernel"],stride=a["stride"],padding=a["padding"],
                     ceil_mode=False,count_include_pad=a["count_include_pad"])
            else: raise ValueError("Unknown Torch reference operator")
            for name,value in zip(node["outputs"],y if op=="split" else [y]):
                values[name]=value
        return tuple(values[o["value"]] for o in g["outputs"])


def evaluate(model, inputs):
    module=Model(model).eval()
    with torch.no_grad():
        result=module(*(torch.from_numpy(inputs[s["name"]].copy()) for s in model["inputs"]))
    return {o["name"]:v.detach().numpy().copy() for o,v in zip(model["outputs"],result)}
