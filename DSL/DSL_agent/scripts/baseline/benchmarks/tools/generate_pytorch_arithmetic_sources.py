"""Owned V8 source contexts, syntax variants and float32 normalization probe."""
import argparse,hashlib,io,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest

SOURCE='''"""Owned deterministic arithmetic module factories."""
import torch
from torch import nn
class Base(nn.Module):
    def __init__(self,public_shape,side,style):
        super().__init__()
        self.side,self.style=side,style
        if public_shape is None:self.public=0.375
        else:self.register_buffer("public",torch.zeros(public_shape,dtype=torch.float64))
    def operands(self,x):
        return (self.public,x) if self.side=="left" else (x,self.public)
class Add(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.add(other=b,alpha=1)
        if self.style=="function":return torch.add(input=a,other=b,alpha=1)
        return a+b
class Subtract(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.sub(other=b,alpha=1)
        if self.style=="function":return torch.sub(input=a,other=b,alpha=1)
        return a-b
class Multiply(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.mul(other=b)
        if self.style=="function":return torch.mul(input=a,other=b)
        return a*b
def make_model(op,public_shape,side,style="operator",float32=False):
    module={"add":Add,"subtract":Subtract,"multiply":Multiply}[op](public_shape,side,style).eval()
    return module.float() if float32 else module.double()
'''
def generate(output):
    import numpy as np
    output=Path(output)
    if output.exists():raise ValueError("Fresh V8 source fixtures")
    output.mkdir(parents=True)
    inputs={"scalar":[4],"tensor":[2,2],"broadcast":[2,2,2]}
    shapes={"scalar":None,"tensor":[2,2],"broadcast":[2]}
    values={"tensor":[[.125,-.25],[.375,-.5]],"broadcast":[.125,-.25]}
    contexts=[]
    for op in ("add","subtract","multiply"):
        for context in ("scalar","tensor","broadcast"):
            for side in ("left","right"):contexts.append((op,context,side,"operator",False))
    for op in ("add","subtract","multiply"):
        contexts.append((op,"tensor","right","method",False))
        contexts.append((op,"tensor","left","function",False))
    contexts.append(("add","tensor","left","operator",True))
    cases=[]
    for op,context,side,style,float32 in contexts:
        name="v8_"+op+"_"+context+"_"+side+"_"+style+("_float32" if float32 else "")
        folder=output/name;folder.mkdir();(folder/"models.py").write_text(SOURCE)
        weights={}
        if context!="scalar":
            value=np.asarray(values[context],dtype=np.float32 if float32 else np.float64)
            stream=io.BytesIO();np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
            (folder/"state_0000.npy").write_bytes(raw)
            weights["public"]=dict(file="state_0000.npy",dtype=value.dtype.str,shape=list(value.shape),
                                  bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        spec=dict(format="poseidon-trusted-pytorch-v8",entry="models.make_model",
            files={"models.py":hashlib.sha256(SOURCE.encode()).hexdigest()},
            factory=dict(args=[op,shapes[context],side,style,float32],kwargs={}),
            inputs=[dict(name="x",shape=inputs[context])],outputs=["result"],weights=weights,
            dtype_policy="convert-float64-v1" if float32 else "strict-float64-v1",
            approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
        (folder/"manifest.json").write_bytes(canonical(spec))
        cases.append(dict(id=name,manifest=name+"/manifest.json",partition=op,context=context,side=side,
                          syntax=style,float32_source=float32,manifest_sha256=digest(spec),
                          variant_kind="dtype" if float32 else "syntax" if style!="operator" else "mathematical_context"))
    catalog=dict(format="poseidon-trusted-arithmetic-source-fixtures-v1",planned=25,
        mathematical_contexts=18,syntax_variants=6,dtype_variants=1,cases=cases,frozen_benchmark_changed=False)
    catalog["binding"]=digest(catalog);(output/"catalog.json").write_bytes(canonical(catalog))
    return catalog
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(generate(a.output)))
