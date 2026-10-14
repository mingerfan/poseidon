"""Owned PyTorch arithmetic/shape syntax contexts; no model code runs here."""
import argparse,ast,hashlib,io,json
from pathlib import Path
from benchmark_graph import canonical,digest,require

VERSION="poseidon-pytorch-arithmetic-shape-fixtures-v1"
SHAPES=("literal","input_numel","result_shape")
STYLES=("operator","function","method")
OPS=("add","subtract","multiply")
QUERIES={
 "literal":"y.reshape(4)",
 "input_numel":"y.reshape(x.numel())",
 "result_shape":"y.reshape(y.shape[0]*y.shape[1])",
 "result_size":"y.reshape(y.size(0),-1).flatten()",
 "result_numel":"y.reshape(y.numel())",
}

def source(op,style,query,rejection=None):
    target={"add":"add","subtract":"sub","multiply":"mul"}[op]
    if style=="operator":
        call="x "+{"add":"+","subtract":"-","multiply":"*"}[op]+" self.public"
    elif style=="function":
        call=f"torch.{target}(input=x, other=self.public"+(", alpha=1)" if op!="multiply" else ")")
    else:
        call=f"x.{target}(other=self.public"+(", alpha=1)" if op!="multiply" else ")")
    body=f"y={call}\n        return "+QUERIES[query]
    if rejection=="encrypted_branch":
        body="if x.sum()>0:\n            return x+self.public\n        return x-self.public"
    elif rejection=="inplace":
        body="return x.add_(self.public)"
    elif rejection=="out_mutation":
        body="return torch.add(x,self.public,out=x)"
    elif rejection is not None:
        raise ValueError("Unknown rejection fixture")
    text='''"""Owned static source; explicitly trusted capture only."""
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.zeros(2,dtype=torch.float64))
    def forward(self,x):
        '''+body+'''
def make_model():
    return Model().double().eval()
'''
    ast.parse(text)
    return text

def mathematical_graph(case_id,op):
    return dict(format="poseidon-operator-graph-v3",id=case_id,
        inputs=[dict(name="x",shape=[2,2])],constants=dict(public=[.125,-.25]),
        nodes=[dict(id="arithmetic",op=op,inputs=["x","public"],attrs={},outputs=["y"]),
               dict(id="shape",op="reshape",inputs=["y"],attrs=dict(shape=[4]),outputs=["result"])],
        outputs=[dict(name="result",value="result")])

def generate(output):
    import numpy as np
    output=Path(output);require(not output.exists(),"Fresh shape fixtures")
    output.mkdir(parents=True);rows=[]
    plans=[(op,style,query,None) for op in OPS for style in STYLES for query in SHAPES]
    plans += [(op,"method",query,None) for op in OPS for query in ("result_size","result_numel")]
    plans += [("add","function","literal",reason)
              for reason in ("encrypted_branch","inplace","out_mutation")]
    for op,style,query,rejection in plans:
        name="shape_"+op+"_"+style+"_"+query if rejection is None else "reject_"+rejection
        folder=output/name;folder.mkdir()
        text=source(op,style,query,rejection)
        (folder/"models.py").write_text(text)
        array=np.asarray([.125,-.25],dtype=np.float64)
        stream=io.BytesIO();np.save(stream,array,allow_pickle=False);raw=stream.getvalue()
        (folder/"state_0000.npy").write_bytes(raw)
        weights=dict(public=dict(file="state_0000.npy",shape=[2],dtype=array.dtype.str,
                                 bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        manifest=dict(format="poseidon-trusted-pytorch-v8",entry="models.make_model",
            files={"models.py":hashlib.sha256(text.encode()).hexdigest()},
            factory=dict(args=[],kwargs={}),inputs=[dict(name="x",shape=[2,2])],
            outputs=["result"],weights=weights,dtype_policy="strict-float64-v1",
            approximations=dict(format="poseidon-pytorch-approximations-v1",
                                defaults={},modules=[],nodes=[]))
        (folder/"manifest.json").write_bytes(canonical(manifest))
        graph=None if rejection else mathematical_graph(name,op)
        if graph is not None:(folder/"operator-model.json").write_bytes(canonical(graph))
        rows.append(dict(id=name,operator=op,style=style,query=query,
            kind="rejection" if rejection else "positive",rejection=rejection,
            manifest=name+"/manifest.json",manifest_sha256=digest(manifest),
            graph=None if rejection else name+"/operator-model.json",
            graph_sha256=None if rejection else digest(graph),
            mathematical_group=None if rejection else "broadcast_"+op+"_flatten"))
    result=dict(format=VERSION,planned_positive=33,planned_rejections=3,
                mathematical_groups=3,cases=rows,frozen_benchmark_changed=False)
    result["binding"]=digest(result)
    (output/"catalog.json").write_bytes(canonical(result))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(generate(args.output)))
