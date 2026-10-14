"""Small owned pooling/normalization/padding shape contexts, separate from Benchmark."""
import ast,hashlib,io,json,math
from pathlib import Path
from benchmark_graph import canonical,digest,require
VERSION="poseidon-pytorch-operator-shape-contexts-v1"
PLANS=[
 dict(group="adaptive_pool",rank=1,shape=[1,1,5],output=[2]),
 dict(group="adaptive_pool",rank=2,shape=[1,2,3,4],output=[2,2]),
 dict(group="adaptive_pool",rank=2,shape=[2,1,4,5],output=[1,3]),
 dict(group="instance_norm",rank=1,shape=[1,2,3]),
 dict(group="instance_norm",rank=2,shape=[1,2,2,3]),
 dict(group="instance_norm",rank=1,shape=[2,1,4]),
 dict(group="padding",rank=1,shape=[1,1,3],mode="constant",pads=[1,2],value=.125),
 dict(group="padding",rank=2,shape=[1,2,2,3],mode="reflect",pads=[1,0,0,1],value=0.),
 dict(group="padding",rank=2,shape=[1,1,3,2],mode="replicate",pads=[0,1,1,0],value=0.),
 dict(group="average_pool_control",rank=1,shape=[1,1,5],kernel=[2],stride=[1],padding=[0],include=True),
 dict(group="average_pool_control",rank=2,shape=[1,2,3,4],kernel=[2,2],stride=[1,1],padding=[1,1],include=False),
 dict(group="average_pool_control",rank=2,shape=[2,1,4,5],kernel=[2,3],stride=[2,1],padding=[0,0],include=True),
]
def definition(plan,name):
    import numpy as np
    group=plan["group"];shape=plan["shape"];constants={};weights={}
    if group=="adaptive_pool":
        constructor="nn.AdaptiveAvgPool"+str(plan["rank"])+"d("+repr(tuple(plan["output"]))+")"
        call="F.adaptive_avg_pool"+str(plan["rank"])+"d(x, output_size="+repr(tuple(plan["output"]))+")"
        op="adaptive_avg_pool";attrs=dict(output_size=plan["output"]);arguments=["x"]
        output_shape=shape[:2]+plan["output"]
    elif group=="instance_norm":
        c=shape[1];index=np.arange(c,dtype=np.float64)
        constants=dict(mean=(.125+.0625*index).tolist(),variance=(1+.125*index).tolist(),
                       gain=(.75+.125*index).tolist(),offset=(-.125+.0625*index).tolist())
        for key,constant in (("running_mean","mean"),("running_var","variance"),("weight","gain"),("bias","offset")):
            weights["operation."+key]=np.asarray(constants[constant],dtype=np.float64)
        weights["operation.num_batches_tracked"]=np.asarray(0,dtype=np.int64)
        constructor="nn.InstanceNorm"+str(plan["rank"])+"d("+str(c)+", affine=True, track_running_stats=True, eps=1e-5)"
        call=("F.instance_norm(x, running_mean=self.operation.running_mean, running_var=self.operation.running_var, "
              "weight=self.operation.weight, bias=self.operation.bias, use_input_stats=False, momentum=.1, eps=1e-5)")
        op="batch_norm";attrs=dict(eps=1e-5);arguments=["x","mean","variance","gain","offset"];output_shape=shape
    elif group=="padding":
        prefix={"constant":"Constant","reflect":"Reflection","replicate":"Replication"}[plan["mode"]]
        constructor="nn."+prefix+"Pad"+str(plan["rank"])+"d("+repr(tuple(plan["pads"]))+(", .125)" if plan["mode"]=="constant" else ")")
        call="F.pad(x, "+repr(tuple(plan["pads"]))+", mode="+repr(plan["mode"])+", value="+repr(plan["value"])+")"
        pads=[[0,0] for _ in shape]
        for i in range(plan["rank"]):pads[-1-i]=plan["pads"][2*i:2*i+2]
        op="padding";attrs=dict(pads=pads,mode=plan["mode"],value=plan["value"]);arguments=["x"]
        output_shape=[n+sum(p) for n,p in zip(shape,pads)]
    else:
        r=str(plan["rank"]);constructor="nn.AvgPool"+r+"d("+repr(tuple(plan["kernel"]))+", stride="+repr(tuple(plan["stride"]))+", padding="+repr(tuple(plan["padding"]))+", count_include_pad="+repr(plan["include"])+")"
        call="F.avg_pool"+r+"d(x, kernel_size="+repr(tuple(plan["kernel"]))+", stride="+repr(tuple(plan["stride"]))+", padding="+repr(tuple(plan["padding"]))+", count_include_pad="+repr(plan["include"])+")"
        op="avg_pool"+r+"d";attrs=dict(kernel=plan["kernel"],stride=plan["stride"],padding=plan["padding"],count_include_pad=plan["include"])
        arguments=["x"];output_shape=shape[:2]+[(n+2*p-k)//s+1 for n,p,k,s in zip(shape[2:],plan["padding"],plan["kernel"],plan["stride"])]
    graph=dict(format="poseidon-operator-graph-v3",id=name,inputs=[dict(name="x",shape=shape)],constants=constants,
        nodes=[dict(id="operation",op=op,inputs=arguments,attrs=attrs,outputs=["features"]),
               dict(id="flatten",op="reshape",inputs=["features"],attrs=dict(shape=[math.prod(output_shape)]),outputs=["result"])],
        outputs=[dict(name="result",value="result")])
    return constructor,call,weights,graph
def generate(output):
    import numpy as np
    output=Path(output);require(not output.exists(),"Fresh operator shape contexts")
    output.mkdir(parents=True);rows=[]
    for index,plan in enumerate(PLANS):
        variants=(["module_literal","module_query","functional_literal","functional_query"]
                  if plan["group"] in ("adaptive_pool","instance_norm") else
                  ["module_literal","module_query","functional_query"] if plan["group"]=="padding" else
                  ["module_query","functional_query"])
        for variant in variants:
            name=plan["group"]+"_"+str(index%3)+"_"+variant
            folder=output/name;folder.mkdir()
            constructor,call,weights,graph=definition(plan,name)
            count=graph["nodes"][-1]["attrs"]["shape"][0]
            query=["y.reshape(y.numel())","y.reshape(y.size(0), -1).flatten()","y.reshape(y.shape[0], -1).flatten()"][index%3]
            expression=query if variant.endswith("query") else "y.reshape("+str(count)+")"
            source=('"""Owned explicit-trust inference source."""\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'
                'class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.operation='+constructor+'\n'
                '    def forward(self, x):\n        y='+(call if variant.startswith("functional") else "self.operation(x)")+'\n        return '+expression+'\n'
                'def make_model():\n    return Model().double().eval()\n')
            ast.parse(source);(folder/"models.py").write_text(source)
            descriptors={}
            for j,(key,array) in enumerate(sorted(weights.items())):
                stream=io.BytesIO();np.save(stream,array,allow_pickle=False);raw=stream.getvalue();file="state_"+str(j).zfill(4)+".npy"
                (folder/file).write_bytes(raw)
                descriptors[key]=dict(file=file,shape=list(array.shape),dtype=array.dtype.str,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            spec=dict(format="poseidon-trusted-pytorch-v9",entry="models.make_model",files={"models.py":hashlib.sha256(source.encode()).hexdigest()},
                factory=dict(args=[],kwargs={}),inputs=graph["inputs"],outputs=["result"],weights=descriptors,
                dtype_policy="strict-float64-v1",approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
            (folder/"manifest.json").write_bytes(canonical(spec));(folder/"operator-model.json").write_bytes(canonical(graph))
            rows.append(dict(id=name,group=plan["group"],variant=variant,shape=plan["shape"],rank=plan["rank"],
                manifest=name+"/manifest.json",graph=name+"/operator-model.json",manifest_sha256=digest(spec),graph_sha256=digest(graph),
                mathematical_group=plan["group"]+"_"+str(index%3),
                query="signature_literal" if variant.endswith("literal") else ("numel","size","shape")[index%3]))
    require(len(rows)==39 and len({r["mathematical_group"] for r in rows})==12,"Exact contexts")
    record=dict(format=VERSION,cases=rows,source_variants=39,mathematical_groups=12,frozen_benchmark_changed=False)
    record["binding"]=digest(record);(output/"catalog.json").write_bytes(canonical(record));return record
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();print(json.dumps(generate(args.output)))
