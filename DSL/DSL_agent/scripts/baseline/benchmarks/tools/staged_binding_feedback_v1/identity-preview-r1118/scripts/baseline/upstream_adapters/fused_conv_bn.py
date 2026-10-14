
"""Immutable Conv -> BN subgraph bindings to unchanged upstream fused helpers."""
import copy,math
from benchmark_graph import validate,require,canonical
from upstream_adapters import spatial_mapped

ADAPTER="fused-conv-bn-v1"

def bind_node(model,node_id,period,helper="HE_ConvBN"):
    checked=validate(model)
    nodes=[n for n in model["nodes"] if n["id"]==node_id]
    require(len(nodes)==1 and nodes[0]["op"]=="batch_norm","Fused BN node identity")
    bn=nodes[0];producers={v:n for n in model["nodes"] for v in n["outputs"]}
    conv=producers.get(bn["inputs"][0])
    require(conv is not None and conv["op"]=="conv2d","Fused helper requires direct Conv2d producer")
    require(helper in ("HE_ConvBN","HE_DwConv"),"Fused helper identity")
    shape=checked["shapes"][conv["inputs"][0]];outshape=checked["shapes"][bn["outputs"][0]]
    require(len(shape)==4 and shape[0]==1,"Fused batch-one rank4")
    ci=shape[1];co=outshape[1]
    if helper=="HE_DwConv":
        require(conv["attrs"]["groups"]==ci==co,"HE_DwConv requires one filter per input channel")
    base=spatial_mapped.bind_node(model,conv["id"],period)
    require(base["helper"]=="HE_Conv","Fused convolution binding")
    if base.get("adapter")==spatial_mapped.ADAPTER:
        calls=copy.deepcopy(base["calls"])
    else:
        calls=[dict(inner=copy.deepcopy(base),
                    before=spatial_mapped.mapping([(i,i,1.) for i in range(math.prod(shape))],period),
                    after=spatial_mapped.mapping([(i,i,1.) for i in range(math.prod(outshape))],period))]
    mean,var,gamma,beta=[copy.deepcopy(model["constants"][key]) for key in bn["inputs"][1:]]
    bias=copy.deepcopy(model["constants"][conv["inputs"][2]]) if len(conv["inputs"])==3 else [0.]*co
    original=dict(op="conv2d",attrs=copy.deepcopy(conv["attrs"]),
                  weight=copy.deepcopy(model["constants"][conv["inputs"][1]]),bias=bias)
    fields=dict(running_mean=[m-b for m,b in zip(mean,bias)],running_var=var,weight=gamma,bias=beta,eps=bn["attrs"]["eps"])
    require(all(math.isfinite(v) and abs(v)<=1024 for key,vs in fields.items()
                for v in (vs if isinstance(vs,list) else [vs])),"Fused derived public parameter bound")
    for call in calls:
        inner=copy.deepcopy(call["inner"]);call["inner"]=inner;params=inner["parameters"]
        inner["kind"]="conv_bn" if helper=="HE_ConvBN" else "dw_conv_bn";inner["helper"]=helper
        if helper=="HE_DwConv":
            require(inner["geometry"]["s"]==1,"HE_DwConv stride-one closure only")
            params["weight"]=[[channel[c]] for c,channel in enumerate(params["weight"])]
        params["bias"]=[0.]*co;params["batch_norm"]=copy.deepcopy(fields)
        inner["work"]+=8
    work=sum(c["inner"]["work"]+spatial_mapped.map_work(c["before"])+spatial_mapped.map_work(c["after"])
             for c in calls)+len(calls)-1
    require(work<=1024,"Fused reserved work exceeds 1024")
    return dict(schema=1,adapter=ADAPTER,helper=helper,node_id=node_id,conv_node_id=conv["id"],
                input_value=conv["inputs"][0],output_value=bn["outputs"][0],
                input_shape=shape,output_shape=outshape,period=period,work=work,calls=calls,original=original,
                original_batch_norm=dict(mean=mean,variance=var,gamma=gamma,beta=beta,eps=bn["attrs"]["eps"]),
                parameter_derivation="mean_minus_original_conv_bias_then_actual_abstractBN",
                public_zero_policy="exact_public_zero_intermediates_only_fail_public_only_result",
                transformation="fixed public maps and equivalent BN parameters around unchanged fused helper")

def apply(binding,cipher,helpers,mpcb):
    require(binding["adapter"]==ADAPTER,"Fused adapter identity")
    return spatial_mapped.execute_plan(binding,cipher,helpers,mpcb)

def verify_record(binding,record):
    require(binding["adapter"]==ADAPTER,"Fused adapter identity")
    spatial_mapped.verify_plan(binding,record)

def probe(binding,x):
    # Independent scalar Conv then BN; never used to construct expected answers.
    values=spatial_mapped.probe(binding,x)
    fields=binding["original_batch_norm"];spatial=math.prod(binding["output_shape"][2:])
    out=[]
    for c,(m,v,g,b) in enumerate(zip(fields["mean"],fields["variance"],fields["gamma"],fields["beta"])):
        out.extend((values[c*spatial+i]-m)*g/math.sqrt(v+fields["eps"])+b for i in range(spatial))
    return tuple(out+[0.]*(binding["period"]-len(out)))
