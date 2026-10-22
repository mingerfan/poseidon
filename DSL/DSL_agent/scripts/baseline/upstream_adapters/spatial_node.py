"""Bindings for actual HE_Conv/HE_Avg/HE_Pool in an explicit periodic ring."""
import copy,math
from types import SimpleNamespace
from benchmark_graph import validate,require,canonical
from upstream_adapters.periodic_ring import PeriodicExpr,proxy_type

def pow2(n):return n>0 and n&(n-1)==0

def bind_node(model,node_id,period):
    checked=validate(model);nodes=[n for n in model["nodes"] if n["id"]==node_id]
    require(len(nodes)==1,"Spatial node identity");node=nodes[0];op=node["op"];a=node["attrs"]
    shape=checked["shapes"][node["inputs"][0]]
    require(len(shape)==4 and shape[0]==1,"Spatial helper requires batch-one rank4")
    _,ci,hi,wi=shape
    require(pow2(hi) and pow2(wi),"Spatial helper requires power-of-two spatial dimensions")
    require(type(period) is int and period in (4,8,16,32,64,128,256),"Spatial helper period")
    require(math.prod(shape)<=period,"Spatial intermediate exceeds common period")
    outshape=checked["shapes"][node["outputs"][0]]
    if op=="conv2d":
        require(a["groups"]==1 and a["dilation"]==[1,1],"HE_Conv groups/dilation")
        weight=model["constants"][node["inputs"][1]];co=len(weight);fh=len(weight[0][0]);fw=len(weight[0][0][0])
        require(fh%2==fw%2==1 and fh<=3 and fw<=3,"HE_Conv odd kernels up to 3")
        require(a["stride"] in ([1,1],[2,2]) and a["padding"]==[fh//2,fw//2],"HE_Conv same-padding equal stride")
        stride=a["stride"][0];kind="conv";helper="HE_Conv"
        params=dict(weight=copy.deepcopy(weight),bias=copy.deepcopy(model["constants"][node["inputs"][2]]) if len(node["inputs"])==3 else [0.]*co)
    elif op=="mean":
        require({i%4 for i in a["axes"]}=={2,3},"HE_Pool requires spatial mean axes")
        co=ci;fh=fw=1;params={};stride=1;kind="global_pool";helper="HE_Pool"
    elif op=="avg_pool2d":
        co=ci;fh,fw=a["kernel"];params={}
        if a["kernel"]==[hi,wi] and a["padding"]==[0,0] and outshape==[1,ci,1,1]:
            stride=1;kind="global_pool";helper="HE_Pool"
        else:
            require((a["kernel"]==[2,2] and a["padding"]==[0,0] and a["stride"]==[2,2]) or
                    (a["kernel"]==[3,3] and a["padding"]==[1,1] and a["stride"] in ([1,1],[2,2])
                     and a["count_include_pad"] is True),"HE_Avg kernel/padding/divisor")
            stride=a["stride"][0];kind="avg";helper="HE_Avg"
    else:raise ValueError("No spatial helper for operator")
    require(hi%stride==wi%stride==0 and math.prod(outshape)<=period,"Spatial output shape/period")
    # InferShapes equations reproduced independently, then compared with upstream.
    ho,wo=hi//stride,wi//stride;ki=1;ko=stride;ti=ci;to=(co+ko*ko-1)//(ko*ko)
    in_size=ci*hi*wi;packed_out=ko*ko*ho*wo*to
    require(packed_out<=period,"Spatial output packing exceeds period")
    pi=1<<int(math.floor(math.log2(period/in_size)))
    po=1<<int(math.floor(math.log2(period/packed_out)))
    geometry=dict(nt=period,bb=1.,fh=fh,fw=fw,s=stride,hi=hi,wi=wi,ki=ki,ci=ci,co=co,
                  ho=ho,wo=wo,ko=ko,ti=ti,to=to,ni=1,no=1,pi=pi,po=po,q=(co+pi-1)//pi)
    require(kind!="conv" or co<=period//(hi*wi),"Upstream HE_Conv output register indexing exceeds virtual-ring capacity")
    require(kind!="avg" or stride==2 or ci==1,"Upstream HE_Avg stride-one multi-channel indexing is not equivalent")
    positions=[]
    if kind=="global_pool":positions=list(range(ci))
    else:
        for c in range(co):
            t=c//(ko*ko);s1=c//ko%ko;s2=c%ko
            for h in range(ho):
                for w in range(wo):positions.append((((t*ho+h)*ko+s1)*wo+w)*ko+s2)
    require(len(positions)==math.prod(outshape) and max(positions)<period,"Spatial canonical output mapping")
    # Conservative work reservation, including signed rotations, masks and repacking.
    bits=period.bit_length()-1;q=geometry["q"]
    bound=4+2*bits
    if kind=="conv":bound+=fh*fw*bits+2*q*fh*fw+q*2*ci*(bits+1)+co*(bits+2)+2*bits+1
    elif kind=="avg":bound+=fh*fw*(bits+2)+ci*(bits+2)+2*bits
    else:bound+=2*bits+ci*(bits+2)
    if positions!=list(range(len(positions))):bound+=len(positions)*(bits+2)
    require(bound<=1024,"Spatial helper reserved work exceeds 1024")
    return dict(schema=1,kind=kind,helper=helper,node_id=node_id,input_value=node["inputs"][0],output_value=node["outputs"][0],
        input_shape=shape,output_shape=outshape,period=period,geometry=geometry,parameters=params,
        input_closure_period=period//pi,output_positions=positions,work=bound)

def apply(binding,cipher,helpers,mpcb,*,proxy_factory=None):
    import numpy as np
    import torch
    p=binding["period"];geometry=binding["geometry"]
    base={k:geometry[k] for k in ("nt","bb","fh","fw","s","hi","wi","ki","ci","co")}
    actual=mpcb.InferShapes(base)
    require(canonical(actual)==canonical(geometry),"Actual spatial geometry drift")
    record=dict(operations=[],rotations=[])
    x=(proxy_factory or proxy_type)(helpers.hc.Expr)(cipher,p,record,binding["work"])
    count=math.prod(binding["input_shape"]);closure=binding["input_closure_period"]
    x=x*[float(i<count) for i in range(p)]
    for step in (closure*(1<<j) for j in range((p//closure).bit_length()-1)):
        x=x+x.rotate(-step)
    a=np.array([x],dtype=object);close=mpcb.shapeClosure(**actual);kind=binding["kind"]
    if kind in ("conv","conv_bn","dw_conv_bn"):
        params=binding["parameters"]
        conv=SimpleNamespace(weight=torch.tensor(params["weight"],dtype=torch.float64),
                             bias=torch.tensor(params["bias"],dtype=torch.float64))
        if kind=="conv":result=helpers.HE_Conv(close,a,conv)
        else:
            fields=params["batch_norm"]
            bn=SimpleNamespace(**{key:torch.tensor(fields[key],dtype=torch.float64)
                for key in ("running_mean","running_var","weight","bias")},eps=fields["eps"])
            result=helpers.HE_ConvBN(close,a,conv,bn) if kind=="conv_bn" else helpers.HE_DwConv(close,a,conv,bn)
    elif kind=="downsample":result=helpers.HE_DS(close,a)
    elif kind=="avg":result=helpers.HE_Avg(close,a)
    else:result=helpers.HE_Pool(close,a)
    require(type(result) is np.ndarray and result.shape==(1,) and isinstance(result[0],PeriodicExpr),"Spatial whole-Expr result")
    value=result[0];positions=binding["output_positions"]
    if positions!=list(range(len(positions))):
        out=None
        for dst,src in enumerate(positions):
            term=value*[float(i==src) for i in range(p)]
            term=term.rotate(src-dst)
            out=term if out is None else out+term
        value=out
    value=value*[float(i<len(positions)) for i in range(p)]
    require(len(record["operations"])<=binding["work"],"Spatial helper operation accounting")
    require(value.value is not None,"Public-only helper needs explicit encrypted-zero materialization")
    return value.value,dict(helper=binding["helper"],source="poly/Func.py",geometry=actual,
        slot_embedding=dict(logical_period=p,physical_slots=16384,embedding="exact periodic repetition"),
        input_closure_period=closure,output_positions=positions,operation_count=len(record["operations"]),
        rotations=record["rotations"],unchanged_upstream_functions=True,bootstrap_removed=False,
        candidate_python_executed=False,output_contribution_proven=False,
        **({"normalization_parameters":{**{key:getattr(bn,key).detach().cpu().tolist()
                for key in ("running_mean","running_var","weight","bias")},"eps":bn.eps},
            "convolution_bias":conv.bias.detach().cpu().tolist(),
            "parameter_derivation":"mean_minus_original_conv_bias_then_actual_abstractBN"}
           if kind in ("conv_bn","dw_conv_bn") else {}),
        **({'public_zero_eliminations':record.get('public_zero_eliminations',[])} if proxy_factory else {}))

def verify_record(binding,record):
    expected=dict(helper=binding["helper"],source="poly/Func.py",geometry=binding["geometry"],
        slot_embedding=dict(logical_period=binding["period"],physical_slots=16384,embedding="exact periodic repetition"),
        input_closure_period=binding["input_closure_period"],output_positions=binding["output_positions"],
        operation_count=record.get("operation_count"),rotations=record.get("rotations"),
        unchanged_upstream_functions=True,bootstrap_removed=False,
        candidate_python_executed=False,output_contribution_proven=False,
        **({"normalization_parameters":binding["parameters"]["batch_norm"],
            "convolution_bias":binding["parameters"]["bias"],
            "parameter_derivation":"mean_minus_original_conv_bias_then_actual_abstractBN"}
           if binding["kind"] in ("conv_bn","dw_conv_bn") else {}))
    require(canonical(record)==canonical(expected),"Spatial actual record identity")
    count=record["operation_count"];rotations=record["rotations"];p=binding["period"]
    require(type(count) is int and 1<=count<=binding["work"] and type(rotations) is list
            and len(rotations)<=binding["work"],"Spatial actual work")
    steps=0
    for rotation in rotations:
        require(type(rotation) is dict and set(rotation)=={"requested","normalized","steps"}
                and type(rotation["requested"]) is int,"Spatial actual rotation fields")
        n=rotation["requested"]%p;parts=[1<<j for j in range(p.bit_length()-1) if n&(1<<j)]
        require(canonical(rotation)==canonical(dict(requested=rotation["requested"],normalized=n,steps=parts)),
                "Spatial actual rotation lowering")
        steps+=len(parts)
    require(steps<=count,"Spatial rotation operation count")
