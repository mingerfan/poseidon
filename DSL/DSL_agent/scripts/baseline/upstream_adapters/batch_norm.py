"""Real HE_BN adapter with explicit checked closure geometry.

Initially usable by the manual helper probe. This is not yet an Agent DSL call.
The upstream HE_BN, abstractBN, shapeClosure and MultParBN implementations remain
unchanged; this module only binds public parameters and the whole cipher Expr.
"""
import math
from types import SimpleNamespace
from benchmark_graph import validate,require

SLOTS=16384

def bind(model):
    check=validate(model)
    require(len(model["inputs"])==len(model["nodes"])==len(model["outputs"])==1,
            "Manual HE_BN probe requires one standalone BatchNorm node")
    node=model["nodes"][0];inp=model["inputs"][0]
    require(node["op"]=="batch_norm" and node["inputs"][0]==inp["name"] and
            model["outputs"][0]["value"]==node["outputs"][0],"Standalone BatchNorm binding")
    shape=inp["shape"]
    require(2<=len(shape)<=4 and shape[0]==1,"HE_BN closure currently requires batch one, rank two to four")
    channels=shape[1];height=shape[2] if len(shape)==4 else 1
    width=shape[-1] if len(shape)>2 else 1
    require(math.prod(shape)<=256,"HE_BN logical work bound")
    mean,var,gamma,beta=[model["constants"][name] for name in node["inputs"][1:]]
    require(all(type(v) is list and len(v)==channels for v in (mean,var,gamma,beta)),
            "HE_BN public channel parameter shape")
    period=1<<(math.prod(shape)-1).bit_length()
    require(SLOTS%period==0,"HE_BN closure must divide physical slots")
    geometry=dict(nt=SLOTS,bb=1.0,fh=1,fw=1,s=1,hi=height,wi=width,ki=1,ci=channels,co=channels)
    return dict(input_shape=shape,output_shape=check["output_shapes"][model["outputs"][0]["name"]],
        mean=mean,variance=var,gamma=gamma,beta=beta,eps=node["attrs"]["eps"],
        geometry=geometry,closure_period=period,logical_elements=math.prod(shape),
        model_input=inp["name"],model_output=model["outputs"][0]["name"])

def apply(model,cipher,helpers,mpcb):
    import numpy as np
    import torch
    spec=bind(model)
    geometry=mpcb.InferShapes(dict(spec["geometry"]))
    require(geometry["ni"]==geometry["no"]==1 and geometry["ko"]==1 and
            geometry["nt"]//geometry["po"]==spec["closure_period"],
            "Unexpected upstream BatchNorm closure geometry")
    # Public data only. Let the real upstream abstractBN derive G and H.
    bn=SimpleNamespace(weight=torch.tensor(spec["gamma"],dtype=torch.float64),
        bias=torch.tensor(spec["beta"],dtype=torch.float64),
        running_mean=torch.tensor(spec["mean"],dtype=torch.float64),
        running_var=torch.tensor(spec["variance"],dtype=torch.float64),eps=spec["eps"])
    close=mpcb.shapeClosure(**geometry)
    result=helpers.HE_BN(close,np.array([cipher],dtype=object),bn)
    require(type(result) is np.ndarray and result.shape==(1,),"HE_BN whole-expression return shape")
    record=dict(helper="HE_BN",source="poly/Func.py",closure="MPCB.shapeClosure.BN",
        parameter_derivation="MPCB.abstractBN",geometry=geometry,input_shape=spec["input_shape"],
        closure_period=spec["closure_period"],slot_count=SLOTS,returned_to_golden=True,
        unchanged_upstream_functions=True,bootstrap_removed=False,agent_generated=False)
    return result[0],record
