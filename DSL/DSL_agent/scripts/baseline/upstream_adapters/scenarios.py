"""Versioned asymmetric helper probes, separate from the frozen 1200 models."""
from benchmark_graph import digest
from benchmark_suite import Builder

BN_SHAPES=((1,1),(1,3),(1,3,2),(1,2,2,3),(1,4,4,4),(1,4,8,8),(1,3,3,3),(1,2,1,5))

def batch_norm_model(shape):
    channels=shape[1];b=Builder([shape])
    mean=b.const([(-1)**i*(i+1)/16 for i in range(channels)])
    var=b.const([.5+i/16 for i in range(channels)])
    gamma=b.const([(-1)**i*(1+i/32) for i in range(channels)])
    beta=b.const([(i-2)/32 for i in range(channels)])
    out=b.node("batch_norm",["input0",mean,var,gamma,beta],eps=1e-5)
    return b.finish(out)

def supplemental_rows():
    result=[]
    for i,shape in enumerate(BN_SHAPES):
        model=batch_norm_model(shape)
        model["id"]="helper_bn_asymmetric_v1_"+str(i)
        result.append(dict(model=model,model_sha256=digest(model),
                           metadata=dict(helper="HE_BN",scope="supplemental-asymmetric-v1")))
    return result
