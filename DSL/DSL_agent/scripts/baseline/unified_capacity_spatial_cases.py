"""Supplemental spatial capacity models; frozen semantic-v1 corpus is unchanged."""
import copy,math
from benchmark_graph import CAPACITY_FORMAT,validate

def weights(shape):
    counter=0
    def nested(dims):
        nonlocal counter
        if not dims:
            counter+=1
            return ((-1. if counter%3==0 else 1.)*(1+counter%5))/64.
        return [nested(dims[1:]) for _ in range(dims[0])]
    return nested(shape)

def models():
    result=[]
    def graph(op,i,shape,kernel,stride,padding,dilation=None,groups=1,outputs=None,bias=True,include=True):
        rank=1 if op.endswith("1d") else 2
        constants=dict(offset=0.0625)
        refs=["shifted"];attrs=dict(stride=stride,padding=padding)
        if op.startswith("conv"):
            channels=shape[-rank-1]
            constants["weight"]=weights([outputs,channels//groups,*kernel]);refs.append("weight")
            if bias:constants["bias"]=[((-1.)**i)*(i+1)/32. for i in range(outputs)];refs.append("bias")
            attrs.update(dilation=dilation,groups=groups)
        else:attrs.update(kernel=kernel,count_include_pad=include)
        g=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="capacity_"+op+"_"+str(i),
               inputs=[dict(name="input",shape=shape)],constants=constants,
               nodes=[dict(id="n0",op="add",inputs=["input","offset"],attrs={},outputs=["shifted"]),
                      dict(id="n1",op=op,inputs=refs,attrs=attrs,outputs=["result"])],
               outputs=[dict(name="result",value="result")])
        validate(g);result.append(g)
    for i,args in enumerate((
        ([2,131],[3],[1],[1],[1],1,2,True),
        ([2,2,67],[3],[2],[2],[2],2,2,True),
        ([1,4,129],[2],[1],[0],[3],2,4,False))):
        graph("conv1d",i,*args)
    for i,args in enumerate((
        ([1,17,17],[3,3],[1,1],[1,1],[1,1],1,2,True),
        ([1,2,13,11],[3,3],[2,2],[1,1],[1,1],2,2,True),
        ([2,2,9,9],[2,2],[1,1],[0,0],[2,2],2,2,True))):
        graph("conv2d",i,*args)
    for i,args in enumerate((
        ([2,131],[3],[2],[1],True),
        ([2,2,67],[3],[1],[1],False),
        ([1,3,101],[4],[3],[2],True))):
        shape,k,s,p,include=args
        graph("avg_pool1d",i,shape,k,s,p,include=include)
    for i,args in enumerate((
        ([1,17,17],[3,3],[2,2],[1,1],True),
        ([1,2,13,11],[2,3],[1,2],[1,1],False),
        ([2,2,9,9],[2,2],[2,2],[0,0],True))):
        shape,k,s,p,include=args
        graph("avg_pool2d",i,shape,k,s,p,include=include)
    g=copy.deepcopy(result[0]);g["id"]="capacity_spatial_composition_0"
    g["nodes"].extend([
        dict(id="n2",op="square",inputs=["result"],attrs={},outputs=["squared"]),
        dict(id="n3",op="avg_pool1d",inputs=["squared"],attrs=dict(kernel=[3],stride=[2],padding=[1],count_include_pad=False),outputs=["pooled"])])
    g["outputs"]=[dict(name="pooled",value="pooled")];validate(g);result.append(g)
    g=copy.deepcopy(result[3]);g["id"]="capacity_spatial_composition_1"
    g["constants"]["weight"]=g["constants"]["weight"][:1];g["constants"]["bias"]=g["constants"]["bias"][:1]
    g["nodes"].extend([
        dict(id="n2",op="add",inputs=["result","shifted"],attrs={},outputs=["residual"]),
        dict(id="n3",op="square",inputs=["residual"],attrs={},outputs=["squared"]),
        dict(id="n4",op="avg_pool2d",inputs=["squared"],attrs=dict(kernel=[3,3],stride=[2,2],padding=[1,1],count_include_pad=False),outputs=["pooled"])])
    g["outputs"]=[dict(name="residual",value="residual"),dict(name="pooled",value="pooled")];validate(g);result.append(g)
    g=copy.deepcopy(result[4]);g["id"]="capacity_spatial_composition_2"
    g["nodes"].extend([
        dict(id="n2",op="avg_pool2d",inputs=["result"],attrs=dict(kernel=[2,2],stride=[1,1],padding=[0,0],count_include_pad=True),outputs=["pooled"]),
        dict(id="n3",op="flatten",inputs=["pooled"],attrs={},outputs=["flat"])])
    length=math.prod(validate(g)["shapes"]["flat"])
    g["constants"]["linear_weight"]=weights([2,length])
    g["nodes"].append(dict(id="n4",op="linear",inputs=["flat","linear_weight"],attrs={},outputs=["logits"]))
    g["outputs"]=[dict(name="logits",value="logits")];validate(g);result.append(g)
    return result
