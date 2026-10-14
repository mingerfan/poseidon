"""Versioned capacity weight/Linear supplements; never modifies the frozen corpus."""
from benchmark_graph import CAPACITY_FORMAT,validate

def models():
    result=[]
    def build(kind,index,shape,out):
        g=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="capacity_"+kind+"_"+str(index),
            inputs=[dict(name="input",shape=shape)],constants={},nodes=[],outputs=[])
        def add(op,refs,attrs=None):
            v="v"+str(len(g["nodes"]))
            g["nodes"].append(dict(id="n"+str(len(g["nodes"])),op=op,inputs=refs,attrs=attrs or {},outputs=[v]))
            return v
        if kind=="public_tensor":
            width=shape[-1]
            g["constants"]["gain"]=[(j+1)/(64.*width) for j in range(width)]
            g["constants"]["offset"]=[((j*7)%13-6)/128. for j in range(width)]
            v=add("multiply",["input","gain"])
            v=add("add",[v,"offset"]);v=add("subtract",[v,"gain"])
        else:
            width=shape[-1]
            g["constants"]["offset"]=0.0625
            v=add("add",["input","offset"])
            g["constants"]["weight"]=[[(i*width+j+1)/(64.*out*width) for j in range(width)] for i in range(out)]
            # One row in the third shape is zero to exercise exact public-zero
            # terms without transparent ciphertext construction.
            if index==2:g["constants"]["weight"][-1]=[0.]*width
            g["constants"]["bias"]=[((-1.)**i)*(i+1)/64. for i in range(out)]
            v=add("linear",[v,"weight","bias"])
            if kind=="linear_square_linear":
                v=add("square",[v]);out2=2 if index<2 else 3
                g["constants"]["weight2"]=[[(2*i-j+1)/16. for j in range(out)] for i in range(out2)]
                g["constants"]["bias2"]=[(i-1)/128. for i in range(out2)]
                v=add("linear",[v,"weight2","bias2"])
        g["outputs"]=[dict(name="result",value=v)];validate(g);result.append(g)
    for i,s in enumerate(([257],[2,257],[1,2,1,129])):build("public_tensor",i,list(s),None)
    for i,(s,o) in enumerate((([257],3),([2,257],3),([1,2,1,129],4))):build("linear",i,s,o)
    for i,(s,o) in enumerate((([257],3),([2,257],4),([1,2,1,129],4))):build("linear_square_linear",i,s,o)
    return result
