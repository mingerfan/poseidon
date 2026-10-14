"""Supplemental capacity models; independent of the frozen 1200-model corpus."""
from benchmark_graph import CAPACITY_FORMAT,validate
def models():
    result=[]
    def graph(kind,index,shapes,ops):
        inputs=[dict(name="input"+str(i),shape=shape) for i,shape in enumerate(shapes)]
        constants={"bias":0.0625} if kind in ("elementwise","mean_tail") else {}
        nodes=[];ref=inputs[0]["name"]
        def add(op,refs,attrs=None):
            output="v"+str(len(nodes))
            nodes.append(dict(id="n"+str(len(nodes)),op=op,inputs=refs,attrs=attrs or {},outputs=[output]))
            return output
        if kind=="elementwise":
            ref=add("negate",[ref]);ref=add("add",[ref,"bias"]);ref=add("square",[ref])
        elif kind=="mean_tail":
            ref=add("add",[ref,"bias"])
            ref=add("mean",[ref],dict(axes=list(range(len(shapes[0]))),keepdims=False))
        elif kind=="scalar_broadcast":
            mean=add("mean",[ref],dict(axes=list(range(len(shapes[0]))),keepdims=False))
            ref=add("subtract",[ref,mean])
        elif kind=="binary":
            a=add("subtract",["input0","input1"]);b=add("add",["input0","input1"]);ref=add("multiply",[a,b])
        elif kind=="join_aligned":
            ref=add("concat",["input0","input1"],dict(axis=0))
        elif kind=="reshape_output":
            ref=add("reshape",[ref],dict(shape=ops))
        else:raise ValueError(kind)
        g=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="capacity_"+kind+"_"+str(index),
            inputs=inputs,constants=constants,nodes=nodes,outputs=[dict(name="result",value=ref)])
        validate(g);result.append(g)
    for i,s in enumerate(([257],[2,257],[1,1,1,769])):graph("elementwise",i,[list(s)],None)
    for i,s in enumerate(([257],[3,171],[1,1,1,1024])):graph("mean_tail",i,[list(s)],None)
    for i,s in enumerate(([257],[2,257],[1,1,1,769])):graph("scalar_broadcast",i,[list(s)],None)
    for i,s in enumerate(([257],[2,129],[1,1,1,384])):graph("binary",i,[list(s),list(s)],None)
    for i,s in enumerate(([256],[512],[1,2,256])):graph("join_aligned",i,[list(s),list(s)],None)
    for i,(s,out) in enumerate((([512],[2,256]),([3,256],[1,3,256]),([1,4,256],[1,1,4,256]))):
        graph("reshape_output",i,[s],out)
    return result
