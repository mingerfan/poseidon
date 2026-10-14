"""Supplemental joins/reductions with dirty tails and retained hard boundary."""
from benchmark_graph import CAPACITY_FORMAT,validate

def models():
    result=[]
    def graph(op,index,shapes,attrs):
        inputs=[dict(name="input"+str(i),shape=s) for i,s in enumerate(shapes)]
        nodes=[]
        for i,s in enumerate(shapes):
            nodes.append(dict(id="n"+str(i),op="add",inputs=["input"+str(i),"offset"],attrs={},outputs=["shifted"+str(i)]))
        nodes.append(dict(id="n"+str(len(nodes)),op=op,inputs=["shifted"+str(i) for i in range(len(shapes))],
                          attrs=attrs,outputs=["result"]))
        g=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="capacity_"+op+"_axes_"+str(index),
               inputs=inputs,constants=dict(offset=0.0625),nodes=nodes,outputs=[dict(name="result",value="result")])
        validate(g);result.append(g)
    for i,(shapes,axis) in enumerate((([[257],[263]],0),([[2,129],[2,127]],1),
                                     ([[1,2,5,29],[1,2,5,13]],-1))):
        graph("concat",i,shapes,dict(axis=axis))
    for i,(shape,axis) in enumerate((([257],0),([2,129],1),([2,5,29],2))):
        graph("stack",i,[shape,shape],dict(axis=axis))
    for op in ("sum","mean"):
        for i,(shape,axes,keepdims) in enumerate((([3,129],[0],False),([1,2,5,29],[1,3],True),
                                                ([1,129,3],[-1],False))):
            graph(op,i,[shape],dict(axes=axes,keepdims=keepdims))
    graph("stack",3,[[2,5,29],[2,5,29]],dict(axis=-1))
    return result
