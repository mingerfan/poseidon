"""Versioned small extrema/padding cases, independent of Agent success."""
import copy
from model_decomposition import FORMAT
ABS=dict(method="polynomial",basis="power",domain=[-4.,4.],coefficients=[.5,0.,.25])

def models():
    rows=[]
    def add(op,shape,attrs,second=None):
        ident=op+"_"+str(sum(x["operator"]==op for x in rows))
        g=dict(format=FORMAT,id=ident,inputs=[dict(name="x",shape=list(shape))],constants={},
               nodes=[dict(id="operation",op=op,inputs=["x"],attrs=copy.deepcopy(attrs),outputs=["y"])],
               outputs=[dict(name="result",value="y")])
        if second is not None:
            g["inputs"].append(dict(name="other",shape=list(second)))
            g["nodes"][0]["inputs"].append("other")
        rows.append(dict(id=ident,operator=op,model=g))
    for mode in ("constant","reflect","replicate"):
        for shape,pads in [((3,),[[1,2]]),((2,2),[[0,1],[1,0]]),((1,2,2),[[0,0],[1,1],[0,1]])]:
            add("padding",shape,dict(pads=pads,mode=mode,value=.125 if mode=="constant" else 0))
    for op in ("maximum","minimum"):
        for lhs,rhs in [((3,),(3,)),((2,3),(1,3)),((1,2,2),(1,2,1))]:
            add(op,lhs,dict(approximation=ABS),rhs)
    for op in ("max","min"):
        for shape,axes,keep in [((4,),[0],False),((2,3),[-1],True),((1,2,2),[0,2],False)]:
            add(op,shape,dict(axes=axes,keepdims=keep,approximation=ABS))
    for shape,k,s,p,d in [((1,1,4),[2],[2],[0],[1]),((1,2,3),[2],[1],[1],[1]),((1,1,5),[2],[2],[0],[2])]:
        add("max_pool1d",shape,dict(kernel=k,stride=s,padding=p,dilation=d,approximation=ABS))
    for shape,k,s in [((1,1,2,2),[2,2],[2,2]),((1,1,2,3),[2,2],[1,1]),((1,2,2,2),[2,1],[2,1])]:
        add("max_pool2d",shape,dict(kernel=k,stride=s,padding=[0,0],dilation=[1,1],approximation=ABS))
    return rows
