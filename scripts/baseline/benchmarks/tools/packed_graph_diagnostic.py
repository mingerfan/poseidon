"""Manual vectorized graph probe. Does not change request layouts or Agent answers."""
import math
import numpy as np
from benchmark_graph import validate
from unified_graph_lowering import Emitter,Scalar

def source(request):
    graph=request["model"];info=validate(graph)
    if request["layout"]["execution_abi"]!="unified-periodic-inputs-v1":
        raise ValueError("Single-period diagnostic only")
    em=Emitter(request)
    class View:
        def __init__(self,shape,cells):
            self.shape=tuple(shape);self.cells=list(cells);self.packed=None
            assert len(self.cells)==math.prod(self.shape)
        def pack(self):
            if len(self.cells)>em.period:
                raise ValueError("Intermediate logical view exceeds one ciphertext period")
            if self.packed is not None:return self.packed
            # Preserve contiguous prefixes without extraction or repeated polynomial work.
            if self.cells and all(isinstance(c,tuple) for c in self.cells):
                first=self.cells[0][0]
                if all(c[0].name==first.name and c[1]==i for i,c in enumerate(self.cells)):
                    self.packed=first;return first
            acc=em.zero
            for dst,cell in enumerate(self.cells):
                if isinstance(cell,tuple):
                    expr,src=cell;part=em.rotate(expr,src)
                    part=em.emit(part.name+" * mask0")
                    part=em.rotate(part,-dst)
                else:
                    if float(cell)==0.:continue
                    # Only existing named public masks and prepared scalar values.
                    part=em.zero+float(cell)
                    part=em.emit(part.name+" * mask0")
                    part=em.rotate(part,-dst)
                acc=acc+part
            self.packed=acc
            return acc
    def view(expr,shape):
        return View(shape,[(expr,i) for i in range(math.prod(shape))])
    values={k:np.asarray(v,dtype=np.float64) for k,v in graph["constants"].items()}
    for spec in graph["inputs"]:
        binding=next(b for b in request["layout"]["inputs"] if b["name"]==spec["name"])
        values[spec["name"]]=view(Scalar(em,binding["dsl_name"]),spec["shape"])
    for node in graph["nodes"]:
        op=node["op"];a=node["attrs"];args=[values[n] for n in node["inputs"]]
        assert len(node["outputs"])==1
        shape=info["shapes"][node["outputs"][0]]
        if op=="slice":
            x=args[0];assert isinstance(x,View)
            indices=np.arange(len(x.cells)).reshape(x.shape)
            sl=[slice(None)]*len(x.shape);sl[a["axis"]]=slice(a["start"],a["stop"],a["step"])
            y=View(shape,[x.cells[int(i)] for i in indices[tuple(sl)].flat])
        elif op=="concat":
            arrays=[]
            for x in args:
                array=np.empty(x.shape,dtype=object)
                cells=x.cells if isinstance(x,View) else list(x.flat)
                for index,cell in zip(np.ndindex(x.shape),cells):array[index]=cell
                arrays.append(array)
            y=View(shape,list(np.concatenate(arrays,axis=a["axis"]).flat))
        elif op in ("add","subtract","multiply"):
            def operand(x):
                if isinstance(x,View):
                    if x.shape!=tuple(shape):raise ValueError("No encrypted broadcast in this probe")
                    return x.pack()
                if x.size and np.all(x==x.flat[0]):return float(x.flat[0])
                raise ValueError("No nonuniform public operand in this probe")
            left,right=map(operand,args)
            z=left+right if op=="add" else left-right if op=="subtract" else left*right
            y=view(z,shape)
        elif op=="negate":
            # Unary lane-local arithmetic commutes with this logical gather.
            # Do not pack a padded concat larger than P and alias its tail to slot 0.
            x=args[0];negated={};cells=[]
            for cell in x.cells:
                if isinstance(cell,tuple):
                    expr,index=cell
                    if expr.name not in negated:negated[expr.name]=-expr
                    cells.append((negated[expr.name],index))
                else:cells.append(-cell)
            y=View(shape,cells)
        elif op=="polynomial":
            from chebyshev_lowering import balanced
            if a["basis"]!="chebyshev":raise ValueError("Chebyshev probe only")
            y=view(balanced(args[0].pack(),args[1],em.zero),shape)
        else:raise ValueError("Unsupported packed probe operation")
        values[node["outputs"][0]]=y
    outputs=[values[o["value"]].pack().name for o in graph["outputs"]]
    names=[b["dsl_name"] for b in request["layout"]["inputs"]]+["zero_ct"]
    return '@hc.func("'+",".join("c" for _ in names)+'")\ndef golden('+",".join(names)+'):\n'+"\n".join(em.lines)+"\n    return ["+",".join(outputs)+"]\n"
