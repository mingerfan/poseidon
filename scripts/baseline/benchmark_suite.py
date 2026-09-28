"""Deterministic benchmark construction; no ID-based lowering or model inference."""
import copy
import itertools
import json
import math
from pathlib import Path
from benchmark_graph import FORMAT, OPS, infer, validate, signature, digest

ROOT=Path(__file__).resolve().parents[2]
QUOTAS={"primitive":240,"composition":360,"graph":360,"boundary":120,"helper":120}
VERSION="poseidon-semantic-benchmark-v1"
HELPERS=("HE_BN","HE_MPBN","HE_Conv","HE_ConvBN","HE_MaxPad","HE_Max","HE_Avg",
         "HE_DS","HE_Pool","HE_Linear","HE_ReshapeLinear","HE_DwConv","HE_Concat","HE_ReLU","HE_SiLU")
BOOTSTRAP={"HE_MaxPad","HE_Max","HE_ReLU"}


class Builder:
    def __init__(self, shapes):
        self.g=dict(format=FORMAT,id="pending",inputs=[dict(name="input"+str(i),shape=list(s))
                    for i,s in enumerate(shapes)],constants={},nodes=[],outputs=[])
        self.shapes={x["name"]:tuple(x["shape"]) for x in self.g["inputs"]}
    def const(self, value):
        from benchmark_graph import shape
        name="c"+str(len(self.g["constants"]))
        self.g["constants"][name]=value; self.shapes[name]=shape(value)
        return name
    def node(self, op, refs, **attrs):
        ss=infer(op,[self.shapes[x] for x in refs],attrs)
        name="n"+str(len(self.g["nodes"]))
        names=[name+"v"+str(i) for i in range(len(ss))]
        self.g["nodes"].append(dict(id=name,op=op,inputs=refs,attrs=attrs,outputs=names))
        self.shapes.update(zip(names,ss))
        return names if len(names)>1 else names[0]
    def finish(self, *outputs):
        self.g["outputs"]=[dict(name="output"+str(i),value=x) for i,x in enumerate(outputs)]
        validate(self.g)
        return copy.deepcopy(self.g)


def matrix(rows, cols):
    return [[((i*7+j*3)%9-4)/(16*max(1,cols)) for j in range(cols)] for i in range(rows)]


def apply(b, x, op, variant=0):
    s=b.shapes[x]; n=math.prod(s)
    if op in ("add","subtract","multiply"):
        choices=(0.125,[0.125],nested(s,0.125))
        return b.node(op,[x,b.const(choices[variant%3])])
    if op in ("square","negate","flatten"):
        return b.node(op,[x])
    if op=="power": return b.node(op,[x],exponent=2 if variant%2==0 else 4)
    if op=="linear":
        width=1+variant%min(4,s[-1])
        return b.node(op,[x,b.const(matrix(width,s[-1])),b.const([0.0625]*width)])
    if op=="reshape": return b.node(op,[x],shape=([1,n] if len(s)==1 else [n]))
    if op=="transpose": return b.node(op,[x],dim0=0,dim1=len(s)-1)
    if op=="permute": return b.node(op,[x],dims=list(reversed(range(len(s)))))
    if op=="rotate": return b.node(op,[x],step=(-1 if variant%2 else 1)*min(1+variant%3,n-1))
    if op in ("sum","mean"):
        return b.node(op,[x],axes=[variant%len(s)],keepdims=bool(variant%2))
    if op=="slice":
        return b.node(op,[x],axis=-1,start=0,stop=s[-1],step=1+variant%2)
    if op=="split":
        p=b.node(op,[x],axis=-1,sections=[1,s[-1]-1])
        return b.node("concat",p[::-1] if variant%2 else p,axis=-1)
    if op=="concat": return b.node(op,[x,b.node("negate",[x])],axis=-1)
    if op=="stack": return b.node(op,[x,b.node("negate",[x])],axis=0)
    if op=="batch_norm":
        if len(s)<2: x=b.node("reshape",[x],shape=[1,*s]); s=b.shapes[x]
        c=s[1]
        return b.node(op,[x,b.const([0.125]*c),b.const([1.0]*c),
                         b.const([0.5]*c),b.const([-0.0625]*c)],eps=1e-5)
    if op=="polynomial":
        return b.node(op,[x,b.const([0.0625,-0.125,0.03125,0.015625])],
                      basis="power" if variant%2 else "chebyshev")
    raise ValueError(op)


def nested(shape, value):
    if not shape: return value
    return [nested(shape[1:],value) for _ in range(shape[0])]


def spatial(b, x, rank, variant, kind="conv"):
    s=b.shapes[x]; dims=s[-rank:]; c=s[-rank-1]
    if kind=="conv":
        k=[min(2,n) for n in dims]; groups=c if variant%3==0 else 1
        w=nested((c,c//groups,*k),0.0625)
        return b.node("conv"+str(rank)+"d",[x,b.const(w),b.const([0.03125]*c)],
                      stride=[1+variant%2]*rank,padding=[variant%2]*rank,
                      dilation=[1]*rank,groups=groups)
    return b.node("avg_pool"+str(rank)+"d",[x],kernel=[min(2,n) for n in dims],
                  stride=[1+variant%2]*rank,padding=[variant%2]*rank,count_include_pad=bool(variant%2))


def primitive_candidates():
    operations=[x for x in OPS if not x.startswith(("conv","avg_pool"))]
    for variant in range(24):
        for op in operations:
            n=2+variant%7
            s=(n,) if variant<7 else (1,n) if variant<14 else (2,n)
            b=Builder([s])
            try: yield b.finish(apply(b,"input0",op,variant))
            except ValueError: continue
        for rank in (1,2):
            for kind in ("conv","avg"):
                for batched in (False,True):
                    s=((1,) if batched else ())+((1 if variant%2 else 2),)+(3+variant%2,)*rank
                    b=Builder([s])
                    yield b.finish(spatial(b,"input0",rank,variant,kind))


def composition_candidates():
    # Seed real spatial/shape compositions before the combinatorial chain stream.
    for rank in (1,2):
        for variant in range(12):
            b=Builder([(1,1)+(3,)*rank])
            x=spatial(b,"input0",rank,variant)
            x=b.node("square",[x])
            x=apply(b,x,"batch_norm")
            x=spatial(b,x,rank,variant,"avg")
            x=apply(b,x,"flatten"); x=apply(b,x,"linear",variant)
            yield b.finish(x)
    ops=("linear","square","negate","add","subtract","multiply","reshape","polynomial","mean")
    patterns=list(itertools.product(ops,repeat=3))
    # Distribute shapes across many topologies instead of filling one topology first.
    for variant in range(3):
        for pattern in patterns:
            if len(set(pattern))==1: continue
            b=Builder([(2+variant,)])
            x="input0"
            try:
                for i,op in enumerate(pattern): x=apply(b,x,op,i+variant)
                yield b.finish(x)
            except ValueError: continue
    for rank in (1,2):
        for variant in range(12):
            b=Builder([(1,)+(3,)*rank])
            x=spatial(b,"input0",rank,variant)
            x=b.node("square",[x]); x=spatial(b,x,rank,variant,"avg")
            x=apply(b,x,"flatten"); x=apply(b,x,"linear",variant)
            yield b.finish(x)


def graph_candidates():
    unary=("negate","square","linear","add","multiply","polynomial")
    for variant in range(6):
        for index,(left,right,merge) in enumerate(itertools.product(unary,unary,("add","subtract","multiply","concat","stack"))):
            count=2+(index+variant)%3; n=2+variant%3
            shapes=[(n,)]*count
            if variant>=3: shapes[-1]=(1,n)  # Different input ranks, explicit normalization.
            b=Builder(shapes)
            a=apply(b,"input0",left,variant)
            # Keep merge shapes compatible even when one branch uses Linear.
            width=b.shapes[a][-1]
            r=apply(b,"input1",right,variant)
            if b.shapes[r]!=b.shapes[a]:
                r=b.node("linear",[r,b.const(matrix(width,b.shapes[r][-1])),b.const([0.03125]*width)])
                if b.shapes[r]!=b.shapes[a]: r=b.node("reshape",[r],shape=list(b.shapes[a]))
            if merge in ("concat","stack"): y=b.node(merge,[a,r],axis=0)
            else: y=b.node(merge,[a,r])
            for j in range(2,count):
                extra=b.node("mean",["input"+str(j)],axes=list(range(len(shapes[j]))),keepdims=False)
                y=b.node("add",[y,extra])
            # Explicit split/rejoin and shared dependencies provide non-chain motifs.
            if index%7==0 and b.shapes[y][-1]>1:
                parts=b.node("split",[y],axis=-1,sections=[1,b.shapes[y][-1]-1])
                y=b.node("concat",parts[::-1],axis=-1)
            out=[y]
            if index%3==0: out.append(a)
            if index%5==0: out.append(r)
            yield b.finish(*out)


def boundary_candidates():
    # Explicit four-output, different-rank and total-capacity boundaries come
    # first; subsequent parameter partitions round-robin over all tensor sizes.
    for shapes in ([(4,),(2,2),(1,2,2),(1,1,2,2)],[(64,)]*4):
        b=Builder(shapes)
        outputs=[b.node("square",["input"+str(i)]) for i in range(4)]
        yield b.finish(*outputs)
    for variant in range(12):
        for n in (1,5,7,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256):
            s=(n,) if variant%3==0 else (1,n) if variant%3==1 else (1,1,1,n)
            b=Builder([s]); x="input0"
            x=apply(b,x,("negate","square","add","subtract","multiply","polynomial")[variant%6],variant)
            if variant%4==0: x=b.node("subtract",[x,x])
            elif n>1 and variant%4==1: x=apply(b,x,"rotate",variant)
            elif variant%4==2: x=apply(b,x,"mean",variant)
            if math.prod(b.shapes[x])>16 and variant>=2: x=b.node("mean",[x],axes=list(range(len(b.shapes[x]))),keepdims=False)
            yield b.finish(x)


def coefficient_tables():
    root=ROOT/"third_party/dacapo/python/poly/poly/data"
    raw=[float(x.strip()) for x in (root/"sgn151527.txt").read_text().splitlines() if x.strip()]
    return [(raw[:16],2.0),(raw[16:32],1.7),(raw[32:60],2.0)], [
        float(x.strip()) for x in (root/"coeffStr.txt").read_text().splitlines() if x.strip()]


def sign_graph(b,x,tables):
    for coeff,scale in tables:
        x=b.node("polynomial",[x,b.const([v/scale for v in coeff])],basis="chebyshev")
    return x


def helper_candidates():
    tables,silu=coefficient_tables()
    for helper in HELPERS:
        for variant in range(8):
            meta=dict(helper=helper,actual_helper_executed=False,
                      target="mathematical_specification_not_upstream_execution",
                      blocker="bootstrap_backend" if helper in BOOTSTRAP else "upstream_helper_adapter_unverified")
            n=2+variant
            if helper in ("HE_ReLU","HE_SiLU"):
                b=Builder([(n,)])
                x="input0"
                if helper=="HE_ReLU": y=sign_graph(b,x,tables)
                else: y=b.node("polynomial",[x,b.const([v if i%2 else 0.0 for i,v in enumerate(silu)])],basis="chebyshev")
                y=b.node("add",[y,b.const(0.5)]); y=b.node("multiply",[x,y])
                meta["ideal"]=dict(op="relu" if helper=="HE_ReLU" else "silu")
                meta["coefficient_basis"]="chebyshev"
                meta["coefficient_projection"]="MPCB.GenPoly odd leaves; original even coefficients not evaluated"
            elif helper in ("HE_Max","HE_MaxPad"):
                # Two-element pooling along the last axis; explicit padding distinguishes MaxPad.
                size=2+2*(variant%4)
                b=Builder([(1,1,size)])
                x="input0"
                if helper=="HE_MaxPad":
                    zero=b.node("multiply",[x,b.const(0.0)])
                    zero=b.node("slice",[zero],axis=-1,start=0,stop=1,step=1)
                    x=b.node("concat",[zero,x,zero],axis=-1)
                if variant>=4: x=b.node("negate",[x])
                size=b.shapes[x][-1]
                left=b.node("slice",[x],axis=-1,start=0,stop=size,step=2)
                right=b.node("slice",[x],axis=-1,start=1,stop=size,step=2)
                delta=b.node("subtract",[left,right])
                # Keep difference in declared polynomial approximation domain.
                delta=b.node("multiply",[delta,b.const(0.5)])
                sign=sign_graph(b,delta,tables)
                term=b.node("multiply",[delta,sign])
                term=b.node("multiply",[term,b.const(2.0)])
                avg=b.node("add",[left,right]); avg=b.node("multiply",[avg,b.const(0.5)])
                y=b.node("add",[avg,term])
                meta["ideal"]=dict(op="max_pairs",padding=helper=="HE_MaxPad",negate=variant>=4)
                meta["difference_scale"]=0.5
            elif helper in ("HE_BN","HE_MPBN"):
                shape=(1,n) if helper=="HE_BN" else (1,1,n)
                b=Builder([shape]); y=apply(b,"input0","batch_norm",variant)
            elif helper in ("HE_Linear","HE_ReshapeLinear"):
                shape=(n,) if helper=="HE_Linear" else (1,n)
                b=Builder([shape]); x="input0"
                if helper=="HE_ReshapeLinear": x=b.node("flatten",[x])
                y=apply(b,x,"linear",variant)
                if helper=="HE_Linear": y=b.node("negate",[y])
            elif helper=="HE_Concat":
                b=Builder([(n,),(n,)])
                y=b.node("concat",["input0","input1"],axis=0)
                if 2*n>16: y=b.node("sum",[y],axes=[0],keepdims=True)
            else:
                size=3+variant%2; channels=1+variant//4
                b=Builder([(1,channels,size,size)])
                x="input0"
                if variant%4>=2: x=b.node("negate",[x])
                if helper in ("HE_Conv","HE_ConvBN","HE_DwConv"):
                    y=spatial(b,x,2,variant if helper!="HE_DwConv" else 0)
                    if helper in ("HE_ConvBN","HE_DwConv"): y=apply(b,y,"batch_norm")
                    if helper=="HE_DwConv": y=b.node("square",[y])
                elif helper=="HE_DS":
                    y=b.node("slice",[x],axis=-1,start=0,stop=size,step=2)
                    y=b.node("slice",[y],axis=-2,start=0,stop=size,step=2)
                elif helper=="HE_Pool":
                    y=b.node("mean",[x],axes=[-2,-1],keepdims=True)
                else:
                    y=spatial(b,x,2,variant,"avg")
                if math.prod(b.shapes[y])>16: y=b.node("mean",[y],axes=[-2,-1],keepdims=False)
            yield b.finish(y),meta


def generate():
    from benchmark_bridge import from_legacy
    anchors=json.loads((ROOT/"scripts/baseline/cases/user-graph-suite-v2.json").read_text())["cases"]
    selected=[]; seen={signature(from_legacy(g)) for g in anchors}; counts={k:0 for k in QUOTAS}
    def add(model,category,meta=None):
        sig=signature(model)
        if sig in seen: return False
        model["id"]="bench_"+category+"_"+str(counts[category]).zfill(4)
        seen.add(sig); counts[category]+=1
        selected.append(dict(model=model,category=category,signature=sig,
                             topology=signature(model,True),metadata=meta or {}))
        return True
    # Helpers first, so generic models cannot consume their mathematical signatures.
    for model,meta in helper_candidates():
        if not add(model,"helper",meta):
            raise ValueError("Duplicate helper mathematical specification: "+meta["helper"])
    streams={"primitive":primitive_candidates,"composition":composition_candidates,
             "graph":graph_candidates,"boundary":boundary_candidates}
    for category,stream in streams.items():
        for model in stream():
            if counts[category]==QUOTAS[category]: break
            add(model,category)
        if counts[category]!=QUOTAS[category]: raise ValueError("Insufficient unique models: "+str(counts))
    # Stable topology-group allocation. Size variants cannot cross splits.
    groups={}
    for row in selected: groups.setdefault(row["topology"],[]).append(row)
    assigned={"development":0,"validation":0,"holdout":0}
    targets={"development":720,"validation":240,"holdout":240}
    for key in sorted(groups,key=lambda x:(-len(groups[x]),digest(["split-v1",x]))):
        split=max(targets,key=lambda s:targets[s]-assigned[s])
        for row in groups[key]: row["split"]=split
        assigned[split]+=len(groups[key])
    selected.sort(key=lambda r:r["model"]["id"])
    for row in selected:
        row["model_sha256"]=digest(row["model"])
        row["reference_status"]="unverified"
        row["agent_status"]="unverified"
    return selected


def summary(rows):
    from collections import Counter
    small=0
    for row in rows:
        g=row["model"]; check=validate(g)
        small+=int(sum(math.prod(s["shape"]) for s in g["inputs"])<=32 and
                   sum(math.prod(s) for s in check["output_shapes"].values())<=16 and len(g["nodes"])<=16)
    return dict(version=VERSION,models=len(rows),categories=dict(Counter(r["category"] for r in rows)),
                unique_signatures=len({r["signature"] for r in rows}),
                topology_groups=len({r["topology"] for r in rows}),small_models=small,
                split_counts=dict(Counter(r["split"] for r in rows)),
                model_set_sha256=digest([r["model_sha256"] for r in rows]),
                encrypted_execution=False,agent_calls=0)
