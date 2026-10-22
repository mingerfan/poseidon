"""Manual chunk/helper integration fixtures; no provider output or model whitelist."""
import math,copy
from benchmark_suite import Builder
from benchmark_graph import digest
from upstream_candidate_helpers import CHUNK_PROFILE,manifest
from unified_chunk_layout import layout

def models():
    rows=[]
    for shape,m in (((5,),7),((13,),9),((2,3),2),((3,),7)):
        b=Builder([shape]);n=shape[-1]
        w=[[(i*3+j*5)%11/64-.0625 for i in range(n)] for j in range(m)]
        if shape==(3,):w=[[0.]*n for _ in range(m)]
        y=b.node("linear",["input0",b.const(w),b.const([j/64-.03125 for j in range(m)])])
        rows.append(("HE_Linear",b.finish(y),4))
    for shape in ((1,2,3),(2,3,2)):
        b=Builder([shape]);c=shape[1]
        y=b.node("batch_norm",["input0",b.const([.125]*c),b.const([1.+i/8 for i in range(c)]),
            b.const([.25-.5*(i%2) for i in range(c)]),b.const([i/32 for i in range(c)])],eps=1e-5)
        rows.append(("HE_MPBN",b.finish(y),4))
    b=Builder([(1,4,2,2)]);y=b.node("flatten",["input0"])
    y=b.node("linear",[y,b.const([[(i+j)%9/64-.0625 for i in range(16)] for j in range(3)]),b.const([.01,-.02,.03])])
    rows.append(("HE_ReshapeLinear",b.finish(y),4))
    b=Builder([(5,),(3,)])
    a=b.node("linear",["input0",b.const([[.125]*5]*7),b.const([j/32 for j in range(7)])])
    c=b.node("linear",["input1",b.const([[.25]*3]*4),b.const([.03125]*4)])
    g=b.finish(a);g["outputs"].append(dict(name="output1",value=c));rows.append(("HE_Linear",g,4))
    # One real cross-chunk helper inside a nontrivial graph, not a direct alias.
    b=Builder([(5,)]);x=b.node("negate",["input0"])
    y=b.node("linear",[x,b.const([[.125,-.0625,.25,.03125,-.125]]*3),b.const([.01,.02,.03])])
    y=b.node("square",[y]);rows.append(("HE_Linear",b.finish(y),4))
    for family in ("HE_MPBN","HE_ReshapeLinear"):
        original=next(g for f,g,p in rows if f==family)
        g=copy.deepcopy(original)
        g["nodes"].insert(0,dict(id="before_helper",op="negate",inputs=["input0"],outputs=["before_value"],attrs={}))
        g["nodes"][1]["inputs"][0]="before_value"
        rows.append((family,g,4))
        g=copy.deepcopy(original);value=g["outputs"][0]["value"]
        g["nodes"].append(dict(id="after_helper",op="square",inputs=[value],outputs=["after_value"],attrs={}))
        g["outputs"][0]["value"]="after_value"
        rows.append((family,g,4))
    return rows

def source(g,cap,p,family):
    plan=layout(g,p);env={}
    for entry in plan["inputs"]:env.setdefault(entry["name"],[]).append(entry["dsl_name"])
    names=[e["dsl_name"] for e in plan["inputs"]]+["zero_ct"]
    lines=['@hc.func("'+','.join(["c"]*len(names))+'")','def golden('+','.join(names)+'):']
    bound={}
    for name,spec in cap["helpers"].items():
        if name.startswith(family) and "binding" in spec:
            b=spec["binding"];bound.setdefault(b["node_id"],[]).append((name,b))
    needed=set();producers={v:n for n in g["nodes"] for v in n["outputs"]}
    def visit(value):
        if value not in producers:return
        node=producers[value]
        if node["id"] in needed:return
        needed.add(node["id"])
        args=([bound[node["id"]][0][1]["input_value"]] if node["id"] in bound else node["inputs"])
        for v in args:visit(v)
    for out in g["outputs"]:visit(out["value"])
    required=[]
    for i,node in enumerate(g["nodes"]):
        if node["id"] not in needed:continue
        values=[]
        if node["id"] in bound:
            for name,b in sorted(bound[node["id"]],key=lambda row:row[1]["output_chunk"]["index"]):
                lhs="v"+str(i)+"_"+str(b["output_chunk"]["index"])
                args=env[b["input_value"]]+["zero_ct"];lines.append("    "+lhs+" = "+name+"("+",".join(args)+")")
                values.append(lhs);required.append(name)
        else:
            for j,arg in enumerate(env[node["inputs"][0]]):
                lhs="v"+str(i)+"_"+str(j)
                if node["op"]=="negate":expr="-"+arg
                elif node["op"]=="square":expr=arg+"*"+arg
                else:raise ValueError("Unimplemented manual fixture rendering")
                lines.append("    "+lhs+" = "+expr);values.append(lhs)
        env[node["outputs"][0]]=values
    out=[v for o in g["outputs"] for v in env[o["value"]]]
    lines.append("    return "+(out[0] if len(out)==1 else "("+",".join(out)+")"))
    return chr(10).join(lines)+chr(10),required

def cases():
    rows=[]
    for i,(family,g,p) in enumerate(models()):
        g["id"]="chunk_helper_"+str(i)
        cap=manifest(g,CHUNK_PROFILE,p)
        program,required=source(g,cap,p,family)
        rows.append(dict(name=g["id"],model=g,model_sha256=digest(g),source=program,required_helpers=required,
                         configuration="seal-cpu-eva-w45-v1",profile=CHUNK_PROFILE,chunk_period=p,expected_calls=len(required)))
    return rows
