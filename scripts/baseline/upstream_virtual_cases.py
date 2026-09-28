import copy,math
from benchmark_suite import Builder,generate
from benchmark_graph import digest
from upstream_adapters import virtual_node as adapter
from upstream_candidate_helpers import manifest,VR_PROFILE
def models(compositions=False):
    result=[]
    for row in generate():
        family=row["metadata"].get("helper")
        if family in adapter.FAMILIES:result.append((family,row["model"]))
    for shape in ((2,3,2),(1,2,3,2),(3,1),(1,4,1,1)):
        b=Builder([shape]);c=shape[1]
        y=b.node("batch_norm",["input0",b.const([.125]*c),b.const([1.+i/8 for i in range(c)]),
                b.const([(-.5 if i%2 else .25) for i in range(c)]),b.const([-.0625]*c)],eps=1e-5)
        result.append(("HE_MPBN",b.finish(y)))
    for shape,m in (((1,),4),((2,),7),((2,3),4),((2,2,3),2),((15,),3)):
        b=Builder([shape]);n=shape[-1];w=[[(j*n+i)%13/32-.125 for i in range(n)] for j in range(m)]
        y=b.node("linear",["input0",b.const(w),b.const([.03125*j for j in range(m)])])
        result.append(("HE_Linear",b.finish(y)))
    for shape in ((1,4,2,2),(1,8,2,2),(2,2,2,2)):
        b=Builder([shape]);n=math.prod(shape);y=b.node("flatten",["input0"])
        y=b.node("linear",[y,b.const([[(i*3+j*5)%17/64-.125 for i in range(n)] for j in range(3)]),b.const([.03125,-.0625,.125])])
        result.append(("HE_ReshapeLinear",b.finish(y)))
    for family in ("HE_MPBN","HE_Linear"):
        b=Builder([(1,2,2)] if family=="HE_MPBN" else [(3,)])
        if family=="HE_MPBN":
            y=b.node("batch_norm",["input0",b.const([.1,.2]),b.const([1.,2.]),b.const([0.,0.]),b.const([.25,-.5])],eps=1e-5)
        else:y=b.node("linear",["input0",b.const([[0.]*3]*2),b.const([.25,-.5])])
        result.append((family,b.finish(y)))
    if not compositions:return result
    for family in adapter.FAMILIES:
        g=copy.deepcopy(next(g for f,g in result if f==family and
            (family!='HE_ReshapeLinear' or g["inputs"][0]["shape"]==[1,4,2,2])))
        g["nodes"].insert(0,dict(id="prefix_neg",op="negate",inputs=["input0"],attrs={},outputs=["prefix_value"]))
        g["nodes"][1]["inputs"][0]="prefix_value"
        result.append((family,g))
        g=copy.deepcopy(g);value=g["outputs"][0]["value"]
        g["nodes"].append(dict(id="suffix_square",op="square",inputs=[value],attrs={},outputs=["squared_value"]))
        g["outputs"][0]["value"]="squared_value"
        result.append((family,g))
    return result

def source(model,cap,family):
    bound={s["binding"]["node_id"]:(name,s["binding"]) for name,s in cap["helpers"].items()
           if "binding" in s and not name.startswith(("HE_ConvBN","HE_DwConv"))}
    for name,s in cap["helpers"].items():
        if name.startswith(family):bound[s["binding"]["node_id"]]=(name,s["binding"])
    producers={v:n for n in model["nodes"] for v in n["outputs"]};needed=set()
    def visit(value):
        if value not in producers:return
        n=producers[value]
        if n["id"] in needed:return
        needed.add(n["id"])
        args=[bound[n["id"]][1]["input_value"]] if n["id"] in bound else n["inputs"]
        for v in args:visit(v)
    for output in model["outputs"]:visit(output["value"])
    env={entry["name"]:("x","y","z","t")[i] for i,entry in enumerate(model["inputs"])}
    lines=['@hc.func("'+','.join(['c']*(len(env)+1))+'")','def golden('+','.join([*env.values(),'zero_ct'])+'):']
    required=[]
    for i,node in enumerate(model["nodes"]):
        if node["id"] not in needed:continue
        lhs="v"+str(i)
        if node["id"] in bound:
            name,binding=bound[node["id"]];expr=name+'('+env[binding["input_value"]]+(',zero_ct' if binding.get('adapter')=='virtual-prefix-v1' else '')+')';required.append(name)
        else:
            arg=env[node["inputs"][0]]
            if node["op"]=="negate":expr="-"+arg
            elif node["op"]=="square":expr=arg+"*"+arg
            else:raise ValueError("Fixture operator needs explicit renderer")
        lines.append("    "+lhs+" = "+expr);env[node["outputs"][0]]=lhs
    values=[env[o["value"]] for o in model["outputs"]]
    lines.append("    return "+(",".join(values) if len(values)==1 else "("+",".join(values)+")"))
    return chr(10).join(lines)+chr(10),required

def cases():
    rows=[]
    for i,(family,model) in enumerate(models(compositions=True)):
        g=copy.deepcopy(model)
        if i>=24:g["id"]="virtual_"+str(i)+"_"+family
        program,required=source(g,manifest(g,VR_PROFILE),family)
        rows.append(dict(name=g["id"],model=g,model_sha256=digest(g),source=program,required_helpers=required,
                         expected_calls=len(required),configuration="seal-cpu-eva-w45-v1",profile=VR_PROFILE))
    return rows

def directed_cases():
    rows=cases()
    return [rows[i] for i in (0,8,16,38,39,40,41,42,43)]
