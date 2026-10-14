
"""Manual fused-helper fixtures; original corpus graphs are not rewritten."""
import copy
from benchmark_suite import generate
from benchmark_graph import digest
from upstream_candidate_helpers import manifest,FUSED_PROFILE

def corpus():
    return [(r["model"],r["metadata"]["helper"]) for r in generate()
            if r["metadata"].get("helper") in ("HE_ConvBN","HE_DwConv")]

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
            name,binding=bound[node["id"]];expr=name+'('+env[binding["input_value"]]+')';required.append(name)
        else:
            arg=env[node["inputs"][0]]
            if node["op"]=="negate":expr="-"+arg
            elif node["op"]=="square":expr=arg+"*"+arg
            else:raise ValueError("Fixture operator needs explicit renderer")
        lines.append("    "+lhs+" = "+expr);env[node["outputs"][0]]=lhs
    values=[env[o["value"]] for o in model["outputs"]]
    lines.append("    return "+(",".join(values) if len(values)==1 else "("+",".join(values)+")"))
    return chr(10).join(lines)+chr(10),required

def row(model,family,name=None):
    g=copy.deepcopy(model)
    if name:g["id"]=name
    program,required=source(g,manifest(g,FUSED_PROFILE),family)
    return dict(name=g["id"],model=g,model_sha256=digest(g),source=program,required_helpers=required,
                expected_calls=len(required),configuration="seal-cpu-eva-w45-v1",profile=FUSED_PROFILE)

def asymmetric():
    from upstream_spatial_mapped_cases import asymmetric as spatial
    rows=[]
    for index,g0 in enumerate(spatial()[:7]):
        g=copy.deepcopy(g0);conv=g["nodes"][0];co=len(g["constants"][conv["inputs"][1]])
        params=[[.125*(c-1) for c in range(co)],[.7+c*.2 for c in range(co)],
                [(.5 if c%2 else -.75) for c in range(co)],[.03*(c+1) for c in range(co)]]
        refs=[]
        for j,values in enumerate(params):
            key="fused_bn_param_"+str(j);g["constants"][key]=values;refs.append(key)
        g["nodes"].append(dict(id="fused_bn",op="batch_norm",attrs=dict(eps=.0001),
                             inputs=[conv["outputs"][0],*refs],outputs=["fused_bn_value"]))
        g["outputs"]=[dict(name="normalized",value="fused_bn_value")]
        if index==1:g["outputs"].append(dict(name="raw_conv",value=conv["outputs"][0]))
        rows.append(row(g,"HE_ConvBN","fused_asymmetric_"+str(index)))
        if conv["attrs"]["groups"]==co==g["inputs"][0]["shape"][1]:
            rows.append(row(g,"HE_DwConv","fused_dw_asymmetric_"+str(index)))
    g=copy.deepcopy(rows[0]["model"]);g["inputs"].append(dict(name="input1",shape=[1,1,3,3]))
    # An unrelated node between Conv and BN proves producer-edge binding,
    # and both named outputs remain independently decoded.
    g["nodes"].insert(1,dict(id="other_branch",op="negate",attrs={},inputs=["input1"],outputs=["other_value"]))
    g["outputs"].append(dict(name="other",value="other_value"))
    rows.append(row(g,"HE_ConvBN","fused_two_input_nonadjacent"))
    rows.append(row(g,"HE_DwConv","fused_dw_two_input_nonadjacent"))
    return rows

def cases():
    return [row(g,h) for g,h in corpus()]+asymmetric()

def directed_cases():
    rows={r["name"]:r for r in cases()}
    return [rows["bench_helper_"+str(i).zfill(4)] for i in (24,26,29,88,90,93)]
