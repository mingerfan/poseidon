
"""Original HE_DS corpus plus shape/edge/offset manual acceptance fixtures."""
import copy
from benchmark_suite import generate,Builder
from benchmark_graph import digest
from upstream_candidate_helpers import manifest,DS_PROFILE
from upstream_fused_cases import source

def corpus():
    return [r["model"] for r in generate() if r["metadata"].get("helper")=="HE_DS"]

def row(model,name=None):
    g=copy.deepcopy(model)
    if name:g["id"]=name
    program,required=source(g,manifest(g,DS_PROFILE),"HE_DS")
    return dict(name=g["id"],model=g,model_sha256=digest(g),source=program,required_helpers=required,
                expected_calls=len(required),configuration="seal-cpu-eva-w45-v1",profile=DS_PROFILE)

def graph(shape,*,reverse=False,start=0,square=False,raw=False):
    b=Builder([shape]);y="input0";h,w=shape[2:]
    axes=(2,3) if reverse else (3,2)
    for axis in axes:y=b.node("slice",[y],axis=axis,start=start,stop=shape[axis],step=2)
    if square:y=b.node("square",[y])
    g=b.finish(y)
    if raw:g["outputs"].append(dict(name="raw_input",value="input0"))
    return g

def extra():
    shapes=[(1,1,1,1),(1,1,1,7),(1,3,3,5),(1,4,4,4),(1,5,3,3),(1,8,2,2),
            (1,1,8,8),(1,2,8,8),(1,1,16,16),(1,16,1,1),(1,1,1,256)]
    rows=[row(graph(shape),"ds_shape_"+str(i)) for i,shape in enumerate(shapes)]
    rows += [row(graph((1,2,5,7),reverse=True,start=1),"ds_offset_reversed_axes"),
             row(graph((1,2,4,4),square=True),"ds_then_square"),
             row(graph((1,3,3,3),raw=True),"ds_shared_raw_output")]
    g=graph((1,1,3,3));g["inputs"].append(dict(name="input1",shape=[1,1,3,3]))
    g["nodes"].insert(1,dict(id="unrelated",op="negate",inputs=["input1"],attrs={},outputs=["unrelated_value"]))
    g["outputs"].append(dict(name="other",value="unrelated_value"))
    rows.append(row(g,"ds_nonadjacent_two_inputs"))
    g=graph((1,1,4,4));g["inputs"].append(dict(name="input1",shape=[1,1,4,4]))
    for i,original in enumerate(copy.deepcopy(g["nodes"])):
        n=original;n["id"]="second_"+str(i);n["inputs"]=["input1" if i==0 else "second_value_0"]
        n["outputs"]=["second_value_"+str(i)];g["nodes"].append(n)
    g["outputs"].append(dict(name="second",value="second_value_1"))
    rows.append(row(g,"ds_two_bound_outputs"))
    return rows

def cases():
    return [row(g) for g in corpus()]+extra()

def directed_cases():
    rows={r["name"]:r for r in cases()}
    return [rows[name] for name in ("bench_helper_0056","bench_helper_0058","ds_then_square")]
