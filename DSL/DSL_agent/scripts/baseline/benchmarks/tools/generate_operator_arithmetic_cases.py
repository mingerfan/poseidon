"""Generate declared V3 arithmetic contexts without selecting on Agent success."""
import argparse,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
FORMAT="poseidon-operator-graph-v3"

def generate(output):
    output=Path(output)
    if output.exists():raise ValueError("Fresh arithmetic fixture directory")
    output.mkdir(parents=True)
    shapes={"scalar":[4],"tensor":[2,2],"broadcast":[2,2,2]}
    values={"scalar":.375,"tensor":[[.125,-.25],[.375,-.5]],"broadcast":[.125,-.25]}
    models=[]
    for op in ("add","subtract","multiply"):
        for context in ("scalar","tensor","broadcast"):
            for side in ("left","right"):
                name="arithmetic_"+op+"_"+context+"_"+side
                model=dict(format=FORMAT,id=name,inputs=[dict(name="x",shape=shapes[context])],
                    constants=dict(public=values[context]),
                    nodes=[dict(id="arithmetic",op=op,inputs=["public","x"] if side=="left" else ["x","public"],
                                attrs={},outputs=["y"])],outputs=[dict(name="result",value="y")])
                models.append((model,op,context,side))
    node=lambda name,op,refs,attrs={}:dict(id=name,op=op,inputs=refs,attrs=attrs,outputs=[name])
    fanout=dict(format=FORMAT,id="arithmetic_shared_multi_output",
        inputs=[dict(name="x",shape=[4]),dict(name="other",shape=[4])],
        constants=dict(scalar=.375,vector=[.125,-.25,.375,-.5]),
        nodes=[node("shared","multiply",["scalar","x"]),
               node("plus","add",["vector","shared"]),
               node("minus","subtract",["scalar","shared"]),
               node("fused","add",["plus","other"])],
        outputs=[dict(name="first",value="fused"),dict(name="second",value="minus")])
    matrix=dict(format=FORMAT,id="arithmetic_matrix_chain",
        inputs=[dict(name="x",shape=[2,2])],
        constants=dict(weight=[[.25,-.125],[.5,.375]],bias=[.125,-.25],gain=.375),
        nodes=[node("projection","matmul",["x","weight"]),
               node("contrast","subtract",["bias","projection"]),
               node("result","multiply",["gain","contrast"])],
        outputs=[dict(name="result",value="result")])
    branch=dict(format=FORMAT,id="arithmetic_broadcast_branches",
        inputs=[dict(name="x",shape=[2,2,2])],
        constants=dict(vector=[.125,-.25],bias=.375),
        nodes=[node("scaled","multiply",["vector","x"]),
               node("squared","square",["scaled"]),
               node("added","add",["bias","squared"]),
               node("difference","subtract",["bias","scaled"])],
        outputs=[dict(name="first",value="added"),dict(name="second",value="difference")])
    for model,context in ((fanout,"shared_multi_output"),(matrix,"matrix_chain"),(branch,"broadcast_branches")):
        models.append((model,"composition",context,None))
    cases=[]
    for model,op,context,side in models:
        name=model["id"]+".json";(output/name).write_bytes(canonical(model))
        cases.append(dict(id=model["id"],graph=name,partition=op,context=context,public_side=side,
                          original_sha256=digest(model)))
    catalog=dict(format="poseidon-operator-arithmetic-fixtures-v1",planned=21,cases=cases,
                 frozen_benchmark_changed=False,generated_from_mathematical_definitions=True)
    catalog["binding"]=digest(catalog);(output/"catalog.json").write_bytes(canonical(catalog))
    return catalog

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(generate(a.output)))
