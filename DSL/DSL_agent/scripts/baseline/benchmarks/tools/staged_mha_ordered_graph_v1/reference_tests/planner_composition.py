"""Test-only graph composition around the fixed planner, not production intake."""
import copy,hashlib,importlib.util,sys
from pathlib import Path
from cases import masks
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[3]
LEGACY=BASE/"benchmarks/tools/static_mha_source_draft_v1"
FROZEN={"mha_source_plan.py":"f1f845815f671091a71b9b86217c5b68644d704b736ed7a237f638dec6164f49",
        "mha_source_reference.py":"25bdaf114101b57fe3b3f8b78f6fd2eb4e77e97f0e357851aef4e86b849817c5"}

def legacy(name):
 path=LEGACY/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN[name]:raise ValueError("frozen planner/reference changed")
 if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
 spec=importlib.util.spec_from_file_location("_ordered_mha_math_"+path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def build_graph(case,module,signature,approximation):
 from model_decomposition import FORMAT,decompose
 planner=legacy("mha_source_plan.py")
 qshape=list(signature[0]["shape"]);qshape[-1]=case["e"];qname="pre_projected" if module.pre is not None else "q"
 shapes={x["name"]:x["shape"] for x in signature};operands={}
 for role,name in (("query",qname),("key",qname if case.get("self_attention") else "k"),("value",qname if case.get("self_attention") else "v")):
  operands[role]=dict(kind="cipher",name=name,shape=qshape if name==qname else shapes[name],dtype="float64")
 am,pm=masks(case);options=dict(attn_mask=am,key_padding_mask=pm,need_weights=case["need"],average_attn_weights=case["average"],is_causal=False)
 mini=planner.plan(planner.snapshot_module(module.attention),operands,options,approximation,resource_profile="general-tensors-v1")
 graph=copy.deepcopy(mini["mathematical_graph"]);graph["id"]="ordered_mha_composition";graph["inputs"]=copy.deepcopy(signature)
 if module.pre is not None:
  graph["constants"]["pre_weight"]=module.pre.weight.detach().tolist();refs=["q","pre_weight"]
  if module.pre.bias is not None:graph["constants"]["pre_bias"]=module.pre.bias.detach().tolist();refs.append("pre_bias")
  graph["nodes"].insert(0,dict(id="pre_linear",op="linear",inputs=refs,attrs={},outputs=[qname]))
 output=graph["outputs"][0]["value"]
 if module.post is not None:
  graph["constants"]["post_weight"]=module.post.weight.detach().tolist();refs=[output,"post_weight"]
  if module.post.bias is not None:graph["constants"]["post_bias"]=module.post.bias.detach().tolist();refs.append("post_bias")
  graph["nodes"].append(dict(id="post_linear",op="linear",inputs=refs,attrs={},outputs=["post_projected"]));output="post_projected"
 if case.get("residual"):
  graph["nodes"].append(dict(id="residual_add",op="add",inputs=[output,qname],attrs={},outputs=["residual_output"]));output="residual_output"
 graph["outputs"][0]=dict(name="result",value=output)
 if case["need"]:graph["outputs"][1]["name"]="weights"
 if case.get("projected_output"):graph["outputs"].append(dict(name="projected",value=qname))
 lowered,binding=decompose(graph,resource_profile="general-tensors-v1")
 return graph,lowered,binding,mini
