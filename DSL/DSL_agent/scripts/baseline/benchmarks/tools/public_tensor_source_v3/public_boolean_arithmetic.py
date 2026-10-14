"""Exact, authenticated public boolean multiplication; inactive Agent-owned draft."""
import copy,re,struct,hashlib
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from public_tensor_division import shape,broadcast,metadata,sha,PROFILE
from pytorch_exact_arithmetic import _roles
POLICY="poseidon-public-boolean-arithmetic-draft-v1"
FUNCTIONS={"_operator.mul","torch.mul","torch._VariableFunctionsClass.mul"}
def mask_plan(input_shape,mask_shape,payload,*,inplace=False,resource_profile=PROFILE):
 limit=resolve(resource_profile);a=shape(input_shape,limit);b=shape(mask_shape,limit)
 require(type(inplace)is bool and type(payload)is bytes and len(payload)==__import__("math").prod(b),
  "Authenticated boolean mask payload")
 result=shape(broadcast(a,b),limit)
 require(not inplace or result==a,"Inplace boolean multiplication cannot expand shape")
 raw=b"".join(struct.pack("<d",1.0 if value else 0.0)for value in payload)
 p=dict(format=POLICY,input_shape=a,mask_shape=b,result_shape=result,inplace=inplace,
  mask_sha256=sha(payload),numeric_mask_sha256=sha(raw),operation="multiply",mapping="false=0,true=1",
  approximation="none",resource_profile=resource_profile)
 p["binding"]=digest(p);return p,raw
def operands(node):
 op=node["op"];target=node["target"]
 method=op=="call_method"and target in("mul","mul_")
 function=op=="call_function"and target in FUNCTIONS
 if not(method or function):return None
 require(node["args"].get("kind")=="tuple"and node["kwargs"].get("kind")=="dict","Boolean arithmetic call structure")
 pos=node["args"]["items"];kw={}
 for key,value in node["kwargs"]["items"]:
  require(type(key)is str and key not in kw,"Boolean arithmetic duplicate keyword");kw[key]=value
 if target=="_operator.mul":
  require(len(pos)==2 and not kw,"Operator multiplication signature");return pos[0],pos[1],False
 require(len(pos)<=2 and (not method or len(pos)>=1),"Boolean multiplication positional signature")
 if len(pos):
  require("input"not in kw,"Duplicate multiplication input");left=pos[0]
 else:
  require("input"in kw,"Multiplication input required");left=kw.pop("input")
 if len(pos)==2:
  require("other"not in kw,"Duplicate multiplication other");right=pos[1]
 else:
  require("other"in kw,"Multiplication other required");right=kw.pop("other")
 if not method and "out"in kw:require(kw.pop("out")==dict(kind="none"),"Public boolean multiplication out is forbidden")
 require(not kw,"Public boolean multiplication keyword");return left,right,method and target=="mul_"
def selected_entries(source,state,payloads,*,resource_profile=PROFILE):
 limit=resolve(resource_profile)
 require(type(source)is list and len(source)<=limit.expanded_nodes+limit.constants+limit.inputs+limit.outputs
  and len({n["name"]for n in source})==len(source),"Unique bounded boolean source inventory")
 nodes={n["name"]:n for n in source};roles=_roles(source);entries=[]
 for n in source:
  values=operands(n)
  if values is None:continue
  left,right,inplace=values;matches=[]
  for ordinal,v in enumerate((left,right)):
   if type(v)is dict and v.get("kind")=="fx_value"and v.get("name")in nodes:
    attr=nodes[v["name"]]
    if attr["op"]=="get_attr"and state.get(attr["target"],{}).get("dtype")=="|b1":matches.append((ordinal,attr))
  if not matches:continue
  require(len(matches)==1,"One public boolean operand required")
  ordinal,mask=matches[0];base_arg=(right,left)[ordinal]
  require(type(base_arg)is dict and set(base_arg)=={"kind","name"}and base_arg["kind"]=="fx_value"
   and base_arg["name"]in nodes and roles[base_arg["name"]]and not roles[mask["name"]],
   "Boolean multiplication requires encrypted float input and direct public mask")
  base=nodes[base_arg["name"]];name=mask["target"]
  require(not inplace or ordinal==1,"Public boolean state cannot be mutated")
  require(name in payloads,"Authenticated public mask bytes required")
  selected,raw=mask_plan(base["tensor"]["shape"],state[name]["shape"],payloads[name],
   inplace=inplace,resource_profile=resource_profile)
  require(base["tensor"]==metadata(selected["input_shape"])and n["tensor"]==metadata(selected["result_shape"])
   and mask["tensor"]==dict(policy=metadata([])["policy"],shape=selected["mask_shape"],dtype="bool"),
   "Boolean arithmetic typed shape binding")
  require(sha(payloads[name])==state[name]["sha256"]and sha(raw)==state[name]["normalized_float64_sha256"],
   "Boolean public state bytes or numeric identity changed")
  derived="_poseidon_public_arithmetic_mask_"+n["name"]
  require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",derived)and derived not in state,"Derived boolean state name collision")
  entries.append(dict(source_node=n["name"],input=base["name"],mask_node=mask["name"],mask_state_name=name,
   derived_state_name=derived,operand_order=ordinal,inplace=inplace,plan=selected))
 return entries
def refs(value):
 if type(value)is dict:
  if value.get("kind")=="fx_value":return {value["name"]}
  return set().union(*(refs(v)for v in value.values()))
 if type(value)is list:return set().union(*(refs(v)for v in value))
 return set()
def projected(source,entries):
 nodes={n["name"]:n for n in source};changes={}
 for e in entries:
  source_node=nodes[e["source_node"]];name=e["derived_fx_node"]
  changes[e["source_node"]]=[
   dict(name=name,op="get_attr",target=e["derived_state_name"],args=dict(kind="tuple",items=[]),
    kwargs=dict(kind="dict",items=[]),tensor=metadata(e["plan"]["mask_shape"])),
   dict(name=source_node["name"],op="call_method"if e["inplace"]else"call_function",
    target="mul_"if e["inplace"]else"_operator.mul",
    args=dict(kind="tuple",items=[dict(kind="fx_value",name=e["input"]),dict(kind="fx_value",name=name)]),
    kwargs=dict(kind="dict",items=[]),tensor=metadata(e["plan"]["result_shape"]))]
 expanded=[v for n in source for v in changes.get(n["name"],[n])]
 used=set().union(*(refs(n["args"])|refs(n["kwargs"])for n in expanded))
 removed=[n["name"]for n in source if n["name"]in {e["mask_node"]for e in entries}and n["name"]not in used]
 return [n for n in expanded if n["name"]not in removed],removed
def check(record,payloads,*,expected_source_identity,resource_profile=PROFILE):
 limit=resolve(resource_profile)
 fields={"format","source_inventory","normalized_inventory","source_fx_text","source_fx_sha256","normalized_fx_text",
  "normalized_fx_sha256","public_state","normalized_state","entries","removed_get_attrs","source_identity",
  "implementation_sha256","resource_profile","production_activation","original_state_preserved","binding"}
 require(type(record)is dict and set(record)==fields and record["format"]==POLICY
  and record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Boolean normalization record binding")
 require(record["resource_profile"]==resource_profile and record["production_activation"]is False
  and record["original_state_preserved"]is True and record["implementation_sha256"]==sha(Path(__file__).read_bytes()),
  "Boolean normalization trusted identity")
 for text,key in[("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")]:
  require(sha(record[text].encode())==record[key],"Boolean normalization FX text binding")
 identity=digest(dict(source_inventory=record["source_inventory"],public_state=record["public_state"],
  source_fx_sha256=record["source_fx_sha256"],resource_profile=resource_profile))
 require(expected_source_identity==record["source_identity"]==identity,"Authenticated boolean source identity")
 expected=selected_entries(record["source_inventory"],record["public_state"],payloads,resource_profile=resource_profile)
 require(len(record["entries"])==len(expected),"Complete boolean arithmetic use inventory")
 names={n["name"]for n in record["source_inventory"]};derived={}
 for actual,e in zip(record["entries"],expected):
  require(set(actual)==set(e)|{"derived_fx_node"}and {k:v for k,v in actual.items()if k!="derived_fx_node"}==e,
   "Boolean arithmetic source entry")
  fresh=actual["derived_fx_node"];require(type(fresh)is str and fresh not in names,"Fresh boolean FX node")
  names.add(fresh);_,raw=mask_plan(e["plan"]["input_shape"],e["plan"]["mask_shape"],payloads[e["mask_state_name"]],
   inplace=e["inplace"],resource_profile=resource_profile)
  derived[e["derived_state_name"]]=dict(shape=e["plan"]["mask_shape"],dtype="<f8",sha256=sha(raw),normalized_float64_sha256=sha(raw))
 require(len(record["normalized_inventory"])<=limit.expanded_nodes+limit.constants+limit.inputs+limit.outputs,"Boolean expanded graph budget")
 normalized,removed=projected(record["source_inventory"],record["entries"])
 require(record["normalized_inventory"]==normalized and record["removed_get_attrs"]==removed,"Complete boolean graph projection")
 require(record["normalized_state"]=={**record["public_state"],**derived},"Derived numeric mask state binding")
 require(len(record["normalized_state"])<=limit.constants and
  sum(__import__("math").prod(v["shape"])for v in record["normalized_state"].values())<=limit.total_constant_elements
  and len(canonical(record))<=limit.json_bytes,"Boolean normalization resource budget")
 return record
def normalize(module,*,resource_profile=PROFILE):
 import torch,numpy as np
 from pytorch_arithmetic_normalization import clone_public_state
 from pytorch_fused_matrix_normalization import inventory,_state
 require(isinstance(module,torch.fx.GraphModule),"Trusted static boolean graph")
 before=_state(module);source=inventory(module);text=str(module.graph)
 payloads={}
 for name,state in before.items():
  if state["dtype"]=="|b1":
   value=module
   for part in name.split("."):value=getattr(value,part)
   payloads[name]=value.detach().contiguous().numpy().tobytes()
 selected=selected_entries(source,before,payloads,resource_profile=resource_profile)
 result=clone_public_state(module);source_metadata={n.name:copy.deepcopy(n.meta)for n in module.graph.nodes}
 for node in result.graph.nodes:node.meta=source_metadata[node.name]
 graph_nodes={n.name:n for n in result.graph.nodes};entries=[]
 for e in selected:
  node=graph_nodes[e["source_node"]];base=graph_nodes[e["input"]];name=e["derived_state_name"]
  require(not hasattr(result,name),"Derived boolean state name collision")
  _,raw=mask_plan(e["plan"]["input_shape"],e["plan"]["mask_shape"],payloads[e["mask_state_name"]],
   inplace=e["inplace"],resource_profile=resource_profile)
  result.register_buffer(name,torch.from_numpy(np.frombuffer(raw,dtype="<f8").copy().reshape(e["plan"]["mask_shape"])))
  with result.graph.inserting_before(node):constant=result.graph.get_attr(name)
  constant.meta["poseidon_static_tensor"]=metadata(e["plan"]["mask_shape"])
  node.op="call_method"if e["inplace"]else"call_function";node.target="mul_"if e["inplace"]else __import__("operator").mul
  node.args=(base,constant);node.kwargs={};entries.append(dict(e,derived_fx_node=constant.name))
 _,removed=projected(source,entries)
 for name in removed:
  node=graph_nodes[name];require(not node.users,"Public mask retains a computational use");result.graph.erase_node(node)
 result.graph.lint();result.recompile();after=_state(result)
 require(all(after.get(n)==v for n,v in before.items())and _state(module)==before and str(module.graph)==text,
  "Original boolean state or source graph changed")
 record=dict(format=POLICY,source_inventory=source,normalized_inventory=inventory(result),source_fx_text=text,
  source_fx_sha256=sha(text.encode()),normalized_fx_text=str(result.graph),normalized_fx_sha256=sha(str(result.graph).encode()),
  public_state=before,normalized_state=after,entries=entries,removed_get_attrs=removed,
  implementation_sha256=sha(Path(__file__).read_bytes()),resource_profile=resource_profile,
  production_activation=False,original_state_preserved=True)
 record["source_identity"]=digest(dict(source_inventory=source,public_state=before,source_fx_sha256=record["source_fx_sha256"],resource_profile=resource_profile))
 record["binding"]=digest(record);check(record,payloads,expected_source_identity=record["source_identity"],resource_profile=resource_profile)
 return result,record
