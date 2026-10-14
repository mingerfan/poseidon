"""Isolated Agent-layer prototype. Active capture contracts do not import this module."""
import copy,hashlib,math,operator,re,struct
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from pytorch_exact_arithmetic import classification,target_name,arguments,_roles
from pytorch_static_trace import ALIAS_POLICY
POLICY="poseidon-public-tensor-division-source-draft-v2"
PLAN="poseidon-public-tensor-division-plan-v1"
PROFILE="general-tensors-v1"
def sha(data):return hashlib.sha256(data).hexdigest()
def shape(value,limit):
 require(type(value)in(list,tuple)and len(value)<=limit.rank and
  all(type(d)is int and 0<d<=limit.axis_size for d in value),"Public division static shape")
 result=list(value)
 require(math.prod(result)<=limit.tensor_elements,"Public division tensor budget")
 return result
def broadcast(a,b):
 result=[]
 for i in range(1,max(len(a),len(b))+1):
  x=a[-i]if i<=len(a)else 1;y=b[-i]if i<=len(b)else 1
  require(x==y or x==1 or y==1,"Public division broadcast")
  result.append(max(x,y))
 return result[::-1]
def plan(input_shape,divisor_shape,payload,*,inplace=False,resource_profile=PROFILE):
 limits=resolve(resource_profile);a=shape(input_shape,limits);b=shape(divisor_shape,limits)
 require(type(inplace)is bool,"Public division inplace flag")
 require(math.prod(b)<=limits.constant_elements and type(payload)is bytes and len(payload)==8*math.prod(b),
  "Public float64 divisor payload")
 result=shape(broadcast(a,b),limits)
 require(not inplace or result==a,"Inplace public division cannot expand shape")
 values=[x[0]for x in struct.iter_unpack("<d",payload)];coefficients=[]
 for d in values:
  require(math.isfinite(d)and d!=0 and abs(d)<=1024,"Public divisor finite nonzero bound")
  c=1.0/d
  require(math.isfinite(c)and abs(c)<=1024,"Public reciprocal coefficient bound")
  coefficients.append(c)
 raw=b"".join(struct.pack("<d",c)for c in coefficients)
 record=dict(format=PLAN,input_shape=a,divisor_shape=b,result_shape=result,inplace=inplace,
  divisor_sha256=sha(payload),reciprocal_sha256=sha(raw),elements=len(values),
  operation="multiply",approximation="none",resource_profile=resource_profile)
 record["binding"]=digest(record)
 return record,raw
def state_payload(module,name):
 import torch
 value=module
 for part in name.split("."):value=getattr(value,part)
 require(isinstance(value,torch.Tensor)and value.dtype is torch.float64 and value.device.type=="cpu",
  "Declared public float64 CPU tensor")
 return value.detach().contiguous().numpy().astype("<f8",copy=False).tobytes()
def metadata(value):
 return dict(policy=ALIAS_POLICY,shape=list(value),dtype="float64")
def source_identity(inputs,source,state,text,resource_profile=PROFILE):
 return digest(dict(input_signature=inputs,source_inventory=source,public_state=state,
  source_fx_sha256=sha(text.encode()),resource_profile=resource_profile))
def source_checked(module,declared):
 from pytorch_fused_matrix_normalization import _state
 before=_state(module)
 require(type(declared)is dict and declared and all(type(n)is str and n in before for n in declared),
  "Explicit authenticated public divisor inventory")
 require(all(declared[n]==before[n]and before[n]["dtype"]=="<f8"for n in declared),
  "Declared public divisor state binding")
 payloads={n:state_payload(module,n)for n in declared}
 require(all(sha(v)==declared[n]["normalized_float64_sha256"]for n,v in payloads.items()),
  "Declared public divisor payload binding")
 return before,payloads
def normalize(module,inputs,declared,*,resource_profile=PROFILE):
 import torch,numpy as np
 from pytorch_arithmetic_normalization import clone_public_state
 from pytorch_fused_matrix_normalization import inventory,_state
 require(isinstance(module,torch.fx.GraphModule)and module.meta.get("public_tensor_division_policy")==POLICY,
  "Explicit isolated division policy")
 limits=resolve(resource_profile);before,payloads=source_checked(module,declared)
 source=inventory(module);roles=_roles(source);nodes={n["name"]:n for n in source}
 placeholders=[n for n in source if n["op"]=="placeholder"]
 require(type(inputs)is list and len(inputs)==len(placeholders)<=limits.inputs and
  all(n["target"]==i["name"]and n["tensor"]==metadata(i["shape"])for n,i in zip(placeholders,inputs)),
  "Public division input signature")
 text=str(module.graph);result=clone_public_state(module);entries=[]
 # Torch 2.0.1 FX deepcopy drops output-node metadata. Preserve the immutable typed inventory.
 source_metadata={n.name:copy.deepcopy(n.meta.get("poseidon_static_tensor"))for n in module.graph.nodes}
 for copied in result.graph.nodes:
  copied.meta["poseidon_static_tensor"]=source_metadata[copied.name]
 for node in list(result.graph.nodes):
  signature=classification(node.op,target_name(node.op,node.target))
  if signature is None or signature[0]!="divide":continue
  values=arguments(node.args,node.kwargs,signature);base=values["input"];divisor=values["parameter"]
  # Literal scalar division stays for the separately verified retained exact-arithmetic pass.
  if not isinstance(divisor,torch.fx.Node):continue
  require(isinstance(base,torch.fx.Node)and roles[base.name]and isinstance(divisor,torch.fx.Node)and
   divisor.op=="get_attr"and str(divisor.target)in declared and not roles[divisor.name],
   "Direct declared public tensor divisor required")
  state=str(divisor.target);base_meta=base.meta.get("poseidon_static_tensor")
  require(base_meta==metadata(nodes[base.name]["tensor"]["shape"]),"Public division input float64 metadata")
  selected,raw=plan(base_meta["shape"],declared[state]["shape"],payloads[state],
   inplace=signature[1],resource_profile=resource_profile)
  require(divisor.meta.get("poseidon_static_tensor")==metadata(selected["divisor_shape"])and
   node.meta.get("poseidon_static_tensor")==metadata(selected["result_shape"]),"Public division output metadata")
  name="_poseidon_public_reciprocal_"+node.name
  require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",name)and not hasattr(result,name),"Derived public state name collision")
  value=torch.from_numpy(np.frombuffer(raw,dtype="<f8").copy().reshape(selected["divisor_shape"]))
  result.register_buffer(name,value)
  with result.graph.inserting_before(node):constant=result.graph.get_attr(name)
  constant.meta["poseidon_static_tensor"]=metadata(selected["divisor_shape"])
  node.op="call_method"if selected["inplace"]else"call_function"
  node.target="mul_"if selected["inplace"]else operator.mul;node.args=(base,constant);node.kwargs={}
  entries.append(dict(source_node=node.name,input=base.name,divisor=divisor.name,state_name=state,
   reciprocal_state_name=name,reciprocal_fx_node=constant.name,plan=selected))
 result.graph.lint();result.recompile()
 after=_state(result)
 require(all(after.get(n)==v for n,v in before.items())and _state(module)==before and str(module.graph)==text,
  "Original division graph or public state changed")
 require(len(after)<=limits.constants and sum(math.prod(v["shape"])for v in after.values())<=limits.total_constant_elements,
  "Public reciprocal aggregate state budget")
 record=dict(format=POLICY,inputs=copy.deepcopy(inputs),resource_profile=resource_profile,
  source_fx_text=text,source_fx_sha256=sha(text.encode()),normalized_fx_text=str(result.graph),
  normalized_fx_sha256=sha(str(result.graph).encode()),source_inventory=source,normalized_inventory=inventory(result),
  declared_divisors=copy.deepcopy(declared),public_state=before,normalized_state=after,entries=entries,
  original_state_preserved=True,production_activation=False,implementation_sha256=sha(Path(__file__).read_bytes()),
  source_identity=source_identity(inputs,source,before,text,resource_profile))
 record["binding"]=digest(record)
 check(record,inputs,declared,payloads,expected_source_identity=record["source_identity"],resource_profile=resource_profile)
 return result,record
def check(record,inputs,declared,payloads,*,expected_source_identity,resource_profile=PROFILE):
 """Recompute from authenticated public bytes; no source or FX program execution."""
 limits=resolve(resource_profile)
 fields={"format","inputs","resource_profile","source_fx_text","source_fx_sha256","normalized_fx_text",
 "normalized_fx_sha256","source_inventory","normalized_inventory","declared_divisors","public_state",
 "normalized_state","entries","original_state_preserved","production_activation","implementation_sha256","source_identity","binding"}
 require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
  record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Public division record binding")
 require(record["inputs"]==inputs and record["resource_profile"]==resource_profile and
  record["declared_divisors"]==declared and record["production_activation"]is False and record["original_state_preserved"]is True and
  record["implementation_sha256"]==sha(Path(__file__).read_bytes()),"Public division trusted identity")
 require(len(canonical(record))<=limits.json_bytes and type(payloads)is dict and set(payloads)==set(declared),
  "Public division record or payload inventory")
 for n,payload in payloads.items():
  require(type(payload)is bytes and sha(payload)==declared[n]["normalized_float64_sha256"]and
   record["public_state"].get(n)==declared[n],"Public divisor payload substitution")
 for key,h in[("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")]:
  require(type(record[key])is str and sha(record[key].encode())==record[h],"Public division FX text binding")
 source=record["source_inventory"];normalized=record["normalized_inventory"]
 require(type(expected_source_identity)is str and len(expected_source_identity)==64 and
  record["source_identity"]==expected_source_identity==source_identity(inputs,source,record["public_state"],
   record["source_fx_text"],resource_profile),"Authenticated source graph identity")
 placeholders=[n for n in source if n["op"]=="placeholder"]
 require(len(placeholders)==len(inputs)<=limits.inputs and
  all(n["target"]==i["name"]and n["tensor"]==metadata(shape(i["shape"],limits))for n,i in zip(placeholders,inputs)),
  "Public division immutable input signature")
 cap=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs
 require(type(source)is list and type(normalized)is list and len(source)<=cap and len(normalized)<=cap,
  "Public division inventory budget")
 nodes={n["name"]:n for n in source};require(len(nodes)==len(source),"Public division source names")
 roles=_roles(source);entries={};used=set(nodes);derived={}
 def decode(v):
  require(type(v)is dict,"Public division call argument")
  if v==dict(kind="none"):return None
  require(set(v)=={"kind","name"}and v["kind"]=="fx_value"and v["name"]in nodes,"Direct tensor division operands")
  return v
 for entry in record["entries"]:
  require(type(entry)is dict and set(entry)=={"source_node","input","divisor","state_name","reciprocal_state_name","reciprocal_fx_node","plan"}and
   entry["source_node"]in nodes and entry["source_node"]not in entries,"Public division source entry")
  node=nodes[entry["source_node"]];signature=classification(node["op"],node["target"])
  require(signature is not None and signature[0]=="divide","Public division registered call")
  kwargs={}
  for k,v in node["kwargs"]["items"]:
   require(type(k)is str and k not in kwargs,"Public division duplicate keywords");kwargs[k]=decode(v)
  values=arguments(tuple(decode(v)for v in node["args"]["items"]),kwargs,signature)
  require(values["input"]==dict(kind="fx_value",name=entry["input"])and
   values["parameter"]==dict(kind="fx_value",name=entry["divisor"])and roles[entry["input"]]and not roles[entry["divisor"]],
   "Public division source operands")
  base=nodes[entry["input"]];divisor=nodes[entry["divisor"]];state=entry["state_name"]
  require(divisor["op"]=="get_attr"and divisor["target"]==state and state in declared,"Public division direct state origin")
  selected,raw=plan(base["tensor"]["shape"],declared[state]["shape"],payloads[state],
   inplace=signature[1],resource_profile=resource_profile)
  require(entry["plan"]==selected and base["tensor"]==metadata(selected["input_shape"])and
   divisor["tensor"]==metadata(selected["divisor_shape"])and node["tensor"]==metadata(selected["result_shape"]),
   "Public division plan recomputation")
  name=entry["reciprocal_state_name"];fresh=entry["reciprocal_fx_node"]
  require(name=="_poseidon_public_reciprocal_"+node["name"]and name not in record["public_state"]and
   fresh not in used,"Public division derived names")
  used.add(fresh)
  derived[name]=dict(shape=selected["divisor_shape"],dtype="<f8",sha256=sha(raw),normalized_float64_sha256=sha(raw))
  constant=dict(name=fresh,op="get_attr",target=name,args=dict(kind="tuple",items=[]),
   kwargs=dict(kind="dict",items=[]),tensor=metadata(selected["divisor_shape"]))
  product=dict(name=node["name"],op="call_method"if selected["inplace"]else"call_function",
   target="mul_"if selected["inplace"]else"_operator.mul",
   args=dict(kind="tuple",items=[dict(kind="fx_value",name=entry["input"]),dict(kind="fx_value",name=fresh)]),
   kwargs=dict(kind="dict",items=[]),tensor=metadata(selected["result_shape"]))
  entries[node["name"]]=[constant,product]
 actual=set()
 for n in source:
  signature=classification(n["op"],n["target"])
  if signature is None or signature[0]!="divide":continue
  kwargs={}
  for k,v in n["kwargs"]["items"]:
   require(type(k)is str and k not in kwargs,"Public division duplicate keywords")
   if k not in ("rounding_mode","out"):kwargs[k]=v
   else:require(v==dict(kind="none"),"Exact arithmetic rounding/out must be absent or None")
  values=arguments(tuple(n["args"]["items"]),kwargs,signature)
  parameter=values["parameter"]
  if parameter.get("kind")=="fx_value":actual.add(n["name"])

 require(set(entries)==actual and normalized==[v for n in source for v in entries.get(n["name"],[n])],
  "Public division complete graph projection")
 require(record["normalized_state"]=={**record["public_state"],**derived},"Public division derived state binding")
 require(len(record["normalized_state"])<=limits.constants and
  sum(math.prod(v["shape"])for v in record["normalized_state"].values())<=limits.total_constant_elements,
  "Public reciprocal aggregate state budget")
 return record
