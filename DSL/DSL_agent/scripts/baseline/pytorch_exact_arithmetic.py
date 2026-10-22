"""Registered exact PyTorch power/division spellings over immutable FX records."""
import copy,hashlib,math,struct
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
POLICY="poseidon-pytorch-exact-arithmetic-v1"
PLAN="poseidon-exact-arithmetic-plan-v1"
_CALLS={
 ("call_function","_operator.pow"):("power",False,"operator"),
 ("call_function","torch.pow"):("power",False,"function"),
 ("call_method","pow"):("power",False,"method"),
 ("call_method","pow_"):("power",True,"method"),
 ("call_function","_operator.truediv"):("divide",False,"operator"),
 ("call_function","torch.div"):("divide",False,"function"),
 ("call_function","torch.divide"):("divide",False,"function"),
 ("call_function","torch.true_divide"):("divide",False,"true_function"),
 **{("call_method",n):("divide",n.endswith("_"),"method")for n in("div","divide","div_","divide_")},
 **{("call_method",n):("divide",n.endswith("_"),"true_method")for n in("true_divide","true_divide_")}
}
def classification(kind,target):
 return _CALLS.get((kind,target))
def target_name(kind,target):
 if kind!="call_function":return str(target)
 return target.__module__+"."+target.__name__
def arguments(args,kwargs,signature):
 operation,inplace,style=signature
 parameter="exponent"if operation=="power"else"other"
 if style=="operator":
  require(len(args)==2 and not kwargs,"Exact arithmetic operator arguments")
  return dict(input=args[0],parameter=args[1])
 names=("input",parameter)
 allowed={parameter}if style in("method","true_method")else set(names)
 if operation=="divide"and style not in("true_function","true_method"):allowed.add("rounding_mode")
 if style in("function","true_function"):allowed.add("out")
 require(len(args)<=2 and not(set(kwargs)-allowed),"Exact arithmetic arguments")
 values=dict(zip(names,args))
 for n,v in kwargs.items():
  if n in ("rounding_mode","out"):
   require(v is None,"Exact arithmetic rounding/out must be absent or None");continue
  require(n not in values,"Duplicate exact arithmetic argument");values[n]=v
 require(set(values)==set(names),"Exact arithmetic input and parameter required")
 return dict(input=values["input"],parameter=values[parameter])
def plan(shape,operation,parameter,*,inplace=False,resource_profile="general-tensors-v1"):
 limits=resolve(resource_profile)
 require(type(shape)in(list,tuple)and len(shape)<=limits.rank and
  all(type(v)is int and 0<v<=limits.axis_size for v in shape)and math.prod(shape)<=limits.tensor_elements,
  "Exact arithmetic tensor shape budget")
 require(type(inplace)is bool and operation in("power","divide"),"Registered exact arithmetic operation")
 require(type(parameter)in(int,float)and math.isfinite(parameter),"Static finite arithmetic parameter")
 if operation=="power":
  require(parameter in(2,4),"Registered exact powers are 2 and 4")
  steps=[dict(operation="square",inplace=inplace)]*(1 if parameter==2 else 2)
 else:
  require(parameter!=0 and abs(parameter)<=1024,"Nonzero bounded public divisor")
  coefficient=1.0/parameter
  require(math.isfinite(coefficient)and abs(coefficient)<=1024,"Public reciprocal constant budget")
  steps=[dict(operation="multiply",inplace=inplace,coefficient=coefficient)]
 return dict(format=PLAN,operation=operation,parameter=parameter,inplace=inplace,
  shape=list(shape),steps=steps,resource_profile=resource_profile,approximation="none")
def _refs(value):
 if type(value)is dict and value.get("kind")=="fx_value":return[value["name"]]
 if type(value)is dict:
  return [n for v in value.values()for n in _refs(v)]
 if type(value)is list:return[n for v in value for n in _refs(v)]
 return []
def _roles(inventory):
 roles={}
 for n in inventory:
  refs=_refs(n["args"])+_refs(n["kwargs"])
  require(all(x in roles for x in refs),"Exact arithmetic FX dependency order")
  roles[n["name"]]=n["op"]=="placeholder"or(n["op"]!="get_attr"and any(roles[x]for x in refs))
 return roles
def _scalar(value,source,nodes,state,raw_value=None):
 """Only a directly declared scalar float64 state tensor can be a tensor divisor."""
 from pytorch_static_trace import ALIAS_POLICY
 if type(value)in(int,float):return value,None
 require(type(value)is dict and value==dict(kind="fx_value",name=value.get("name"))and
         value["name"]in nodes,"Public scalar divisor source")
 node=nodes[value["name"]];meta=node["tensor"];name=node["target"]
 require(node["op"]=="get_attr"and meta==dict(policy=ALIAS_POLICY,shape=[],dtype="float64")and
         name in state and state[name]["shape"]==[]and state[name]["dtype"]=="<f8",
         "Only a direct public scalar float64 tensor divisor is registered")
 require(type(raw_value)in(int,float)and math.isfinite(raw_value),"Finite public scalar divisor")
 require(hashlib.sha256(struct.pack("<d",float(raw_value))).hexdigest()==state[name]["normalized_float64_sha256"],
         "Public scalar divisor state binding")
 return float(raw_value),dict(fx_node=value["name"],state_name=name,value=float(raw_value))
def step_inventory(step,source,name,metadata):
 target=("square_"if step["inplace"]else"square")if step["operation"]=="square"else(
  "mul_"if step["inplace"]else"_operator.mul")
 args=[dict(kind="fx_value",name=source)]
 if step["operation"]=="multiply":args.append(dict(kind="float",value=step["coefficient"]))
 return dict(name=name,op="call_method"if target!="_operator.mul"else"call_function",
  target=target,args=dict(kind="tuple",items=args),kwargs=dict(kind="dict",items=[]),tensor=copy.deepcopy(metadata))
def normalize(traced,inputs,*,resource_profile="general-tensors-v1"):
 import torch,operator
 from pytorch_static_trace import ALIAS_POLICY,policy_record
 from pytorch_arithmetic_normalization import clone_public_state
 from pytorch_fused_matrix_normalization import inventory,_state
 require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("exact_arithmetic_policy")==POLICY,
         "Explicit exact arithmetic trace policy")
 require(traced.meta.get("ordered_alias_capture_policy")==policy_record(inputs,resource_profile,policy=ALIAS_POLICY),
         "Exact arithmetic ordered trace policy")
 original=str(traced.graph)
 source_inventory=inventory(traced);roles=_roles(source_inventory);state=_state(traced)
 nodes={n["name"]:n for n in source_inventory};result=clone_public_state(traced);entries=[]
 for node in list(result.graph.nodes):
  signature=classification(node.op,target_name(node.op,node.target))
  if signature is None:continue
  values=arguments(node.args,node.kwargs,signature);base=values["input"]
  require(isinstance(base,torch.fx.Node)and roles[base.name],"Exact arithmetic requires an encrypted tensor base")
  metadata=base.meta.get("poseidon_static_tensor")
  require(type(metadata)is dict and set(metadata)=={"policy","shape","dtype"} and metadata.get("policy")==ALIAS_POLICY and metadata.get("dtype")=="float64",
          "Exact arithmetic float64 metadata")
  amount=values["parameter"];binding=None
  if isinstance(amount,torch.fx.Node):
   require(signature[0]=="divide","Encrypted or tensor exponents are not registered")
   value=result
   require(amount.op=="get_attr","Only a direct public scalar tensor divisor is registered")
   for part in str(amount.target).split("."):value=getattr(value,part)
   require(isinstance(value,torch.Tensor)and value.ndim==0 and value.dtype is torch.float64,
           "Public scalar divisor dtype and shape")
   amount,binding=_scalar(dict(kind="fx_value",name=amount.name),source_inventory,nodes,state,float(value.detach().cpu()))
  selected=plan(metadata["shape"],signature[0],amount,inplace=signature[1],resource_profile=resource_profile)
  require(node.meta.get("poseidon_static_tensor")==metadata,"Exact arithmetic result metadata")
  emitted=[];operand=base
  for i,step in enumerate(selected["steps"]):
   target=("square_"if step["inplace"]else"square")if step["operation"]=="square"else(
      "mul_"if step["inplace"]else operator.mul)
   kind="call_function"if target is operator.mul else"call_method"
   args=(operand,)if step["operation"]=="square"else(operand,step["coefficient"])
   if i==len(selected["steps"])-1:
    created=node;created.op=kind;created.target=target;created.args=args;created.kwargs={}
   else:
    with result.graph.inserting_before(node):created=result.graph.create_node(kind,target,args,{})
   created.meta["poseidon_static_tensor"]=copy.deepcopy(metadata);operand=created;emitted.append(created.name)
  entries.append(dict(source_node=node.name,input=base.name,plan=selected,scalar_divisor=binding,emitted_nodes=emitted))
 result.graph.lint();result.recompile()
 require(sum(n.op in("call_function","call_method","call_module")for n in result.graph.nodes)<=resolve(resource_profile).original_nodes,
         "Exact arithmetic normalized expansion budget")
 require(_state(result)==state and str(traced.graph)==original,"Exact arithmetic public state preserved")
 text=str(traced.graph);normalized=str(result.graph)
 record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
  source_fx_text=text,source_fx_sha256=hashlib.sha256(text.encode()).hexdigest(),
  normalized_fx_text=normalized,normalized_fx_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
  source_inventory=source_inventory,normalized_inventory=inventory(result),public_state=state,entries=entries,
  source_order_preserved=True,public_state_preserved=True,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 record["binding"]=digest(record);check_transport(record,inputs=inputs,resource_profile=resource_profile)
 return result,record
def check_transport(record,*,inputs,resource_profile):
 from pytorch_static_trace import ALIAS_POLICY
 fields={"format","input_signature","resource_profile","source_fx_text","source_fx_sha256","normalized_fx_text",
  "normalized_fx_sha256","source_inventory","normalized_inventory","public_state","entries",
  "source_order_preserved","public_state_preserved","implementation_sha256","binding"}
 limits=resolve(resource_profile)
 require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
  record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Exact arithmetic transport binding")
 require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
  record["source_order_preserved"]is True and record["public_state_preserved"]is True and
  record["implementation_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"Exact arithmetic source identity")
 require(len(canonical(record))<=limits.json_bytes,"Exact arithmetic record budget")
 for k,h in(("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
  require(type(record[k])is str and hashlib.sha256(record[k].encode()).hexdigest()==record[h],"Exact arithmetic FX text binding")
 source=record["source_inventory"];normalized=record["normalized_inventory"];state=record["public_state"]
 cap=limits.original_nodes+limits.constants+limits.inputs+limits.outputs
 require(type(source)is list and type(normalized)is list and 1<=len(source)<=cap and 1<=len(normalized)<=cap and
  all(type(n)is dict and set(n)=={"name","op","target","args","kwargs","tensor"}and type(n["name"])is str
      for n in source+normalized),"Exact arithmetic inventory budget")
 require(type(state)is dict and len(state)<=limits.constants,"Exact arithmetic public state budget")
 for name,value in state.items():
  require(type(name)is str and type(value)is dict and set(value)=={"shape","dtype","sha256","normalized_float64_sha256"}and
          type(value["shape"])is list and type(value["dtype"])is str,"Exact arithmetic public state fields")
  require(all(type(value[k])is str and len(value[k])==64 and all(c in"0123456789abcdef"for c in value[k])
              for k in("sha256","normalized_float64_sha256")),"Exact arithmetic public state hashes")
 nodes={n["name"]:n for n in source};require(len(nodes)==len(source),"Exact arithmetic unique source names")
 roles=_roles(source);fresh=set();entries={}
 require(type(record["entries"])is list and len(record["entries"])<=limits.original_nodes,"Exact arithmetic entry budget")
 def decode(v):
  require(type(v)is dict,"Exact arithmetic literal")
  if v==dict(kind="none"):return None
  if v.get("kind")in("int","float")and set(v)=={"kind","value"}and type(v["value"])is(
    int if v["kind"]=="int"else float):return v["value"]
  if v.get("kind")=="fx_value"and set(v)=={"kind","name"}and v["name"]in nodes:return v
  raise ValueError("Unregistered exact arithmetic argument")
 for entry in record["entries"]:
  require(type(entry)is dict and set(entry)=={"source_node","input","plan","scalar_divisor","emitted_nodes"}and
          entry["source_node"]in nodes and entry["source_node"]not in entries,"Exact arithmetic entry source")
  node=nodes[entry["source_node"]];signature=classification(node["op"],node["target"])
  require(signature is not None and node["args"].get("kind")=="tuple"and set(node["args"])=={"kind","items"}and
   node["kwargs"].get("kind")=="dict"and set(node["kwargs"])=={"kind","items"},"Exact arithmetic source call")
  kwargs={}
  for pair in node["kwargs"]["items"]:
   require(type(pair)is list and len(pair)==2 and type(pair[0])is str and pair[0]not in kwargs,"Exact arithmetic keywords")
   kwargs[pair[0]]=decode(pair[1])
  values=arguments(tuple(decode(v)for v in node["args"]["items"]),kwargs,signature)
  require(values["input"]==dict(kind="fx_value",name=entry["input"])and entry["input"]in nodes and roles[entry["input"]],
          "Exact arithmetic encrypted input binding")
  metadata=nodes[entry["input"]]["tensor"]
  require(type(metadata)is dict and set(metadata)=={"policy","shape","dtype"} and metadata.get("policy")==ALIAS_POLICY and metadata.get("dtype")=="float64"and
          node["tensor"]==metadata,"Exact arithmetic metadata binding")
  parameter=values["parameter"];binding=entry["scalar_divisor"]
  if type(parameter)is dict:
   require(signature[0]=="divide"and type(binding)is dict and set(binding)=={"fx_node","state_name","value"},
           "Exact arithmetic public scalar divisor binding")
   parameter,expected_binding=_scalar(parameter,source,nodes,state,binding["value"])
   require(binding==expected_binding,"Exact arithmetic scalar source projection")
  else:require(binding is None,"Exact arithmetic literal divisor binding")
  selected=plan(metadata["shape"],signature[0],parameter,inplace=signature[1],resource_profile=resource_profile)
  require(entry["plan"]==selected,"Exact arithmetic plan recomputation")
  emitted=entry["emitted_nodes"]
  require(type(emitted)is list and len(emitted)==len(selected["steps"])and all(type(n)is str for n in emitted)and
   len(set(emitted))==len(emitted)and emitted[-1]==node["name"]and not set(emitted[:-1]).intersection(set(nodes)|fresh),
   "Exact arithmetic emitted names")
  fresh.update(emitted[:-1]);replacement=[];operand=entry["input"]
  for step,name in zip(selected["steps"],emitted):
   replacement.append(step_inventory(step,operand,name,metadata));operand=name
  entries[node["name"]]=replacement
 actual={n["name"]for n in source if classification(n["op"],n["target"])is not None}
 require(set(entries)==actual,"Exact arithmetic missing rewrite")
 expected=[item for n in source for item in(entries[n["name"]]if n["name"]in entries else[n])]
 require(normalized==expected,"Exact arithmetic complete FX projection")
 require(sum(n["op"]in("call_function","call_method","call_module")for n in normalized)<=limits.original_nodes,
         "Exact arithmetic normalized budget")
 return record
