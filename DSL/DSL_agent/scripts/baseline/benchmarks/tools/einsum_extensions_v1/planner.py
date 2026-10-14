"""Draft extension of static einsum decomposition; not a production source policy.

Repeated input labels are a checked public diagonal gather. Scalar factors are
separate tensor multiplications. Existing einsum planning and operator lowering
remain immutable; source-view mutation needs separate integration before use.
"""
import copy,hashlib,itertools,math,re
from pathlib import Path
from benchmark_graph import canonical,digest,require
from model_resources import resolve
import pytorch_einsum_normalization as existing
POLICY="poseidon-pytorch-static-einsum-extension-draft-v1"
FORMAT="poseidon-einsum-extension-plan-v1"
def _source():
 return {"planner.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  "pytorch_einsum_normalization.py":hashlib.sha256(Path(existing.__file__).read_bytes()).hexdigest()}
def plan(equation,shapes,*,resource_profile="general-tensors-v1"):
 limits=resolve(resource_profile)
 require(type(equation)is str and len(equation)<=256,"Static einsum equation budget")
 require(type(shapes)is list and 1<=len(shapes)<=4,"Static einsum operand budget")
 def valid(shape):
  return type(shape)is list and len(shape)<=limits.rank and all(type(n)is int and 0<n<=limits.axis_size for n in shape)and math.prod(shape)<=limits.tensor_elements
 require(all(valid(s)for s in shapes),"Static einsum tensor shape budget")
 text="".join(c for c in equation if not c.isspace())
 require(text.count("->")<=1,"Static einsum output delimiter")
 parts=text.split("->");terms=parts[0].split(",")
 require(len(terms)==len(shapes),"Static einsum operand count")
 collapsed=[];diagonals=[];scalars=[];letter_count={}
 for i,(term,shape)in enumerate(zip(terms,shapes)):
  require(re.fullmatch(r"[A-Za-z]*(?:\.\.\.[A-Za-z]*)?",term)is not None,"Static einsum operand labels")
  letters=[c for c in term if c.isalpha()]
  hidden=len(shape)-len(letters)
  require(hidden>=0 and("..."in term or hidden==0),"Static einsum operand rank")
  for label in letters:letter_count[label]=letter_count.get(label,0)+1
  if not shape:
   scalars.append(i);collapsed.append(dict(term="",shape=[],labels=[]));continue
  if "..."in term:
   first,last=term.split("...")
   labels=list(first)+["@"+str(j)for j in range(hidden)]+list(last)
  else:labels=list(term)
  unique=list(dict.fromkeys(labels));sizes={}
  for label,n in zip(labels,shape):
   require(label not in sizes or sizes[label]==n,"Repeated einsum label dimensions must be equal")
   sizes[label]=n
  new_shape=[sizes[label]for label in unique]
  if len(labels)!=len(unique):
   strides=[];stride=1
   for n in reversed(shape):strides.insert(0,stride);stride*=n
   positions={label:j for j,label in enumerate(unique)}
   count=math.prod(new_shape)
   require(2*count+1<=limits.expanded_nodes,"Diagonal gather expansion budget")
   indices=[sum(coord[positions[label]]*s for label,s in zip(labels,strides))
    for coord in itertools.product(*(range(n)for n in new_shape))]
   diagonals.append(dict(operand=i,labels=labels,unique_labels=unique,source_shape=list(shape),
    result_shape=new_shape,indices=indices,exact_gather_core_nodes=2*count+1))
  prefix=term.split("...")[0]if"..."in term else term
  suffix=term.split("...")[1]if"..."in term else""
  seen=set()
  def keep(s):
   result=""
   for c in s:
    if c not in seen:result+=c;seen.add(c)
   return result
  normalized=keep(prefix)
  if "..."in term:normalized+="..."+keep(suffix)
  collapsed.append(dict(term=normalized,shape=new_shape,labels=unique))
 output=parts[1]if len(parts)==2 else("..."if any("..."in t for t in terms)else"")+  "".join(sorted(c for c,n in letter_count.items()if n==1))
 require(re.fullmatch(r"[A-Za-z]*(?:\.\.\.[A-Za-z]*)?",output)is not None,"Static einsum output labels")
 out_letters=[c for c in output if c.isalpha()]
 require(len(set(out_letters))==len(out_letters)and all(c in letter_count for c in out_letters),"Static einsum output label inventory")
 steps=[];refs={"operand_"+str(i):"operand_"+str(i)for i in range(len(shapes))}
 def emit(op,inputs,attrs,shape):
  require(valid(shape)and len(steps)<limits.original_nodes,"Static einsum extension step budget")
  name="step_"+str(len(steps));steps.append(dict(op=op,inputs=list(inputs),attrs=copy.deepcopy(attrs),shape=list(shape)));return name
 for d in diagonals:
  origin="operand_"+str(d["operand"])
  value=emit("reshape",[origin],dict(shape=[math.prod(d["source_shape"])]),[math.prod(d["source_shape"])])
  value=emit("public_gather",[value],dict(axis=0,indices=d["indices"]),[len(d["indices"])])
  value=emit("reshape",[value],dict(shape=d["result_shape"]),d["result_shape"]);refs[origin]=value
 non_scalar=[i for i,s in enumerate(shapes)if s]
 if non_scalar:
  core_equation=",".join(collapsed[i]["term"]for i in non_scalar)+"->"+output
  core=existing.plan(core_equation,[collapsed[i]["shape"]for i in non_scalar],resource_profile=resource_profile)
  mapped={"operand_"+str(j):refs["operand_"+str(i)]for j,i in enumerate(non_scalar)}
  for j,s in enumerate(core["steps"]):
   mapped["step_"+str(j)]=emit(s["op"],[mapped[v]for v in s["inputs"]],s["attrs"],s["shape"])
  value=mapped["step_"+str(len(core["steps"])-1)];result_shape=core["result_shape"]
 else:
  require(not out_letters,"Scalar einsum output labels")
  core=None;result_shape=[];value="operand_"+str(scalars.pop(0))
 for i in scalars:value=emit("multiply",[value,"operand_"+str(i)],{},result_shape)
 if not steps or value!="step_"+str(len(steps)-1):
  value=emit("reshape",[value],dict(shape=result_shape),result_shape)
 require(valid(result_shape),"Static einsum result shape budget")
 require(sum(d["exact_gather_core_nodes"]-3 for d in diagonals)+len(steps)<=limits.expanded_nodes,
  "Static einsum combined expansion lower bound")
 record=dict(format=FORMAT,policy=POLICY,equation=text,source_shapes=copy.deepcopy(shapes),result_shape=list(result_shape),
  resource_profile=resource_profile,diagonals=diagonals,scalar_operands=[i for i,s in enumerate(shapes)if not s],
  core_plan=core,steps=steps,output=value,implementation_sources=_source(),
  source_view_semantics="read-only values; alias mutation integration not activated",production_source_policy=False)
 require(len(canonical(record))<=limits.json_bytes,"Static einsum plan data budget")
 record["binding"]=digest(record);return record
def check(record):
 require(type(record)is dict and record.get("format")==FORMAT and
  record.get("binding")==digest({k:v for k,v in record.items()if k!="binding"}),"Static einsum plan binding")
 require(record==plan(record.get("equation"),record.get("source_shapes"),
  resource_profile=record.get("resource_profile")),"Static einsum complete plan recomputation")
 return record
def graph(record,*,model_id="einsum_extension"):
 check(record)
 require(type(model_id)is str,"Model identity")
 from operator_graph import decompose,FORMAT as OPERATOR_FORMAT
 physical=lambda shape:shape or[1]
 nodes=[];names={"operand_"+str(i):"input_"+str(i)for i in range(len(record["source_shapes"]))}
 logical={name:s for name,s in zip(names,record["source_shapes"])}
 for i,s in enumerate(record["steps"]):
  attrs=copy.deepcopy(s["attrs"]);op=s["op"]
  if op=="reshape":attrs["shape"]=physical(attrs["shape"])
  elif op=="permute"and not logical[s["inputs"][0]]:attrs["dims"]=[0]
  elif op=="sum":attrs["keepdims"]=False
  output="step_"+str(i);nodes.append(dict(id="node_"+str(i),op=op,
   inputs=[names[v]for v in s["inputs"]],attrs=attrs,outputs=[output]))
  names[output]=output;logical[output]=s["shape"]
 original=dict(format=OPERATOR_FORMAT,id=model_id,
  inputs=[dict(name=names["operand_"+str(i)],shape=physical(shape))for i,shape in enumerate(record["source_shapes"])],
  constants={},nodes=nodes,outputs=[dict(name="out",value=names[record["output"]])])
 lowered,binding=decompose(original,resource_profile=record["resource_profile"])
 from benchmark_graph import validate
 require(validate(lowered)["output_shapes"]["out"]==physical(record["result_shape"]),"Einsum logical/carrier output shape")
 provenance=dict(format="poseidon-einsum-extension-graph-v1",plan_binding=record["binding"],
  original_sha256=digest(original),lowered_sha256=digest(lowered),decomposition_binding=binding,
  logical_input_shapes=record["source_shapes"],logical_output_shape=record["result_shape"],scalar_carrier_shape=[1],
  production_source_policy=False)
 provenance["binding"]=digest(provenance)
 return original,lowered,provenance
