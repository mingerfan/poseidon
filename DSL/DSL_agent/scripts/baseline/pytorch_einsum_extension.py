"""Static einsum diagonal views and scalar factors for the explicit v19 source contract."""
import copy,hashlib,itertools,math,re
from pathlib import Path
from benchmark_graph import canonical,digest,require
from model_resources import resolve
import pytorch_einsum_normalization as existing
POLICY="poseidon-pytorch-static-einsum-v2"
arguments=existing.arguments
FORMAT="poseidon-einsum-view-plan-v2"
def _source():
 return {"pytorch_einsum_extension.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
  origin="operand_"+str(d["operand"]);value=origin
  labels=list(d["labels"]);dims=list(d["source_shape"])
  while len(labels)!=len(set(labels)):
   label=next(c for c in labels if labels.count(c)>1)
   a=labels.index(label);z=labels.index(label,a+1)
   result_dims=[n for j,n in enumerate(dims)if j not in(a,z)]+[dims[a]]
   value=emit("diagonal",[value],dict(offset=0,dim1=a,dim2=z),result_dims)
   labels=[c for j,c in enumerate(labels)if j not in(a,z)]+[label];dims=result_dims
  perm=[labels.index(c)for c in d["unique_labels"]]
  if perm!=list(range(len(perm))):
   value=emit("permute",[value],dict(dims=perm),d["result_shape"])
  refs[origin]=value
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
  source_view_semantics="diagonal and identity single-operand results preserve shared storage",production_source_policy=True)
 require(len(canonical(record))<=limits.json_bytes,"Static einsum plan data budget")
 record["binding"]=digest(record);return record
def check(record):
 require(type(record)is dict and record.get("format")==FORMAT and
  record.get("binding")==digest({k:v for k,v in record.items()if k!="binding"}),"Static einsum plan binding")
 require(record==plan(record.get("equation"),record.get("source_shapes"),
  resource_profile=record.get("resource_profile")),"Static einsum complete plan recomputation")
 return record

def _literal(value):
    if type(value) is str:return dict(kind="str",value=value)
    if type(value) is int:return dict(kind="int",value=value)
    if type(value) in (list,tuple):return dict(kind="tuple",items=[_literal(v)for v in value])
    raise ValueError("Unregistered einsum plan literal")

def step_inventory(step, references, name):
    value = lambda n: dict(kind="fx_value",name=references[n])
    op = step["op"]; attrs = step["attrs"]
    args = [value(n) for n in step["inputs"]]
    kwargs = {}
    if op=="diagonal":
        kind,target="call_method","diagonal"
        kwargs={k:_literal(v)for k,v in attrs.items()}
    elif op in ("permute","reshape"):
        kind,target="call_method",op
        args.append(_literal(attrs["dims" if op=="permute"else "shape"]))
    elif op=="sum":
        kind,target="call_function","torch.sum"
        kwargs={"dim":_literal(attrs["axes"]),"keepdim":dict(kind="bool",value=False)}
    elif op=="matmul":kind,target="call_function","torch.matmul"
    elif op=="multiply":kind,target="call_function","_operator.mul"
    else:raise ValueError("Unregistered einsum primitive")
    return dict(name=name,op=kind,target=target,args=dict(kind="tuple",items=args),
                kwargs=dict(kind="dict",items=[[k,v]for k,v in kwargs.items()]),
                tensor=dict(policy="signature-static-ordered-alias-staging-v1",
                            shape=step["shape"],dtype="float64"))

def normalize(traced, inputs, *, resource_profile="general-tensors-v1"):
    import torch, operator
    from pytorch_static_trace import ALIAS_POLICY, policy_record
    from pytorch_arithmetic_normalization import clone_public_state
    from pytorch_fused_matrix_normalization import inventory, _state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("einsum_source_policy")==POLICY,
            "Explicit einsum trace policy")
    require(traced.meta.get("ordered_alias_capture_policy")==
            policy_record(inputs,resource_profile,policy=ALIAS_POLICY),"Einsum ordered trace policy")
    original = str(traced.graph);state=_state(traced)
    result = clone_public_state(traced);entries=[]
    for node in list(result.graph.nodes):
        if node.op!="call_function"or node.target is not torch.einsum:continue
        equation,operands=arguments(node.args,node.kwargs)
        require(all(isinstance(v,torch.fx.Node)for v in operands),"Static einsum tensor operands")
        metas=[v.meta.get("poseidon_static_tensor")for v in operands]
        require(all(type(m)is dict and m.get("policy")==ALIAS_POLICY and m.get("dtype")=="float64"
                    for m in metas),"Static einsum float64 metadata")
        selected=plan(equation,[m["shape"]for m in metas],resource_profile=resource_profile)
        require(node.meta.get("poseidon_static_tensor")==
                dict(policy=ALIAS_POLICY,shape=selected["result_shape"],dtype="float64"),
                "Static einsum result metadata")
        references={"operand_"+str(i):v for i,v in enumerate(operands)};emitted=[]
        for i,step in enumerate(selected["steps"]):
            refs={k:v.name for k,v in references.items()}
            proto=step_inventory(step,refs,node.name)
            args=tuple(references[n]for n in step["inputs"]);kwargs={}
            if step["op"]=="permute":args+=(tuple(step["attrs"]["dims"]),)
            elif step["op"]=="reshape":args+=(tuple(step["attrs"]["shape"]),)
            elif step["op"]=="diagonal":kwargs=dict(step["attrs"])
            elif step["op"]=="sum":kwargs=dict(dim=tuple(step["attrs"]["axes"]),keepdim=False)
            target={"sum":torch.sum,"matmul":torch.matmul,"multiply":operator.mul}.get(step["op"],step["op"])
            if i==len(selected["steps"])-1:
                created=node;created.op=proto["op"];created.target=target
                created.args=args;created.kwargs=kwargs
            else:
                with result.graph.inserting_before(node):
                    created=result.graph.create_node(proto["op"],target,args,kwargs)
            created.meta["poseidon_static_tensor"]=proto["tensor"]
            if step["op"]=="diagonal":created.meta["poseidon_einsum_diagonal_policy"]=POLICY
            references["step_"+str(i)]=created;emitted.append(created.name)
        entries.append(dict(source_node=node.name,operands=[v.name for v in operands],
                            plan=selected,emitted_nodes=emitted))
    result.graph.lint();result.recompile()
    require(sum(n.op in("call_function","call_method","call_module")for n in result.graph.nodes)
            <=resolve(resource_profile).original_nodes,"Einsum normalized FX budget")
    require(original==str(traced.graph)and state==_state(result),"Einsum source or public state changed")
    text=str(result.graph)
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
                source_fx_text=original,source_fx_sha256=hashlib.sha256(original.encode()).hexdigest(),
                normalized_fx_text=text,normalized_fx_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_inventory=inventory(traced),normalized_inventory=inventory(result),entries=entries,
                source_order_preserved=True,public_state_preserved=True,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["binding"]=digest(record);check_transport(record,inputs=inputs,resource_profile=resource_profile)
    return result,record

def check_transport(record, *, inputs, resource_profile):
    fields={"format","input_signature","resource_profile","source_fx_text","source_fx_sha256",
            "normalized_fx_text","normalized_fx_sha256","source_inventory","normalized_inventory",
            "entries","source_order_preserved","public_state_preserved","implementation_sha256","binding"}
    limits=resolve(resource_profile)
    require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
            record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Einsum record binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["source_order_preserved"]is True and record["public_state_preserved"]is True and
            record["implementation_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Einsum record source identity")
    require(len(canonical(record))<=limits.json_bytes,"Einsum record budget")
    for a,b in[("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")]:
        require(type(record[a])is str and hashlib.sha256(record[a].encode()).hexdigest()==record[b],
                "Einsum FX text binding")
    source=record["source_inventory"];normalized=record["normalized_inventory"]
    cap=limits.original_nodes+limits.constants+limits.inputs+limits.outputs
    keys={"name","op","target","args","kwargs","tensor"}
    require(type(source)is list and type(normalized)is list and 1<=len(source)<=cap and
            1<=len(normalized)<=cap and all(type(n)is dict and set(n)==keys and type(n["name"])is str
            for n in source+normalized),"Einsum inventory fields and budget")
    nodes={n["name"]:n for n in source};require(len(nodes)==len(source),"Einsum source names")
    require(type(record["entries"])is list and len(record["entries"])<=limits.original_nodes,"Einsum entry budget")
    expected=[];entries={};fresh=set()
    def unpack(value):
        require(type(value)is dict,"Einsum inventory argument")
        if value.get("kind")=="str"and set(value)=={"kind","value"}and type(value["value"])is str:return value["value"]
        if value.get("kind")=="fx_value"and set(value)=={"kind","name"}and value["name"]in nodes:return value
        if value.get("kind")in("tuple","list")and set(value)=={"kind","items"}and type(value["items"])is list:
            return tuple(unpack(v)for v in value["items"])
        raise ValueError("Unregistered einsum source argument")
    for entry in record["entries"]:
        require(type(entry)is dict and set(entry)=={"source_node","operands","plan","emitted_nodes"}and
                entry["source_node"]in nodes and entry["source_node"]not in entries,"Einsum entry source")
        node=nodes[entry["source_node"]]
        require(node["op"]=="call_function"and node["target"]=="torch.functional.einsum"and
                node["kwargs"]==dict(kind="dict",items=[]),"Einsum registered source call")
        equation,operands=arguments(unpack(node["args"]),{})
        require(entry["operands"]==[v["name"]for v in operands]and all(type(v)is dict and
                v==dict(kind="fx_value",name=v["name"])for v in operands),"Einsum operand binding")
        metas=[nodes[v["name"]]["tensor"]for v in operands]
        require(all(type(m)is dict and set(m)=={"policy","shape","dtype"}and
                    m["policy"]=="signature-static-ordered-alias-staging-v1"and m["dtype"]=="float64"
                    for m in metas),"Einsum operand metadata")
        selected=plan(equation,[m["shape"]for m in metas],resource_profile=resource_profile)
        require(entry["plan"]==selected and node["tensor"]==
                dict(policy="signature-static-ordered-alias-staging-v1",shape=selected["result_shape"],dtype="float64"),
                "Einsum plan recomputation")
        emitted=entry["emitted_nodes"]
        require(type(emitted)is list and len(emitted)==len(selected["steps"])and
                all(type(n)is str for n in emitted)and len(set(emitted))==len(emitted)and
                emitted[-1]==node["name"]and not set(emitted[:-1]).intersection(set(nodes)|fresh),
                "Einsum emission inventory")
        fresh.update(emitted[:-1]);references={"operand_"+str(i):v["name"]for i,v in enumerate(operands)}
        replacement=[]
        for i,(step,name)in enumerate(zip(selected["steps"],emitted)):
            replacement.append(step_inventory(step,references,name));references["step_"+str(i)]=name
        entries[node["name"]]=replacement
    registered={n["name"]for n in source if n["op"]=="call_function"and n["target"]=="torch.functional.einsum"}
    require(set(entries)==registered,"Einsum source call inventory")
    for node in source:expected.extend(entries[node["name"]]if node["name"]in entries else[copy.deepcopy(node)])
    require(expected==normalized,"Einsum complete normalization projection")
    require(sum(n["op"]in("call_function","call_method","call_module")for n in normalized)
            <=limits.original_nodes,"Einsum normalized expansion budget")
    return record

def check_alias_projection(record,emissions,origins,*,inputs):
    """Bind generated view effects to recomputed FX calls, not user annotations."""
    nodes={n["name"]:n for n in record["normalized_inventory"]}
    placeholders=[n["name"]for n in record["normalized_inventory"]if n["op"]=="placeholder"]
    input_refs=dict(zip(placeholders,[i["name"]for i in inputs]))
    view_calls={n["name"]:n for n in nodes.values()if n["op"]=="call_method"and n["target"]=="diagonal"}
    def literal(v):
        require(type(v)is dict and v.get("kind")=="int"and set(v)=={"kind","value"}and type(v["value"])is int,
                "Generated diagonal argument")
        return v["value"]
    seen=set()
    for item in emissions["nodes"]:
        annotation=origins[item["id"]];name=annotation["fx_node"]
        if annotation.get("action")!="diagonal":continue
        require(name in view_calls and name not in seen,"Diagonal effect must match a generated view")
        call=view_calls[name];args=call["args"]
        require(args["kind"]=="tuple"and len(args["items"])==1 and
                args["items"][0]["kind"]=="fx_value","Generated diagonal source")
        source=args["items"][0]["name"];source_node=nodes[source]
        attrs={k:literal(v)for k,v in call["kwargs"]["items"]}
        require(annotation["diagonal"]==attrs and item["op"]=="public_gather"and
                item["inputs"]==[input_refs.get(source,"value_"+source)],"Diagonal emission source or axes")
        import numpy as np
        shape=source_node["tensor"]["shape"]
        indices=np.diagonal(np.arange(math.prod(shape)).reshape(shape),
                            offset=attrs["offset"],axis1=attrs["dim1"],axis2=attrs["dim2"])
        require(item["attrs"]==dict(axis=0,indices=indices.ravel().tolist()),"Diagonal emission indices")
        seen.add(name)
    public_cache={}
    def fx_refs(value):
        if type(value)is dict:
            if value.get("kind")=="fx_value":return [value["name"]]
            return [name for item in value.values()for name in fx_refs(item)]
        if type(value)is list:return [name for item in value for name in fx_refs(item)]
        return []
    def is_public(name):
        if name in public_cache:return public_cache[name]
        n=nodes[name]
        if n["op"]=="placeholder":value=False
        elif n["op"]=="get_attr":value=True
        else:
            refs=fx_refs(n["args"])+fx_refs(n["kwargs"])
            value=bool(refs)and all(is_public(ref)for ref in refs)
        public_cache[name]=value;return value
    require(seen=={name for name in view_calls if not is_public(name)},
            "Every encrypted diagonal must retain its generated shared-view effect")
    return True
