"""Draft static FX input normalization. No backend imports or runtime patching."""
import builtins,copy,hashlib,json,math,operator
import torch
VERSION="poseidon-static-broadcast-source-draft-v1"
class NormalizationError(ValueError):
 def __init__(self,code,node):self.code=code;self.node=node;super().__init__(code+":"+node)
def require(value,code,node):
 if not value:raise NormalizationError(code,node)
def validated_shape(value,node):
 require(type(value)in(tuple,list,torch.Size)and 1<=len(value)<=32,"normalization_rank",node)
 require(all(type(n)is int and n>0 for n in value),"normalization_dimension",node)
 result=tuple(value);require(math.prod(result)<=65536,"normalization_capacity",node);return result
def axis(value,rank,node):
 require(type(value)is int and -rank<=value<rank,"normalization_axis",node);return value%rank
def reshape(source,target,node):
 require(type(target)in(tuple,list)and all(type(n)is int for n in target),"normalization_static_shape",node)
 require(target.count(-1)<=1 and all(n>0 or n==-1 for n in target),"normalization_reshape",node)
 target=list(target);total=math.prod(source)
 if -1 in target:
  known=math.prod(n for n in target if n!=-1);require(total%known==0,"normalization_reshape",node)
  target[target.index(-1)]=total//known
 require(math.prod(target)==total,"normalization_reshape",node);return validated_shape(target,node)
def expanded(source,target,node):
 target=validated_shape(target,node)
 require(len(target)>=len(source),"normalization_broadcast",node)
 padded=(1,)*(len(target)-len(source))+source
 require(all(a==1 or a==b for a,b in zip(padded,target)),"normalization_broadcast",node);return target
def normalize(module,input_shapes):
 require(isinstance(module,torch.fx.GraphModule),"normalization_graph","graph")
 before=module.code;result=copy.deepcopy(module);shapes={};records=[]
 placeholders=[n for n in result.graph.nodes if n.op=="placeholder"]
 require(set(input_shapes)=={n.target for n in placeholders},"normalization_signature","graph")
 for n in placeholders:shapes[n.name]=validated_shape(input_shapes[n.target],n.name)
 def size(value,node):
  if isinstance(value,torch.fx.Node):
   if value.op=="call_function"and value.target is builtins.getattr and len(value.args)==2 and value.args[1]=="shape"and not value.kwargs:
    return known(value.args[0])
   if value.op=="call_method"and value.target=="size"and not value.kwargs:
    s=known(value.args[0]);return s if len(value.args)==1 else s[axis(value.args[1],len(s),node)]
   if value.op=="call_function"and value.target is operator.getitem and len(value.args)==2 and not value.kwargs:
    seq=size(value.args[0],node);idx=value.args[1]
    require(type(seq)in(tuple,list)and type(idx)is int and -len(seq)<=idx<len(seq),"normalization_static_shape",node)
    return seq[idx]
   raise NormalizationError("normalization_dynamic_shape",node)
  if type(value)in(tuple,list,torch.Size):return tuple(size(v,node)for v in value)
  require(type(value)is int,"normalization_static_shape",node);return value
 def known(node):
  require(isinstance(node,torch.fx.Node),"normalization_tensor",str(node))
  if node.name in shapes:return shapes[node.name]
  if node.op=="get_attr":
   value=result
   for name in node.target.split("."):value=getattr(value,name)
   require(isinstance(value,torch.Tensor),"normalization_public_tensor",node.name)
   shape=validated_shape(tuple(value.shape),node.name)
  elif node.op=="call_method"and node.target in("clone","contiguous","detach","float","double","neg","square"):
   shape=known(node.args[0])
  elif node.op=="call_method"and node.target in("reshape","view"):
   target=node.args[1]if len(node.args)==2 else node.args[1:]
   require(not node.kwargs,"normalization_shape_arguments",node.name)
   shape=reshape(known(node.args[0]),size(target,node.name),node.name)
  elif node.op=="call_method"and node.target=="transpose":
   require(len(node.args)==3 and not node.kwargs,"normalization_shape_arguments",node.name)
   shape=list(known(node.args[0]));a=axis(node.args[1],len(shape),node.name);b=axis(node.args[2],len(shape),node.name)
   shape[a],shape[b]=shape[b],shape[a];shape=tuple(shape)
  elif node.op=="call_method"and node.target=="permute":
   source=known(node.args[0]);dims=size(node.args[1]if len(node.args)==2 else node.args[1:],node.name)
   require(type(dims)is tuple and not node.kwargs,"normalization_shape_arguments",node.name)
   dims=[axis(n,len(source),node.name)for n in dims]
   require(sorted(dims)==list(range(len(source))),"normalization_axis",node.name);shape=tuple(source[n]for n in dims)
  elif node.op=="call_method"and node.target=="expand":
   source=known(node.args[0]);target=size(node.args[1]if len(node.args)==2 else node.args[1:],node.name)
   require(type(target)is tuple and not node.kwargs,"normalization_shape_arguments",node.name)
   require(len(target)>=len(source),"normalization_broadcast",node.name)
   padded=(1,)*(len(target)-len(source))+source
   target=tuple(padded[i]if n==-1 and i>=len(target)-len(source)else n for i,n in enumerate(target))
   shape=expanded(source,target,node.name)
  elif node.op=="call_function"and node.target in(operator.add,operator.sub,operator.mul):
   require(not node.kwargs and len(node.args)==2,"normalization_shape_arguments",node.name)
   operands=[known(a)if isinstance(a,torch.fx.Node)else ()for a in node.args]
   try:shape=tuple(torch.broadcast_shapes(*operands))
   except RuntimeError:raise NormalizationError("normalization_broadcast",node.name)
  elif node.op=="call_function"and node.target is operator.neg:
   shape=known(node.args[0])
  elif node.op=="call_module":
   child=result.get_submodule(node.target);shape=known(node.args[0])
   if isinstance(child,torch.nn.Linear):
    require(shape[-1]==child.in_features,"normalization_linear_shape",node.name);shape=(*shape[:-1],child.out_features)
   elif not isinstance(child,(torch.nn.Identity,torch.nn.Dropout)):
    raise NormalizationError("normalization_unresolved_static_shape",node.name)
  else:raise NormalizationError("normalization_unresolved_static_shape",node.name)
  shape=validated_shape(shape,node.name);shapes[node.name]=shape;return shape
 for node in result.graph.nodes:
  if node.op!="call_method"or node.target not in("expand_as","broadcast_to"):continue
  method=node.target;source=known(node.args[0]);args=node.args;kwargs=dict(node.kwargs)
  if method=="expand_as":
   require((len(args)==2 and not kwargs)or(len(args)==1 and set(kwargs)=={"other"}),"normalization_helper_arguments",node.name)
   target=known(args[1]if len(args)==2 else kwargs["other"])
  else:
   require((len(args)>=2 and not kwargs)or(len(args)==1 and set(kwargs)=={"size"}),"normalization_helper_arguments",node.name)
   if kwargs:raw_target=kwargs["size"]
   elif len(args)==2 and isinstance(args[1],(tuple,list,torch.Size,torch.fx.Node)):raw_target=args[1]
   else:raw_target=args[1:]
   target=size(raw_target,node.name)
  target=expanded(source,target,node.name)
  entry=dict(node=node.name,original_operation=method,normalized_operation="expand",
             input_shape=list(source),output_shape=list(target),alias_semantics="broadcast_view",dtype_semantics="source_dtype",
             shape_source="declared_signature_or_registered_static_shape_rule")
  node.target="expand";node.args=(args[0],target);node.kwargs={};node.meta["normalization_origin"]=entry
  shapes[node.name]=target;records.append(entry)
 result.graph.lint();result.recompile()
 require(module.code==before,"normalization_original_mutated","graph")
 manifest=dict(version=VERSION,original_graph_sha256=hashlib.sha256(before.encode()).hexdigest(),
  normalized_graph_sha256=hashlib.sha256(result.code.encode()).hexdigest(),records=records,
  backend_changes=False,source_admission_registered=False)
 manifest["binding"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 return result,manifest
