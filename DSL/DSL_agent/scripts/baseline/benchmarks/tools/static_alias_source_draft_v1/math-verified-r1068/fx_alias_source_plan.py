"""Fixed-target FX mapping for explicitly known local fixtures, without source imports."""
import hashlib, math, operator
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.fx.immutable_collections import immutable_list, immutable_dict
from benchmark_graph import NAME, digest, axis, broadcast, dimensions
from model_resources import resolve
from alias_source_plan import SOURCE_FORMAT, plan, check, AliasError, reshape_dims, decode_index

def index_data(index):
    values = index if type(index) is tuple else (index,)
    out = []
    for value in values:
        if type(value) is slice:out.append(dict(kind="slice",start=value.start,stop=value.stop,step=value.step))
        elif type(value) is int:out.append(dict(kind="int",value=value))
        elif value is Ellipsis:out.append(dict(kind="ellipsis"))
        elif value is None:out.append(dict(kind="newaxis"))
        else:raise AliasError("alias_call_rejected",-1)
    return out
def args_bound(args,kwargs,names,defaults=None,position=-1,required=None,max_pos=None):
    check(type(args) in (tuple,list) and hasattr(kwargs,"items") and len(args) <= (len(names) if max_pos is None else max_pos)
          and not(set(kwargs)-set(names)) and not(set(names[:len(args)])&set(kwargs)), "alias_call_rejected",position)
    result = dict(defaults or {});result.update(zip(names,args));result.update(kwargs)
    check(all(n in result for n in (names if required is None else required)), "alias_call_rejected",position)
    return result
def from_fx(traced,signature,approximations,*,resource_profile="general-tensors-v1"):
    check(isinstance(traced,torch.fx.GraphModule),"alias_call_rejected")
    limits=resolve(resource_profile);nodes=list(traced.graph.nodes)
    check(len(nodes) <= limits.original_nodes+limits.constants+limits.inputs+1,"alias_resource_rejected")
    check(type(signature) is list and 1<=len(signature)<=limits.inputs,"alias_shape_rejected")
    for inp in signature:
        check(type(inp) is dict and set(inp)=={"name","shape"} and type(inp["name"]) is str and NAME.fullmatch(inp["name"]),"alias_shape_rejected")
        try:dimensions(inp["shape"],resource_profile=resource_profile)
        except (TypeError,ValueError,OverflowError):raise AliasError("alias_shape_rejected",-1)
    check(len({x["name"] for x in signature})==len(signature) and sum(math.prod(x["shape"]) for x in signature)<=limits.input_elements,"alias_resource_rejected")
    source=dict(format=SOURCE_FORMAT,inputs=signature,constants={},nodes=[],outputs=[])
    names={n["name"]:tuple(n["shape"]) for n in signature};values={};shapes=dict(names);position=-1
    used=set(names);counter=0;locations=[]
    def fresh(prefix):
        nonlocal counter
        counter+=1;n=prefix+"_"+str(counter)
        while n in used:counter+=1;n=prefix+"_"+str(counter)
        used.add(n);return n
    def ref(value):
        if type(value) is torch.fx.Node:
            check(value in values and type(values[value]) is str,"alias_call_rejected",position);return values[value]
        check(type(value) in (float,int) and abs(value)<=1024 and math.isfinite(value),"alias_call_rejected",position)
        name=fresh("scalar");source["constants"][name]=float(value);shapes[name]=();return name
    def emit(node,op,refs,attrs,result_shape):
        name=fresh("alias");source["nodes"].append(dict(id=fresh("source"),op=op,inputs=refs,attrs=attrs,output=name))
        shapes[name]=tuple(result_shape);locations.append(dict(position=len(source["nodes"])-1,fx_node=node.name,fx_kind=node.op,operation=op))
        return name
    def attr(node):
        value=traced
        for segment in str(node.target).split("."):value=getattr(value,segment)
        check(type(value) in (torch.Tensor,torch.nn.Parameter) and value.device.type=="cpu" and
              value.dtype is torch.float64 and value.numel()<=limits.constant_elements,"alias_call_rejected",position)
        check(torch.isfinite(value).all().item() and torch.max(torch.abs(value)).item()<=1024,"alias_call_rejected",position)
        name=fresh("attribute");source["constants"][name]=value.detach().tolist();shapes[name]=tuple(value.shape);return name
    for position,node in enumerate(nodes):
        if node.op=="placeholder":
            if node.target not in names and not node.users:continue
            check(node.target in names,"alias_shape_rejected",position);values[node]=str(node.target);continue
        if node.op=="get_attr":values[node]=attr(node);continue
        if node.op=="output":
            def structure(value):
                if type(value) is torch.fx.Node:
                    check(value in values and type(values[value]) is str,"alias_call_rejected",position)
                    name="output_"+str(len(source["outputs"]));source["outputs"].append(dict(name=name,value=values[value]))
                    return dict(kind="tensor",name=name)
                if value is None:return dict(kind="none")
                if type(value) in (tuple,list,immutable_list):
                    return dict(kind="tuple" if type(value) is tuple else "list",items=[structure(x) for x in value])
                if type(value) in (dict,immutable_dict) and all(type(k) is str for k in value):
                    return dict(kind="dict",items=[dict(key=k,value=structure(x)) for k,x in value.items()])
                raise AliasError("alias_call_rejected",position)
            check(len(node.args)==1 and not node.kwargs,"alias_call_rejected",position)
            output_schema=structure(node.args[0]);continue
        check(node.op in ("call_method","call_function","call_module"),"alias_call_rejected",position)
        target=node.target;kind=node.op;args=node.args;kwargs=node.kwargs
        binary_functions={operator.add:"add",operator.sub:"subtract",operator.mul:"multiply",
                          torch.add:"add",torch.sub:"subtract",torch.subtract:"subtract",torch.mul:"multiply",torch.multiply:"multiply"}
        binary_methods={"add":"add","add_":"add","sub":"subtract","sub_":"subtract",
                        "subtract":"subtract","subtract_":"subtract","mul":"multiply","mul_":"multiply","multiply":"multiply","multiply_":"multiply"}
        if (kind=="call_function" and target in binary_functions) or (kind=="call_method" and target in binary_methods):
            op=binary_functions[target] if kind=="call_function" else binary_methods[target]
            if kind=="call_function" and target in (operator.add,operator.sub,operator.mul):
                check(len(args)==2 and not kwargs,"alias_call_rejected",position);bound=dict(input=args[0],other=args[1],alpha=1)
            else:
                fields=("input","other") if op=="multiply" else ("input","other","alpha")
                if kind=="call_method":check(not(set(kwargs)&{"input"}),"alias_call_rejected",position)
                bound=args_bound(args,kwargs,fields,dict(alpha=1),position,required=("input","other"),max_pos=2)
            r=[ref(bound["input"]),ref(bound["other"])]
            check(type(bound["alpha"]) in (int,float) and abs(bound["alpha"])<=1024 and math.isfinite(bound["alpha"]),"alias_call_rejected",position)
            values[node]=emit(node,op,r,dict(inplace=kind=="call_method" and target.endswith("_"),alpha=bound["alpha"]),broadcast(shapes[r[0]],shapes[r[1]]));continue
        if kind=="call_module":
            child=traced.get_submodule(target)
            check(type(child) is torch.nn.ReLU and child.training is False and not child._forward_hooks and not child._forward_pre_hooks,
                  "alias_call_rejected",position)
            b=args_bound(args,kwargs,("input",),position=position);r=ref(b["input"])
            values[node]=emit(node,"relu",[r],dict(inplace=child.inplace),shapes[r]);continue
        if (kind=="call_method" and target in ("relu","relu_")) or (kind=="call_function" and target in (F.relu,torch.relu,torch.relu_)):
            if kind=="call_function" and target is F.relu:
                b=args_bound(args,kwargs,("input","inplace"),dict(inplace=False),position)
            else:b=args_bound(args,kwargs,("input",),position=position);b["inplace"]=(target=="relu_" or target is torch.relu_)
            r=ref(b["input"]);values[node]=emit(node,"relu",[r],dict(inplace=b["inplace"]),shapes[r]);continue
        if (kind=="call_method" and target=="square") or (kind=="call_function" and target is torch.square):
            b=args_bound(args,kwargs,("input",),position=position);r=ref(b["input"])
            values[node]=emit(node,"square",[r],{},shapes[r]);continue
        if kind=="call_method" and target in ("clone","contiguous"):
            check(len(args)==1 and not kwargs,"alias_call_rejected",position);r=ref(args[0])
            values[node]=emit(node,target,[r],{},shapes[r]);continue
        if (kind=="call_method" and target in ("reshape","view")) or (kind=="call_function" and target is torch.reshape):
            check(not kwargs and len(args)>=2,"alias_call_rejected",position)
            r=ref(args[0]);wanted=list(args[1]) if len(args)==2 and type(args[1]) in (tuple,list,immutable_list) else list(args[1:])
            result=reshape_dims(shapes[r],wanted,limits,position)
            values[node]=emit(node,"view" if target=="view" else "reshape",[r],dict(shape=wanted),result);continue
        if (kind=="call_method" and target=="flatten") or (kind=="call_function" and target is torch.flatten):
            b=args_bound(args,kwargs,("input","start_dim","end_dim"),dict(start_dim=0,end_dim=-1),position);r=ref(b["input"])
            first,last=axis(b["start_dim"],len(shapes[r])),axis(b["end_dim"],len(shapes[r]))
            check(first<=last,"alias_shape_rejected",position);s=shapes[r]
            values[node]=emit(node,"flatten",[r],dict(start_dim=b["start_dim"],end_dim=b["end_dim"]),
                s[:first]+(math.prod(s[first:last+1]),)+s[last+1:]);continue
        if (kind=="call_method" and target=="transpose") or (kind=="call_function" and target is torch.transpose):
            b=args_bound(args,kwargs,("input","dim0","dim1"),position=position);r=ref(b["input"]);s=list(shapes[r])
            i,j=axis(b["dim0"],len(s)),axis(b["dim1"],len(s));s[i],s[j]=s[j],s[i]
            values[node]=emit(node,"transpose",[r],dict(dim0=b["dim0"],dim1=b["dim1"]),s);continue
        if (kind=="call_method" and target=="permute") or (kind=="call_function" and target is torch.permute):
            check(len(args)>=2 and not kwargs,"alias_call_rejected",position);r=ref(args[0])
            dims=list(args[1]) if len(args)==2 and type(args[1]) in (tuple,list,immutable_list) else list(args[1:])
            norm=[axis(x,len(shapes[r])) for x in dims]
            check(sorted(norm)==list(range(len(shapes[r]))),"alias_shape_rejected",position)
            values[node]=emit(node,"permute",[r],dict(dims=dims),[shapes[r][i] for i in norm]);continue
        if kind=="call_method" and target=="expand":
            check(len(args)>=2 and not kwargs,"alias_call_rejected",position);r=ref(args[0])
            wanted=list(args[1]) if len(args)==2 and type(args[1]) in (tuple,list,immutable_list) else list(args[1:])
            check(len(wanted)>=len(shapes[r]),"alias_shape_rejected",position)
            old=(1,)*(len(wanted)-len(shapes[r]))+shapes[r]
            expanded=[old[i] if v==-1 and i>=len(wanted)-len(shapes[r]) else v for i,v in enumerate(wanted)]
            check(all(type(n) is int and n>0 for n in expanded) and broadcast(shapes[r],tuple(expanded))==tuple(expanded),"alias_shape_rejected",position)
            values[node]=emit(node,"expand",[r],dict(shape=expanded),expanded);continue
        if ((kind=="call_method" and target=="split") or (kind=="call_function" and target is torch.split)):
            b=args_bound(args,kwargs,("input","split_size_or_sections","dim"),dict(dim=0),position);r=ref(b["input"])
            dim=axis(b["dim"],len(shapes[r]));size=shapes[r][dim];sections=b["split_size_or_sections"]
            if type(sections) is int:
                check(sections>0,"alias_shape_rejected",position);sections=[min(sections,size-i) for i in range(0,size,sections)]
            check(type(sections) in (tuple,list,immutable_list) and sections and all(type(n) is int and n>0 for n in sections) and sum(sections)==size,
                  "alias_shape_rejected",position)
            port=[];offset=0
            for count in sections:
                spec=[dict(kind="slice",start=None,stop=None,step=None) for _ in shapes[r]]
                spec[dim]=dict(kind="slice",start=offset,stop=offset+count,step=None);shape=list(shapes[r]);shape[dim]=count
                port.append(emit(node,"slice",[r],dict(index=spec),shape));offset+=count
            values[node]=port;continue
        if kind=="call_function" and target is operator.getitem:
            check(len(args)==2 and not kwargs and type(args[0]) is torch.fx.Node and args[0] in values,"alias_call_rejected",position)
            if type(values[args[0]]) is list:
                check(type(args[1]) is int and -len(values[args[0]])<=args[1]<len(values[args[0]]),"alias_shape_rejected",position)
                values[node]=values[args[0]][args[1]];continue
            r=ref(args[0]);spec=index_data(args[1])
            try:result_shape=np.empty(shapes[r],dtype=np.uint8)[decode_index(spec,len(shapes[r]),position)].shape
            except (IndexError,ValueError,TypeError):raise AliasError("alias_shape_rejected",position)
            values[node]=emit(node,"slice",[r],dict(index=spec),result_shape);continue
        raise AliasError("alias_call_rejected",position)
    check("output_schema" in locals(),"alias_source_rejected")
    value=plan(source,approximations,resource_profile=resource_profile)
    metadata=dict(format="poseidon-alias-fx-draft-v1",source_graph_sha256=hashlib.sha256(str(traced.graph).encode()).hexdigest(),
        source_ir_sha256=digest(source),fx_mapping=locations,output_schema=output_schema,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata["binding"]=digest(metadata);return value,metadata
