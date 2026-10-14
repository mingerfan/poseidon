"""Trusted nn.Module to operator graph. Never call on untrusted code in the host."""
import hashlib
import operator
from benchmark_graph import require, digest
from model_decomposition import FORMAT, decompose

REGISTERED_FUNCTION_POLICY="registered-v2"
ADAPTIVE_FUNCTION_POLICY="static-adaptive-average-v1"

def capture(module, inputs, outputs, *, approximations=None, resource_profile="legacy-v1",
            dtype_policy="legacy-float64-v1", preserve_output_structure=False, record_state_sources=False,
            capture_policy="symbolic-v1", approximation_policy="operator-defaults-v1", matrix_policy="rank-two-three-v1", functional_policy=REGISTERED_FUNCTION_POLICY,
            capture_aliases=False):
    require(type(capture_aliases) is bool,"Explicit ordered alias capture policy")
    require(functional_policy in (REGISTERED_FUNCTION_POLICY,ADAPTIVE_FUNCTION_POLICY),"Functional capture policy")
    from model_resources import resolve
    import math
    limits=resolve(resource_profile)
    require(type(inputs) is list and 1<=len(inputs)<=limits.inputs,"Input count")
    total=0
    for item in inputs:
        require(type(item) is dict and set(item)=={"name","shape"},"Input declaration")
        dims=item["shape"]
        require(type(dims) is list and 1<=len(dims)<=limits.rank and
                all(type(n) is int and 0<n<=limits.axis_size for n in dims),"Input shape")
        total+=math.prod(dims)
    require(total<=limits.input_elements,"Input element budget")
    require(type(outputs) is list and 1<=len(outputs)<=limits.outputs,"Output count")
    import torch
    from torch.fx import symbolic_trace
    from torch.fx.immutable_collections import immutable_list
    from torch.fx.passes.shape_prop import ShapeProp
    approximations={} if approximations is None else approximations
    from pytorch_approximation import LEGACY_POLICY as LEGACY_APPROXIMATION, POLICY as SCOPED_APPROXIMATION, Resolver, module_paths
    require(approximation_policy in (LEGACY_APPROXIMATION,SCOPED_APPROXIMATION),"Approximation capture policy")
    selection=Resolver(approximations) if approximation_policy==SCOPED_APPROXIMATION else None
    require(selection is None or capture_policy in ("signature-static-v1","signature-static-index-v1") and preserve_output_structure,
            "Scoped approximation requires static capture metadata")
    require(isinstance(module,torch.nn.Module),"PyTorch nn.Module required")
    from pytorch_source_contract import FLOAT_POLICIES,output_tree
    require(dtype_policy=="legacy-float64-v1" or dtype_policy in FLOAT_POLICIES,"Float policy")
    source_dtypes={name:str(value.dtype).removeprefix("torch.")
                   for name,value in module.state_dict().items()}
    if dtype_policy!="legacy-float64-v1":
        for value in module.state_dict().values():
            if value.is_floating_point():
                require(value.dtype in (torch.float32,torch.float64),"Supported source floating dtype")
                if dtype_policy=="strict-float64-v1":
                    require(value.dtype is torch.float64,"Strict float64 source policy")
            else:
                require(value.dtype in (torch.int32,torch.int64,torch.bool),"Supported public dtype")
    if capture_aliases:
        import copy
        module=copy.deepcopy(module)
    module=module.cpu().double().eval()
    import random
    import numpy as np
    rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    before={k:v.detach().clone() for k,v in module.state_dict().items()}
    from pytorch_static_trace import LEGACY_POLICY,POLICY,INDEX_POLICY,trace,policy_record
    from pytorch_matrix_plan import LEGACY_POLICY as LEGACY_MATRIX,POLICY as STATIC_MATRIX
    require(matrix_policy in (LEGACY_MATRIX,STATIC_MATRIX),"Matrix capture policy")
    require(matrix_policy==LEGACY_MATRIX or capture_policy==INDEX_POLICY and preserve_output_structure,
            "Static matrix normalization requires v7 capture metadata")
    require(capture_policy in (LEGACY_POLICY,POLICY,INDEX_POLICY), "Capture shape policy")
    require(capture_policy==LEGACY_POLICY or preserve_output_structure, "Static capture requires output metadata")
    if capture_aliases:
        require(isinstance(module,torch.fx.GraphModule) and capture_policy==INDEX_POLICY and
                preserve_output_structure,"Ordered capture requires an explicit trusted FX graph")
        traced=module
        from pytorch_static_trace import ALIAS_POLICY
        record=getattr(traced,"meta",{}).get("ordered_alias_capture_policy")
        require(record==policy_record(inputs,resource_profile,policy=ALIAS_POLICY),
                "Ordered capture source policy binding")
        fx_nodes=list(traced.graph.nodes)
        require(sum(n.op in ("call_function","call_method","call_module") for n in fx_nodes)<=limits.original_nodes
                and len(fx_nodes)<=limits.original_nodes+limits.constants+limits.inputs+limits.outputs,
                "Ordered capture FX budget")
    else:
        traced=symbolic_trace(module) if capture_policy==LEGACY_POLICY else trace(module,inputs,resource_profile=resource_profile,policy=capture_policy)
    ShapeProp(traced).propagate(*(torch.zeros(i["shape"],dtype=torch.float64) for i in inputs))
    require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py_rng
            and all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng)),
            "Random forward is not supported")
    require(before.keys()==module.state_dict().keys() and
            all(torch.equal(v,module.state_dict()[k]) for k,v in before.items()),
            "State-mutating forward is not supported")
    graph=dict(format=FORMAT,id="pytorch_model",inputs=inputs,constants={},nodes=[],outputs=[])
    refs={};public_values={};public_linear_weights={};static_values={};constant_dtypes={};state_constant_sources={};count=0;input_iter=iter(inputs)
    alias_origins={}
    if capture_aliases:from pytorch_alias_capture_bridge import origin,normalize_emissions
    reserved_fx_names={node.name for node in traced.graph.nodes}
    def constant(value,state_source=None):
        nonlocal count
        dtype=None
        if isinstance(value,torch.Tensor):
            dtype=str(value.dtype).removeprefix("torch.")
            if dtype_policy!="legacy-float64-v1":
                require(value.dtype in (torch.float32,torch.float64,torch.int32,torch.int64),
                        "Boolean/complex public arithmetic is not supported")
                if dtype_policy=="strict-float64-v1" and value.is_floating_point():
                    require(value.dtype is torch.float64,"Strict float64 constant policy")
                if value.dtype in (torch.int32,torch.int64):
                    exact=value.to(torch.int64)
                    require(bool((exact>=-(2**53)).all() and (exact<=2**53).all()),
                            "Public integer precision")
            value=value.detach().cpu().double().numpy().tolist()
        name="weight_"+str(count);count+=1
        if dtype is not None:constant_dtypes[name]=dtype
        graph["constants"][name]=value
        if record_state_sources and state_source in source_dtypes:
            state_constant_sources[name]=state_source
        return name
    def ref(value):
        if isinstance(value,torch.fx.Node):
            if value in static_values:return constant(static_values[value])
            if value not in refs and value in public_values:
                refs[value]=constant(public_values[value])
            return refs[value]
        return constant(value)
    def static_value(value):
        if isinstance(value,torch.fx.Node):
            require(value in static_values,"Data-dependent shape argument")
            return static_values[value]
        if type(value) in (tuple,list,immutable_list):return [static_value(v) for v in value]
        require(type(value) in (int,str,bool) or value is None,"Static shape argument")
        return value
    def approximation(name,op=None):
        if selection is not None:return selection.select(node.name,op or name,name,module_paths(node))
        require(name in approximations,"Explicit approximation required for "+name)
        return approximations[name]
    def emit(node,op,values,attrs=None,*,suffix=""):
        label=node.name+suffix
        if suffix:
            base=label;index=0
            while label in reserved_fx_names:
                index+=1;label=base+"_"+str(index)
            reserved_fx_names.add(label)
        name="value_"+label
        item=dict(id="fx_"+label,op=op,inputs=values,attrs=attrs or {},outputs=[name])
        graph["nodes"].append(item)
        if capture_aliases:alias_origins[item["id"]]=origin(node,item,traced)
        refs[node]=name
    def shape(node):
        return tuple(node.meta["tensor_meta"].shape)
    def pair(v,rank):
        return [v]*rank if isinstance(v,int) else list(v)

    def static_arguments(node,names,defaults,label):
        require(len(node.args)<=len(names) and not(set(node.kwargs)-set(names)),label+" arguments")
        values=dict(zip(names,defaults))
        for key,value in zip(names,node.args):
            require(key not in node.kwargs,"Duplicate "+label+" argument: "+key)
            values[key]=value
        values.update(node.kwargs)
        return values

    index_records=[];matrix_records=[];split_values={};split_public_values={}
    def public_index(value):
        if isinstance(value,torch.fx.Node):
            if value in static_values:return static_values[value]
            require(value.op=="get_attr" and value in public_values,"Public static tensor index required")
            value=public_values[value]
        if isinstance(value,torch.Tensor):
            require(value.dtype in (torch.int32,torch.int64) and value.ndim==1,"Public index vector type")
            return value.tolist()
        if type(value) in (tuple,list,immutable_list):return (tuple if type(value) is tuple else list)(public_index(v) for v in value)
        if isinstance(value,slice):return slice(*(public_index(v) if v is not None else None for v in (value.start,value.stop,value.step)))
        require(type(value) is int or value is None or value is Ellipsis,"Public static tensor index required")
        return value

    def emit_index(node,x,selected,kind):
        source=ref(x);previous=source;ids=[]
        for i,step in enumerate(selected["steps"]):
            emit(node,step["op"],[previous],step["attrs"],suffix="_index_"+str(i))
            previous=ref(node);ids.append(graph["nodes"][-1]["id"])
        emit(node,"reshape",[previous],dict(shape=selected["result_shape"] or [1]))
        ids.append(graph["nodes"][-1]["id"])
        require(tuple(selected["result_shape"])==shape(node),"Static index PyTorch result shape")
        index_records.append(dict(operation=kind,source=source,source_shape=list(shape(x)),plan=selected,node_ids=ids))

    def public_affine(gain,bias,dims):
        require(type(dims) in (tuple,list) and dims and
                all(type(n) is int and n>0 for n in dims),"Static affine shape")
        # Defaults are mathematical public constants, never model parameters.
        return [constant(torch.ones(dims,dtype=torch.float64)) if gain is None else ref(gain),
                constant(torch.zeros(dims,dtype=torch.float64)) if bias is None else ref(bias)]

    def instance_normalization(node,tensor,affine,statistics,eps,*,spatial_rank=None):
        require(isinstance(tensor,torch.fx.Node),"InstanceNorm tensor input")
        dims=shape(tensor)
        unbatched=spatial_rank is not None and len(dims)==spatial_rank+1
        require((spatial_rank is None and len(dims) in (3,4)) or
                spatial_rank in (1,2) and len(dims) in (spatial_rank+1,spatial_rank+2),
                "InstanceNorm rank-3/4 batched or explicit rank-2/3 unbatched input")
        require(type(eps) in (int,float) and math.isfinite(eps) and 0<eps<=1,
                "InstanceNorm epsilon")
        x=ref(tensor)
        if unbatched:
            emit(node,"reshape",[x],dict(shape=[1,*dims]),suffix="_instance_batch")
            x=ref(node)
        if statistics is None:
            emit(node,"instance_norm",[x,*affine],
                 dict(eps=eps,approximation=approximation("rsqrt","instance_norm")))
        else:
            emit(node,"batch_norm",[x,*statistics,*affine],dict(eps=eps))
        if unbatched:
            emit(node,"reshape",[ref(node)],dict(shape=list(dims)),suffix="_instance_restore")

    def reduction_axes(dim,x):
        dim=static_value(dim)
        if dim is None or isinstance(dim,(tuple,list)) and len(dim)==0:
            return list(range(len(shape(x))))
        axes=[dim] if type(dim) is int else list(dim) if isinstance(dim,(tuple,list)) else []
        require(axes and all(type(i) is int for i in axes),"Static reduction axes")
        return axes

    def padding_attrs(node,pad,mode,value):
        require(isinstance(pad,(tuple,list)) and len(pad)%2==0 and len(pad)<=2*len(shape(node.args[0])),
                "Static PyTorch padding")
        pairs=[[0,0] for _ in shape(node.args[0])]
        for i in range(len(pad)//2):pairs[-1-i]=list(pad[2*i:2*i+2])
        return dict(pads=pairs,mode=mode,value=0 if value is None else value)
    for node in traced.graph.nodes:
        if node.op=="placeholder":
            item=next(input_iter,None);require(item is not None,"Input declaration count")
            refs[node]=item["name"]
        elif node.op=="get_attr":
            value=traced
            for segment in node.target.split("."):value=getattr(value,segment)
            require(isinstance(value,torch.Tensor),"Only tensor attributes")
            index_only=bool(node.users) and all(
                user.target in ("index_select",torch.index_select) and len(user.args)==3
                and user.args[2] is node for user in node.users)
            if capture_policy==INDEX_POLICY:
                def used_as_index(user):
                    if user.target in ("index_select",torch.index_select):
                        return (len(user.args)==3 and user.args[2] is node) or user.kwargs.get("index") is node
                    if user.target is operator.getitem and len(user.args)==2:
                        def contains(v):
                            return v is node or type(v) in (tuple,list,immutable_list) and any(contains(k) for k in v)
                        return contains(user.args[1])
                    return False
                index_only=bool(node.users) and all(used_as_index(user) for user in node.users)
            if index_only:
                require(value.dtype in (torch.int32,torch.int64) and value.ndim==1,
                        "Public index vector type")
                if capture_policy==INDEX_POLICY:public_values[node]=value
            else:
                public_values[node]=value
                refs[node]=constant(value,node.target)
        elif node.op=="call_module":
            m=traced.get_submodule(node.target)
            argument=static_arguments(node,("input",),(None,),"Module input")["input"]
            require(isinstance(argument,torch.fx.Node),"Module tensor input")
            x=ref(argument)
            if isinstance(m,torch.nn.Linear):
                emit(node,"linear",[x,constant(m.weight,str(node.target)+".weight")]+([] if m.bias is None else [constant(m.bias,str(node.target)+".bias")]))
            elif isinstance(m,(torch.nn.Conv1d,torch.nn.Conv2d)):
                rank=1 if isinstance(m,torch.nn.Conv1d) else 2
                require(m.padding_mode=="zeros" and not isinstance(m.padding,str),"Explicit zero convolution padding")
                emit(node,"conv"+str(rank)+"d",[x,constant(m.weight,str(node.target)+".weight")]+([] if m.bias is None else [constant(m.bias,str(node.target)+".bias")]),
                     dict(stride=pair(m.stride,rank),padding=pair(m.padding,rank),dilation=pair(m.dilation,rank),groups=m.groups))
            elif isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
                require(m.track_running_stats and m.running_mean is not None,"Frozen BatchNorm statistics required")
                gain=m.weight if m.affine else torch.ones(m.num_features,dtype=torch.float64)
                bias=m.bias if m.affine else torch.zeros(m.num_features,dtype=torch.float64)
                emit(node,"batch_norm",[x,constant(m.running_mean,str(node.target)+".running_mean"),constant(m.running_var,str(node.target)+".running_var"),constant(gain,str(node.target)+".weight"),constant(bias,str(node.target)+".bias")],dict(eps=m.eps))
            elif isinstance(m,(torch.nn.InstanceNorm1d,torch.nn.InstanceNorm2d)):
                rank=1 if isinstance(m,torch.nn.InstanceNorm1d) else 2
                dims=shape(argument)
                require(len(dims) in (rank+1,rank+2),"InstanceNorm module rank")
                channels=dims[0] if len(dims)==rank+1 else dims[1]
                require(channels==m.num_features,"InstanceNorm declared channel count")
                affine=[constant(m.weight,str(node.target)+".weight") if m.weight is not None
                        else constant(torch.ones(channels,dtype=torch.float64)),
                        constant(m.bias,str(node.target)+".bias") if m.bias is not None
                        else constant(torch.zeros(channels,dtype=torch.float64))]
                statistics=None
                if m.track_running_stats:
                    require(m.running_mean is not None and m.running_var is not None,
                            "Frozen InstanceNorm statistics required")
                    statistics=[constant(m.running_mean,str(node.target)+".running_mean"),
                                constant(m.running_var,str(node.target)+".running_var")]
                instance_normalization(node,argument,affine,statistics,m.eps,spatial_rank=rank)
            elif isinstance(m,torch.nn.LayerNorm):
                gain=constant(m.weight,str(node.target)+".weight") if m.weight is not None else constant(torch.ones(m.normalized_shape,dtype=torch.float64))
                bias=constant(m.bias,str(node.target)+".bias") if m.bias is not None else constant(torch.zeros(m.normalized_shape,dtype=torch.float64))
                emit(node,"layer_norm",[x,gain,bias],dict(normalized_shape=list(m.normalized_shape),eps=m.eps,approximation=approximation("rsqrt","layer_norm")))
            elif isinstance(m,torch.nn.GroupNorm):
                gain=constant(m.weight,str(node.target)+".weight") if m.weight is not None else constant(torch.ones(m.num_channels,dtype=torch.float64))
                bias=constant(m.bias,str(node.target)+".bias") if m.bias is not None else constant(torch.zeros(m.num_channels,dtype=torch.float64))
                emit(node,"group_norm",[x,gain,bias],dict(groups=m.num_groups,eps=m.eps,approximation=approximation("rsqrt","group_norm")))

            elif isinstance(m,torch.nn.LeakyReLU):
                require(not m.inplace or capture_aliases,"In-place activation not yet imported")
                emit(node,"leaky_relu",[x],dict(negative_slope=m.negative_slope,approximation=approximation("relu","leaky_relu")))
            elif isinstance(m,torch.nn.PReLU):
                emit(node,"prelu",[x,constant(m.weight,str(node.target)+".weight")],dict(approximation=approximation("relu","prelu")))
            elif selection is not None and isinstance(m,torch.nn.Softmax):
                require(type(m.dim) is int,"Explicit Softmax axis required")
                emit(node,"softmax",[x],dict(axis=m.dim,exp=approximation("exp","softmax"),
                                           reciprocal=approximation("reciprocal","softmax")))
            elif isinstance(m,torch.nn.LogSoftmax):
                require(m.dim is not None,"Explicit LogSoftmax axis required")
                emit(node,"logsoftmax",[x],dict(axis=m.dim,exp=approximation("exp","logsoftmax"),log=approximation("log","logsoftmax")))
            elif isinstance(m,(torch.nn.ReLU,torch.nn.GELU,torch.nn.SiLU,torch.nn.Sigmoid,torch.nn.Tanh)):
                op={torch.nn.ReLU:"relu",torch.nn.GELU:"gelu",torch.nn.SiLU:"silu",torch.nn.Sigmoid:"sigmoid",torch.nn.Tanh:"tanh"}[type(m)]
                if isinstance(m,torch.nn.GELU):require(m.approximate=="none","Exact GELU source required")
                require(not getattr(m,"inplace",False) or capture_aliases,"In-place activation not yet imported")
                emit(node,op,[x],dict(approximation=approximation(op)))

            elif isinstance(m,(torch.nn.MaxPool1d,torch.nn.MaxPool2d)):
                rank=1 if isinstance(m,torch.nn.MaxPool1d) else 2
                require(not m.ceil_mode and not m.return_indices,"MaxPool floor mode and values only")
                emit(node,"max_pool"+str(rank)+"d",[x],dict(kernel=pair(m.kernel_size,rank),
                     stride=pair(m.stride or m.kernel_size,rank),padding=pair(m.padding,rank),
                     dilation=pair(m.dilation,rank),approximation=approximation("abs","max_pool"+str(rank)+"d")))
            elif isinstance(m,(torch.nn.ConstantPad1d,torch.nn.ConstantPad2d,torch.nn.ZeroPad2d,
                               torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d,
                               torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d)):
                mode=("reflect" if isinstance(m,(torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d))
                      else "replicate" if isinstance(m,(torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d))
                      else "constant")
                emit(node,"padding",[x],padding_attrs(node,m.padding,mode,getattr(m,"value",0)))
            elif isinstance(m,(torch.nn.AvgPool1d,torch.nn.AvgPool2d)):
                rank=1 if isinstance(m,torch.nn.AvgPool1d) else 2
                require(not m.ceil_mode and getattr(m,"divisor_override",None) is None,"Pool ceil/divisor")
                emit(node,"avg_pool"+str(rank)+"d",[x],dict(kernel=pair(m.kernel_size,rank),stride=pair(m.stride or m.kernel_size,rank),
                    padding=pair(m.padding,rank),count_include_pad=m.count_include_pad))
            elif isinstance(m,(torch.nn.AdaptiveAvgPool1d,torch.nn.AdaptiveAvgPool2d)):
                rank=1 if isinstance(m,torch.nn.AdaptiveAvgPool1d) else 2
                emit(node,"adaptive_avg_pool",[x],dict(output_size=pair(m.output_size,rank)))
            elif isinstance(m,torch.nn.Flatten):
                emit(node,"reshape",[x],dict(shape=list(shape(node))))
            elif isinstance(m,(torch.nn.Dropout,torch.nn.Identity)):
                emit(node,"dropout",[x],dict(training=False))
            else:raise ValueError("Unsupported PyTorch module: "+type(m).__name__)
        elif node.op in ("call_function","call_method"):
            binary={operator.add:"add",operator.sub:"subtract",operator.mul:"multiply",
                    torch.add:"add",torch.sub:"subtract",torch.mul:"multiply"}
            target=node.target
            if capture_aliases and node.op=="call_method" and target in ("add_","sub_","subtract_","mul_","multiply_"):
                op="add" if target=="add_" else "subtract" if target in ("sub_","subtract_") else "multiply"
                values=static_arguments(node,("input","other","alpha"),(None,None,1),"In-place arithmetic")
                require(len(node.args)<=2 and isinstance(values["input"],torch.fx.Node),
                        "In-place arithmetic positional tensor target")
                alpha=values["alpha"]
                require(type(alpha) in (int,float) and math.isfinite(alpha) and abs(alpha)<=1024 and
                        (op!="multiply" or alpha==1),"In-place arithmetic alpha")
                target_ref=ref(values["input"]);other=values["other"];other_ref=ref(other)
                weighted=other_ref
                if alpha!=1:
                    if not isinstance(other,torch.fx.Node):weighted=constant(other*alpha)
                    elif other in public_values:weighted=constant(public_values[other]*alpha)
                    else:
                        emit(node,"multiply",[other_ref,constant(alpha)],suffix="_alias_alpha")
                        weighted=ref(node)
                emit(node,op,[target_ref,weighted])
                alias_origins[graph["nodes"][-1]["id"]].update(mutation_target=target_ref,mutation_other=other_ref)
                continue
            if capture_aliases and node.op=="call_method" and target in ("copy_","clone","detach"):
                if target=="copy_":
                    values=static_arguments(node,("input","source","non_blocking"),(None,None,False),"Tensor copy")
                    require(len(node.args)<=2 and values["non_blocking"] is False,"CPU blocking tensor copy")
                    emit(node,"dropout",[ref(values["input"]),ref(values["source"])],dict(training=False))
                    alias_origins[graph["nodes"][-1]["id"]]["action"]="copy"
                else:
                    require(len(node.args)==1 and not node.kwargs,"Default tensor clone/detach")
                    emit(node,"dropout",[ref(node.args[0])],dict(training=False))
                continue
            if capture_aliases and ((node.op=="call_method" and target in ("relu_","square_")) or
                                    (node.op=="call_function" and target is torch.relu_)):
                require(len(node.args)==1 and not node.kwargs,"Unary in-place tensor signature")
                name="square" if target=="square_" else "relu"
                target_ref=ref(node.args[0])
                attrs={} if name=="square" else dict(approximation=approximation("relu","relu"))
                emit(node,name,[target_ref],attrs)
                alias_origins[graph["nodes"][-1]["id"]]["mutation_target"]=target_ref
                continue
            matrix_functions=(operator.matmul,torch.matmul,torch.mm,torch.bmm,torch.dot)
            matrix_methods=("matmul","mm","bmm","dot")
            if target is getattr and len(node.args)==2 and node.args[1]=="T":
                require(not node.kwargs and isinstance(node.args[0],torch.fx.Node),"Tensor.T arguments")
                source=node.args[0];dims=list(reversed(range(len(shape(source)))))
                if source in public_values:
                    public_values[node]=public_values[source].permute(dims)
                    # A rank-2 public R.T supplies the existing, hash-bound
                    # R directly as the Linear weight for x @ R.T.
                    if len(dims)==2:public_linear_weights[node]=ref(source)
                else:
                    emit(node,"permute",[ref(source)],dict(dims=dims))
                continue
            if target is getattr and len(node.args)==2 and node.args[1] in ("shape","ndim"):
                require(not node.kwargs and isinstance(node.args[0],torch.fx.Node),
                        "Static tensor shape access")
                dims=shape(node.args[0]);static_values[node]=dims if node.args[1]=="shape" else len(dims)
                continue
            if target is operator.getitem and isinstance(node.args[0],torch.fx.Node) and node.args[0] in static_values:
                require(not node.kwargs and len(node.args)==2 and type(node.args[1]) is int,
                        "Static shape indexing")
                static_values[node]=static_values[node.args[0]][node.args[1]]
                continue
            if node.op=="call_method" and target in ("size","dim"):
                require(not node.kwargs and 1<=len(node.args)<=2,"Static tensor size arguments")
                dims=shape(node.args[0])
                if target=="dim":
                    require(len(node.args)==1,"Static tensor dim arguments")
                    static_values[node]=len(dims)
                elif len(node.args)==1:static_values[node]=dims
                else:
                    axis=static_value(node.args[1])
                    require(type(axis) is int and -len(dims)<=axis<len(dims),"Static size axis")
                    static_values[node]=dims[axis]
                continue
            split_call=capture_policy==INDEX_POLICY and ((node.op=="call_function" and target is torch.split)
                                                       or (node.op=="call_method" and target=="split"))
            if split_call:
                from pytorch_split_plan import arguments,plan
                values=arguments(node.args,node.kwargs,method=node.op=="call_method")
                x=values["input"]
                require(isinstance(x,torch.fx.Node),"Static split tensor input")
                selected=plan(list(shape(x)),static_value(values["sections"]),static_value(values["axis"]),
                              resource_profile=resource_profile)
                metadata=node.meta.get("tensor_meta")
                require(type(metadata) in (tuple,list) and len(metadata)==len(selected["result_shapes"]) and
                        all(tuple(m.shape)==tuple(s) for m,s in zip(metadata,selected["result_shapes"])),
                        "Static split PyTorch result shapes")
                if x in public_values:
                    split_public_values[node]=tuple(public_values[x].narrow(selected["axis"],a["start"],n)
                        for a,n in zip(selected["slices"],selected["sections"]))
                elif selected["lowering"]=="split":
                    names=["value_"+node.name+"_port_"+str(i) for i in range(len(selected["sections"]))]
                    graph["nodes"].append(dict(id="fx_"+node.name,op="split",inputs=[ref(x)],
                        attrs=dict(axis=selected["axis"],sections=selected["sections"]),outputs=names))
                    if capture_aliases:
                        alias_origins[graph["nodes"][-1]["id"]]=origin(node,graph["nodes"][-1],traced)
                    split_values[node]=tuple(names)
                else:
                    names=[]
                    for i,attrs in enumerate(selected["slices"]):
                        emit(node,"slice",[ref(x)],attrs,suffix="_split_port_"+str(i))
                        names.append(ref(node))
                    split_values[node]=tuple(names);refs.pop(node,None)
                continue
            if (capture_policy==INDEX_POLICY and target is operator.getitem and
                    isinstance(node.args[0],torch.fx.Node) and
                    (node.args[0] in split_values or node.args[0] in split_public_values)):
                require(not node.kwargs and len(node.args)==2,"Static split selection arguments")
                x=node.args[0];port=static_value(node.args[1])
                values=split_public_values[x] if x in split_public_values else split_values[x]
                require(type(port) is int and -len(values)<=port<len(values),"Static split selection port")
                if x in split_public_values:public_values[node]=values[port]
                else:refs[node]=values[port]
                continue
            if capture_policy==INDEX_POLICY and target is operator.getitem:
                require(not node.kwargs and len(node.args)==2 and isinstance(node.args[0],torch.fx.Node),"Static tensor indexing arguments")
                from pytorch_static_index import plan
                x=node.args[0];selected=plan(shape(x),public_index(node.args[1]),resource_profile=resource_profile)
                emit_index(node,x,selected,"getitem");continue
            if capture_policy==INDEX_POLICY and target in ("select",torch.select,"narrow",torch.narrow,"index_select",torch.index_select):
                kind="select" if target in ("select",torch.select) else "narrow" if target in ("narrow",torch.narrow) else "index_select"
                names=("input","dim","index") if kind=="select" else ("input","dim","start","length") if kind=="narrow" else ("input","dim","index","out")
                values=static_arguments(node,names,(None,)*len(names),"Static "+kind)
                require(isinstance(values["input"],torch.fx.Node),"Static tensor index input")
                require(kind!="index_select" or values["out"] is None,"Index selection out mutation is not supported")
                x=values["input"];axis=public_index(values["dim"])
                if kind=="narrow":
                    start,length=public_index(values["start"]),public_index(values["length"])
                    require(type(axis) is int and -len(shape(x))<=axis<len(shape(x)),"Static tensor index axis")
                    width=shape(x)[axis]
                    require(type(start) is int and -width<=start<=width and type(length) is int
                            and length>0 and (start if start>=0 else start+width)+length<=width,"Static narrow interval")
                    start=start if start>=0 else start+width;index=slice(start,start+length)
                else:index=public_index(values["index"])
                from pytorch_static_index import axis_selection
                selected=axis_selection(shape(x),axis,index,kind=kind,resource_profile=resource_profile)
                emit_index(node,x,selected,kind);continue
            if ((node.op=="call_function" and target in matrix_functions) or
                    (node.op=="call_method" and target in matrix_methods)):
                kind=("dot" if target in (torch.dot,"dot") else
                      "mm" if target in (torch.mm,"mm") else
                      "bmm" if target in (torch.bmm,"bmm") else "matmul")
                names=("input","mat2" if kind in ("mm","bmm") else "tensor" if kind=="dot" else "other","out")
                values=static_arguments(node,names,(None,None,None),"Matrix product")
                left,right,out=(values[key] for key in names)
                require(out is None,"Matrix product out mutation is not supported")
                require(isinstance(left,torch.fx.Node) and isinstance(right,torch.fx.Node),
                        "Matrix product tensor operands")
                first,second=shape(left),shape(right)
                if kind=="mm":
                    require(len(first)==len(second)==2,"mm requires rank-2 tensors")
                elif kind=="bmm":
                    require(len(first)==len(second)==3 and first[0]==second[0],
                            "bmm requires matching rank-3 batches")
                elif kind=="dot":
                    require(len(first)==len(second)==1 and first==second,"dot requires matching vectors")
                plain_left=left in public_values;plain_right=right in public_values
                if matrix_policy==STATIC_MATRIX:
                    require(not (plain_left and plain_right),"A matrix product must depend on an encrypted tensor")
                    from pytorch_matrix_plan import plan,constant_value
                    side="left" if plain_left else "right" if plain_right else None
                    planned=plan(list(first),list(second),operation=kind,public_side=side,resource_profile=resource_profile)
                    require(tuple(planned["result_shape"])==shape(node),"Static matrix PyTorch result shape")
                    sources=[ref(left),ref(right)];mapped=dict(zip(("left","right"),sources));derived=None
                    if planned["constant"] is not None:
                        public=left if plain_left else right
                        derived=constant(constant_value(public_values[public].detach().cpu().numpy(),planned))
                        mapped["constant"]=derived
                    ids=[]
                    for i,step in enumerate(planned["steps"]):
                        emit(node,step["op"],[mapped[v] for v in step["inputs"]],step["attrs"],
                             suffix="_matrix_"+str(i) if i+1<len(planned["steps"]) else "")
                        mapped["step_"+str(i)]=ref(node);ids.append(graph["nodes"][-1]["id"])
                    matrix_records.append(dict(sources=sources,plan=planned,node_ids=ids,constant=derived))
                    continue
                if kind!="dot" and (plain_left or plain_right):
                    require(plain_left != plain_right,"A matrix product must depend on an encrypted tensor")
                    public=left if plain_left else right
                    encrypted=right if plain_left else left
                    require(len(shape(encrypted)) in (2,3),"Public matrix product supports encrypted rank-2/3 tensors")
                    require(len(shape(public))==2,"Public matrix product requires a fixed rank-2 matrix")
                    if plain_right:
                        weight=public_linear_weights.get(right)
                        if weight is None:weight=constant(public_values[right].transpose(0,1))
                        emit(node,"linear",[ref(left),weight])
                    else:
                        emit(node,"transpose",[ref(right)],dict(dim0=-2,dim1=-1),suffix="_public_lhs_input")
                        emit(node,"linear",[ref(node),ref(left)],suffix="_public_lhs_linear")
                        emit(node,"transpose",[ref(node)],dict(dim0=-2,dim1=-1))
                else:
                    emit(node,"dot" if kind=="dot" else "matmul",[ref(left),ref(right)])
            elif node.op=="call_function" and target in binary:
                op=binary[target]
                if target in (operator.add,operator.sub,operator.mul):
                    require(not node.kwargs and len(node.args)==2,"Binary arity/kwargs")
                    args=list(node.args)
                else:
                    names=("input","other","out") if op=="multiply" else ("input","other","alpha","out")
                    values=static_arguments(node,names,
                        (None,None,None) if op=="multiply" else (None,None,1,None),"Binary")
                    require(values["out"] is None and values.get("alpha",1)==1,"Binary out/alpha")
                    args=[values["input"],values["other"]]
                if not isinstance(args[0],torch.fx.Node) and op in ("add","multiply"):args.reverse()
                emit(node,op,[ref(a) for a in args])


            elif target in (torch.nn.functional.conv1d,torch.nn.functional.conv2d):
                rank=1 if target is torch.nn.functional.conv1d else 2
                values=static_arguments(node,("input","weight","bias","stride","padding","dilation","groups"),
                    (None,None,None,1,0,1,1),"Functional convolution")
                require(isinstance(values["input"],torch.fx.Node) and values["weight"] is not None,
                        "Convolution operands")
                require(not isinstance(values["padding"],str),"Explicit zero convolution padding")
                attrs={key:pair(static_value(values[key]),rank) for key in ("stride","padding","dilation")}
                attrs["groups"]=static_value(values["groups"])
                emit(node,"conv"+str(rank)+"d",
                    [ref(values[key]) for key in ("input","weight","bias") if values[key] is not None],attrs)
            elif target in (torch.nn.functional.avg_pool1d,torch.nn.functional.avg_pool2d):
                rank=1 if target is torch.nn.functional.avg_pool1d else 2
                names=("input","kernel_size","stride","padding","ceil_mode","count_include_pad")
                defaults=(None,None,None,0,False,True)
                if rank==2:names+=("divisor_override",);defaults+=(None,)
                values=static_arguments(node,names,defaults,"Functional average pool")
                require(isinstance(values["input"],torch.fx.Node),"Pool tensor input")
                require(values["ceil_mode"] is False and values.get("divisor_override") is None,
                        "Pool ceil/divisor")
                require(type(values["count_include_pad"]) is bool,"Pool count_include_pad")
                kernel=static_value(values["kernel_size"])
                stride=kernel if values["stride"] is None else static_value(values["stride"])
                emit(node,"avg_pool"+str(rank)+"d",[ref(values["input"])],
                     dict(kernel=pair(kernel,rank),stride=pair(stride,rank),
                          padding=pair(static_value(values["padding"]),rank),
                          count_include_pad=values["count_include_pad"]))
            elif functional_policy==ADAPTIVE_FUNCTION_POLICY and target in (
                    torch.nn.functional.adaptive_avg_pool1d,torch.nn.functional.adaptive_avg_pool2d):
                rank=1 if target is torch.nn.functional.adaptive_avg_pool1d else 2
                values=static_arguments(node,("input","output_size"),(None,None),"Functional adaptive average pool")
                require(isinstance(values["input"],torch.fx.Node),"Adaptive pool tensor input")
                require(len(shape(values["input"]))==rank+2,"Adaptive pool N,C,spatial rank")
                emit(node,"adaptive_avg_pool",[ref(values["input"])],
                     dict(output_size=pair(static_value(values["output_size"]),rank)))
            elif target is torch.nn.functional.batch_norm:
                values=static_arguments(node,
                    ("input","running_mean","running_var","weight","bias","training","momentum","eps"),
                    (None,None,None,None,None,False,.1,1e-5),"Functional BatchNorm")
                x=values["input"]
                require(isinstance(x,torch.fx.Node) and len(shape(x)) in (2,3,4),
                        "Inference BatchNorm rank-2/3/4 input")
                require(values["training"] is False,"Only inference BatchNorm is supported")
                require(values["running_mean"] is not None and values["running_var"] is not None,
                        "Frozen BatchNorm statistics required")
                channels=[shape(x)[1]]
                emit(node,"batch_norm",[ref(x),ref(values["running_mean"]),ref(values["running_var"]),
                     *public_affine(values["weight"],values["bias"],channels)],dict(eps=values["eps"]))

            elif target is torch.nn.functional.instance_norm:
                values=static_arguments(node,
                    ("input","running_mean","running_var","weight","bias","use_input_stats","momentum","eps"),
                    (None,None,None,None,None,True,.1,1e-5),"Functional InstanceNorm")
                x=values["input"]
                require(isinstance(x,torch.fx.Node) and len(shape(x)) in (3,4),
                        "Functional InstanceNorm N,C,spatial input")
                require(type(values["use_input_stats"]) is bool,"Static InstanceNorm statistics mode")
                momentum=values["momentum"]
                require(type(momentum) in (int,float) and math.isfinite(momentum) and 0<=momentum<=1,
                        "Static InstanceNorm momentum")
                statistics=None
                if values["use_input_stats"]:
                    require(values["running_mean"] is None and values["running_var"] is None,
                            "InstanceNorm input statistics cannot update public state")
                else:
                    require(values["running_mean"] is not None and values["running_var"] is not None,
                            "Frozen InstanceNorm statistics required")
                    for key in ("running_mean","running_var"):
                        value=values[key]
                        require(not isinstance(value,torch.fx.Node) or value in public_values,
                                "Public InstanceNorm statistics required")
                    statistics=[ref(values[key]) for key in ("running_mean","running_var")]
                instance_normalization(node,x,
                    public_affine(values["weight"],values["bias"],[shape(x)[1]]),
                    statistics,values["eps"])

            elif target in (torch.nn.functional.layer_norm,torch.nn.functional.group_norm):
                layer=target is torch.nn.functional.layer_norm
                names=("input","normalized_shape" if layer else "num_groups","weight","bias","eps")
                values=static_arguments(node,names,(None,None,None,None,1e-5),"Functional normalization")
                x=values["input"]
                require(isinstance(x,torch.fx.Node),"Normalization tensor input")
                if layer:
                    dims=static_value(values["normalized_shape"])
                    dims=[dims] if type(dims) is int else list(dims) if type(dims) in (tuple,list) else []
                    attrs=dict(normalized_shape=dims,eps=values["eps"],
                               approximation=approximation("rsqrt","layer_norm"))
                else:
                    require(len(shape(x)) in (3,4),"GroupNorm N,C,spatial shape")
                    dims=[shape(x)[1]]
                    attrs=dict(groups=static_value(values["num_groups"]),eps=values["eps"],
                               approximation=approximation("rsqrt","group_norm"))
                emit(node,"layer_norm" if layer else "group_norm",
                     [ref(x),*public_affine(values["weight"],values["bias"],dims)],attrs)

            elif (selection is not None or capture_aliases) and target in (torch.relu,torch.nn.functional.relu,"relu",
                    torch.nn.functional.leaky_relu,torch.nn.functional.prelu,torch.nn.functional.gelu,
                    torch.nn.functional.silu,torch.sigmoid,torch.nn.functional.sigmoid,"sigmoid",
                    torch.tanh,torch.nn.functional.tanh,"tanh",torch.exp,"exp",
                    torch.reciprocal,"reciprocal",torch.rsqrt,"rsqrt",torch.sqrt,"sqrt"):
                if target in (torch.relu,torch.nn.functional.relu,"relu"):op="relu"
                elif target is torch.nn.functional.leaky_relu:op="leaky_relu"
                elif target is torch.nn.functional.prelu:op="prelu"
                elif target is torch.nn.functional.gelu:op="gelu"
                elif target is torch.nn.functional.silu:op="silu"
                else:op=next(n for n in ("sigmoid","tanh","exp","reciprocal","rsqrt","sqrt")
                    if target==n or target in (getattr(torch,n),getattr(torch.nn.functional,n,None)))
                if target in (torch.nn.functional.relu,torch.nn.functional.silu):
                    values=static_arguments(node,("input","inplace"),(None,False),"Functional activation")
                    require(values["inplace"] is False or capture_aliases and type(values["inplace"]) is bool,"In-place activation not yet imported")
                elif op=="leaky_relu":
                    values=static_arguments(node,("input","negative_slope","inplace"),(None,.01,False),"Functional activation")
                    require(values["inplace"] is False or capture_aliases and type(values["inplace"]) is bool,"In-place activation not yet imported")
                elif op=="prelu":
                    values=static_arguments(node,("input","weight"),(None,None),"Functional activation")
                    require(values["weight"] is not None,"PReLU public weight")
                elif op=="gelu":
                    values=static_arguments(node,("input","approximate"),(None,"none"),"Functional activation")
                    require(values["approximate"]=="none","Exact GELU source required")
                else:
                    values=static_arguments(node,("input",),(None,),"Functional activation")
                require(isinstance(values["input"],torch.fx.Node),"Functional activation tensor input")
                function="relu" if op in ("leaky_relu","prelu") else "rsqrt" if op=="sqrt" else op
                attrs=dict(approximation=approximation(function,op))
                if op=="leaky_relu":attrs["negative_slope"]=values["negative_slope"]
                operands=[ref(values["input"])]
                if op=="prelu":operands.append(ref(values["weight"]))
                emit(node,op,operands,attrs)
            elif selection is not None and target in (torch.softmax,torch.nn.functional.softmax,"softmax",
                                                      torch.log_softmax,torch.nn.functional.log_softmax,"log_softmax"):
                op="softmax" if target in (torch.softmax,torch.nn.functional.softmax,"softmax") else "logsoftmax"
                if target in (torch.nn.functional.softmax,torch.nn.functional.log_softmax):
                    values=static_arguments(node,("input","dim","_stacklevel","dtype"),(None,None,3,None),"Functional softmax")
                    require(values["_stacklevel"]==3,"Softmax stacklevel")
                else:
                    values=static_arguments(node,("input","dim","dtype"),(None,None,None),"Softmax")
                require(isinstance(values["input"],torch.fx.Node) and type(values["dim"]) is int and
                        values["dtype"] in (None,torch.float64),"Static softmax input/axis/dtype")
                second="reciprocal" if op=="softmax" else "log"
                emit(node,op,[ref(values["input"])],dict(axis=values["dim"],
                    exp=approximation("exp",op),**{second:approximation(second,op)}))

            elif target in (torch.var,"var"):
                require(len(node.args)<=4,"Variance positional arguments")
                values=static_arguments(node,("input","dim","unbiased","keepdim","correction"),
                    (None,None,True,False,None),"Variance")
                x=values["input"]
                require(isinstance(x,torch.fx.Node),"Variance tensor input")
                if "correction" in node.kwargs:
                    require("unbiased" not in node.kwargs and len(node.args)<=2,
                            "Variance correction/unbiased conflict")
                    correction=static_value(values["correction"])
                    require(type(correction) is int and correction==0,
                            "Only population variance (correction=0) is supported")
                else:
                    require(values["unbiased"] is False,
                            "Only population variance (unbiased=False) is supported")
                keep=static_value(values["keepdim"])
                require(type(keep) is bool,"Static variance keepdim")
                emit(node,"variance",[ref(x)],dict(axes=reduction_axes(values["dim"],x),keepdims=keep))

            elif target in (torch.sum,torch.mean,"sum","mean"):
                values=static_arguments(node,("input","dim","keepdim","dtype"),(None,None,False,None),"Reduction")
                x,dim,keep,dtype=(values[k] for k in ("input","dim","keepdim","dtype"))
                require(isinstance(x,torch.fx.Node),"Reduction tensor input")
                require(dtype is None or dtype is torch.float64,"Reduction dtype conversion is not supported")
                require(type(keep) is bool,"Static reduction keepdim")
                if dim is None or isinstance(dim,(tuple,list)) and len(dim)==0:
                    axes=list(range(len(shape(x))))
                else:
                    axes=[dim] if type(dim) is int else list(dim) if isinstance(dim,(tuple,list)) else []
                    require(axes and all(type(i) is int for i in axes),"Static reduction axes")
                emit(node,"sum" if target in (torch.sum,"sum") else "mean",[ref(x)],
                     dict(axes=axes,keepdims=keep))
            elif target in (torch.cat,torch.concat,torch.concatenate,torch.stack):
                values=static_arguments(node,("tensors","dim","out"),(None,0,None),"Tensor join")
                tensors,dim,out=(values[k] for k in ("tensors","dim","out"))
                require(out is None,"Tensor join out mutation is not supported")
                require(isinstance(tensors,(tuple,list)) and 1<=len(tensors)<=8 and
                        all(isinstance(x,torch.fx.Node) for x in tensors),"Static tensor sequence")
                require(type(dim) is int,"Static tensor join axis")
                if target is torch.stack and all(not shape(x) for x in tensors):
                    # Graph scalars use shape [1]. Remove only their synthetic
                    # singleton axis so torch.stack keeps its real rank-1 shape.
                    emit(node,"stack",[ref(x) for x in tensors],dict(axis=dim),suffix="_scalar_stack")
                    emit(node,"reshape",[ref(node)],dict(shape=list(shape(node))))
                else:
                    emit(node,"stack" if target is torch.stack else "concat",[ref(x) for x in tensors],
                         dict(axis=dim))

            elif target in (torch.maximum,torch.minimum):
                require(not node.kwargs and len(node.args)==2,"Maximum/minimum positional arguments")
                emit(node,"maximum" if target is torch.maximum else "minimum",
                     [ref(x) for x in node.args],dict(approximation=approximation("abs","maximum" if target is torch.maximum else "minimum")))
            elif target in (torch.amax,torch.amin,"amax","amin"):
                require(len(node.args)<=3 and not(set(node.kwargs)-{"dim","keepdim"}),"Reduction arguments")
                require(not(len(node.args)>1 and "dim" in node.kwargs) and
                        not(len(node.args)>2 and "keepdim" in node.kwargs),"Duplicate reduction arguments")
                dim=node.args[1] if len(node.args)>1 else node.kwargs.get("dim",())
                axes=list(range(len(shape(node.args[0])))) if dim is None or dim==() else ([dim] if type(dim) is int else list(dim))
                keep=node.args[2] if len(node.args)>2 else node.kwargs.get("keepdim",False)
                emit(node,"max" if target in (torch.amax,"amax") else "min",[ref(node.args[0])],
                     dict(axes=axes,keepdims=keep,approximation=approximation("abs","max" if target in (torch.amax,"amax") else "min")))
            elif target is torch.nn.functional.pad:
                require(len(node.args)<=4 and not(set(node.kwargs)-{"pad","mode","value"}),"Padding arguments")
                for i,key in enumerate(("pad","mode","value"),1):
                    require(not(len(node.args)>i and key in node.kwargs),"Duplicate padding argument")
                pad=node.args[1] if len(node.args)>1 else node.kwargs.get("pad")
                mode=node.args[2] if len(node.args)>2 else node.kwargs.get("mode","constant")
                value=node.args[3] if len(node.args)>3 else node.kwargs.get("value",0)
                emit(node,"padding",[ref(node.args[0])],padding_attrs(node,pad,mode,value))
            elif target in ("expand",torch.broadcast_to):
                require(not node.kwargs,"Broadcast positional arguments required")
                emit(node,"broadcast",[ref(node.args[0])],dict(shape=list(shape(node))))
            elif target in ("index_select",torch.index_select):
                require(not node.kwargs and len(node.args)==3,"Index selection positional arguments")
                idx=node.args[2]
                require(isinstance(idx,torch.fx.Node) and idx.op=="get_attr","Public constant indices required")
                tensor=traced
                for segment in idx.target.split("."):tensor=getattr(tensor,segment)
                require(isinstance(tensor,torch.Tensor) and tensor.dtype in (torch.int32,torch.int64) and tensor.ndim==1,
                        "Public index vector type")
                emit(node,"public_gather",[ref(node.args[0])],dict(axis=node.args[1],indices=tensor.tolist()))
            elif target in ("reshape","view","flatten",torch.reshape,torch.flatten,"squeeze","unsqueeze",torch.squeeze,torch.unsqueeze):
                for value in (*node.args[1:],*node.kwargs.values()):static_value(value)
                emit(node,"reshape",[ref(node.args[0])],dict(shape=list(shape(node))))
            elif target=="contiguous":
                require(len(node.args)==1 and (not node.kwargs or
                    node.kwargs==dict(memory_format=torch.contiguous_format)),"C-order contiguous only")
                emit(node,"dropout",[ref(node.args[0])],dict(training=False))
            elif target in (operator.neg,torch.neg,torch.negative,"neg"):
                require(not node.kwargs and len(node.args)==1,"Negation arguments")
                emit(node,"negate",[ref(node.args[0])])
            elif target in ("square",torch.square):emit(node,"square",[ref(node.args[0])])
            elif target in ("transpose",torch.transpose):
                emit(node,"transpose",[ref(node.args[0])],dict(dim0=node.args[1],dim1=node.args[2]))
            elif target in ("permute",torch.permute):
                dims=node.args[1] if len(node.args)==2 else node.args[1:]
                emit(node,"permute",[ref(node.args[0])],dict(dims=list(dims)))
            elif target in (torch.nn.functional.linear,):
                values=static_arguments(node,("input","weight","bias"),(None,None,None),"Functional linear")
                require(values["input"] is not None and values["weight"] is not None,"Linear operands")
                emit(node,"linear",[ref(values[name]) for name in ("input","weight","bias") if values[name] is not None])
            else:raise ValueError("Unsupported PyTorch call: "+str(target))
        elif node.op=="output":
            if preserve_output_structure:
                values,structure=output_tree(node.args[0],lambda v:isinstance(v,torch.fx.Node) and "tensor_meta" in v.meta,outputs,
                    scalar_tensor=(lambda v:not shape(v)) if capture_policy in (POLICY,INDEX_POLICY) else None)
            else:
                values=node.args[0];values=list(values) if isinstance(values,(tuple,list)) else [values]
                require(len(values)==len(outputs) and all(isinstance(v,torch.fx.Node) for v in values),"Named tensor outputs")
            scalar_outputs=any(not shape(v) for v in values)
            graph["outputs"]=[dict(name=n,value=ref(v)) for n,v in zip(outputs,values)]
        else:raise ValueError("Unsupported FX node")
    require(next(input_iter,None) is None,"Extra declared input")
    alias_metadata=None
    if capture_aliases:
        graph,lowered,binding,alias_metadata=normalize_emissions(graph,alias_origins,
            resource_profile=resource_profile,constant_dtypes=constant_dtypes,
            state_constant_sources=state_constant_sources)
        constant_dtypes=alias_metadata["constant_dtypes"]
        state_constant_sources=alias_metadata["state_constant_sources"]
    else:
        lowered,binding=decompose(graph,resource_profile=resource_profile)
    require(not scalar_outputs or capture_policy in (POLICY,INDEX_POLICY),"Scalar outputs require reshape(1)")
    binding["pytorch_capture"]=dict(torch_version=torch.__version__,trusted_code=True,training=False,
                                     graph_sha256=digest(graph))
    if preserve_output_structure:
        binding["pytorch_capture"].update(dtype_policy=dtype_policy,input_dtype="float64",
            source_state_dtypes=source_dtypes,constant_dtypes=constant_dtypes,output_structure=structure,
            source_precision_execution_compared=False)
    if capture_policy in (POLICY,INDEX_POLICY):
        binding["pytorch_capture"]["static_capture"]=policy_record(inputs,resource_profile,policy=capture_policy)
    if capture_policy==INDEX_POLICY:
        from pytorch_static_index import VERSION as INDEX_VERSION
        binding["pytorch_capture"]["index_capture"]=dict(version=INDEX_VERSION,graph_sha256=digest(graph),entries=index_records)
    if matrix_policy==STATIC_MATRIX:
        binding["pytorch_capture"]["matrix_capture"]=dict(version=STATIC_MATRIX,graph_sha256=digest(graph),entries=matrix_records)
    if selection is not None:
        binding["pytorch_capture"]["approximation_selection"]=selection.finish()
    if record_state_sources:
        require(preserve_output_structure,"State source records require static output metadata")
        binding["pytorch_capture"]["state_constant_sources"]=state_constant_sources
    if alias_metadata is not None:
        alias_metadata["source_capture_policy"]=record
        alias_metadata.pop("binding",None);alias_metadata["binding"]=digest(alias_metadata)
        binding["pytorch_capture"]["ordered_alias_capture"]=alias_metadata
    binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    return graph,lowered,binding

def functional_input_rules():
    """Declare the functional input subset; execution evidence is separate."""
    return dict(version="poseidon-pytorch-functional-input-v2",
        conv1d=dict(calls=["torch.nn.functional.conv1d"],ranks=[2,3],weights="public",
                    padding="explicit integers"),
        conv2d=dict(calls=["torch.nn.functional.conv2d"],ranks=[3,4],weights="public",
                    padding="explicit integers"),
        avg_pool1d=dict(calls=["torch.nn.functional.avg_pool1d"],ranks=[2,3],
                        ceil_mode=False,divisor_override=None),
        avg_pool2d=dict(calls=["torch.nn.functional.avg_pool2d"],ranks=[3,4],
                        ceil_mode=False,divisor_override=None),
        batch_norm=dict(calls=["torch.nn.functional.batch_norm"],ranks=[2,3,4],
                        training=False,statistics="public fixed"),
        matrix_normalization=dict(public_rank=2,cipher_ranks=[2,3],
            right_public="linear with transposed public matrix",
            left_public="transpose, linear, transpose"),
        tensor_T=dict(calls=["Tensor.T"],semantics="reverse all tensor axes"),
        module_normalization=dict(affine_false=["nn.LayerNorm","nn.GroupNorm"],
                                  default_gain=1,default_bias=0),
        layer_norm=dict(calls=["torch.nn.functional.layer_norm"],
            normalized_shape="positive static suffix", affine="public gain/bias or unit/zero defaults",
            approximation="explicit rsqrt", epsilon="0 < eps <= 1"),
        group_norm=dict(calls=["torch.nn.functional.group_norm"], ranks=[3,4],
            groups="positive static divisor of channels", affine="public gain/bias or unit/zero defaults",
            approximation="explicit rsqrt", epsilon="0 < eps <= 1"),
        tensor_split=dict(version="poseidon-pytorch-static-split-v1",
            calls=["torch.split","Tensor.split"],policy="signature-static-index-v1",
            sections="positive static integer size or explicit positive sections",
            axes="static valid axis; rank determined by resource profile",
            output="tuple of tensors",lowering="existing split for 2-4 ports; otherwise existing slice",
            limits="existing original-node, tensor and output budgets"),
        instance_norm=dict(version="poseidon-pytorch-instance-normalization-v1",
            calls=["nn.InstanceNorm1d","nn.InstanceNorm2d","torch.nn.functional.instance_norm"],
            module_ranks=dict(one_dimensional=[2,3],two_dimensional=[3,4]),
            functional_ranks=[3,4],affine="public gain/bias or unit/zero defaults",
            input_statistics="explicit rsqrt; no running-statistic mutation",
            frozen_statistics="public running mean/variance; exact affine inference",
            epsilon="0 < eps <= 1",declared_module_channels="must match signature"),
        variance=dict(calls=["torch.var","Tensor.var"], axes="distinct static axes or all axes",
            correction=0, legacy_unbiased=False, keepdim="static bool",
            reference="population variance"))
