"""Agent staging copy of the pinned signature tracer; explicit public tensor divisor extension only."""
"""Declared-shape FX capture; never use tensor data to resolve Python control flow."""
import math
from benchmark_graph import require, digest
from model_resources import resolve

LEGACY_POLICY = "symbolic-v1"
POLICY = "signature-static-v1"
INDEX_POLICY = "signature-static-index-v1"
ARITHMETIC_POLICY = "signature-static-arithmetic-shape-v1"
OPERATOR_POLICY = "signature-static-operator-shape-v1"
COMMON_POLICY = "signature-static-common-call-shape-v1"
ALIAS_POLICY = "signature-static-ordered-alias-staging-v1"

def policy_record(inputs, resource_profile, *, policy=POLICY):
    limits = resolve(resource_profile)
    require(policy in (POLICY,INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY), "Static capture policy")
    record=dict(version=policy, input_signature_sha256=digest(inputs),
                max_fx_compute_nodes=limits.original_nodes,
                shape_sources="declared_signature_and_registered_meta_rules",
                scalar_outputs="rank_zero_tree_v1")
    if policy in (INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY):
        from pytorch_static_index import VERSION
        record["indexing_policy"]=VERSION
    if policy in (COMMON_POLICY,ALIAS_POLICY):
        record["common_call_rules"]="poseidon-fx-common-calls-v1"
        record["tensor_metadata_record"]="registered-signature-shapes-v1"
    return record

def trace(module, inputs, *, resource_profile, policy=POLICY, fused_matrix_policy=None, mha_source_policy=None, partition_source_policy=None, einsum_source_policy=None, exact_arithmetic_policy=None, public_tensor_division_policy=None):
    """Use metadata-only rules for shape queries; ordinary tensors remain proxies."""
    import operator
    import torch
    from torch.fx import Tracer, Proxy, GraphModule
    from torch.nn import functional as F
    limits = resolve(resource_profile)
    require(policy in (POLICY,INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY), "Static capture policy")
    require(fused_matrix_policy is None or
            fused_matrix_policy == "poseidon-fx-fused-matrix-draft-v1" and policy == ALIAS_POLICY,
            "Explicit fused matrix metadata policy")
    require(mha_source_policy is None or mha_source_policy=="poseidon-mha-ordered-source-v1" and
            policy==ALIAS_POLICY,"Explicit ordered MHA metadata policy")
    require(partition_source_policy is None or partition_source_policy in ("poseidon-pytorch-static-partition-v1","poseidon-pytorch-static-partition-v2") and
            policy==ALIAS_POLICY,"Explicit ordered partition metadata policy")
    require(einsum_source_policy is None or einsum_source_policy in ("poseidon-pytorch-static-einsum-v1","poseidon-pytorch-static-einsum-v2") and
            policy==ALIAS_POLICY,"Explicit ordered einsum metadata policy")
    require(exact_arithmetic_policy is None or
            exact_arithmetic_policy=="poseidon-pytorch-exact-arithmetic-v1" and policy==ALIAS_POLICY,
            "Explicit exact arithmetic metadata policy")
    require(public_tensor_division_policy is None or
            public_tensor_division_policy=="poseidon-public-tensor-division-source-draft-v3" and
            exact_arithmetic_policy=="poseidon-pytorch-exact-arithmetic-v1" and policy==ALIAS_POLICY,
            "Explicit draft public tensor division metadata policy")
    unknown = object()

    class SignatureProxy(Proxy):
        @property
        def shape(self):
            value = self.tracer.metadata.get(self.node, unknown)
            require(isinstance(value, torch.Tensor), "Static shape unavailable for this operation")
            return torch.Size(value.shape)
        @property
        def ndim(self):
            return len(self.shape)
        def size(self, dim=None):
            dims = self.shape
            if dim is None:
                return dims
            require(type(dim) is int and -len(dims) <= dim < len(dims), "Static size axis")
            return dims[dim]
        def dim(self):
            return self.ndim
        def numel(self):
            return math.prod(self.shape)
        def __len__(self):
            dims = self.shape
            require(bool(dims), "Scalar tensor has no len")
            return dims[0]

    class SignatureTracer(Tracer):
        def __init__(self):
            super().__init__()
            if policy==ALIAS_POLICY:self.proxy_buffer_attributes=True
            self.metadata = {}
            self.placeholder_count = 0
            self.compute_count = 0
        def proxy(self, node):
            return SignatureProxy(node, self)
        def meta_tensor(self, tensor):
            return torch.empty(tuple(tensor.shape), device="meta", dtype=tensor.dtype)
        def value(self, item):
            if isinstance(item, Proxy):
                value = self.metadata.get(item.node, unknown)
                require(value is not unknown, "Static shape unavailable for operand")
                return value
            if isinstance(item, torch.Tensor):
                return self.meta_tensor(item)
            if type(item) in (list, tuple):
                return type(item)(self.value(v) for v in item)
            if type(item) is dict:
                return {k:self.value(v) for k,v in item.items()}
            return item
        def attribute(self, target):
            value = self.root
            for part in target.split("."):
                if isinstance(value,torch.nn.Module):
                    tables=(value._parameters,value._buffers,value._modules,value.__dict__)
                    value=next((table[part] for table in tables if part in table),unknown)
                    require(value is not unknown, "Tensor metadata attribute unavailable")
                else:
                    value = getattr(value, part)
            return value
        def meta_result(self, kind, target, args, kwargs):
            if kind == "placeholder":
                require(self.placeholder_count < len(inputs), "Input declaration count")
                dims = inputs[self.placeholder_count]["shape"]
                self.placeholder_count += 1
                return torch.empty(dims, device="meta", dtype=torch.float64)
            if kind == "get_attr":
                value = self.attribute(target)
                return self.meta_tensor(value) if isinstance(value, torch.Tensor) else unknown
            # Only the registered shape operations run on metadata tensors.
            # No user callable or module forward is executed for this inference.
            functions = {operator.add, operator.sub, operator.mul, operator.neg, operator.matmul,
                operator.getitem, torch.add, torch.sub, torch.mul, torch.neg, torch.square,
                torch.matmul, torch.mm, torch.bmm, torch.dot, torch.sum, torch.mean, torch.var,
                torch.cat, torch.concat, torch.concatenate, torch.stack, torch.reshape, torch.flatten,
                torch.squeeze, torch.unsqueeze, torch.transpose, torch.permute,
                torch.broadcast_to, torch.index_select, torch.maximum, torch.minimum,
                torch.amax, torch.amin, F.linear, F.conv1d, F.conv2d,
                F.avg_pool1d, F.avg_pool2d, F.batch_norm, F.layer_norm, F.group_norm, F.pad,
                torch.relu, F.relu, F.leaky_relu, F.prelu, F.gelu, F.silu, torch.sigmoid,
                F.sigmoid, torch.tanh, F.tanh, torch.exp, torch.reciprocal, torch.rsqrt, torch.sqrt,
                torch.softmax, F.softmax, torch.log_softmax, F.log_softmax}
            methods = {"reshape","view","flatten","squeeze","unsqueeze","transpose","permute",
                       "contiguous","square","neg","sum","mean","var","matmul","mm","bmm",
                       "dot","expand","index_select","amax","amin","relu","sigmoid","tanh",
                       "exp","reciprocal","rsqrt","sqrt","softmax","log_softmax"}
            if fused_matrix_policy is not None:
                functions |= {torch.addmm,torch.baddbmm,torch.addbmm}
                methods |= {"addmm","baddbmm","addbmm"}
            if policy in (ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY):
                functions|={torch.subtract,torch.multiply}
                methods|={"add","sub","subtract","mul","multiply"}
            if policy in (OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY):
                functions|={F.instance_norm,F.adaptive_avg_pool1d,F.adaptive_avg_pool2d}
            if policy in (INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY):
                functions|={torch.select,torch.narrow}
                methods|={"select","narrow"}
                if kind=="call_function" and target is operator.getitem and isinstance(self.value(args[0]),torch.Tensor):
                    from pytorch_static_index import plan
                    def key(item):
                        if isinstance(item,Proxy):
                            require(item.node.op=="get_attr","Public static tensor index required")
                            value=self.attribute(item.node.target)
                            require(isinstance(value,torch.Tensor) and value.dtype in (torch.int32,torch.int64)
                                    and value.ndim==1,"Public index vector type")
                            return value.tolist()
                        if type(item) in (tuple,list):return type(item)(key(v) for v in item)
                        return item
                    selected=plan(tuple(self.value(args[0]).shape),key(args[1]),resource_profile=resource_profile)
                    return torch.empty(selected["result_shape"],device="meta",dtype=self.value(args[0]).dtype)
            if policy in (COMMON_POLICY,ALIAS_POLICY) and kind=="call_function":
                from pytorch_arithmetic_normalization import common_arguments,common_scalar,sdpa_shape
                dropout={F.dropout:0,F.dropout1d:1,F.dropout2d:2,F.dropout3d:3,
                         F.alpha_dropout:0,F.feature_alpha_dropout:0}
                if target in dropout:
                    values=common_arguments(args,kwargs,("input","p","training","inplace"),
                                            {"p":0.5,"training":True,"inplace":False})
                    tensor=self.value(values["input"])
                    require(values["training"] is False and values["inplace"] is False and
                            0<=common_scalar(values["p"])<=1,"Inference Dropout metadata")
                    if dropout[target]:
                        require(tensor.ndim in (dropout[target]+1,dropout[target]+2),"Spatial Dropout input rank")
                    return tensor
                if target is F.scaled_dot_product_attention:
                    values=common_arguments(args,kwargs,("query","key","value","attn_mask","dropout_p","is_causal"),
                                            {"attn_mask":None,"dropout_p":0.,"is_causal":False})
                    require(common_scalar(values["dropout_p"])==0 and type(values["is_causal"]) is bool,
                            "Static inference SDPA metadata")
                    require(not values["is_causal"] or values["attn_mask"] is None,"SDPA mask and causal flags")
                    tensors=[self.value(values[name]) for name in ("query","key","value")]
                    require(all(isinstance(v,torch.Tensor) and v.dtype is torch.float64 for v in tensors),
                            "SDPA float64 tensor metadata")
                    scores,output=sdpa_shape(*(list(v.shape) for v in tensors),resource_profile=resource_profile)
                    if values["attn_mask"] is not None:
                        raw=values["attn_mask"]
                        public_proxy=isinstance(raw,Proxy) and raw.node.op=="get_attr"
                        public_buffer=isinstance(raw,torch.Tensor) and any(
                            raw is value for _,value in self.root.named_buffers())
                        public_parameter=isinstance(raw,torch.Tensor) and any(
                            raw is value for _,value in self.root.named_parameters())
                        require(public_proxy or public_buffer or public_parameter,
                                "Public SDPA mask metadata")
                        mask=self.value(raw)
                        require(list(torch.broadcast_shapes(list(mask.shape),scores))==scores,
                                "SDPA mask broadcast")
                    return torch.empty(output,device="meta",dtype=torch.float64)
            if policy==ALIAS_POLICY:
                functions|={torch.relu_}
                methods|={"add_","sub_","subtract_","mul_","multiply_","copy_",
                          "relu_","square_","clone","detach"}
                if kind=="call_method" and target in ("add_","sub_","subtract_","mul_","multiply_","copy_","relu_","square_"):
                    values=self.value(args);options=self.value(kwargs)
                    # Metadata tensors carry only signature shape/dtype. Validate
                    # the fixed Torch signature and broadcast without user data.
                    result=getattr(values[0],target)(*values[1:],**options)
                    require(tuple(result.shape)==tuple(values[0].shape),"In-place metadata shape")
                    return result
            if kind == "call_function" and target is getattr:
                require(len(args) == 2 and args[1] == "T", "Unregistered metadata attribute")
                return self.value(args[0]).T
            if kind == "call_function" and target in functions:
                return target(*self.value(args), **self.value(kwargs))
            if kind == "call_method" and target in methods:
                values = self.value(args)
                return getattr(values[0], target)(*values[1:], **self.value(kwargs))
            if kind == "call_module":
                child = self.root.get_submodule(target)
                if type(child) is torch.nn.MultiheadAttention and mha_source_policy is not None:
                    from pytorch_mha_source import metadata
                    return metadata(child,args,kwargs,self.value,resource_profile=resource_profile)
                x = self.value(args[0] if args else kwargs["input"])
                if type(child) is torch.nn.Linear:
                    return F.linear(x, self.meta_tensor(child._parameters["weight"]),
                                    None if child._parameters["bias"] is None else self.meta_tensor(child._parameters["bias"]))
                if type(child) in (torch.nn.Conv1d,torch.nn.Conv2d):
                    fn = F.conv1d if type(child) is torch.nn.Conv1d else F.conv2d
                    return fn(x,self.meta_tensor(child._parameters["weight"]),
                              None if child._parameters["bias"] is None else self.meta_tensor(child._parameters["bias"]),
                              child.stride,child.padding,child.dilation,child.groups)
                if policy in (OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY):
                    if type(child) in (torch.nn.InstanceNorm1d,torch.nn.InstanceNorm2d):
                        return x
                    if type(child) in (torch.nn.AdaptiveAvgPool1d,torch.nn.AdaptiveAvgPool2d):
                        fn=F.adaptive_avg_pool1d if type(child) is torch.nn.AdaptiveAvgPool1d else F.adaptive_avg_pool2d
                        return fn(x,child.output_size)
                    if type(child) in (torch.nn.ConstantPad1d,torch.nn.ConstantPad2d,torch.nn.ZeroPad2d,
                                      torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d,
                                      torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d):
                        mode=("reflect" if type(child) in (torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d)
                              else "replicate" if type(child) in (torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d)
                              else "constant")
                        return F.pad(x,child.padding,mode,getattr(child,"value",0.))
                if policy in (COMMON_POLICY,ALIAS_POLICY):
                    drops={torch.nn.Dropout:0,torch.nn.Dropout1d:1,torch.nn.Dropout2d:2,torch.nn.Dropout3d:3,
                           torch.nn.AlphaDropout:0,torch.nn.FeatureAlphaDropout:0}
                    if type(child) in drops:
                        from pytorch_arithmetic_normalization import common_scalar
                        require(child.training is False and child.inplace is False and 0<=common_scalar(child.p)<=1,
                                "Inference Dropout module metadata")
                        if drops[type(child)]:
                            require(x.ndim in (drops[type(child)]+1,drops[type(child)]+2),"Spatial Dropout input rank")
                        return x
                if type(child) is torch.nn.Flatten:
                    return x.flatten(child.start_dim,child.end_dim)
                if type(child) in (torch.nn.Identity,torch.nn.Dropout,torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,torch.nn.LayerNorm,torch.nn.GroupNorm,torch.nn.ReLU,
                    torch.nn.LeakyReLU,torch.nn.PReLU,torch.nn.GELU,torch.nn.SiLU,
                    torch.nn.Sigmoid,torch.nn.Tanh,torch.nn.LogSoftmax,torch.nn.Softmax):
                    return x
                if type(child) in (torch.nn.AvgPool1d,torch.nn.AvgPool2d):
                    fn = F.avg_pool1d if type(child) is torch.nn.AvgPool1d else F.avg_pool2d
                    return fn(x,child.kernel_size,child.stride,child.padding,
                              child.ceil_mode,child.count_include_pad)
            return unknown
        def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None,
                         proxy_factory_fn=None):
            if kind in ("call_module","call_function","call_method"):
                self.compute_count += 1
                require(self.compute_count <= limits.original_nodes, "Static FX expansion budget")
            split_call=policy in (INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY,ALIAS_POLICY) and ((kind=="call_function" and target is torch.split)
                                                or (kind=="call_method" and target=="split"))
            mha_call=(mha_source_policy is not None and kind=="call_module" and
                      type(self.root.get_submodule(target)) is torch.nn.MultiheadAttention)
            partition_call=partition_source_policy is not None and (
                kind=="call_function"and target in(torch.chunk,torch.unbind)or
                kind=="call_method"and target in("chunk","unbind"))
            einsum_call=einsum_source_policy is not None and kind=="call_function" and target is torch.einsum
            exact_plan=None
            if exact_arithmetic_policy is not None:
                from pytorch_exact_arithmetic import classification,target_name,arguments as exact_arguments,plan as exact_select
                signature=classification(kind,target_name(kind,target))
                if signature is not None:
                    values=exact_arguments(args,kwargs,signature)
                    exact_tensor=self.value(values["input"])
                    require(isinstance(exact_tensor,torch.Tensor) and exact_tensor.dtype is torch.float64,
                            "Exact arithmetic float64 tensor metadata")
                    parameter=values["parameter"]
                    if isinstance(parameter,Proxy) and public_tensor_division_policy is not None:
                        require(signature[0]=="divide" and parameter.node.op=="get_attr",
                                "Only a direct declared public tensor divisor is registered")
                        public=self.attribute(parameter.node.target)
                        require(isinstance(public,torch.Tensor) and public.device.type=="cpu" and public.dtype is torch.float64,
                                "Public tensor divisor dtype and device")
                        from public_tensor_division import plan as public_select
                        payload=public.detach().contiguous().numpy().astype("<f8",copy=False).tobytes()
                        selected,_=public_select(list(exact_tensor.shape),list(public.shape),payload,
                                                inplace=signature[1],resource_profile=resource_profile)
                        exact_plan=dict(shape=selected["result_shape"],steps=[dict(operation="multiply")])
                    else:
                        if isinstance(parameter,Proxy):
                            require(signature[0]=="divide" and parameter.node.op=="get_attr",
                                    "Only a direct public scalar tensor divisor is registered")
                            public=self.attribute(parameter.node.target)
                            require(isinstance(public,torch.Tensor) and public.ndim==0 and public.dtype is torch.float64,
                                    "Public scalar divisor dtype and shape")
                            parameter=float(public.detach().cpu())
                        exact_plan=exact_select(list(exact_tensor.shape),signature[0],parameter,
                                                inplace=signature[1],resource_profile=resource_profile)
                    require(self.compute_count+len(exact_plan["steps"])-1<=limits.original_nodes,
                            "Static FX expansion budget")
            einsum_plan=None
            if einsum_call:
                from pytorch_einsum_normalization import arguments as einsum_arguments,plan as einsum_select
                if einsum_source_policy=="poseidon-pytorch-static-einsum-v2":
                    from pytorch_einsum_extension import plan as einsum_select
                equation,operands=einsum_arguments(args,kwargs)
                tensors=[self.value(v)for v in operands]
                require(all(isinstance(v,torch.Tensor)and v.dtype is torch.float64 for v in tensors),
                        "Static einsum tensor metadata")
                einsum_plan=einsum_select(equation,[list(v.shape)for v in tensors],resource_profile=resource_profile)
                require(self.compute_count+len(einsum_plan["steps"])<=limits.original_nodes,
                        "Static FX expansion budget")
            partition_plan=None
            if partition_call:
                from pytorch_partition_normalization import arguments as partition_arguments,plan as partition_select
                operation=("chunk"if target is torch.chunk else "unbind")if kind=="call_function"else target
                values=partition_arguments(args,kwargs,operation,method=kind=="call_method")
                source=self.value(values["input"])
                require(isinstance(source,torch.Tensor),"Static partition tensor metadata")
                partition_plan=partition_select(list(source.shape),operation,axis=values["axis"],chunks=values["chunks"],
                                                resource_profile=resource_profile,source_policy=partition_source_policy)
                require(self.compute_count+len(partition_plan["ports"])<=limits.original_nodes,
                        "Static FX expansion budget")
            split_plan=None
            if split_call:
                from pytorch_split_plan import arguments,plan
                values=arguments(args,kwargs,method=kind=="call_method")
                source=self.value(values["input"])
                require(isinstance(source,torch.Tensor),"Static split tensor metadata")
                split_plan=plan(list(source.shape),values["sections"],values["axis"],
                                resource_profile=resource_profile)
                require(self.compute_count+len(split_plan["sections"])<=limits.original_nodes,
                        "Static FX expansion budget")
            result = super().create_proxy(kind,target,args,kwargs,name,type_expr,proxy_factory_fn)
            try:
                value = (torch.empty(exact_plan["shape"],device="meta",dtype=torch.float64) if exact_plan is not None else
                         torch.empty(einsum_plan["result_shape"],device="meta",dtype=torch.float64) if einsum_call else
                         tuple(torch.empty(v["result_shape"],device="meta",dtype=source.dtype)
                               for v in partition_plan["ports"]) if partition_call else
                         tuple(torch.empty(dims,device="meta",dtype=source.dtype)
                               for dims in split_plan["result_shapes"]) if split_call else
                         self.meta_result(kind,target,args,kwargs))
            except (ValueError,TypeError,RuntimeError,KeyError,IndexError):
                if kind == "placeholder" or mha_call:
                    raise
                value = unknown
            if isinstance(value,torch.Tensor):
                require(value.ndim <= limits.rank and value.numel() <= (limits.constant_elements if kind=="get_attr" else limits.tensor_elements),
                        "Static FX tensor budget")
            if mha_call:
                require(type(value) is tuple and len(value)==2 and isinstance(value[0],torch.Tensor) and
                        (value[1] is None or isinstance(value[1],torch.Tensor)),"Static MHA tuple metadata")
                require(all(v is None or v.numel()<=limits.tensor_elements for v in value),"MHA tuple tensor budget")
            self.metadata[result.node] = value
            if mha_call:
                require(self.compute_count+sum(v is not None for v in value)<=limits.original_nodes,
                        "MHA tuple proxy budget")
                return tuple(None if v is None else self.create_proxy(
                    "call_function",operator.getitem,(result,i),{}) for i,v in enumerate(value))
            if partition_call:
                return tuple(self.create_proxy("call_function",operator.getitem,(result,i),{})
                             for i in range(len(partition_plan["ports"])))
            if split_call:
                # A real Python tuple contains tensor proxies; iteration is over
                # public, signature-derived ports rather than encrypted values.
                return tuple(self.create_proxy("call_function",operator.getitem,(result,i),{})
                             for i in range(len(split_plan["sections"])))
            return result

    tracer = SignatureTracer()
    graph = tracer.trace(module)
    require(tracer.placeholder_count == len(inputs), "Input declaration count")
    if policy in (COMMON_POLICY,ALIAS_POLICY):
        for node in graph.nodes:
            node.meta.pop("poseidon_static_tensor",None)
            tensor=tracer.metadata.get(node,unknown)
            if isinstance(tensor,torch.Tensor):
                node.meta["poseidon_static_tensor"]=dict(policy=policy,shape=list(tensor.shape),
                    dtype=str(tensor.dtype).removeprefix("torch."))
    result=GraphModule(module,graph)
    if policy==ALIAS_POLICY:
        result.meta={"ordered_alias_capture_policy":policy_record(inputs,resource_profile,policy=policy)}
    if fused_matrix_policy is not None:
        result.meta["fused_matrix_policy"] = fused_matrix_policy
    if mha_source_policy is not None:
        result.meta["mha_source_policy"]=mha_source_policy
    if partition_source_policy is not None:result.meta["partition_source_policy"]=partition_source_policy
    if einsum_source_policy is not None:result.meta["einsum_source_policy"]=einsum_source_policy
    if exact_arithmetic_policy is not None:result.meta["exact_arithmetic_policy"]=exact_arithmetic_policy
    if public_tensor_division_policy is not None:result.meta["public_tensor_division_policy"]=public_tensor_division_policy
    return result
