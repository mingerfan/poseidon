"""Declared-shape FX capture; never use tensor data to resolve Python control flow."""
import math
from benchmark_graph import require, digest
from model_resources import resolve

LEGACY_POLICY = "symbolic-v1"
POLICY = "signature-static-v1"
INDEX_POLICY = "signature-static-index-v1"
ARITHMETIC_POLICY = "signature-static-operator-shape-prototype-v1"

def policy_record(inputs, resource_profile, *, policy=POLICY):
    limits = resolve(resource_profile)
    require(policy in (POLICY,INDEX_POLICY,ARITHMETIC_POLICY), "Static capture policy")
    record=dict(version=policy, input_signature_sha256=digest(inputs),
                max_fx_compute_nodes=limits.original_nodes,
                shape_sources="declared_signature_and_registered_meta_rules",
                scalar_outputs="rank_zero_tree_v1")
    if policy in (INDEX_POLICY,ARITHMETIC_POLICY):
        from pytorch_static_index import VERSION
        record["indexing_policy"]=VERSION
    return record

def trace(module, inputs, *, resource_profile, policy=POLICY):
    """Use metadata-only rules for shape queries; ordinary tensors remain proxies."""
    import operator
    import torch
    from torch.fx import Tracer, Proxy, GraphModule
    from torch.nn import functional as F
    limits = resolve(resource_profile)
    require(policy in (POLICY,INDEX_POLICY,ARITHMETIC_POLICY), "Static capture policy")
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
            if policy==ARITHMETIC_POLICY:
                functions|={torch.subtract,torch.multiply,F.instance_norm,
                            F.adaptive_avg_pool1d,F.adaptive_avg_pool2d}
                methods|={"add","sub","subtract","mul","multiply"}
            if policy in (INDEX_POLICY,ARITHMETIC_POLICY):
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
                x = self.value(args[0] if args else kwargs["input"])
                if type(child) is torch.nn.Linear:
                    return F.linear(x, self.meta_tensor(child._parameters["weight"]),
                                    None if child._parameters["bias"] is None else self.meta_tensor(child._parameters["bias"]))
                if type(child) in (torch.nn.Conv1d,torch.nn.Conv2d):
                    fn = F.conv1d if type(child) is torch.nn.Conv1d else F.conv2d
                    return fn(x,self.meta_tensor(child._parameters["weight"]),
                              None if child._parameters["bias"] is None else self.meta_tensor(child._parameters["bias"]),
                              child.stride,child.padding,child.dilation,child.groups)
                if policy==ARITHMETIC_POLICY:
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
            split_call=policy in (INDEX_POLICY,ARITHMETIC_POLICY) and ((kind=="call_function" and target is torch.split)
                                                or (kind=="call_method" and target=="split"))
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
                value = (tuple(torch.empty(dims,device="meta",dtype=source.dtype)
                               for dims in split_plan["result_shapes"]) if split_call else
                         self.meta_result(kind,target,args,kwargs))
            except (ValueError,TypeError,RuntimeError,KeyError,IndexError):
                if kind == "placeholder":
                    raise
                value = unknown
            if isinstance(value,torch.Tensor):
                require(value.ndim <= limits.rank and value.numel() <= (limits.constant_elements if kind=="get_attr" else limits.tensor_elements),
                        "Static FX tensor budget")
            self.metadata[result.node] = value
            if split_call:
                # A real Python tuple contains tensor proxies; iteration is over
                # public, signature-derived ports rather than encrypted values.
                return tuple(self.create_proxy("call_function",operator.getitem,(result,i),{})
                             for i in range(len(split_plan["sections"])))
            return result

    tracer = SignatureTracer()
    graph = tracer.trace(module)
    require(tracer.placeholder_count == len(inputs), "Input declaration count")
    return GraphModule(module, graph)
