"""Deterministic, fail-closed FX -> Hecate reference translator (NOT an Agent).

Accepts trusted nn.Module objects only: symbolic tracing executes Python. No
claims that inspecting the resulting graph detects every external side effect.
Translation uses operators/types/shapes, never catalog family/case names.
"""
from dataclasses import dataclass
import math
import operator
import torch
import torch.nn.functional as F

from hecate_contract import validate_function
from seal_artifact_gate import require
from spatial_ops import INPUT_SHAPES
from model_graph import LINEAR_WIDTH_LIMIT
from cipher_abi import ZERO_NAME, ZERO_ARGUMENT


class UnsupportedModel(ValueError):
    pass


def supported(condition, reason):
    if not condition:
        raise UnsupportedModel(reason)


@dataclass(frozen=True)
class CipherValue:
    names: tuple
    shape: tuple
    packed: bool


class Emitter:
    def __init__(self, *, slot_period=4, packed_abi=False):
        self.lines, self.constants, self.origins = [], {}, {}
        self.needs_zero = False
        self.slot_period,self.packed_abi=slot_period,packed_abi
        self.width_limit=16 if packed_abi else LINEAR_WIDTH_LIMIT

    def encrypted_zero(self):
        self.needs_zero = True
        return ZERO_NAME

    def assign(self, expression):
        name = f"v{len(self.lines)}"
        self.lines.append(f"    {name} = {expression}")
        return name

    def public(self, value, origin):
        name = f"c{len(self.constants)}"
        self.constants[name] = value
        self.origins[name] = origin
        return name

    def binary(self, left, right, op, node):
        # Canonicalize public-left add/multiply using their commutativity.
        if not isinstance(left, CipherValue):
            left, right = right, left
        supported(isinstance(left, CipherValue), f"{node}: public-only arithmetic is not lowered")
        supported(self.packed_abi or len(left.shape) == 1 or not left.packed, f"{node}: flatten packed multidimensional inputs explicitly first")
        if isinstance(right, CipherValue):
            supported(left.shape == right.shape and left.packed == right.packed,
                      f"{node}: incompatible cipher shape/packing (no implicit repacking)")
            rhs = right.names
        else:
            plain = torch.as_tensor(right, dtype=torch.float64)
            if self.packed_abi:
                from packed_model import broadcast_to_shape
                supported(broadcast_to_shape(tuple(plain.shape),left.shape),f"{node}: unsupported plaintext broadcast")
                if tuple(plain.shape) not in ((),(1,),left.shape):
                    plain=torch.broadcast_to(plain,left.shape)
            else:
                supported(tuple(plain.shape) in ((), (1,), left.shape), f"{node}: unsupported broadcasting")
            values = plain.reshape(-1).tolist()
            if op == '*' and all(v == 0.0 for v in values):
                zero = self.encrypted_zero()
                return CipherValue(tuple(zero for _ in left.names), left.shape, left.packed)
            if left.packed:
                if self.packed_abi and len(values)>1:
                    values += [0.0]*(self.slot_period-len(values))
                rhs = (self.public(values, node),)
            else:
                if op == '*':
                    outputs = []
                    for i, source in enumerate(left.names):
                        scalar = values[0] if len(values) == 1 else values[i]
                        if scalar == 0.0:
                            outputs.append(self.encrypted_zero())
                        else:
                            public = self.public(scalar, node)
                            outputs.append(self.assign(f"{source} * {public}"))
                    return CipherValue(tuple(outputs), left.shape, left.packed)
                rhs = tuple(self.public(values[0] if len(values) == 1 else values[i], node)
                            for i in range(len(left.names)))
        return CipherValue(tuple(self.assign(f"{a} {op} {b}") for a, b in zip(left.names, rhs)), left.shape, left.packed)

    def linear(self, value, weight, bias, node):
        if self.packed_abi and isinstance(value,CipherValue) and len(value.shape)>1:
            supported(type(weight) in (torch.Tensor,torch.nn.Parameter) and weight.ndim==2 and
                      weight.shape[1]==value.shape[-1],f"{node}: invalid last-axis Linear weight")
            out=int(weight.shape[0]);width=value.shape[-1];groups=math.prod(value.shape[:-1])
            supported(1<=groups*out<=self.width_limit,f"{node}: batched scalar-neuron output budget")
            supported(bias is None or tuple(bias.shape)==(out,),f"{node}: invalid Linear bias")
            if value.packed:
                matrix=torch.zeros((groups*out,math.prod(value.shape)),dtype=torch.float64)
                for group in range(groups):
                    matrix[group*out:(group+1)*out,group*width:(group+1)*width]=weight
                bias_vector=None if bias is None else bias.repeat(groups)
                flat=CipherValue(value.names,(math.prod(value.shape),),True)
                mapped=self.linear(flat,matrix,bias_vector,node+'.last_axis_block_diagonal')
                names=mapped.names
            else:
                names=tuple(name for group in range(groups) for name in self.linear(
                    CipherValue(value.names[group*width:(group+1)*width],(width,),False),
                    weight,bias,f'{node}.last_axis_group{group}').names)
            return CipherValue(names,(*value.shape[:-1],out),False)
        supported(isinstance(value, CipherValue) and len(value.shape) == 1, f"{node}: Linear needs a vector")
        supported(type(weight) in (torch.Tensor, torch.nn.Parameter) and weight.ndim == 2 and
                  weight.shape[1] == value.shape[0] and 1 <= weight.shape[0] <= self.width_limit, f"{node}: invalid Linear weight/shape")
        supported(bias is None or tuple(bias.shape) == (weight.shape[0],), f"{node}: invalid Linear bias")
        outputs = []
        for row in range(weight.shape[0]):
            if not bool((weight[row] != 0).any()):
                total = self.encrypted_zero()
            elif value.packed:
                supported(1<=value.shape[0]<=self.slot_period if self.packed_abi else value.shape==(4,),
                          f"{node}: packed dot product shape/period mismatch")
                weights=weight[row].detach().tolist()
                if self.packed_abi:
                    weights += [0.0]*(self.slot_period-len(weights))
                w = self.public(weights, f"{node}.weight[{row}]")
                p = self.assign(f"{value.names[0]} * {w}")
                total=p
                for bit in range(self.slot_period.bit_length()-1):
                    total=self.assign(f"{total} + {total}.rotate({1<<bit})")
            else:
                terms = []
                for column, source in enumerate(value.names):
                    # Exact public-zero elimination; never disable SEAL's transparent-ciphertext checks.
                    if float(weight[row, column]) == 0.0:
                        continue
                    w = self.public(float(weight[row, column]), f"{node}.weight[{row},{column}]")
                    terms.append(self.assign(f"{source} * {w}"))
                total = terms[0]
                for term in terms[1:]:
                    total = self.assign(f"{total} + {term}")
            if bias is not None:
                b = self.public(float(bias[row]), f"{node}.bias[{row}]")
                total = self.assign(f"{total} + {b}")
            outputs.append(total)
        return CipherValue(tuple(outputs), (len(outputs),), False)

    def permute(self,value,dims,node):
        from tensor_permutation import routing,source_order
        supported(self.packed_abi and isinstance(value,CipherValue),f'{node}: permutation requires packed ABI')
        shape,order=source_order(value.shape,dims)
        if not value.packed:
            supported(len(value.names)==len(order),f'{node}: scalar layout mismatch')
            return CipherValue(tuple(value.names[i] for i in order),shape,False)
        supported(len(value.names)==1,f'{node}: packed permutation input count')
        if order==list(range(len(order))):return CipherValue(value.names,shape,True)
        _,_,masks=routing(value.shape,dims,self.slot_period)
        rotated={0:value.names[0]}
        def shifted(delta):
            if delta not in rotated:
                bit=delta & -delta
                rotated[delta]=self.assign(f'{shifted(delta-bit)}.rotate({bit})')
            return rotated[delta]
        terms=[]
        for delta,mask in sorted(masks.items()):
            constant=self.public(mask,f'{node}.permutation_shift[{delta}]')
            terms.append(self.assign(f'{shifted(delta)} * {constant}'))
        total=terms[0]
        for term in terms[1:]:total=self.assign(f'{total} + {term}')
        return CipherValue((total,),shape,True)

    def concat(self, values, axis, node):
        from concat_ops import source_order
        supported(type(values) in (tuple,list) and all(isinstance(v,CipherValue) for v in values),
                  f"{node}: concat needs a static ciphertext sequence")
        shape, order = source_order([v.shape for v in values], axis,max_elements=16 if self.packed_abi else 8)
        scalars = []
        for index, value in enumerate(values):
            if value.packed:
                count=math.prod(value.shape)
                supported((count<=self.slot_period if self.packed_abi else count==4) and len(value.names)==1,
                          f"{node}: packed concat input/period mismatch")
                # Explicit extraction: one-hot multiply plus reduction repeats each
                # selected slot in all slots. A rotation alone would not do that.
                vector=CipherValue(value.names,(count,),True)
                extracted=self.linear(vector,torch.eye(count,dtype=torch.float64),None,
                                      f"{node}.extract{index}")
                scalars.append(extracted.names)
            else:
                supported(len(value.names)==math.prod(value.shape),f"{node}: scalar layout mismatch")
                scalars.append(value.names)
        return CipherValue(tuple(scalars[b][i] for b,i in order),shape,False)

    def batch_norm(self,value,mean,variance,weight,bias,eps,node):
        from batch_norm_ops import coefficients,expand_channels
        supported(isinstance(value,CipherValue),f"{node}: BatchNorm needs ciphertext")
        def public_tensor(t):
            if t is None:return None
            supported(type(t) in (torch.Tensor,torch.nn.Parameter) and t.device.type=="cpu" and
                      t.dtype==torch.float64 and t.ndim==1,f"{node}: BatchNorm requires public float64 vectors")
            return t.detach().tolist()
        gain,offset=coefficients(value.shape,*[public_tensor(t) for t in (mean,variance,weight,bias)],eps,
                                 max_elements=256 if self.packed_abi else 8)
        vector=CipherValue(value.names,(math.prod(value.shape),),value.packed)
        scaled=self.binary(vector,expand_channels(value.shape,gain),"*",node+".gain")
        shifted=self.binary(scaled,expand_channels(value.shape,offset),"+",node+".offset")
        return CipherValue(shifted.names,value.shape,value.packed)

    def spatial(self, value, op, weight, bias, kernel, stride, padding, include_pad, node, *, dilation=None, groups=1):
        if self.packed_abi:
            from packed_spatial import lowering
        else:
            from spatial_ops import lowering
        supported(isinstance(value, CipherValue), f"{node}: expected ciphertext tensor")
        matrix, biases, shape = lowering(op, value.shape,
            None if weight is None else weight.detach().tolist(), None if weight is None else tuple(weight.shape),
            None if bias is None else bias.detach().tolist(), kernel, stride, padding, include_pad,
            dilation=dilation, groups=groups)
        vector = CipherValue(value.names, (math.prod(value.shape),), value.packed)
        mapped = self.linear(vector, torch.tensor(matrix, dtype=torch.float64),
                             None if bias is None else torch.tensor(biases, dtype=torch.float64), node)
        return CipherValue(mapped.names, shape, False)


def state_arrays(model, *, max_elements=128):
    result = {}
    counters={((name+".") if name else "")+"num_batches_tracked" for name,m in model.named_modules()
              if type(m) in (torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)}
    for name, tensor in model.state_dict().items():
        if name in counters:
            supported(tensor.device.type=="cpu" and tensor.dtype==torch.int64 and tensor.ndim==0 and
                      int(tensor)>=0,f"{name}: invalid BatchNorm counter")
            result[name]=tensor.detach().numpy().copy()
            continue
        supported(tensor.device.type == "cpu" and tensor.dtype == torch.float64,
                  f"{name}: requires CPU float64 state, no implicit dtype conversion")
        supported(tensor.numel() <= max_elements and bool(torch.isfinite(tensor).all()) and
                  bool((tensor.abs() <= 1024).all()), f"{name}: invalid or oversized state")
        result[name] = tensor.detach().numpy().copy()
    return result


def translate(model, input_shape):
    supported(isinstance(model, torch.nn.Module), "Expected a trusted torch.nn.Module")
    supported(all(not m.training for m in model.modules()), "Model and all submodules must be eval()")
    from packed_input_abi import ABI as PACKED_ABI, CONTRACT as PACKED_CONTRACT, binding as packed_binding
    packed=type(input_shape) is dict and 'execution_abi' in input_shape
    if packed:
        supported(set(input_shape)=={'execution_abi','input_shape'} and input_shape['execution_abi']==PACKED_ABI,
                  'Invalid packed input manifest')
        plan=packed_binding(input_shape['input_shape'])
    multiple = type(input_shape) is dict and not packed
    if multiple:
        supported(set(input_shape) in ({"inputs"}, {"inputs", "model_input_binding"}) and type(input_shape["inputs"]) is list and
                  2 <= len(input_shape["inputs"]) <= 4, "Invalid input manifest")
        specs = input_shape["inputs"]
        supported(all(type(s) is dict and set(s) == {"name", "shape"} and type(s["name"]) is str
                      and type(s["shape"]) is list for s in specs), "Invalid input manifest entry")
        supported(len({s["name"] for s in specs}) == len(specs), "Duplicate input names")
    else:
        specs = [{"name": "x", "shape": input_shape['input_shape'] if packed else input_shape}]
    supported(all(type(s["shape"]) in (list, tuple) and all(type(n) is int for n in s["shape"]) and
                  (packed or tuple(s["shape"]) in INPUT_SHAPES) for s in specs),
              "Supported logical input shapes: [4], [2,2], [1,4]")
    input_names = ("x", "y", "z", "t")[:len(specs)]
    state_limit=4096 if packed else 128
    before = state_arrays(model,max_elements=state_limit)
    try:
        target=torch.nn.Sequential(model) if type(model) in (
            torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d) else model
        graph = torch.fx.symbolic_trace(target)
    except Exception as error:
        raise UnsupportedModel(f"FX tracing rejected model: {error}") from error
    after = state_arrays(model,max_elements=state_limit)
    supported(before.keys() == after.keys() and all((before[k] == after[k]).all() for k in before),
              "Model state changed during FX tracing")
    supported(len(list(graph.graph.nodes)) <= 256, "FX node limit")
    emit, values, placeholders, result = Emitter(slot_period=plan['slot_period'] if packed else 4,packed_abi=packed), {}, 0, None
    rotation_contract = False

    def resolve(value):
        if isinstance(value,torch.fx.Node):return values[value]
        # FX wraps literal lists in immutable_list; normalize that metadata,
        # not arbitrary runtime objects, into the checked static containers.
        if isinstance(value,list):return [resolve(v) for v in value]
        if isinstance(value,tuple):return tuple(resolve(v) for v in value)
        return value

    for node in graph.graph.nodes:
        args = tuple(resolve(x) for x in node.args)
        if node.op=="call_function" and node.target is F.batch_norm:
            import inspect
            bound=inspect.signature(F.batch_norm).bind(*args,**{k:resolve(v) for k,v in node.kwargs.items()})
            bound.apply_defaults()
            args=tuple(bound.arguments.values())
        elif node.op=="call_function" and node.target in (torch.cat,torch.concat,torch.concatenate):
            supported(set(node.kwargs)<= {'dim'} and 1 <= len(args) <= 2 and
                      not (len(args)==2 and 'dim' in node.kwargs),f"{node.name}: unsupported concat arguments")
            args=(args[0],args[1] if len(args)==2 else node.kwargs.get('dim',0))
        elif ((node.op=='call_function' and node.target in (torch.permute,torch.transpose)) or
              (node.op=='call_method' and node.target in ('permute','transpose'))):
            fields=('dims',) if node.target in ('permute',torch.permute) else ('dim0','dim1')
            if node.kwargs:
                supported(set(node.kwargs)<=set(fields) and 1<=len(args)<=len(fields)+1,
                          f'{node.name}: invalid permutation keywords')
                supplied=dict(zip(fields,args[1:]))
                supported(not set(supplied)&set(node.kwargs),f'{node.name}: duplicate permutation argument')
                supplied.update({k:resolve(v) for k,v in node.kwargs.items()})
                supported(set(supplied)==set(fields),f'{node.name}: missing permutation argument')
                args=(args[0],*(supplied[k] for k in fields))
        else:
            supported(not node.kwargs, f"{node.name}: keyword arguments require explicit semantics, currently rejected")
        if node.op == "placeholder":
            placeholders += 1
            supported(placeholders <= len(specs) and not node.args, "Encrypted input count/default mismatch")
            values[node] = CipherValue((input_names[placeholders-1],), tuple(specs[placeholders-1]["shape"]), True)
        elif node.op == "get_attr":
            obj = graph
            for part in node.target.split("."):
                obj = getattr(obj, part)
            supported(type(obj) in (torch.Tensor, torch.nn.Parameter) and obj.device.type == "cpu" and
                      obj.dtype == torch.float64 and obj.numel() <= state_limit and bool(torch.isfinite(obj).all()) and
                      bool((obj.abs() <= 1024).all()), f"{node.name}: unsupported public attribute")
            values[node] = obj.detach()
        elif node.op == "call_module":
            module = graph.get_submodule(node.target)
            supported(len(args)==1,f"{node.name}: module arity")
            if type(module) in (torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d):
                supported(not module.training and module.track_running_stats and
                          module.running_mean is not None and module.running_var is not None,
                          f"{node.name}: BatchNorm requires eval and frozen running statistics")
                rank=len(args[0].shape) if isinstance(args[0],CipherValue) else 0
                supported(rank in ((2,3) if type(module) is torch.nn.BatchNorm1d else
                                   (4,) if type(module) is torch.nn.BatchNorm2d else (5,)),
                          f"{node.name}: BatchNorm module input rank")
                values[node]=emit.batch_norm(args[0],module.running_mean,module.running_var,
                                            module.weight,module.bias,module.eps,node.target)
            elif packed and type(module) in (torch.nn.Conv1d,torch.nn.Conv2d):
                supported(module.padding_mode=='zeros' and type(module.padding) is tuple,
                          f'{node.name}: only explicit zero-padding Conv modules supported')
                op='conv1d' if type(module) is torch.nn.Conv1d else 'conv2d'
                values[node]=emit.spatial(args[0],op,module.weight.detach(),module.bias,module.kernel_size,
                    module.stride,module.padding,True,node.target,dilation=module.dilation,groups=module.groups)
            elif packed and type(module) in (torch.nn.AvgPool1d,torch.nn.AvgPool2d):
                supported(module.ceil_mode is False and getattr(module,'divisor_override',None) is None,
                          f'{node.name}: average pool requires floor mode and native divisor')
                dims=1 if type(module) is torch.nn.AvgPool1d else 2
                def pair(value):return (value,)*dims if type(value) is int else tuple(value)
                kernel=pair(module.kernel_size)
                stride=kernel if module.stride is None else pair(module.stride)
                values[node]=emit.spatial(args[0],'avg_pool'+str(dims)+'d',None,None,kernel,stride,
                    pair(module.padding),module.count_include_pad,node.target)
            else:
                supported(type(module) is torch.nn.Linear, f"{node.name}: unsupported module {type(module).__name__}")
                values[node] = emit.linear(args[0], module.weight.detach(), module.bias, node.target)
        elif node.op == "call_function" and node.target in (torch.cat,torch.concat,torch.concatenate):
            values[node]=emit.concat(args[0],args[1],node.name)
        elif node.op == "call_function" and node.target is F.batch_norm:
            supported(len(args)==8 and args[5] is False,f"{node.name}: BatchNorm training is forbidden")
            values[node]=emit.batch_norm(*args[:5],args[7],node.name)
        elif node.op == "call_function" and node.target in (operator.add, torch.add, operator.mul, torch.mul):
            supported(len(args) == 2, f"{node.name}: binary arity")
            values[node] = emit.binary(args[0], args[1], "+" if node.target in (operator.add, torch.add) else "*", node.name)
        elif node.op == "call_function" and node.target in (operator.neg, torch.neg):
            supported(len(args) == 1 and isinstance(args[0], CipherValue), f"{node.name}: cipher negation arity")
            # Exact algebraic rewrite into the existing verified DSL subset.
            # This may consume more CKKS depth than native negation; not a speed claim.
            values[node] = emit.binary(args[0], -1.0, "*", node.name)
        elif node.op == "call_function" and node.target in (operator.sub, torch.sub):
            supported(len(args) == 2 and isinstance(args[0], CipherValue), f"{node.name}: cipher-left subtraction")
            negative = (emit.binary(args[1], -1.0, "*", node.name + ".negate")
                        if isinstance(args[1], CipherValue) else -args[1])
            values[node] = emit.binary(args[0], negative, "+", node.name)
        elif node.op == "call_function" and node.target is torch.square:
            supported(len(args) == 1, f"{node.name}: square arity")
            values[node] = emit.binary(args[0], args[0], "*", node.name)
        elif node.op == "call_function" and node.target is torch.pow:
            supported(len(args) == 2 and type(args[1]) is int and args[1] in (2, 4), f"{node.name}: unsupported power")
            square = emit.binary(args[0], args[0], "*", node.name)
            values[node] = square if args[1] == 2 else emit.binary(square, square, "*", node.name)
        elif node.op == "call_function" and node.target in (F.conv1d, F.conv2d):
            supported(len(args) == 7 and type(args[1]) in (torch.Tensor, torch.nn.Parameter),
                      f"{node.name}: unsupported Conv arguments")
            op = "conv1d" if node.target is F.conv1d else "conv2d"
            dilation = (args[5],)*(1 if op == "conv1d" else 2) if type(args[5]) is int else args[5]
            values[node] = emit.spatial(args[0], op, args[1], args[2], tuple(args[1].shape[2:]),
                                       args[3], args[4], True, node.name, dilation=dilation, groups=args[6])
        elif node.op == "call_function" and node.target in (F.avg_pool1d, F.avg_pool2d):
            supported(len(args) == 6 and args[4] is False and type(args[5]) is bool,
                      f"{node.name}: requires floor-mode average pooling")
            op = "avg_pool1d" if node.target is F.avg_pool1d else "avg_pool2d"
            values[node] = emit.spatial(args[0], op, None, None, args[1], args[2], args[3], args[5], node.name)
        elif node.op == "call_function" and node.target is F.linear:
            supported(len(args) in (2, 3), f"{node.name}: linear arity")
            values[node] = emit.linear(args[0], args[1], args[2] if len(args) == 3 else None, node.name)
        elif node.op == "call_function" and node.target is torch.roll:
            supported(len(args) == 3 and isinstance(args[0], CipherValue) and args[0].packed and
                      args[0].shape == (4,) and type(args[1]) is int and args[1] in (-3, -2, -1, 1, 2, 3) and
                      type(args[2]) is int and args[2] == 0, f"{node.name}: unsupported roll shape/step/axis")
            name = emit.assign(f"{args[0].names[0]}.rotate({-args[1]})")
            values[node] = CipherValue((name,), (4,), True)
            rotation_contract = True
        elif ((node.op=='call_function' and node.target in (torch.permute,torch.transpose)) or
              (node.op=='call_method' and node.target in ('permute','transpose'))):
            from tensor_permutation import transpose_axes
            supported(packed and len(args)>=2 and isinstance(args[0],CipherValue),f'{node.name}: invalid permutation input')
            if node.target in ('transpose',torch.transpose):
                supported(len(args)==3,f'{node.name}: transpose arity')
                dims=transpose_axes(args[0].shape,args[1],args[2])
            else:
                dims=args[1] if len(args)==2 and type(args[1]) in (tuple,list) else args[1:]
                if node.op=='call_function':supported(len(args)==2 and type(args[1]) in (tuple,list),f'{node.name}: permute needs dims sequence')
            values[node]=emit.permute(args[0],dims,node.name)
        elif (node.op == "call_function" and node.target is torch.reshape or
              node.op == "call_method" and node.target == "reshape"):
            from logical_reshape import reshape_shape
            supported(len(args)>=2 and isinstance(args[0],CipherValue),f"{node.name}: reshape needs ciphertext and static shape")
            target=args[1] if len(args)==2 and type(args[1]) in (tuple,list) else args[1:]
            if node.op=="call_function":
                supported(len(args)==2 and type(args[1]) in (tuple,list),f"{node.name}: torch.reshape shape tuple required")
            shape=reshape_shape(args[0].shape,target,max_elements=256 if packed else 8)
            values[node]=CipherValue(args[0].names,shape,args[0].packed)
        elif node.op == "call_function" and node.target is torch.flatten:
            supported(len(args) == 1 and isinstance(args[0], CipherValue),
                      f"{node.name}: only full row-major flatten of ciphertext tensor allowed")
            values[node] = CipherValue(args[0].names, (math.prod(args[0].shape),), args[0].packed)
        elif node.op == "output":
            supported(len(args) == 1 and isinstance(args[0], CipherValue) and (packed or len(args[0].shape) == 1),
                      "Output must be a single logical vector, not tuple/dict")
            result = args[0]
        else:
            raise UnsupportedModel(f"{node.name}: unsupported FX {node.op} {node.target}")
    supported(placeholders == len(specs) and result is not None, "Missing I/O")
    returned = result.names[0] if len(result.names) == 1 else "[" + ", ".join(result.names) + "]"
    physical_names = (*input_names, ZERO_NAME) if emit.needs_zero else input_names
    source = '@hc.func("' + ','.join(['c']*len(physical_names)) + '")\ndef golden(' + ', '.join(physical_names) + '):\n' + "\n".join(emit.lines) + f"\n    return {returned}\n"
    check = validate_function(source, emit.constants, expected_outputs=len(result.names),
                              contract=PACKED_CONTRACT if packed else "hecate-function-v4" if emit.needs_zero else "hecate-function-v3" if multiple else "hecate-function-v2" if rotation_contract else "hecate-function-v0",
                              input_names=physical_names,slot_period=emit.slot_period)
    selectors = [[0, i] for i in range(math.prod(result.shape) if packed else 4)] if result.packed else [[i, 0] for i in range(len(result.names))]
    layout = dict(input_shape=list(specs[0]['shape']) if packed else list(input_shape), input_slot_period=emit.slot_period, input_order="C-row-major",
                  output_shape=list(result.shape), output_ciphertexts=len(result.names), output_selectors=selectors)
    if multiple:
        layout.pop("input_shape")
        layout["inputs"] = [dict(name=s["name"], dsl_name=n, shape=list(s["shape"])) for s, n in zip(specs, input_names)]
        if 'model_input_binding' in input_shape:
            from chunked_input_abi import validate_request_binding
            import copy
            layout['model_input_binding'] = copy.deepcopy(input_shape['model_input_binding'])
            validate_request_binding(dict(model=dict(schema=4,
                input_shape=layout['model_input_binding'].get('logical_shape')), layout=layout))
    if emit.needs_zero:
        layout['auxiliary_ciphertexts'] = [dict(ZERO_ARGUMENT)]
    if packed:
        from packed_input_abi import validate_layout,zero_argument
        layout.update(execution_abi=PACKED_ABI,model_input_binding=plan,
                      output_representation='packed_prefix' if result.packed else 'scalar_neurons')
        if emit.needs_zero:
            layout['auxiliary_ciphertexts']=[zero_argument(emit.slot_period)]
        validate_layout(layout)
    return dict(generator="deterministic-fx-v0", hecate_source=source, public_constants=emit.constants,
                constant_origins=emit.origins, fx_graph=str(graph.graph), static_check=check,
                layout=layout,
                warnings=["Trusted model objects only; FX tracing is not an untrusted Python sandbox",
                          "Compiler/runtime/real numerical comparison still required"])
