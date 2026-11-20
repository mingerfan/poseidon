"""Trusted float64 PT-Graph/R17 approximation adapter for native HEIR linalg.

This adapter supplies model arithmetic to HEIR. Approximation selection remains
the saved frontend's responsibility. It never runs user or LLM Python text.
"""
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace


OFFICIAL_TAG = 'v2026.11.01'
OFFICIAL_CONV = 'https://github.com/google/heir/blob/v2026.11.01/tests/Examples/common/conv2d_nchw_padded.mlir'
OFFICIAL_PIPELINE = 'https://github.com/google/heir/blob/v2026.11.01/tests/Examples/lattigo/ckks/conv2d_nchw_chain/BUILD'
SUPPORTED = frozenset(('conv2d', 'batchnorm', 'act', 'add', 'slice', 'pad', 'mean', 'linear'))


def load_graph(directory):
    """Read the saved JSON/NPZ without importing the Torch intake package."""
    import numpy as np
    directory = Path(directory)
    record = json.loads((directory/'graph.json').read_text())
    with np.load(directory/'weights.npz',allow_pickle=False) as archive:
        weights = {}
        for name in archive.files:
            value = archive[name]
            if value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('Saved graph weights must retain finite float64 values')
            value.setflags(write=False)
            weights[name] = value
    nodes = [SimpleNamespace(id=n,op=op,inputs=inputs,outputs=outputs,attrs=attrs,src=src)
             for n,op,inputs,outputs,attrs,src in record['nodes']]
    tensors = {name:SimpleNamespace(name=name,shape=tuple(shape),kind=kind)
               for name,shape,kind in record['tensors']}
    return SimpleNamespace(name=record['name'],nodes=nodes,tensors=tensors,weights=weights,
                           inputs=record['inputs'],outputs=record['outputs'])


def tensor_type(shape):
    return 'tensor<' + 'x'.join(str(int(v)) for v in shape) + 'xf64>'


def scalar_literal(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('HEIR input constants must be finite float64')
    return format(value, '.17e')


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple

    @property
    def ty(self):
        return tensor_type(self.shape)


def optiona_selector_weights(channels, low, high):
    """Exact public 0/1 coefficients; no trainable graph weights are changed."""
    import numpy as np
    if any(type(x) is not int for x in (channels,low,high)) or channels<=0 or min(low,high)<0:
        raise ValueError('Exact OptionA selector channel dimensions differ')
    weight=np.zeros((low+channels+high,channels,1,1),dtype=np.float64)
    for channel in range(channels):weight[low+channel,channel,0,0]=1.
    weight.setflags(write=False)
    return weight


def exact_optiona_selectors(graph):
    """Find only complete, exclusive H/W stride2 then zero-channelpad triples."""
    from collections import Counter
    uses=Counter(name for node in graph.nodes for name in node.inputs)
    matches={}
    for first,second,pad in zip(graph.nodes,graph.nodes[1:],graph.nodes[2:]):
        if (first.op,second.op,pad.op)!=('slice','slice','pad'):continue
        if (len(first.inputs)!=1 or len(first.outputs)!=1 or tuple(second.inputs)!=tuple(first.outputs)
                or len(second.outputs)!=1 or tuple(pad.inputs)!=tuple(second.outputs) or len(pad.outputs)!=1):continue
        shape=tuple(graph.tensors[first.inputs[0]].shape)
        if len(shape)!=4 or shape[0]!=1 or min(shape)<=0 or shape[2]%2 or shape[3]%2:
            raise ValueError('Exact OptionA requires batch1 and even spatial dimensions')
        _,channels,height,width=shape
        expected=[(1,channels,height//2,width),(1,channels,height//2,width//2)]
        for node,axis,size,want in ((first,2,height,expected[0]),(second,3,width,expected[1])):
            if (node.attrs!={'axis':axis,'start':0,'end':size,'step':2}
                    or tuple(graph.tensors[node.outputs[0]].shape)!=want):
                raise ValueError('Exact OptionA slice axes/ranges/shapes differ')
            if uses[node.outputs[0]]!=1 or node.outputs[0] in graph.outputs:
                raise ValueError('Exact OptionA intermediate has another consumer/output')
        attrs=pad.attrs
        if set(attrs)-{'pads','mode','value'} or attrs.get('mode','constant')!='constant':
            raise ValueError('Exact OptionA requires original constant padding')
        value=float(attrs.get('value',0.))
        if not math.isfinite(value) or value!=0.:raise ValueError('Exact OptionA requires zero padding')
        pads=attrs.get('pads')
        if not isinstance(pads,(tuple,list)) or len(pads) not in (6,8) or any(type(x) is not int or x<0 for x in pads):
            raise ValueError('Exact OptionA padding dimensions differ')
        low,high=[0]*4,[0]*4
        for i in range(len(pads)//2):low[-1-i],high[-1-i]=pads[2*i:2*i+2]
        if any(low[i] or high[i] for i in (0,2,3)):
            raise ValueError('Exact OptionA permits channel padding only')
        output=(1,low[1]+channels+high[1],height//2,width//2)
        if tuple(graph.tensors[pad.outputs[0]].shape)!=output:
            raise ValueError('Exact OptionA padded shape differs')
        matches[first.id]=dict(original_nodes=[first.id,second.id,pad.id],input=first.inputs[0],
            output=pad.outputs[0],input_shape=list(shape),output_shape=list(output),
            pad_low=low[1],pad_high=high[1],stride=[2,2],
            coefficients='Exact float64 0/1; W[oc,ic,0,0]=1 iff oc=ic+pad_low',
            trainable_weights_changed=False)
    return matches


class Scalar:
    def __init__(self, tape, index):
        self.tape, self.index = tape, index

    def _binary(self, other, op):
        return self.tape.binary(op, self, other)

    def __add__(self, other): return self._binary(other, 'add')
    def __radd__(self, other): return self.tape.binary('add', other, self)
    def __sub__(self, other): return self._binary(other, 'sub')
    def __rsub__(self, other): return self.tape.binary('sub', other, self)
    def __mul__(self, other): return self._binary(other, 'mul')
    def __rmul__(self, other): return self.tape.binary('mul', other, self)


class ScalarTape:
    """Record the existing small-step/giant-step Chebyshev arithmetic verbatim."""
    def __init__(self):
        self.ops = [('input',)]
        self.constants = {}
        self.input = Scalar(self, 0)

    def value(self, value):
        if isinstance(value, Scalar):
            if value.tape is not self:
                raise ValueError('Scalar tapes must not be mixed')
            return value
        value = float(value)
        scalar_literal(value)
        # float.hex preserves -0.0 separately; this is a numerical cache key.
        key = value.hex()
        if key not in self.constants:
            self.constants[key] = Scalar(self, len(self.ops))
            self.ops.append(('constant', value))
        return self.constants[key]

    def binary(self, kind, left, right):
        left, right = self.value(left), self.value(right)
        result = Scalar(self, len(self.ops))
        self.ops.append((kind, left.index, right.index))
        return result

    def evaluate(self, value, output):
        values = [value]
        for op in self.ops[1:]:
            if op[0] == 'constant': values.append(op[1])
            elif op[0] == 'add': values.append(values[op[1]] + values[op[2]])
            elif op[0] == 'sub': values.append(values[op[1]] - values[op[2]])
            elif op[0] == 'mul': values.append(values[op[1]] * values[op[2]])
            else: raise ValueError('Unexpected scalar operation')
        return values[output.index]

    def mlir(self, output):
        lines = []
        for i, op in enumerate(self.ops[1:], 1):
            if op[0] == 'constant':
                text = 'arith.constant ' + scalar_literal(op[1]) + ' : f64'
            else:
                text = 'arith.' + {'add':'addf', 'sub':'subf', 'mul':'mulf'}[op[0]]
                text += f' %s{op[1]}, %s{op[2]} : f64'
            lines.append(f'      %s{i} = {text}')
        lines.append(f'      linalg.yield %s{output.index} : f64')
        return lines


def chebyshev_tape(fn, degree, domain):
    from ..kernels.poly import cheb_eval, cheb_fit
    lo, hi = domain
    coefficients = cheb_fit(fn, degree, lo, hi)
    tape = ScalarTape()
    normalized = tape.input * (2 / (hi-lo)) + (-(hi+lo) / (hi-lo))
    output = cheb_eval(None, normalized, coefficients)
    if not isinstance(output, Scalar):
        raise ValueError('Expected a nonconstant saved activation')
    return tape, output, coefficients


def spatial_mean_matrix(shape):
    """Exact public block-average coefficients; original model weights stay separate."""
    import numpy as np
    shape=tuple(shape)
    if (len(shape)!=4 or shape[0]!=1 or any(type(n) is not int or n<=0 for n in shape)
            or shape[1]*math.prod(shape[1:])>16*1024**2):
        raise ValueError('Fixed spatial mean needs a bounded positive batch-one NCHW shape')
    _,channels,height,width=shape;spatial=height*width
    result=np.zeros((channels,channels*spatial),dtype=np.float64)
    for channel in range(channels):result[channel,channel*spatial:(channel+1)*spatial]=1/spatial
    result.flags.writeable=False
    return result


class Emitter:
    def __init__(self):
        self.lines, self.counter = [], 0
        self.operations = []

    def fresh(self, shape, stem='t'):
        self.counter += 1
        return Tensor(f'%{stem}{self.counter}', tuple(shape))

    def emit(self, text):
        self.lines.append('    ' + text)

    def constant(self, array):
        import numpy as np
        array = np.asarray(array)
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError('Constants must retain finite float64 values')
        result = self.fresh(array.shape, 'c')
        def nested(value):
            if isinstance(value, list): return '[' + ', '.join(nested(v) for v in value) + ']'
            return scalar_literal(value)
        self.emit(f'{result.name} = arith.constant dense<{nested(array.tolist())}> : {result.ty}')
        return result

    def empty(self, shape, zero=False):
        result = self.fresh(shape, 'empty')
        self.emit(f'{result.name} = tensor.empty() : {result.ty}')
        if zero:
            scalar = self.fresh((), 'zero').name
            self.emit(f'{scalar} = arith.constant 0.00000000000000000e+00 : f64')
            filled = self.fresh(shape)
            self.emit(f'{filled.name} = linalg.fill ins({scalar} : f64) outs({result.name} : {result.ty}) -> {result.ty}')
            result = filled
        return result

    @staticmethod
    def identity(rank):
        dims = ', '.join(f'd{i}' for i in range(rank))
        return f'affine_map<({dims}) -> ({dims})>'

    def generic(self, inputs, shape, maps, body, *, reduction_axes=()):
        output = self.fresh(shape)
        initial = self.empty(shape, zero=bool(reduction_axes))
        rank = max(len(v.shape) for v in inputs)
        rank = max(rank, max(reduction_axes, default=-1)+1)
        iters = ', '.join('"reduction"' if i in reduction_axes else '"parallel"' for i in range(rank))
        self.emit(f'{output.name} = linalg.generic {{indexing_maps = [{", ".join(maps)}], iterator_types = [{iters}]}}')
        self.emit('  ins(' + ', '.join(v.name for v in inputs) + ' : ' + ', '.join(v.ty for v in inputs) + ')')
        self.emit(f'  outs({initial.name} : {initial.ty}) {{')
        names = ', '.join(f'%s{i}: f64' for i in range(len(inputs)))
        self.emit(f'^bb0({names}, %acc: f64):')
        self.lines.extend(body)
        self.emit(f'}} -> {output.ty}')
        return output

    def elementwise(self, left, right, operation, *, channel=False):
        rank = len(left.shape)
        identity = self.identity(rank)
        maps = [identity, identity, identity]
        if channel:
            dims = ', '.join(f'd{i}' for i in range(rank))
            maps[1] = f'affine_map<({dims}) -> (d1)>'
            if right.shape != (left.shape[1],): raise ValueError('Channel broadcast shape differs')
        elif left.shape != right.shape:
            raise ValueError('Only matching tensors or per-channel constants can be combined')
        body = [f'      %value = arith.{operation} %s0, %s1 : f64', '      linalg.yield %value : f64']
        return self.generic([left, right], left.shape, maps, body)

    def pad(self, value, low, high):
        if len(low) != len(value.shape) or len(high) != len(value.shape):
            raise ValueError('Padding rank differs')
        result = self.fresh(tuple(a+b+c for a,b,c in zip(value.shape,low,high)))
        zero = self.fresh((), 'padzero').name
        self.emit(f'{zero} = arith.constant 0.00000000000000000e+00 : f64')
        self.emit(f'{result.name} = tensor.pad {value.name} low[{", ".join(map(str,low))}] high[{", ".join(map(str,high))}] {{')
        args = ', '.join(f'%p{i}: index' for i in range(len(value.shape)))
        self.emit(f'^bb0({args}):')
        self.emit(f'  tensor.yield {zero} : f64')
        self.emit(f'}} : {value.ty} to {result.ty}')
        return result

    def conv(self, value, weight, attrs, expected):
        if attrs.get('groups', 1) != 1 or tuple(attrs.get('dilation', (1,1))) != (1,1):
            raise ValueError('This adapter supports original groups=1, dilation=1 convolutions')
        pads = tuple(attrs.get('padding', (0,0)))
        if any(pads): value = self.pad(value, (0,0,*pads), (0,0,*pads))
        kernel = self.constant(weight)
        initial = self.empty(expected, zero=True)
        output = self.fresh(expected)
        strides = ', '.join(str(i) for i in attrs.get('stride', (1,1)))
        self.emit(f'{output.name} = linalg.conv_2d_nchw_fchw {{dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[{strides}]> : vector<2xi64>}}')
        self.emit(f'  ins({value.name}, {kernel.name} : {value.ty}, {kernel.ty}) outs({initial.name} : {initial.ty}) -> {output.ty}')
        return output

    def optiona_selector(self,value,spec):
        if tuple(spec['input_shape'])!=value.shape:raise ValueError('Exact OptionA source shape differs')
        weight=optiona_selector_weights(value.shape[1],spec['pad_low'],spec['pad_high'])
        return self.conv(value,weight,dict(stride=(2,2),padding=(0,0),dilation=(1,1),groups=1),tuple(spec['output_shape']))

    def activation(self, value, config):
        tape, result, coefficients = chebyshev_tape(config['fn'], config['deg'], config['domain'])
        identity = self.identity(len(value.shape))
        output = self.generic([value], value.shape, [identity, identity], tape.mlir(result))
        return output, dict(config, coefficients=coefficients.tolist(), scalar_operations=len(tape.ops)-1)

    def mean(self, value, axes):
        if tuple(axes) != (2,3) or len(value.shape) != 4:
            raise ValueError('This adapter supports original global spatial mean')
        flattened = self.fresh((*value.shape[:2],value.shape[2]*value.shape[3]))
        self.emit(f'{flattened.name} = tensor.collapse_shape {value.name} [[0], [1], [2, 3]] : {value.ty} into {flattened.ty}')
        identity = self.identity(3)
        projected = 'affine_map<(d0, d1, d2) -> (d0, d1)>'
        body = ['      %value = arith.addf %s0, %acc : f64', '      linalg.yield %value : f64']
        reduced = self.generic([flattened], value.shape[:2], [identity, projected], body, reduction_axes=(2,))
        factor = 1/(value.shape[2]*value.shape[3])
        body = [f'      %factor = arith.constant {scalar_literal(factor)} : f64',
                '      %value = arith.mulf %s0, %factor : f64', '      linalg.yield %value : f64']
        identity2 = self.identity(2)
        return self.generic([reduced], reduced.shape, [identity2,identity2], body)

    def linear(self, value, weight, bias):
        if value.shape[0] != 1: raise ValueError('The original CNN adapter uses batch one')
        kernel = self.constant(weight)
        vector = self.fresh((value.shape[1],))
        self.emit(f'{vector.name} = tensor.collapse_shape {value.name} [[0, 1]] : {value.ty} into {vector.ty}')
        initial = self.empty((weight.shape[0],),zero=True)
        output = self.fresh(initial.shape)
        self.emit(f'{output.name} = linalg.matvec ins({kernel.name}, {vector.name} : {kernel.ty}, {vector.ty}) outs({initial.name} : {initial.ty}) -> {output.ty}')
        if bias is not None: output = self.elementwise(output,self.constant(bias),'addf')
        expanded = self.fresh((1,weight.shape[0]))
        self.emit(f'{expanded.name} = tensor.expand_shape {output.name} [[0, 1]] output_shape [1, {weight.shape[0]}] : {output.ty} into {expanded.ty}')
        return expanded

    def spatial_mean_matvec(self,value,axes):
        if tuple(axes)!=(2,3):raise ValueError('Fixed spatial mean preserves only original axes2,3')
        coefficients=spatial_mean_matrix(value.shape)
        flat=self.fresh((1,math.prod(value.shape[1:])))
        self.emit(f'{flat.name} = tensor.collapse_shape {value.name} [[0], [1, 2, 3]] : {value.ty} into {flat.ty}')
        return self.linear(flat,coefficients,None)

    def local_avgpool(self,value,attrs,expected):
        from .heir_common_models import avgpool_weights
        coefficients,spec=avgpool_weights(value.shape,attrs,expected)
        return self.conv(value,coefficients,dict(stride=spec['stride'],padding=spec['padding'],
                         dilation=(1,1),groups=1),tuple(expected)),spec

    def c_order_reshape(self,value,attrs,expected):
        from .heir_common_models import reshape_spec
        spec=reshape_spec(value.shape,attrs,expected)
        if value.shape==tuple(expected):return value
        result=self.fresh(expected)
        groups=', '.join('['+', '.join(map(str,g))+']' for g in spec['reassociation'])
        self.emit(f'{result.name} = tensor.collapse_shape {value.name} [{groups}] : {value.ty} into {result.ty}')
        return result


def saved_activations(program_text):
    from ..dsl.parser import parse
    program = parse(program_text)
    acts = [op for op in program.ops if op.kind == 'act']
    if not acts or any(op.attrs.get('fn') != 'silu' or op.attrs.get('method') != 'cheb'
                       or op.attrs.get('deg') != 63 for op in acts):
        raise ValueError('Expected the saved fixed SiLU Chebyshev degree63 frontend')
    return program, [dict(op.attrs, source_op_id=op.id, source_line=op.lineno) for op in acts]


def export_graph(graph, program_text, *, function_name='resnet20_silu',exact_optiona_selector=False,exact_spatial_mean_matvec=False,common_model=None):
    import numpy as np
    from ..kernels.conv import fold_bn
    contract=None
    if common_model is not None:
        from .heir_common_models import graph_contract
        contract=graph_contract(graph,common_model)
        if function_name not in ('resnet20_silu',common_model):raise ValueError('Common-model function identity differs')
        function_name=common_model
    program, acts = saved_activations(program_text)
    if len(graph.inputs) != 1 or len(graph.outputs) != 1 or len(program.params) != 1:
        raise ValueError('HEIR CNN adapter requires one feature tensor and one output')
    if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', function_name):
        raise ValueError('Function name must be an MLIR identifier')
    input_shape = tuple(graph.tensors[graph.inputs[0]].shape)
    if input_shape != program.params[0].shape:
        raise ValueError('Saved frontend and original graph input shapes differ')
    original_acts = [n for n in graph.nodes if n.op == 'act']
    if len(original_acts) != len(acts): raise ValueError('Graph/frontend activation counts differ')
    if type(exact_optiona_selector) is not bool:raise ValueError('Exact OptionA option must be an explicit boolean')
    if type(exact_spatial_mean_matvec) is not bool:raise ValueError('Fixed spatial mean option must be an explicit boolean')
    selectors=exact_optiona_selectors(graph) if exact_optiona_selector else {}
    folded={node_id:spec for spec in selectors.values() for node_id in spec['original_nodes'][1:]}
    e = Emitter()
    values = {graph.inputs[0]: Tensor('%arg0',input_shape)}
    act_index, approximations,mean_lowerings,pool_lowerings = 0, [], [], []
    for node in graph.nodes:
        supported=SUPPORTED if common_model is None else SUPPORTED|{'avgpool','reshape'}
        if node.op not in supported or len(node.outputs) != 1:
            raise ValueError('Unsupported original graph operation: '+node.op)
        a, expected = node.attrs, tuple(graph.tensors[node.outputs[0]].shape)
        original=dict(graph_node=node.id,kind=node.op,input_shapes=[list(graph.tensors[name].shape) for name in node.inputs],output_shape=list(expected))
        e.emit('// Original graph '+node.id+' '+node.op+' -> '+node.outputs[0])
        if node.id in folded:
            spec=folded[node.id]
            e.emit('// Preserved in exact OptionA selector emitted at '+spec['original_nodes'][0])
            e.operations.append(dict(original,lowered_by_exact_optiona_selector=spec['original_nodes'][0]));continue
        value = values[node.inputs[0]]
        if node.id in selectors:
            spec=selectors[node.id];output=e.optiona_selector(value,spec)
            values[spec['output']]=output
            e.operations.append(dict(original,exact_optiona_selector=spec));continue
        if node.op == 'conv2d':
            output = e.conv(value,graph.weights[a['weight']],a,expected)
            if a.get('bias') is not None:
                output = e.elementwise(output,e.constant(graph.weights[a['bias']]),'addf',channel=True)
        elif node.op == 'batchnorm':
            scale, shift = fold_bn(*(graph.weights[a[k]] for k in ('weight','bias','mean','var')),a['eps'])
            output = e.elementwise(value,e.constant(scale),'mulf',channel=True)
            output = e.elementwise(output,e.constant(shift),'addf',channel=True)
        elif node.op == 'act':
            config = acts[act_index]
            if a['fn'] != config['fn']: raise ValueError('Graph/frontend activation functions differ')
            output, record = e.activation(value,config)
            approximations.append(dict(record, graph_node=node.id, graph_output=node.outputs[0]))
            act_index += 1
        elif node.op == 'add':
            output = e.elementwise(value,values[node.inputs[1]],'addf')
        elif node.op == 'slice':
            rank = len(value.shape)
            offsets, strides = [0]*rank, [1]*rank
            offsets[a['axis']], strides[a['axis']] = a['start'], a['step']
            output = e.fresh(expected)
            fmt = lambda q: ', '.join(map(str,q))
            e.emit(f'{output.name} = tensor.extract_slice {value.name}[{fmt(offsets)}] [{fmt(expected)}] [{fmt(strides)}] : {value.ty} to {output.ty}')
        elif node.op == 'pad':
            pads, rank = a['pads'], len(value.shape)
            low, high = [0]*rank,[0]*rank
            for i in range(len(pads)//2): low[-1-i], high[-1-i] = pads[2*i:2*i+2]
            output = e.pad(value,low,high)
        elif node.op == 'mean':
            if exact_spatial_mean_matvec:
                output=e.spatial_mean_matvec(value,a['axes'])
                mean_lowerings.append(dict(original_node=node.id,input=node.inputs[0],output=node.outputs[0],
                    input_shape=list(value.shape),output_shape=list(output.shape),axes=list(a['axes']),
                    coefficient=1/math.prod(value.shape[2:]),matrix_shape=[value.shape[1],math.prod(value.shape[1:])],
                    nonzeros=math.prod(value.shape[1:]),trainable_weights_changed=False,bias=False,
                    formula='W[c,c*H*W+p]=1/(H*W), p=0..H*W-1; all other coefficients0'))
            else:output = e.mean(value,a['axes'])
        elif node.op == 'linear':
            output = e.linear(value,graph.weights[a['weight']],graph.weights[a['bias']] if a.get('bias') else None)
        elif node.op == 'avgpool':
            output,spec=e.local_avgpool(value,a,expected)
            pool_lowerings.append(dict(spec,original_node=node.id,input=node.inputs[0],output=node.outputs[0]))
        elif node.op == 'reshape':
            output=e.c_order_reshape(value,a,expected)
        if output.shape != expected: raise ValueError('Original graph output shape differs at '+node.id)
        values[node.outputs[0]] = output
        e.operations.append(original)
    result = values[graph.outputs[0]]
    text = '\n'.join(['// Trusted original-model adapter; no HEIR-native approximation selection.',
                       'module {',f'  func.func @{function_name}(%arg0: {tensor_type(input_shape)} {{secret.secret}}) -> {result.ty} {{',
                       *e.lines,f'    return {result.name} : {result.ty}','  }','}',''])
    metadata = dict(function=function_name,input_shape=list(input_shape),output_shape=list(result.shape),
                    dtype='float64',graph_nodes=len(graph.nodes),graph_operations=e.operations,
                    approximations=approximations,approximation_source='saved R17 margin1.50 degree63 program',
                    approximation_selected_by_heir=False,bn_formula='conv; x*(gamma/sqrt(var+eps)); x+(beta-scale*mean)',
                    option_a='Exact fixed-public 0/1 1x1 stride2 convolution for proven exclusive original slice/slice/channelpad triples' if exact_optiona_selector else 'original graph strided slices followed by channel zero padding',
                    exact_optiona_selector=exact_optiona_selector,exact_optiona_lowerings=list(selectors.values()),
                    exact_spatial_mean_matvec=exact_spatial_mean_matvec,exact_spatial_mean_lowerings=mean_lowerings,
                    weights_preserved=True,derived_bn_constants=True,official_release=OFFICIAL_TAG,
                    official_sources=[OFFICIAL_CONV,OFFICIAL_PIPELINE],mlir_bytes=len(text.encode()),
                    compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    if common_model is not None:
        metadata.update(common_model=common_model,common_model_contract=contract,
                        fixed_local_avgpool_lowerings=pool_lowerings,
                        approximation_source='saved fixed degree63 program for '+common_model)
    return text, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-dir',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--function-name',default='resnet20_silu')
    from .heir_common_models import COMMON_MODELS
    p.add_argument('--common-model',choices=COMMON_MODELS)
    p.add_argument('--exact-optiona-selector',action='store_true',help='Use the proved exact public0/1 selector only for exclusive original OptionA triples')
    p.add_argument('--exact-spatial-mean-matvec',action='store_true',help='Represent original axes2,3 spatial mean with a fixed public block-average matrix; preserve original model weights')
    args = p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    text, record = export_graph(load_graph(args.graph_dir),args.program.read_text(),function_name=args.function_name,exact_optiona_selector=args.exact_optiona_selector,exact_spatial_mean_matvec=args.exact_spatial_mean_matvec,common_model=args.common_model)
    (args.output/'model.mlir').write_text(text)
    record['sources'] = dict(graph=str(args.graph_dir),program=str(args.program))
    (args.output/'export.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
