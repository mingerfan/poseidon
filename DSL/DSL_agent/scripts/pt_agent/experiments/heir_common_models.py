"""Source-derived contracts for the explicit common-model HEIR variant."""
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace


COMMON_MODELS = ('resnet20_silu', 'vgg16_silu_avgpool', 'alexnet_silu_avgpool')


def graph_contract(graph, selected_model):
    if selected_model not in COMMON_MODELS or graph.name != selected_model:
        raise ValueError('Selected common model and original graph identity differ')
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError('Common-model HEIR requires one input and one output')
    shapes = {name: tuple(t.shape) for name, t in graph.tensors.items()}
    if any(not s or any(type(d) is not int or d <= 0 for d in s) for s in shapes.values()):
        raise ValueError('Common-model tensors require positive static shapes')
    input_shape, output_shape = shapes[graph.inputs[0]], shapes[graph.outputs[0]]
    if input_shape[0] != 1 or len(output_shape) != 2 or output_shape[0] != 1:
        raise ValueError('Common-model input/output batch or classifier rank differs')
    constants = []
    for node in graph.nodes:
        if node.op in ('conv2d', 'linear'):
            constants.append(node.attrs['weight'])
            if node.attrs.get('bias'):
                constants.append(node.attrs['bias'])
        elif node.op == 'batchnorm':
            constants.extend((node.id + ':scale', node.id + ':shift'))
    result = dict(model=selected_model, graph_nodes=len(graph.nodes),
                  operator_counts=dict(sorted(Counter(n.op for n in graph.nodes).items())),
                  activation_count=sum(n.op == 'act' for n in graph.nodes),
                  original_tensor_constant_names=constants,
                  original_tensor_constant_count=len(constants),
                  input_shape=list(input_shape), output_shape=list(output_shape),
                  nodes=[dict(id=n.id, op=n.op, inputs=list(n.inputs), outputs=list(n.outputs),
                              attrs=n.attrs, input_shapes=[shapes[x] for x in n.inputs],
                              output_shapes=[shapes[x] for x in n.outputs]) for n in graph.nodes])
    # JSON normalization keeps tuples from in-memory fixtures equivalent to saved JSON.
    return json.loads(json.dumps(result))


def read_graph_contract(directory, selected_model):
    record = json.loads((Path(directory) / 'graph.json').read_text())
    graph = SimpleNamespace(name=record['name'], inputs=record['inputs'], outputs=record['outputs'],
        tensors={n: SimpleNamespace(shape=tuple(s)) for n, s, _ in record['tensors']},
        nodes=[SimpleNamespace(id=n, op=op, inputs=ins, outputs=outs, attrs=a)
               for n, op, ins, outs, a, _ in record['nodes']])
    return graph_contract(graph, selected_model)


def checked_export_contract(metadata, contract):
    if (metadata.get('common_model') != contract['model']
            or metadata.get('common_model_contract') != contract
            or metadata.get('function') != contract['model']
            or metadata.get('graph_nodes') != contract['graph_nodes']
            or metadata.get('input_shape') != contract['input_shape']
            or metadata.get('output_shape') != contract['output_shape']
            or len(metadata.get('approximations', [])) != contract['activation_count']):
        raise ValueError('Common-model export and original graph contract differ')
    pools=[]
    means=[]
    for node in contract['nodes']:
        if node['op']=='avgpool':
            pools.append(dict(avgpool_spec(node['input_shapes'][0],node['attrs'],node['output_shapes'][0]),
                              original_node=node['id'],input=node['inputs'][0],output=node['outputs'][0]))
        elif node['op']=='mean' and metadata.get('exact_spatial_mean_matvec'):
            shape=node['input_shapes'][0]
            if node['attrs'].get('axes')!=[2,3] or len(shape)!=4 or node['output_shapes'][0]!=shape[:2]:
                raise ValueError('Common-model fixed spatial mean source differs')
            means.append(dict(original_node=node['id'],input=node['inputs'][0],output=node['outputs'][0],
                input_shape=shape,output_shape=shape[:2],axes=[2,3],coefficient=1/math.prod(shape[2:]),
                matrix_shape=[shape[1],math.prod(shape[1:])],nonzeros=math.prod(shape[1:]),
                trainable_weights_changed=False,bias=False,
                formula='W[c,c*H*W+p]=1/(H*W), p=0..H*W-1; all other coefficients0'))
    if metadata.get('fixed_local_avgpool_lowerings')!=pools or metadata.get('exact_spatial_mean_lowerings',[])!=means:
        raise ValueError('Common-model derived public pool source inventory differs')
    return contract


def expected_constant_names(contract, metadata):
    names=[]
    selectors={s['original_nodes'][0]:s for s in metadata.get('exact_optiona_lowerings',[])}
    for node in contract['nodes']:
        a=node['attrs']
        if node['id'] in selectors:
            names.append('derived-exact-optiona:'+selectors[node['id']]['original_nodes'][-1])
        elif node['op'] in ('conv2d','linear'):
            names.append(a['weight'])
            if a.get('bias'):names.append(a['bias'])
        elif node['op']=='batchnorm':names.extend((node['id']+':scale',node['id']+':shift'))
        elif node['op']=='avgpool':names.append('derived-exact-local-avgpool:'+node['id'])
        elif node['op']=='mean' and metadata.get('exact_spatial_mean_matvec'):
            names.append('derived-exact-spatial-mean:'+node['id'])
    return names


def checked_math_record(adapter, contract, metadata):
    worker=adapter.get('worker') or {}
    if (adapter.get('status')!='ok' or adapter.get('model')!=contract['model']
            or worker.get('status')!='ok' or worker.get('common_model')!=contract['model']
            or worker.get('common_model_contract')!=contract
            or worker.get('activation_pairs')!=3*contract['activation_count']
            or worker.get('reference_forward_completed')!=3 or worker.get('adapter_forward_completed')!=3
            or worker.get('quality_selection') is not False):
        raise ValueError('Common-model saved arithmetic provenance or counts differ')
    constants=worker.get('constants',[])
    if ([r.get('name') for r in constants]!=expected_constant_names(contract,metadata)
            or not all(r.get('passed') is True and r.get('numerically_equal') is True for r in constants)):
        raise ValueError('Common-model original and derived constant inventory differs')
    references=worker.get('probes') or []
    if [r.get('sample_id') for r in references]!=['train:0','train:1','train:2']:
        raise ValueError('Common-model original training probe identities differ')
    for row in references:
        ref=row.get('reference')
        if (not isinstance(ref,list) or len(ref)!=1 or not isinstance(ref[0],list)
                or len(ref[0])!=contract['output_shape'][1]
                or not all(type(x) in (int,float) and math.isfinite(x) for x in ref[0])
                or row.get('logit_check',{}).get('passed') is not True
                or len(row.get('activation_checks',[]))!=contract['activation_count']
                or not all(r.get('passed') is True for r in row['activation_checks'])):
            raise ValueError('Common-model original strict reference or activation checks differ')
    return references


def checked_saved_frontend_identity(saved, program, selected_model):
    expected = 'pt_agent.experiments.r13_common_models:' + selected_model
    worker = saved.get('worker') or {}
    if (selected_model not in COMMON_MODELS or saved.get('status')!='ok' or saved.get('round') not in ('R17','R17a')
            or worker.get('status')!='ok' or worker.get('heldout_acceptance') is not True
            or worker.get('completed_samples')!=1000 or worker.get('math_probes_passed') is not True
            or worker.get('rule_options')!={'deg':63,'margin':1.5,'slots':16384,'placement':'depth-dp','refresh':'compiler','act_method':'cheb'}
            or saved.get('model') != expected or worker.get('model') != expected
            or Path(worker.get('artifacts', {}).get('program', '')).resolve() != Path(program).resolve()):
        raise ValueError('Selected common model and saved frontend provenance differ')


def avgpool_spec(shape, attrs, expected):
    shape, expected = tuple(shape), tuple(expected)
    if (len(shape) != 4 or shape[0] != 1
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError('Local average pool requires positive batch-one NCHW')
    if (set(attrs) - {'kernel', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override'}
            or attrs.get('ceil_mode', False) is not False
            or attrs.get('count_include_pad', True) is not True
            or attrs.get('divisor_override') is not None):
        raise ValueError('Local average pool requires floor output, include-pad and original divisor')
    def pair(key, minimum):
        value = attrs.get(key)
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) is not int or x < minimum for x in value)):
            raise ValueError('Local average pool ' + key + ' differs')
        return tuple(value)
    kernel, stride, padding = pair('kernel', 1), pair('stride', 1), pair('padding', 0)
    if any(p > k // 2 for p, k in zip(padding, kernel)):
        raise ValueError('Local average pool padding exceeds the original supported range')
    output = (1, shape[1], *((n + 2*p - k) // s + 1 for n, p, k, s in zip(shape[2:], padding, kernel, stride)))
    if min(output) <= 0 or output != expected:
        raise ValueError('Local average pool output shape differs')
    return dict(input_shape=list(shape), output_shape=list(output), kernel=list(kernel),
                stride=list(stride), padding=list(padding), divisor=math.prod(kernel),
                ceil_mode=False, count_include_pad=True, divisor_override=None,
                representation='fixed channel-diagonal convolution',
                matrix_shape=[shape[1], shape[1], *kernel], nonzeros=shape[1]*math.prod(kernel),
                trainable_weights_changed=False)


def avgpool_weights(shape, attrs, expected):
    import numpy as np
    spec = avgpool_spec(shape, attrs, expected)
    if math.prod(spec['matrix_shape']) > 16 * 1024**2:
        raise ValueError('Fixed average pool public coefficients exceed the existing source bound')
    weight = np.zeros(spec['matrix_shape'], dtype=np.float64)
    for c in range(shape[1]):
        weight[c, c, :, :] = 1 / spec['divisor']
    weight.setflags(write=False)
    return weight, spec


def reshape_spec(shape, attrs, expected):
    shape, expected = tuple(shape), tuple(expected)
    if (set(attrs) != {'shape'} or tuple(attrs['shape']) != expected
            or len(shape) < 2 or shape[0] != 1
            or any(type(x) is not int or x <= 0 for x in (*shape, *expected))
            or expected != (shape[0], math.prod(shape[1:]))):
        raise ValueError('Static C-order reshape requires the original batch-preserving flatten')
    return dict(input_shape=list(shape), output_shape=list(expected), order='C',
                reassociation=[[0], list(range(1, len(shape)))])
