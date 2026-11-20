"""Generic pre-generation projection fusion over a public mathematical DAG.

Only static unit-step slices are moved through same-axis concatenation and
shape-preserving elementwise arithmetic. Every other operator keeps its full
original semantics. No model name, sample, compiler response or DSL is consulted.
"""
import copy
import hashlib
import json

from benchmark_graph import validate, require

VERSION = 'poseidon-broadcast-and-transform-projection-fusion-v2'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def normalize(model, *, maximum_nodes=1024):
    require(type(maximum_nodes) is int and 1 <= maximum_nodes <= 1024, 'Finite projection graph budget')
    checked = validate(model); original = canonical(model)
    result = copy.deepcopy(model); result['nodes'] = []
    producers = {value: node for node in model['nodes'] for value in node['outputs']}
    inputs = {spec['name'] for spec in model['inputs']}
    occupied = set(checked['shapes']) | {node['id'] for node in model['nodes']}
    serial = 0; memo = {}; projected = {}; copied = set(); actions = []

    def fresh():
        nonlocal serial
        while True:
            name = 'projection' + str(serial); serial += 1
            if name not in occupied and name + 'v0' not in occupied:
                occupied.add(name); occupied.add(name + 'v0')
                return name

    def emit(op, refs, attrs):
        name = fresh(); output = name + 'v0'
        result['nodes'].append(dict(id=name,op=op,inputs=list(refs),attrs=copy.deepcopy(attrs),outputs=[output]))
        require(len(result['nodes']) <= maximum_nodes, 'Projection fusion node budget')
        return output

    def join(parts, axis):
        require(bool(parts), 'Nonempty projected concatenation')
        while len(parts) > 8:
            parts = [emit('concat',parts[i:i+8],dict(axis=axis)) if len(parts[i:i+8])>1
                     else parts[i] for i in range(0,len(parts),8)]
        return parts[0] if len(parts)==1 else emit('concat',parts,dict(axis=axis))

    def project(value, axis, lo, hi):
        shape=checked['shapes'][value];axis%=len(shape)
        require(0<=lo<hi<=shape[axis], 'Bounded static projection')
        if lo==0 and hi==shape[axis]:return build(value)
        key=value,axis,lo,hi
        if key in projected:return projected[key]
        node=producers.get(value);output=None;kind='materialized_slice'
        if node is not None and node['op']=='concat' and node['attrs']['axis']%len(shape)==axis:
            parts=[];offset=0
            for ref in node['inputs']:
                width=checked['shapes'][ref][axis]
                begin,end=max(lo,offset),min(hi,offset+width)
                if begin<end:parts.append(project(ref,axis,begin-offset,end-offset))
                offset+=width
            output=join(parts,axis);kind='concat_projection'
        elif node is not None and node['op']in('square','negate','power'):
            output=emit(node['op'],[project(node['inputs'][0],axis,lo,hi)],node['attrs']);kind='elementwise_unary_projection'
        elif node is not None and node['op']in('permute','transpose'):
            source=node['inputs'][0];source_shape=list(checked['shapes'][source]);dims=list(range(len(source_shape)))
            if node['op']=='permute':dims=[a%len(dims)for a in node['attrs']['dims']]
            else:
                a,b=node['attrs']['dim0']%len(dims),node['attrs']['dim1']%len(dims);dims[a],dims[b]=dims[b],dims[a]
            source_axis=dims[axis];child=project(source,source_axis,lo,hi);source_shape[source_axis]=hi-lo
            if [a for a in dims if source_shape[a]!=1]==[a for a,n in enumerate(source_shape)if n!=1]:
                output=emit('reshape',[child],dict(shape=[source_shape[a]for a in dims]));kind='transform_projection_to_view'
            else:
                output=emit('permute',[child],dict(dims=dims));kind='transform_projection'
        elif node is not None and node['op']in('add','subtract','multiply'):
            refs=[];safe=True
            for ref in node['inputs']:
                operand=checked['shapes'][ref];mapped=axis-(len(shape)-len(operand))
                needs=(mapped>=0 and operand[mapped]!=1)
                # Public array slices need explicit public-data projection;
                # retain the original full operator for that separate case.
                if checked['kinds'][ref]=='plain'and operand: safe=False;break
                refs.append(project(ref,mapped,lo,hi)if needs else build(ref))
            if safe:
                output=emit(node['op'],refs,node['attrs'])
                kind='elementwise_binary_projection'if all(checked['shapes'][r]in(shape,[])for r in node['inputs'])else'elementwise_broadcast_projection'
        if output is None:
            output=emit('slice',[build(value)],dict(axis=axis,start=lo,stop=hi,step=1))
        projected[key]=output
        actions.append(dict(value=value,axis=axis,start=lo,stop=hi,kind=kind,normalized_value=output))
        return output

    def build(value):
        if value in memo:return memo[value]
        if value in inputs or value in model['constants']:
            memo[value]=value;return value
        node=producers[value]
        if node['op']=='slice' and node['attrs']['step']==1:
            source=node['inputs'][0];shape=checked['shapes'][source];axis=node['attrs']['axis']%len(shape)
            lo,hi,step=slice(node['attrs']['start'],node['attrs']['stop'],1).indices(shape[axis])
            memo[value]=project(source,axis,lo,hi);return memo[value]
        if node['id']not in copied:
            refs=[build(ref)for ref in node['inputs']]
            replacement=copy.deepcopy(node);replacement['inputs']=refs
            result['nodes'].append(replacement)
            require(len(result['nodes'])<=maximum_nodes,'Projection fusion node budget')
            copied.add(node['id'])
            for out in node['outputs']:memo[out]=out
        return memo[value]

    result['outputs']=[dict(name=out['name'],value=build(out['value']))for out in model['outputs']]
    if not result['nodes']:
        first=result['outputs'][0]
        first['value']=emit('reshape',[first['value']],dict(shape=checked['output_shapes'][first['name']]))
    # Aliases can make a node built for another path redundant. Keep only nodes
    # reachable from the final named outputs, with their original topological order.
    by_value={out:n for n in result['nodes']for out in n['outputs']}
    needed=set();pending=[out['value']for out in result['outputs']]
    while pending:
        value=pending.pop();node=by_value.get(value)
        if node is not None and node['id']not in needed:
            needed.add(node['id']);pending.extend(node['inputs'])
    result['nodes']=[n for n in result['nodes']if n['id']in needed]
    final=validate(result)
    require(result['inputs']==model['inputs'] and result['constants']==model['constants']
            and [o['name']for o in result['outputs']]==[o['name']for o in model['outputs']]
            and final['output_shapes']==checked['output_shapes'], 'Projection fusion changed public function signature/weights')
    require(all(final['shapes'][new]==checked['shapes'][old]for old,new in memo.items()if new in final['shapes']),
            'Projected value alias changed shape')
    require(canonical(model)==original,'Projection fusion mutated input')
    receipt=dict(format=VERSION,original_model_sha256=hashlib.sha256(original).hexdigest(),
        normalized_model_sha256=hashlib.sha256(canonical(result)).hexdigest(),maximum_nodes=maximum_nodes,
        original_nodes=len(model['nodes']),normalized_nodes=len(result['nodes']),actions=actions,
        original_named_output_shapes_preserved=True,original_public_weights_preserved=True,model_name_dispatch=False,
        numerical_reference_used=False,compiler_feedback_used=False,DSL_generated=False,resource_limits_increased=False)
    receipt['binding']=hashlib.sha256(canonical(receipt)).hexdigest()
    return result,receipt
