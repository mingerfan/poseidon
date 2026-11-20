"""Data-only exact coefficient cover for Conv1d/Conv2d component models.

This verifies a mathematical partition's stored coefficients and geometry. It
does not execute a model, qualify an Agent package, or grant Runtime authority.
"""
import hashlib
import itertools
import json
import math
import re
import struct

MAX_FILE=16*1024**2
MAX_ELEMENTS=1_000_000
MAX_POSITIONS=4096
MAX_PARTS=9
MAX_DESCRIPTOR=128*1024


def need(ok,why):
    if not ok:raise ValueError(why)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def integer(value,lo,hi,why):
    need(type(value) is int and lo<=value<=hi,why)


def exact(value,keys,why):
    need(type(value) is dict and set(value)==set(keys),why)


def loads(raw):
    need(type(raw) is bytes and len(raw)<=MAX_FILE,'bounded model bytes')
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value,'duplicate model JSON key')
            value[key]=item
        return value
    def number(text):
        value=float(text)
        need(math.isfinite(value),'finite JSON number')
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_float=number,
        parse_constant=lambda _:need(False,'finite JSON number'))


def name(value):
    need(type(value) is str and 1<=len(value)<=128,'bounded graph name')
    return value


def tensor(value,rank):
    """Validate every row, preserving exact semantic binary64 including -0."""
    shape=[];leaf=value
    for _ in range(rank):
        need(type(leaf) is list and leaf,'nonempty tensor rank')
        shape.append(len(leaf));leaf=leaf[0]
    need(type(leaf) in (int,float) and math.prod(shape)<=MAX_ELEMENTS,'tensor work/rank bound')
    data=bytearray()
    def append(node,depth):
        if depth<rank:
            need(type(node) is list and len(node)==shape[depth],'rectangular tensor shape')
            for item in node:append(item,depth+1)
        else:
            need(type(node) in (int,float),'numeric tensor coefficient')
            try:finite=math.isfinite(node);raw=struct.pack('<d',node)
            except (OverflowError,struct.error):finite=False;raw=b''
            need(finite,'finite binary64 tensor coefficient')
            data.extend(raw)
    append(value,0)
    return shape,bytes(data)


def geometry(parent,node):
    exact(node,{'id','op','inputs','outputs','attrs'},'source convolution node fields')
    name(node['id'])
    need(node['op'] in ('conv1d','conv2d'),'source convolution operator')
    rank=1 if node['op']=='conv1d' else 2
    need(type(node['inputs']) is list and len(node['inputs']) in (2,3) and
        len(set(node['inputs']))==len(node['inputs']),'source convolution operands')
    for item in node['inputs']:name(item)
    need(type(node['outputs']) is list and len(node['outputs'])==1,'source one output')
    name(node['outputs'][0])
    need(type(parent['inputs']) is list and 1<=len(parent['inputs'])<=4,'source input declarations')
    matches=[v for v in parent['inputs'] if type(v) is dict and v.get('name')==node['inputs'][0]]
    need(len(matches)==1,'convolution input must be a declared source input')
    input_spec=matches[0];exact(input_spec,{'name','shape'},'input geometry fields')
    shape=input_spec['shape']
    need(type(shape) is list and len(shape)==rank+2,'convolution input rank')
    for n in shape:integer(n,1,65536,'positive input dimension')
    need(math.prod(shape)<=MAX_ELEMENTS,'input geometry work bound')
    attrs=node['attrs']
    need(type(attrs) is dict and set(attrs)<={'stride','padding','dilation','groups'},'convolution geometry attrs')
    fields={}
    for field,default,lower in [('stride',1,1),('padding',0,0),('dilation',1,1)]:
        values=attrs.get(field,[default]*rank)
        need(type(values) is list and len(values)==rank,'convolution attribute rank')
        for value in values:integer(value,lower,65536,'convolution attribute value')
        fields[field]=values
    groups=attrs.get('groups',1);integer(groups,1,65536,'convolution groups')
    constants=parent['constants']
    need(type(constants) is dict and all(key in constants for key in node['inputs'][1:]),'public source weights/bias')
    weight_shape,weight=tensor(constants[node['inputs'][1]],rank+2)
    need(weight_shape[1]*groups==shape[1] and weight_shape[0]%groups==0,'grouped channel geometry')
    bias=None
    if len(node['inputs'])==3:
        bias_shape,bias=tensor(constants[node['inputs'][2]],1)
        need(bias_shape==[weight_shape[0]],'source bias shape')
    output=[shape[0],weight_shape[0]]
    for n,k,s,p,d in zip(shape[2:],weight_shape[2:],fields['stride'],fields['padding'],fields['dilation']):
        extent=(n+2*p-d*(k-1)-1)//s+1
        need(extent>0,'positive convolution output extent')
        output.append(extent)
    return rank,input_spec,weight_shape,weight,bias,output


def verify_cover(parent_raw,*,source_node_id,pieces):
    """Each piece declares its source kernel positions; stored masks prove them."""
    parent=loads(parent_raw)
    required={'format','id','inputs','constants','nodes','outputs'}
    need(type(parent) is dict and set(parent) in (required,required|{'resource_profile'}) and
        parent['format']=='poseidon-model-graph-v2','source model fields/format')
    name(parent['id']);name(source_node_id)
    need(type(parent['nodes']) is list and 1<=len(parent['nodes'])<=1024,'source node work bound')
    ids=[v.get('id') for v in parent['nodes'] if type(v) is dict]
    need(len(ids)==len(parent['nodes']) and all(type(x) is str for x in ids) and
        len(set(ids))==len(ids),'unique source node identities')
    matches=[v for v in parent['nodes'] if v['id']==source_node_id]
    need(len(matches)==1,'unique selected source node')
    source=matches[0]
    rank,input_spec,weight_shape,weight,bias,output_shape=geometry(parent,source)
    kernel=weight_shape[2:];position_count=math.prod(kernel)
    need(position_count<=MAX_POSITIONS,'kernel coordinate work bound')
    need(type(pieces) is list and 2<=len(pieces)<=MAX_PARTS,'partition count2..9')
    universe=set(itertools.product(*(range(n)for n in kernel)))
    covered=set();records=[];model_ids=set();model_hashes=set();bias_owners=[];output_name=None
    zero=b'\0'*8
    for index,piece in enumerate(pieces):
        exact(piece,{'index','model_raw','kernel_positions'},'partition evidence fields')
        integer(piece['index'],index,index,'stable partition index')
        positions=piece['kernel_positions']
        need(type(positions) is list and 1<=len(positions)<=position_count,'nonempty partition coordinates')
        selected=set()
        for coord in positions:
            need(type(coord) is list and len(coord)==rank,'kernel coordinate rank')
            for value,extent in zip(coord,kernel):integer(value,0,extent-1,'kernel coordinate value')
            point=tuple(coord)
            need(point not in selected and point not in covered,'duplicate kernel coordinate')
            selected.add(point)
        need([tuple(v)for v in positions]==sorted(selected),'stable coordinate order')
        covered.update(selected)
        model=loads(piece['model_raw'])
        need(type(model) is dict and set(model)==set(parent),'component model fields')
        need(model['format']==parent['format'] and model.get('resource_profile')==parent.get('resource_profile'),
             'component profile/format unchanged')
        model_id=name(model['id']);need(model_id not in model_ids,'unique component model ID');model_ids.add(model_id)
        model_hash=sha(canonical(model));need(model_hash not in model_hashes,'unique component model bytes');model_hashes.add(model_hash)
        need(canonical(model['inputs'])==canonical([input_spec]),'exact source input geometry')
        need(type(model['nodes']) is list and len(model['nodes'])==1,'no operation or activation before sum')
        node=model['nodes'][0]
        exact(node,{'id','op','inputs','outputs','attrs'},'component convolution node fields')
        name(node['id'])
        need(type(node['outputs']) is list and len(node['outputs'])==1,'component one direct output')
        name(node['outputs'][0])
        includes_bias=len(source['inputs'])==3 and node['inputs']==source['inputs']
        expected=dict(source,id=node['id'],outputs=node['outputs'],
            inputs=source['inputs'] if includes_bias else source['inputs'][:2])
        need(canonical(node)==canonical(expected),'original operation/orientation/stride/padding/dilation/groups')
        expected_constants=source['inputs'][1:3] if includes_bias else source['inputs'][1:2]
        need(type(model['constants']) is dict and set(model['constants'])==set(expected_constants),
             'only original public convolution constants')
        need(type(model['outputs']) is list and len(model['outputs'])==1,'one logical component output')
        declaration=model['outputs'][0];exact(declaration,{'name','value'},'component output binding')
        logical_name=name(declaration['name'])
        need(declaration['value']==node['outputs'][0],'output is the unactivated convolution')
        if output_name is None:output_name=logical_name
        need(logical_name==output_name,'corresponding logical output name')
        child_shape,child_weight=tensor(model['constants'][source['inputs'][1]],rank+2)
        need(child_shape==weight_shape,'component weight shape')
        selected_indices=set()
        for coord in selected:
            flat=0
            for value,extent in zip(coord,kernel):flat=flat*extent+value
            selected_indices.add(flat)
        for offset in range(0,len(weight),8):
            expected=weight[offset:offset+8] if (offset//8)%position_count in selected_indices else zero
            need(child_weight[offset:offset+8]==expected,'exact coefficient or positive-zero mask')
        if includes_bias:
            shape,child_bias=tensor(model['constants'][source['inputs'][2]],1)
            need(shape==[weight_shape[0]] and child_bias==bias,'exact original bias bytes')
            bias_owners.append(index)
        records.append(dict(index=index,model_id=model_id,model_sha256=model_hash,
            model_raw_sha256=sha(piece['model_raw']),kernel_positions=[list(p)for p in positions],
            weight_f64_sha256=sha(child_weight),includes_original_bias=includes_bias))
    need(covered==universe,'complete kernel coordinate cover')
    need(len(bias_owners)==(1 if bias is not None else 0),'original bias exactly once')
    result=dict(format='poseidon-exact-public-Conv-kernel-cover-v1',
        source_model_raw_sha256=sha(parent_raw),source_model_sha256=sha(canonical(parent)),
        source_node_id=source_node_id,source_operator=source['op'],source_weight=source['inputs'][1],
        source_weight_shape=weight_shape,source_weight_f64_sha256=sha(weight),
        source_coefficient_count=len(weight)//8,kernel_positions_covered=len(covered),
        input=input_spec,output=dict(name=output_name,shape=output_shape),
        preserved_attributes=source['attrs'],bias_owners=bias_owners,pieces=records,
        algebraic_cover=True,mathematical_forward_executed=False,execution_authority=False,
        complete_source_graph_accepted=False,numerical_sum_accepted=False)
    need(len(canonical(result))<=MAX_DESCRIPTOR,'partition descriptor byte bound')
    result['binding']=sha(canonical(result))
    return result


def qualification_gaps(cover,qualified_model_hashes):
    """Coverage accounting only; the caller separately verifies genuine receipts."""
    need(type(cover) is dict and cover.get('binding')==sha(canonical({k:v for k,v in cover.items()if k!='binding'})),
         'coefficient cover binding')
    need(cover.get('format')=='poseidon-exact-public-Conv-kernel-cover-v1' and
         cover.get('execution_authority') is False,'coefficient cover kind/scope')
    rows=cover.get('pieces')
    need(type(rows) is list and 2<=len(rows)<=MAX_PARTS,'coefficient cover part bound')
    identities=[]
    for index,row in enumerate(rows):
        need(type(row) is dict,'coefficient cover part record')
        integer(row.get('index'),index,index,'coefficient cover stable index')
        value=row.get('model_sha256')
        need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None and
             value not in identities,'coefficient cover unique model identity')
        identities.append(value)
    need(type(qualified_model_hashes) is list and len(qualified_model_hashes)<=MAX_PARTS,'qualification list bound')
    for value in qualified_model_hashes:
        need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'qualified model hash')
    need(len(set(qualified_model_hashes))==len(qualified_model_hashes),'duplicate qualified model identity')
    known=set(identities)
    need(set(qualified_model_hashes)<=known,'unrelated qualified model')
    missing=[p['index']for p in cover['pieces']if p['model_sha256']not in qualified_model_hashes]
    return dict(qualified_pieces=len(qualified_model_hashes),missing_piece_indices=missing,
        qualified_model_coverage_complete=not missing,execution_authority=False,
        requires_original_package_and_source_admission=True)
