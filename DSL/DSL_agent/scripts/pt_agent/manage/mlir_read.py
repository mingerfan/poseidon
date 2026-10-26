"""S1 §10 generic Earth reader: shared SSA identity survives frontend preprocessing."""

from pathlib import Path
import re
import struct

import numpy as np

from .ir import Function, Node, Type
from .streams import open_text


RAW_OPS = {'constant','add','sub','mul','negate','rotate'}
MANAGED_OPS = RAW_OPS | {'rescale','upscale','modswitch','bootstrap'}
TYPE = re.compile(r'tensor<1x!earth\.(ci|pl)<(\d+)\s*\*\s*(\d+)>>')
VALUE = re.compile(r'%[A-Za-z0-9_]+')


def _types(text):
    return [Type(k=='ci',int(s),int(l)) for k,s,l in TYPE.findall(text)]


def _number(text):
    return struct.unpack('>d',bytes.fromhex(text[2:]))[0] if text.startswith('0x') else float(text)


def read(path, profile, *, locations=None, allowed=None):
    """Read generic/custom function wrappers and reject every unknown operation."""
    allowed = MANAGED_OPS if allowed is None else allowed
    locations = locations or {}
    mapping, inputs, nodes, outputs, attrs, pending, aliases = {},[],[],[],{},[],{}
    name, results = None,[]
    with open_text(path) as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            alias = re.match(r'(#loc\w*) = (.*)',text)
            if alias:
                aliases[alias[1]] = alias[2]
                continue
            if 'func.func @' in text:
                m = re.search(r'func.func @_hecate_([A-Za-z0-9_]+)\((.*?)\)\s*->\s*(.*)',text)
                if m is None:
                    raise ValueError('unsupported Earth function signature')
                name,inputs,results = m[1],_types(m[2]),_types(m[3].split(' attributes ',1)[0])
                mapping.update((v,-i-1) for i,v in enumerate(VALUE.findall(m[2])))
            elif '"func.func"' in text:
                symbol = re.search(r'sym_name = "_hecate_([A-Za-z0-9_]+)"',text)
                signature = re.search(r'function_type = \((.*?)\) -> (.*?)(?:, sym_name|\}>|\})',text)
                if symbol is None or signature is None:
                    raise ValueError('unsupported generic Earth function signature')
                name,inputs,results = symbol[1],_types(signature[1]),_types(signature[2])
            elif text.startswith('^bb'):
                mapping.update((v,-i-1) for i,v in enumerate(VALUE.findall(text.split(' : ',1)[0])))
                continue
            elif text.startswith('return ') or '"func.return"' in text:
                outputs = [mapping[v] for v in VALUE.findall(text.split(' : ',1)[0])]
                continue
            elif '"earth.' in text:
                m = re.search(r'(%[\w]+) = "earth\.([\w]+)"\(([^)]*)\)',text)
                if m is None or m[2] not in allowed:
                    raise ValueError('unsupported Earth operation '+(m[2] if m else text[:100]))
                kind,args,nid = m[2],[mapping[v] for v in VALUE.findall(m[3])],len(nodes)
                mapping[m[1]] = nid
                result = _types(text.split(' -> ',1)[1])
                if len(result)!=1:
                    raise ValueError('Earth operations require one result')
                properties,data = {},None
                if kind=='constant':
                    dense = re.search(r'value = dense<(.*?)> : tensor<(\d+)xf64>',text)
                    if dense is None:
                        raise ValueError('unsupported dense Earth constant')
                    payload,size = dense[1],int(dense[2])
                    if payload.startswith('"0x'):
                        data = np.frombuffer(bytes.fromhex(payload[3:-1]),dtype='<f8').copy()
                    elif payload.startswith('['):
                        data = np.array([_number(v.strip()) for v in payload.strip('[]').split(',')])
                    else:
                        data = np.full(size,_number(payload))
                    if data.size!=size:
                        raise ValueError('dense constant size differs')
                    rms = re.search(r'rms_var = ([^ ]+) : f64',text)
                    if rms is None:
                        raise ValueError('Earth constant lacks rms_var')
                    properties['rms_var'] = _number(rms[1])
                else:
                    for key in ('upFactor','downFactor','targetLevel'):
                        value = re.search(key+r' = (-?\d+) : i64',text)
                        if value:
                            properties[key] = int(value[1])
                    offset = re.search(r'offset = array<i64: ([^>]+)>',text)
                    if offset:
                        properties['offset'] = [int(v) for v in offset[1].split(',')]
                owner = locations.get(nid)
                node = Node(nid,kind,args,result[0],properties,data,getattr(owner,'id',None),getattr(owner,'lineno',0))
                nodes.append(node)
                pending.append((node,text[-240:]))
                continue
            elif text.startswith(('module ','module {','"builtin.module"','})', '}')):
                continue
            else:
                raise ValueError('unsupported MLIR operation or syntax: '+text[:120])
            # Function-level management attributes are absent at stage 0.
            for key in ('arg_scale','res_scale'):
                value = re.search(key+r' = array<i64: ([^>]+)>',text)
                if value:
                    attrs[key] = [int(v) for v in value[1].split(',')]
            value = re.search(r'init_level = (\d+) : i64',text)
            if value:
                attrs['init_level'] = int(value[1])

    def owner_text(text, seen=None):
        seen = set() if seen is None else seen
        match = re.search(r'ptdsl-op-(None|\d+)":(\d+):',text)
        if match:
            return (None if match[1]=='None' else int(match[1]),int(match[2]))
        for alias in re.findall(r'#loc\w*',text):
            if alias not in seen and alias in aliases:
                seen.add(alias)
                found = owner_text(aliases[alias],seen)
                if found:
                    return found
        return None

    for node,loc in pending:
        owner = owner_text(loc)
        if owner:
            node.op_id,node.lineno = owner
    if name is None or not outputs or len(inputs)!=sum(k<0 for k in mapping.values()):
        raise ValueError('missing Earth function, block arguments or return')
    return Function(name,inputs,nodes,outputs,results,attrs,dict(profile))


def load_graph(path, slots, *, locations=None):
    """Return the preprocessed shared DAG in the recording backend's interface."""
    return graph_backend(read(path,{},locations=locations,allowed=RAW_OPS),slots,locations=locations)


def graph_backend(function, slots, *, locations=None):
    """Reuse an already parsed stage-0 function without copying its constants."""
    from .greedy import RecordingBackend, RecordedExpr
    backend = RecordingBackend(slots,locations=locations)
    for i,t in enumerate(function.inputs):
        if t!=Type(True,0,0):
            raise ValueError('unmanaged graph requires ci<0 * 0> inputs')
        backend.input(i)
    backend.nodes = function.nodes
    backend.output([RecordedExpr(backend,v) for v in function.outputs])
    return backend
