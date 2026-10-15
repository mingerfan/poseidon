"""Lower audited CPU graph to literal CKKS DSL for the real compiler.

Manifest records primitive expansion. Multi-rescale groups require native runtime
coalescing before encrypted acceptance; this generator does not declare that gate.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def f64(bits):
    return struct.unpack('>d', bytes.fromhex(bits.zfill(16)))[0]


def lower(graph, prefix):
    ops = graph['operations']
    lines, raw, mapping, constants = [], [], {}, []
    cipher_count, plain_count, serial = 27, 0, 0
    q_by_value = {}
    def ty(q, plain=False):
        return f'tensor<1x!ckks.poly<{1 if plain else 2} * {q-1}>>'

    def emit(kind, inputs, q, origin, attrs='', plain=False, rhs=None):
        nonlocal serial, cipher_count, plain_count
        serial += 1
        buf, out = f'%buf{serial}', f'%v{serial}'
        t = ty(q, plain)
        register = plain_count if plain else cipher_count
        if plain:
            plain_count += 1
        else:
            cipher_count += 1
        lines.append(f'    {buf} = tensor.empty() : {t}')
        raw.append(dict(opcode=-1, origin=origin))
        args = ', '.join([buf]+[v[0] for v in inputs])
        types = ', '.join([t]+[ty(v[2], v[3]) for v in inputs])
        lines.append(f'    {out} = "ckks.{kind}"({args}) {attrs} : ({types}) -> {t}')
        codes = {'encode':0,'rotatec':1,'negatec':2,'rescalec':3,'modswitchc':4,
                 'addcc':6,'addcp':7,'mulcc':8,'mulcp':9,'bootstrapc':10}
        raw.append(dict(opcode=codes[kind], dst=register,
                        lhs=rhs[0] if kind=='encode' else inputs[0][1],
                        rhs=rhs[1] if kind=='encode' else (rhs if rhs is not None else inputs[1][1] if len(inputs)>1 else 0),
                        origin=origin, kind=kind))
        q_by_value[out] = q
        return out, register, q, plain

    def drop(v, q, origin):
        return emit('modswitchc', [v], q, origin,
                    f'<{{downFactor = {v[2]-q} : i64}}>', rhs=v[2]-q)

    def encode(op, q):
        attr = op['attributes']
        key = ('scalar', attr['value']) if 'value' in attr else ('payload', attr['payload_offset'], attr['payload_count'])
        if key not in constants:
            constants.append(key)
        index = constants.index(key)
        s = f64(attr['plain_scale_bits'])
        log_scale = round(math.log2(s))
        attrs = f'<{{value = {index} : i64, scale = {log_scale} : i64, level = {q-1} : i64}}> {{poseidon_native_scale = {s:.17e} : f64}}'
        result=emit('encode', [], q, op['id'], attrs, True, (index, ((q-1)<<10)|log_scale))
        raw[-1]['native_scale_bits']=struct.pack('>d',s).hex()
        return result

    groups = []
    arg_index = 0
    for op in ops:
        kind, origin, q, attr = op['kind'], op['id'], op['q_count'], op['attributes']
        inputs = [mapping[i] for i in op['inputs']]
        if kind=='input_layout':
            mapping[origin] = (f'%arg{arg_index}', arg_index, q, False)
            arg_index += 1
            continue
        if kind in ('drop_or_clone', 'drop_to_q_prefix'):
            result = drop(inputs[0], q, origin)
        elif kind=='rotate':
            step=attr['parameter']
            result=emit('rotatec', inputs, q, origin, f'<{{offset = array<i64: {step}>}}>', rhs=step)
        elif kind=='bootstrap':
            result=emit('bootstrapc', inputs, q, origin, f'<{{level = {q-1} : i64}}>', rhs=q-1)
        elif kind in ('rescale', 'rescale_many'):
            count=inputs[0][2]-q
            start=len(raw)
            result=inputs[0]
            for _ in range(count):
                result=emit('rescalec', [result], result[2]-1, origin)
            groups.append(dict(origin=origin, raw_begin=start, raw_end=len(raw), count=count,
                               native_rule='Conv sequential' if kind=='rescale' else 'ReLU rescale_many default x2'))
        elif kind in ('multiply_plain', 'multiply_plain_accumulate', 'multiply_plain_q_prefix'):
            source=inputs[0]
            if source[2]!=q:
                source=drop(source,q,origin)
            result=emit('mulcp',[source,encode(op,q)],q,origin)
            if kind=='multiply_plain_accumulate':
                result=emit('addcc',[inputs[1],result],q,origin)
        elif kind=='multiply_q_prefix_relinearize':
            sources=[v if v[2]==q else drop(v,q,origin) for v in inputs]
            result=emit('mulcc',sources,q,origin)
        elif kind in ('add','sub'):
            sources=inputs
            if kind=='sub':
                sources=[inputs[0],emit('negatec',[inputs[1]],q,origin)]
            result=emit('addcc',sources,q,origin)
        elif kind in ('add_plain','sub_plain'):
            source=inputs[0]
            if kind=='sub_plain':
                source=emit('negatec',[source],q,origin)
            result=emit('addcp',[source,encode(op,q)],q,origin)
            if kind=='sub_plain':
                result=emit('negatec',[result],q,origin)
        else:
            raise ValueError(kind)
        mapping[origin]=result
    assert arg_index==27
    result=mapping[graph['result_id']]
    arguments=', '.join(f'%arg{i}: {ty(32)}' for i in range(27))
    levels=', '.join(['31']*27); scales=', '.join(['40']*27)
    text='module {\n  func.func @strict_native_resnet20('+arguments+') -> '+ty(result[2])+f' attributes {{arg_scale = array<i64: {scales}>, arg_level = array<i64: {levels}>, res_scale = array<i64: 40>, res_level = array<i64: {result[2]-1}>, init_level = 49 : i64}} {{\n'
    text+='\n'.join(lines)+f'\n    return {result[0]} : {ty(result[2])}\n  }}\n}}\n'
    Path(prefix+'.mlir').write_text(text)
    manifest=dict(scope='literal CKKS lowering proposal; pending real compiler audit/runtime native rescale support',
                  raw_operations=raw, constants=constants, rescale_groups=groups,
                  cipher_registers=cipher_count, plain_registers=plain_count,
                  input_count=27, result_register=result[1], source_graph=graph['scope'])
    Path(prefix+'.lowering.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return len(raw), len(constants)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph');parser.add_argument('prefix')
    args=parser.parse_args()
    print(lower(json.loads(Path(args.graph).read_text()),args.prefix))
