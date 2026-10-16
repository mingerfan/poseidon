"""Stream a validated native trace into literal CKKS MLIR and audit sidecars.

The output is a real-compiler input, not an encrypted-execution result. F64
scales are captured as exact per-operation metadata. Constants refer to the
lossless payload table and remain lazy; this tool never expands 32768-value CST
vectors in memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def f64(bits):
    return struct.unpack('<d', struct.pack('<Q', bits))[0]


class Writer:
    def __init__(self, output, graph, args, constants, scales, mapping, function):
        self.out, self.graph, self.args = output, graph, args
        self.constants, self.scales, self.mapping, self.function = constants, scales, mapping, function
        self.scales_file = scales
        self.serial = self.cipher_regs = self.plain_regs = self.instructions = 0
        self.op_counts = {}
        self.stage = ''

    @staticmethod
    def ty(q, plain=False, components=2):
        return f'tensor<1x!ckks.poly<{1 if plain else components} * {q-1}>>'

    def emit(self, kind, operands, q, source_id, attrs='', plain=False,
             components=2, constant=None, rhs=None):
        raw_start = self.instructions
        self.serial += 1
        temp, result = f'%buf{self.serial}', f'%v{self.serial}'
        ty = self.ty(q, plain, components)
        reg = self.plain_regs if plain else self.cipher_regs
        if plain:
            self.plain_regs += 1
        else:
            self.cipher_regs += 1
        self.out.write(f'    {temp} = tensor.empty() : {ty}\n')
        self.instructions += 1  # allocator sentinel
        values = ', '.join([temp] + [x[0] for x in operands])
        types = ', '.join([ty] + [self.ty(x[2], x[3], x[4]) for x in operands])
        self.out.write(f'    {result} = "ckks.{kind}"({values}) {attrs} : ({types}) -> {ty}\n')
        self.instructions += 1
        count = self.op_counts.get(kind, 0) + 1
        self.op_counts[kind] = count
        raw = dict(source_id=source_id, stage=self.stage, op=kind, q=q, raw_dst=reg,
                   plain=plain, components=components,
                   raw_start=raw_start, raw_end=self.instructions,
                   hevm_instruction_index=raw_start + 1,
                   operands=[dict(register=x[1], plain=x[3], q=x[2],
                                  scale_bits=f'{self.scales[(x[3], x[1])]:016x}')
                             for x in operands])
        if constant is not None:
            raw['constant_index'] = constant
        if rhs is not None:
            raw['parameter'] = rhs
        self.mapping.write(json.dumps(raw, separators=(',', ':')) + '\n')
        if hasattr(self, 'current_scale_bits'):
            self.scales[(plain, reg)] = self.current_scale_bits
        return result, reg, q, plain, components

    def scale_edge(self, value, edge, owner):
        q = edge['q']
        if q != value[2]:
            raise ValueError(f"Q mismatch at op{owner}: producer {value[2]}, edge {q}")
        if edge['scale_bits'] == self.scales[(value[3], value[1])]:
            return value
        target = f64(edge['scale_bits'])
        attrs = f'<{{scale = {target:.17e} : f64}}>'
        previous = getattr(self, 'current_scale_bits', None)
        self.current_scale_bits = edge['scale_bits']
        result = self.emit('setscalec', [value], q, owner, attrs, components=value[4])
        if previous is None:
            del self.current_scale_bits
        else:
            self.current_scale_bits = previous
        return result

    def encode(self, row, q):
        index = self.constants[row['id']]
        scale_bits = row['scale_bits']
        scale = f64(scale_bits)
        log_scale = round(__import__('math').log2(scale))
        attrs = (f'<{{value = {index} : i64, scale = {log_scale} : i64, '
                 f'level = {q-1} : i64}}> {{poseidon_native_scale = {scale:.17e} : f64}}')
        result = self.emit('encode', [], q, row['id'], attrs, plain=True,
                           constant=index, rhs=((q-1) << 10) | log_scale)
        self.scales[(True, result[1])] = scale_bits
        bits = f'{scale_bits:016x}'
        self.scales_file.write(f'{result[1]} {index} {q-1} {bits}\n')
        return result


def lower(trace, payload, prefix, model):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    expected = [prefix.with_suffix(s) for s in
                ('.native_constants.jsonl', '.mlir', '.native_trace_map.jsonl',
                 '.native_expected_scales', '.lowering_report.json')]
    if any(path.exists() for path in expected):
        raise FileExistsError('refusing to overwrite existing lowering artifacts')
    # The first pass fixes function arguments and all table indices before any
    # compiler IR is emitted; values are input layouts plus cached encrypted ONE.
    args, outputs, constants = [], None, {}
    table = (prefix.with_suffix('.native_constants.jsonl')).open('x')
    with trace.open() as src:
        for line in src:
            row = json.loads(line)
            kind = row['op']
            if kind in ('input', 'encrypted_constant'):
                args.append(row)
            elif kind in ('encode_vector', 'encode_scalar'):
                idx = len(constants)
                constants[row['id']] = idx
                table.write(json.dumps(dict(index=idx, source_id=row['id'], op=kind,
                    scale_bits=row['scale_bits'], value_bits=row.get('value_bits'),
                    elements=row.get('elements'), payload_offset=row.get('payload_offset'),
                    payload_compressed_bytes=row.get('payload_compressed_bytes'),
                    fnv64_words=row.get('fnv64_words')), separators=(',', ':'))+'\n')
            elif kind == 'outputs':
                outputs = row['values']
    table.close()
    if len(args) != 148 or not outputs or len(outputs) != 1000:
        raise ValueError(f'expected 147 image layouts, encrypted ONE and 1000 outputs; got {len(args)}')
    if sum(x['op'] == 'input' for x in args) != 147 or sum(x['op'] == 'encrypted_constant' for x in args) != 1:
        raise ValueError('native input/encrypted constant contract changed')

    mlir = prefix.with_suffix('.mlir').open('x')
    mapping = prefix.with_suffix('.native_trace_map.jsonl').open('x')
    scale_file = prefix.with_suffix('.native_expected_scales').open('x')
    graph_map, expected_scales = {}, {}
    for i, row in enumerate(args):
        q = row['q']
        graph_map[row['id']] = (f'%arg{i}', i, q, False, 2)
        expected_scales[(False, i)] = row['scale_bits']
    for i, row in enumerate(args):
        scale_file.write(f'{i} -1 {row["q"]-1} {row["scale_bits"]:016x}\n')
    arg_types = ', '.join(f'%arg{i}: {Writer.ty(x["q"])}' for i, x in enumerate(args))
    arg_scales = ', '.join(str(40) for _ in args)
    arg_levels = ', '.join(str(x['q']-1) for x in args)
    result_types = ', '.join(Writer.ty(4) for _ in outputs)
    mlir.write('module {\n')
    mlir.write(f'  func.func @native_resnet{model}({arg_types}) -> ({result_types}) '
        f'attributes {{poseidon.hevm_wide, arg_scale = array<i64: {arg_scales}>, '
        f'arg_level = array<i64: {arg_levels}>, res_scale = array<i64: '+', '.join(['40']*1000)+
        f'>, res_level = array<i64: '+', '.join(['3']*1000)+'>, init_level = 49 : i64} {\n')
    writer = Writer(mlir, trace, args, constants, scale_file, mapping, model)
    # HEVM reserves the first ciphertext registers for function arguments.
    # Reusing 0 here aliases input scale metadata with produced SSA values.
    writer.cipher_regs = len(args)
    writer.scales = expected_scales
    # Emit all 5M-scale programs incrementally; only SSA ids/metadata stay in memory.
    with trace.open() as src:
        for line in src:
            row = json.loads(line)
            kind, origin, q = row['op'], row.get('id'), row.get('q')
            writer.stage = row.get('stage', 'setup')
            writer.current_scale_bits = row.get('scale_bits', 0)
            if kind == 'outputs':
                continue
            if kind in ('input', 'encrypted_constant'):
                continue
            ins = [writer.scale_edge(graph_map[e['id']], e, origin) for e in row['inputs']]
            attr = row
            if kind in ('encode_vector', 'encode_scalar'):
                value = writer.encode(row, q)
            elif kind == 'rotate':
                step = attr['step']
                value = writer.emit('rotatec', ins, q, origin,
                    f'<{{offset = array<i64: {step}>}}>', rhs=step)
            elif kind == 'negate':
                value = writer.emit('negatec', ins, q, origin)
            elif kind in ('add', 'sub'):
                if kind == 'sub':
                    ins[1] = writer.emit('negatec', [ins[1]], q, origin)
                value = writer.emit('addcc', ins, q, origin)
            elif kind in ('add_plain', 'sub_plain'):
                if kind == 'sub_plain':
                    ins[0] = writer.emit('negatec', [ins[0]], q, origin)
                value = writer.emit('addcp', ins, q, origin)
                if kind == 'sub_plain':
                    value = writer.emit('negatec', [value], q, origin)
            elif kind == 'multiply_integer':
                n = row['integer']
                if not 0 <= n <= 0xffffffff:
                    raise ValueError(f'integer multiplier outside uint32 at {origin}')
                value = writer.emit('multiplyscalarc', ins, q, origin,
                    f'<{{multiplier = {n} : i64}}>', rhs=n)
            elif kind == 'set_scale':
                scale = f64(row['scale_bits'])
                value = writer.emit('setscalec', ins, q, origin,
                    f'<{{scale = {scale:.17e} : f64}}>')
            elif kind == 'drop':
                down = ins[0][2] - q
                value = writer.emit('modswitchc', ins, q, origin,
                    f'<{{downFactor = {down} : i64}}>', rhs=down)
            elif kind == 'rescale_many':
                count = row['count']
                if count == 1:
                    value = writer.emit('rescalec', ins, q, origin)
                else:
                    value = writer.emit('rescalemanyc', ins, q, origin,
                        f'<{{count = {count} : i64}}>', rhs=count)
            elif kind == 'bootstrap_native_api':
                value = writer.emit('bootstrapc', ins, q, origin,
                    f'<{{level = {q-1} : i64}}>', rhs=q-1)
            elif kind == 'multiply_plain':
                value = writer.emit('mulcp', ins, q, origin)
            elif kind == 'multiply_raw':
                value = writer.emit('rawmulcc', ins, q, origin,
                                    components=3)
            elif kind == 'relinearize':
                value = writer.emit('relinearizec', ins, q, origin)
            else:
                raise ValueError(f'unsupported native trace opcode {kind} at {origin}')
            graph_map[origin] = value
            writer.scales[(value[3], value[1])] = row['scale_bits']
            if len(graph_map) % 250000 == 0:
                mlir.flush(); mapping.flush(); scale_file.flush()
    result_values = [graph_map[i][0] for i in outputs]
    result_value_types = ', '.join(Writer.ty(graph_map[i][2]) for i in outputs)
    mlir.write('    return '+', '.join(result_values)+' : '+result_value_types+'\n  }\n}\n')
    mlir.close(); mapping.close(); scale_file.close()
    report = dict(status='real_compiler_input_emitted', model=model,
        source_sha256=sha(trace), payload_sha256=sha(payload), mlir_sha256=sha(prefix.with_suffix('.mlir')),
        constants_sha256=sha(prefix.with_suffix('.native_constants.jsonl')),
        trace_map_sha256=sha(prefix.with_suffix('.native_trace_map.jsonl')),
        inputs=147, encrypted_constant_inputs=1, outputs=1000,
        source_operations=sum(writer.op_counts.values()), compiler_instructions=writer.instructions,
        plain_registers=writer.plain_regs, cipher_registers=writer.cipher_regs,
        constants=len(constants), operations=writer.op_counts,
        wide_format='poseidon.hevm_wide / V2 uint32 records',
        note='not yet compiled by real Hecate compiler; no encrypted execution')
    prefix.with_suffix('.lowering_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=int, choices=(18, 50))
    p.add_argument('trace', type=Path)
    p.add_argument('payload', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    lower(a.trace, a.payload, a.output, a.model)
