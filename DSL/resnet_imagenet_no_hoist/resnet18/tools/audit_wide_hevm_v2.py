"""Audit the real Hecate V2 HEVM bytes against trace/compiler sidecars."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def fnv(path):
    value = 14695981039346656037
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def audit(hevm, scales, constants, lowering, trace_map, native_trace, report_path):
    raw = hevm.read_bytes()
    magic, header_size, argc, resultc = struct.unpack_from('<IIQQ', raw)
    body_len, count, cipherc, plainc, init_level = struct.unpack_from('<QQQQQ', raw, header_size)
    if magic != 0x48455632:
        raise ValueError('not an opt-in wide HEVM file')
    body = raw[header_size + 40:header_size + body_len]
    values = struct.unpack('<' + 'Q' * (len(body)//8), body)
    if len(values) != argc*2 + resultc*2 + resultc:
        raise ValueError('header array lengths do not match function signature')
    arg_scales, arg_levels = values[:argc], values[argc:argc*2]
    res_scales = values[argc*2:argc*2+resultc]
    res_levels = values[argc*2+resultc:argc*2+resultc*2]
    ret_dst = values[-resultc:]
    if argc != 148 or resultc != 1000 or init_level != 49:
        raise ValueError('native ResNet signature/config mismatch')
    if any(x != 40 for x in arg_scales+res_scales) or any(x != 31 for x in arg_levels) or any(x != 3 for x in res_levels):
        raise ValueError('Q/scale contract mismatch')
    inst_off = header_size + body_len
    if len(raw) != inst_off + count*16:
        raise ValueError('wide operation stream length mismatch')
    instructions = list(struct.iter_unpack('<IIII', raw[inst_off:]))
    ops = Counter(x[0] for x in instructions if x[0] != 0xffffffff)
    if not instructions or max(x[1] for x in instructions) >= cipherc or cipherc < 65536:
        raise ValueError('cipher register bound not exercised or out of range')
    if max(x[1] for x in instructions if x[0] == 0) >= plainc:
        raise ValueError('plaintext register outside table')
    sidecar = scales.read_text().splitlines()
    if len(sidecar) < 1 or sidecar[0].split()[0] != 'POSEIDON_HEVM_NATIVE_SCALE_V1':
        raise ValueError('native scale sidecar header missing')
    _, declared, bound_hash = sidecar[0].split()
    if int(declared) != count or int(bound_hash, 16) != fnv(hevm):
        raise ValueError('native scale sidecar is not byte-bound')
    scale_rows = 0
    for line in sidecar[1:]:
        index, constant, level, _bits = line.split()
        op = instructions[int(index)]
        if op[0] != 0 or op[2] != int(constant) or op[3] >> 10 != int(level):
            raise ValueError('native scale row does not identify its encoded constant')
        scale_rows += 1
    const_rows = sum(1 for _ in constants.open())
    lowering_doc = json.loads(lowering.read_text())
    # The v5 trace-map binds emitted destinations to source IDs/stages but
    # predates an explicit `op` field. Count bootstrap locations from the
    # original native trace, whose SHA is already recorded by the lowering
    # report, then compare those calls with the actual emitted opcode stream.
    source_hash = hashlib.sha256()
    trace_map_hash = hashlib.sha256()
    trace_map_rows = 0
    with trace_map.open('rb') as mapping:
        for chunk in iter(lambda: mapping.read(1 << 20), b''):
            trace_map_hash.update(chunk)
            trace_map_rows += chunk.count(b'\n')
    if trace_map_rows != lowering_doc['compiler_instructions'] // 2:
        raise ValueError('trace-map row count does not match lowering instruction pairs')
    if trace_map_hash.hexdigest() != lowering_doc['trace_map_sha256']:
        raise ValueError('trace-map SHA does not match lowering report')
    positions = set()
    native_boot_calls = 0
    with native_trace.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            source_hash.update(chunk)
    if source_hash.hexdigest() != lowering_doc['source_sha256']:
        raise ValueError('native trace SHA does not match lowering source')
    with native_trace.open() as source:
        for line in source:
            row = json.loads(line)
            if row.get('op') == 'bootstrap_native_api':
                positions.add(row['stage'])
                native_boot_calls += 1
    expected_positions, expected_calls = ((16, 60) if lowering_doc['model'] == 18 else (48, 351))
    if (const_rows != lowering_doc['constants'] or ops[10] != expected_calls or
        native_boot_calls != expected_calls or len(positions) != expected_positions or
        scale_rows != ops[0] or ops[0] != const_rows):
        raise ValueError('constant/bootstrap counts do not match native contract')
    result = dict(status='PASS', hevm_sha256=hashlib.sha256(raw).hexdigest(),
        format='V2 uint32/16-byte ops', args=argc, outputs=resultc, operations=count,
        cipher_registers=cipherc, plaintext_registers=plainc, scale_rows=scale_rows,
        constants=const_rows, bootstrap_positions=len(positions),
        bootstrap_calls=ops[10], result_registers=len(ret_dst),
        native_trace_bootstrap_calls=native_boot_calls,
        opcode_counts=dict(ops), source_graph_sha256=lowering_doc['source_sha256'],
        native_trace_sha256=source_hash.hexdigest(),
        bootstrap_stages=sorted(positions), trace_map_rows=trace_map_rows,
        trace_map_sha256=trace_map_hash.hexdigest(),
        gpu_executed=False, encrypted_runtime=False)
    report_path.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('hevm',type=Path);p.add_argument('scales',type=Path)
    p.add_argument('constants',type=Path);p.add_argument('lowering',type=Path)
    p.add_argument('trace_map',type=Path);p.add_argument('native_trace',type=Path)
    p.add_argument('report',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.hevm,a.scales,a.constants,a.lowering,a.trace_map,a.native_trace,a.report)))
