"""Validate the actual compiler-emitted optional f64 Encode metadata."""
import argparse
import json
import math
import struct
from pathlib import Path


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte)*1099511628211) & ((1 << 64)-1)
    return value


def validate(data, text):
    lines = text.strip().splitlines()
    version, declared_ops, digest = lines[0].split()
    assert version == 'POSEIDON_HEVM_NATIVE_SCALE_V1'
    magic, header = struct.unpack_from('<II', data)
    assert magic == 0x4845564D
    body, nops = struct.unpack_from('<QQ', data, header)
    assert nops == int(declared_ops) and fnv(data) == int(digest, 16)
    entries, seen = [], set()
    for line in lines[1:]:
        index, constant, level, scale_bits = line.split()
        index, constant, level = int(index), int(constant), int(level)
        assert index < nops and index not in seen
        seen.add(index)
        opcode, dst, lhs, rhs = struct.unpack_from('<4H', data, header+body+8*index)
        assert opcode == 0 and lhs == constant and rhs >> 10 == level
        scale, = struct.unpack('<d', int(scale_bits, 16).to_bytes(8, 'little'))
        assert scale > 0 and math.isfinite(scale)
        entries.append(dict(op_index=index, constant_index=constant, level=level,
                            scale=scale, log_scale=math.log2(scale), bits=scale_bits))
    assert entries
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hevm')
    parser.add_argument('--negative-check', action='store_true')
    args = parser.parse_args()
    data = Path(args.hevm).read_bytes()
    text = Path(args.hevm+'.native_scales').read_text()
    result = validate(data, text)
    if args.negative_check:
        tampered = bytearray(data)
        tampered[-1] ^= 1
        try:
            validate(tampered, text)
        except AssertionError:
            pass
        else:
            raise RuntimeError('tampered binding was accepted')
    print(json.dumps(dict(status='PASS', negative_binding_check=args.negative_check,
                         entries=result), indent=2))
