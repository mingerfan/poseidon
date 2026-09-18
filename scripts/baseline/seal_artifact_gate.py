"""Bounded, fail-closed preflight for trusted, tiny SEAL golden artifacts.

This is not a sandbox or a general verifier for hostile Agent programs. Reject
bootstrap before the unmodified runtime (whose release build can simulate it).
Format evidence: HEVMHeader.h, EmitHEVM.cpp, SEAL_HEVM.cpp at pinned Dacapo.
"""
from collections import Counter
import math
import struct


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inspect_artifacts(hevm, constants, *, rotation_steps=(1, 2), expected_inputs=1,
                      execution_abi=None, input_period=4):
    from packed_input_abi import ABI, rotations as packed_rotations
    require(execution_abi in (None,ABI),'Unknown execution ABI')
    packed=execution_abi==ABI
    require(type(input_period) is int and (packed or input_period==4),'Invalid legacy input period')
    policy=packed_rotations(input_period) if packed else (-3,-2,-1,1,2,3)
    require(not packed or expected_inputs in (1,2),'Packed ABI requires one model input and optional zero')
    require(type(expected_inputs) is int and 1 <= expected_inputs <= 5, "Invalid expected input count")
    require(type(rotation_steps) in (tuple, list) and 1 <= len(rotation_steps) <= len(policy) and
            all(type(s) is int and s in policy for s in rotation_steps) and
            len(set(rotation_steps)) == len(rotation_steps), "Invalid rotation key policy")
    require(64 <= len(hevm) <= 1024**2, "Invalid HEVM size")
    require(8 <= len(constants) <= 1024**2, "Invalid CST size")
    magic, header_size, nargs, nresults = struct.unpack_from("<IIQQ", hevm)
    require(magic == 0x4845564D and header_size == 24, "Invalid HEVM header")
    require(nargs == expected_inputs and 1 <= nresults <= (16 if packed else 4), "Encrypted input/output count mismatch")
    body_size, nops, ncipher, nplain, initial = struct.unpack_from("<5Q", hevm, 24)
    require(body_size == 40 + 8 * (2 * nargs + 3 * nresults), "Invalid config length")
    require(24 + body_size + nops * 8 == len(hevm), "HEVM truncation or trailing data")
    require(0 <= nops <= 4096 and nargs <= ncipher <= 128 and nplain <= (256 if packed else 128),
            "Resource limit exceeded")
    require(initial == 13, "Requires stock SEAL profile level upper bound 13")
    offset = 64

    def array(count):
        nonlocal offset
        values = list(struct.unpack_from(f"<{count}Q", hevm, offset))
        offset += 8 * count
        return values

    arg_scale, arg_level = array(nargs), array(nargs)
    res_scale, res_level, res_dst = array(nresults), array(nresults), array(nresults)
    for level, scale in zip(arg_level + res_level, arg_scale + res_scale):
        require(1 <= level <= 13 and 1 <= scale < min(180, 60 * level), "Invalid level/scale")
    require(all(dst < ncipher for dst in res_dst), "Invalid result register")
    operations = list(struct.iter_unpack("<4H", hevm[offset:]))
    # Scan the ENTIRE program before parsing operands or invoking any native API.
    allowed = {0, 1, 2, 3, 4, 6, 7, 8, 9, 65535}
    require(all(op[0] in allowed for op in operations), "Forbidden opcode (bootstrap/upscale/unknown)")

    count, = struct.unpack_from("<q", constants)
    require(0 <= count <= (256 if packed else 128), "Invalid constant count")
    coffset, vectors = 8, []
    for _ in range(count):
        require(coffset + 8 <= len(constants), "Truncated CST vector length")
        size, = struct.unpack_from("<q", constants, coffset)
        coffset += 8
        require(1 <= size <= 16384 and coffset + size * 8 <= len(constants), "Invalid CST vector size")
        require(not packed or size in (1,input_period),'CST vector does not match declared slot period')
        values = struct.unpack_from(f"<{size}d", constants, coffset)
        require(all(math.isfinite(v) for v in values), "Non-finite constant")
        vectors.append(values)
        coffset += size * 8
    require(coffset == len(constants), "Trailing CST data")

    # Level is the number of data primes (13 at chain_index 12), not the
    # Poseidon adapter's level convention. Track availability and chain drops.
    levels = dict(enumerate(arg_level))
    plains, rotations = {}, set()
    for opcode, dst, lhs, rhs in operations:
        if opcode == 65535:
            # EmitHEVM initializes ONLY opcode for tensor.empty. Other fields
            # are unspecified, and stock runtime intentionally ignores them.
            continue
        if opcode == 0:
            level, scale = rhs >> 10, rhs & 1023
            require(dst < nplain and (lhs == 65535 or lhs < len(vectors)), "Invalid encode operand")
            require(dst not in plains, "Repeated plain destination unsafe with eager preprocess")
            require(1 <= level <= 13 and 1 <= scale < min(180, 60 * level), "Invalid encode level/scale")
            plains[dst] = level
            continue
        require(dst < ncipher and lhs in levels, "Invalid or uninitialized cipher register")
        level = levels[lhs]
        if opcode in (6, 8):
            require(rhs in levels and levels[rhs] == level, "Cipher operand level mismatch")
        elif opcode in (7, 9):
            require(rhs in plains and plains[rhs] == level, "Plain operand level mismatch")
        elif opcode == 1:
            signed_step = rhs if rhs < 32768 else rhs - 65536
            require(signed_step in rotation_steps, "Rotation not provisioned by golden key set")
            rotations.add(signed_step)
        elif opcode == 3:
            level -= 1
        elif opcode == 4:
            require(1 <= rhs < level, "Invalid modswitch drop (including no-op)")
            level -= rhs
        require(level >= 1, "Exhausted modulus chain")
        levels[dst] = level
    require(all(levels.get(dst) == level for dst, level in zip(res_dst, res_level)),
            "Result register level mismatch")
    return dict(arg_scale=arg_scale, arg_level=arg_level, res_scale=res_scale, res_level=res_level,
                res_dst=res_dst, initial_level=initial, ciphertext_buffers=ncipher,
                plaintext_buffers=nplain, opcode_counts=dict(Counter(str(op[0]) for op in operations)),
                rotation_steps=sorted(rotations), bootstrap_rejected=True,
                execution_validated=False)
