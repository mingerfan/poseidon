"""Bounded, fail-closed preflight for trusted, tiny SEAL golden artifacts.

This is not a sandbox or a general verifier for hostile Agent programs. Reject
bootstrap before the unmodified runtime (whose release build can simulate it).
Format evidence: HEVMHeader.h, EmitHEVM.cpp, SEAL_HEVM.cpp at pinned Dacapo.
"""
from collections import Counter
import math
import struct
from dataclasses import dataclass
import hashlib
import json



def require(condition, reason):
    if not condition:
        raise ValueError(reason)


# Exact CoeffModulus::Create(32768, [60]*14), SEAL 4.0.0.
# The last prime is the key-switching special prime, not a data level.
STOCK_MODULI = (
    1152921504581419009,1152921504581877761,1152921504583647233,
    1152921504585547777,1152921504586530817,1152921504589938689,
    1152921504592429057,1152921504592822273,1152921504593412097,
    1152921504595640321,1152921504595968001,1152921504597016577,
    1152921504598720513,1152921504606584833)
GATE_VERSION = "seal-4.0.0-dataflow-v2"
# Additional numerical-integrity policy, NOT a SEAL API constraint. A metadata
# rewrite alone must not exceed the already frozen relative error budget.
ALIGNMENT_RELATIVE_LIMIT = 1e-4

def parameter_profile(parameters=None):
    if parameters is None:
        parameters = dict(seal_version="4.0.0", polynomial_degree=32768,
            security_check="tc128", parameters_set=True, data_modulus_count=13,
            modulus_values=[str(q) for q in STOCK_MODULI], modulus_bits=[60]*14)
    require(type(parameters) is dict, "Missing public SEAL parameters")
    require(parameters.get("seal_version")=="4.0.0" and
            parameters.get("polynomial_degree")==32768 and
            parameters.get("security_check")=="tc128" and
            parameters.get("parameters_set") is True and
            parameters.get("data_modulus_count")==13 and
            parameters.get("modulus_bits")==[60]*14 and
            parameters.get("modulus_values")==[str(q) for q in STOCK_MODULI],
            "Public SEAL parameters differ from pinned tc128 profile")
    return dict(seal_version="4.0.0", polynomial_degree=32768, security_check="tc128",
        key_moduli=list(STOCK_MODULI), data_moduli=list(STOCK_MODULI[:-1]))

def modulus_capacity(moduli):
    """SEAL total_coeff_modulus_bit_count: bit length of the PRODUCT."""
    require(bool(moduli) and all(type(q) is int and q>1 for q in moduli), "Invalid modulus chain")
    return math.prod(moduli).bit_length()

def check_capacity(level, scale, moduli, where, operation):
    require(type(level) is int and 1<=level<=len(moduli),
            f"{where}: Exhausted modulus chain or invalid remaining_level={level}")
    require(math.isfinite(scale) and scale>0, f"{where}: Non-finite/nonpositive scale")
    bits=modulus_capacity(moduli[:level])
    # ckks.h vector encode: int(log2(scale))+1 < bit_count.
    # Evaluator multiply/modswitch and CKKS decode: int(log2(scale)) < bit_count.
    extra=1 if operation=="encode" else 0
    exponent=math.log2(scale)
    require(int(exponent)+extra<bits,
        f"Invalid level/scale: {where} remaining_level={level}, log2_scale={exponent:g}; "
        f"{operation} requires int(log2_scale)+{extra} < modulus_product_bits={bits}")
    return bits

@dataclass(frozen=True)
class ScaleState:
    level: int
    scale: float
    nominal: int

def state_record(state):
    return dict(hevm_level=state.level, seal_chain_index=state.level-1,
                data_modulus_count=state.level, scale=state.scale,
                log2_scale=math.log2(state.scale), nominal_log2_scale=state.nominal)

def compatible_scale(lhs, rhs, where):
    # Integer declarations use the compiler's prime-bit convention. Actual
    # doubles use the real primes. Equal nominal exponents are REQUIRED, not
    # rounded-to-nearest with a floating tolerance. Only rescale/multiply can
    # introduce departure from that exact nominal lineage.
    require(lhs.nominal==rhs.nominal,
            f"{where}: Add scale mismatch: nominal {lhs.nominal} != {rhs.nominal}; "
            "runtime would overwrite the left register scale and change its value")
    delta=lhs.scale/rhs.scale-1.0
    require(math.isfinite(delta) and abs(delta)<=ALIGNMENT_RELATIVE_LIMIT,
            f"{where}: Add scale drift exceeds frozen relative numerical budget: {delta}")
    return delta

def verify_parameter_file(path, observer_path):
    """Read only public parm.seal through pinned SEAL; never load any key."""
    import ctypes
    import os
    from hevm_abi import check_elf
    require(path.is_file() and not path.is_symlink() and 16<=path.stat().st_size<=65536,
            "Invalid public parameter file")
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    check_elf(observer_path)
    lib=ctypes.CDLL(str(observer_path))
    fn=lib.inspect_seal_parameters
    ptr=ctypes.POINTER(ctypes.c_uint64)
    fn.argtypes=[ctypes.c_char_p,ptr,ptr,ptr]
    fn.restype=ctypes.c_int
    moduli=(ctypes.c_uint64*14)();bits=(ctypes.c_uint64*13)();indices=(ctypes.c_uint64*13)()
    require(fn(os.fsencode(path),moduli,bits,indices)==0,"Native SEAL public parameter validation failed")
    require(tuple(moduli)==STOCK_MODULI,"Actual runtime moduli differ from pinned profile")
    require(list(bits)==[modulus_capacity(STOCK_MODULI[:n]) for n in range(1,14)]
            and list(indices)==list(range(13)), "SEAL chain mapping mismatch")
    require(hashlib.sha256(path.read_bytes()).hexdigest()==before,"Public parameters changed while checking")
    return dict(parm_sha256=before, observer_sha256=hashlib.sha256(observer_path.read_bytes()).hexdigest(),
                actual_parameters_verified=True, security_check="tc128",
                data_modulus_bits=list(bits), seal_chain_indices=list(indices))


def inspect_artifacts(hevm, constants, *, rotation_steps=(1, 2), expected_inputs=1,
                      execution_abi=None, input_period=4, parameters=None):
    profile=parameter_profile(parameters)
    moduli=profile["data_moduli"]
    from packed_input_abi import ABI, rotations as packed_rotations
    from unified_graph_contract import ABI as UNIFIED_ABI
    from unified_chunk_layout import ABI as CHUNK_ABI
    require(execution_abi in (None,ABI,UNIFIED_ABI,CHUNK_ABI),'Unknown execution ABI')
    packed=execution_abi is not None
    require(type(input_period) is int and (packed or input_period==4),'Invalid legacy input period')
    policy=packed_rotations(input_period) if packed else (-3,-2,-1,1,2,3)
    require(execution_abi != ABI or expected_inputs in (1,2),'Packed ABI requires one model input and optional zero')
    require(type(expected_inputs) is int and 1 <= expected_inputs <= 5, "Invalid expected input count")
    require(type(rotation_steps) in (tuple, list) and 1 <= len(rotation_steps) <= len(policy) and
            all(type(s) is int and s in policy for s in rotation_steps) and
            len(set(rotation_steps)) == len(rotation_steps), "Invalid rotation key policy")
    require(64 <= len(hevm) <= 1024**2, "Invalid HEVM size")
    require(8 <= len(constants) <= 1024**2, "Invalid CST size")
    magic, header_size, nargs, nresults = struct.unpack_from("<IIQQ", hevm)
    require(magic == 0x4845564D and header_size == 24, "Invalid HEVM header")
    require(nargs == expected_inputs and 1 <= nresults <= (4 if execution_abi in (UNIFIED_ABI,CHUNK_ABI) else 16 if packed else 4), "Encrypted input/output count mismatch")
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
    def declared(level, exponent, where, operation):
        require(type(exponent) is int and 0<=exponent<=1023, f"{where}: Invalid binary64 exponent")
        scale=math.ldexp(1.0, exponent)
        # Stock runtime encodes at first_parms_id then modswitches to level.
        if operation=="encode":
            check_capacity(13,scale,moduli,where+" initial encoding","encode")
        check_capacity(level,scale,moduli,where,
                       "modswitch" if operation=="encode" and level<13 else operation)
        return ScaleState(level,scale,exponent)
    states={i:declared(level,scale,f"input[{i}]","encode")
            for i,(level,scale) in enumerate(zip(arg_level,arg_scale))}
    # Output declarations are checked AFTER propagation, so an earlier invalid
    # instruction wins over a downstream bogus declaration.
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

    # Preprocess eagerly encodes ALL constants before any instruction runs.
    plains, rotations, zero_plains = {}, set(), set()
    events, alignments = [], []
    for instruction,(opcode,dst,lhs,rhs) in enumerate(operations):
        if opcode!=0:continue
        where=f"instruction {instruction} EncodeP"
        level,exponent=rhs>>10,rhs&1023
        require(dst<nplain and (lhs==65535 or lhs<len(vectors)),where+": Invalid encode operand")
        require(dst not in plains,where+": Repeated plain destination unsafe with eager preprocess")
        plains[dst]=declared(level,exponent,where,"encode")
        if lhs!=65535 and all(v==0.0 for v in vectors[lhs]):zero_plains.add(dst)

    names={0:"EncodeP",1:"RotateC",2:"NegateC",3:"RescaleC",4:"ModswitchC",
           6:"AddCC",7:"AddCP",8:"MulCC",9:"MulCP",65535:"Empty"}
    for instruction,(opcode,dst,lhs,rhs) in enumerate(operations):
        where=f"instruction {instruction} {names[opcode]}"
        if opcode==65535:continue
        if opcode==0:
            events.append(dict(instruction=instruction,opcode=opcode,dst=dst,result=state_record(plains[dst])))
            continue
        require(dst<ncipher and lhs in states,where+": Invalid or uninitialized cipher register")
        # Snapshot BEFORE writing destinations: dst may alias either operand.
        left=states[lhs]; right=None
        if opcode in (6,8):
            require(rhs in states,where+": Uninitialized rhs")
            right=states[rhs]
        elif opcode in (7,9):
            require(rhs in plains,where+": Uninitialized plain register")
            right=plains[rhs]
            require(opcode!=9 or rhs not in zero_plains,
                f"Transparent ciphertext risk: {where} uses all-zero Plain register {rhs}; use client-encrypted zero_ct")
        if right is not None:
            require(right.level==left.level,where+": Cipher/Plain operand level mismatch")
        if opcode==1:
            signed_step=rhs if rhs<32768 else rhs-65536
            require(signed_step in rotation_steps,where+": Rotation not provisioned by golden key set")
            rotations.add(signed_step)
        result=left
        if opcode==3:
            require(left.level>1,where+": Exhausted modulus chain")
            q=moduli[left.level-1]
            result=ScaleState(left.level-1,left.scale/float(q),left.nominal-q.bit_length())
        elif opcode==4:
            require(1<=rhs<left.level,where+": Invalid modswitch drop (including no-op)")
            result=ScaleState(left.level-rhs,left.scale,left.nominal)
        elif opcode in (8,9):
            result=ScaleState(left.level,left.scale*right.scale,left.nominal+right.nominal)
        elif opcode in (6,7):
            change=compatible_scale(left,right,where)
            # Unmodified HEVM mutates lhs EVEN WHEN dst != lhs.
            result=ScaleState(left.level,right.scale,right.nominal)
            states[lhs]=result
            if change:
                alignments.append(dict(instruction=instruction,lhs=lhs,rhs=rhs,
                    relative_value_change=change,reason="same nominal lineage; real-prime rescale drift",
                    lhs_scale_before=left.scale,lhs_scale_after=right.scale))
        require(math.isfinite(result.scale) and result.scale>0,where+": Non-finite/nonpositive scale")
        # SEAL's rescale itself does NOT apply is_scale_within_bounds to its
        # result. Rotate/negate/add do not impose the decode bound either.
        # Check at the API that actually enforces it and at final decoding.
        if opcode in (8,9,4):
            check_capacity(result.level,result.scale,moduli,where,
                           "multiply" if opcode in (8,9) else "modswitch")
        states[dst]=result
        events.append(dict(instruction=instruction,opcode=opcode,dst=dst,lhs=lhs,rhs=rhs,
                           input=state_record(left),result=state_record(result)))
    outputs=[]
    for index,(dst,level,exponent) in enumerate(zip(res_dst,res_level,res_scale)):
        declared(level,exponent,f"output[{index}]","decode")
        require(dst in states and states[dst].level==level,f"output[{index}]: Result register level mismatch")
        value=states[dst]
        require(value.nominal==exponent,
            f"output[{index}]: Result scale declaration mismatch: declared={exponent}, "
            f"propagated_nominal={value.nominal}, actual_log2={math.log2(value.scale):.17g}")
        require(abs(value.scale/math.ldexp(1.0,exponent)-1)<=ALIGNMENT_RELATIVE_LIMIT,
                f"output[{index}]: Real-prime scale drift exceeds frozen relative numerical budget")
        check_capacity(value.level,value.scale,moduli,f"output[{index}]","decode")
        outputs.append(state_record(value))
    return dict(arg_scale=arg_scale, arg_level=arg_level, res_scale=res_scale, res_level=res_level,
                res_dst=res_dst, initial_level=initial, ciphertext_buffers=ncipher,
                plaintext_buffers=nplain, opcode_counts=dict(Counter(str(op[0]) for op in operations)),
                rotation_steps=sorted(rotations), bootstrap_rejected=True, gate_version=GATE_VERSION,
                parameter_profile=profile,
                parameter_profile_sha256=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                level_capacity_bits=[modulus_capacity(moduli[:n]) for n in range(1,len(moduli)+1)],
                propagated_outputs=outputs, instruction_states=events, scale_alignments=alignments,
                value_magnitude_validated=False, execution_validated=False)

def verify_execution_binding(gate, execution, report):
    """Audit actual execution identity; the static profile alone is insufficient."""
    from seal_cpu_golden import KEY_BUILD
    from hecate_python_env import ROOT, digest
    require(execution.get("artifact_gate_version")==GATE_VERSION, "Execution gate version mismatch")
    bound=execution.get("public_parameter_binding",{})
    require(bound.get("actual_parameters_verified") is True and bound.get("security_check")=="tc128",
            "Missing actual SEAL parameter verification")
    require(type(bound.get("parm_sha256")) is str and len(bound["parm_sha256"])==64 and
            all(c in "0123456789abcdef" for c in bound["parm_sha256"]), "Missing public parameter digest")
    require(bound.get("observer_sha256")==report.get("parameter_observer_sha256")==
            digest(KEY_BUILD/"libseal_artifact_parameters.so"), "Actual parameter observer binding mismatch")
    require(report.get("parameter_observer_source_sha256")==
            digest(ROOT/"scripts/baseline/seal_keys/artifact_parameters.cpp"), "Parameter observer source mismatch")
    require(gate["parameter_profile"]==parameter_profile(report.get("parameters",{})), "Execution parameter profile mismatch")
    require(bound.get("data_modulus_bits")==gate["level_capacity_bits"] and
            bound.get("seal_chain_indices")==list(range(13)), "Actual SEAL chain mapping mismatch")
    for sample in execution["ciphertext_metadata"]:
        observed=sample["outputs"]
        require(len(observed)==len(gate["propagated_outputs"]), "Observed output state count")
        for got,want in zip(observed,gate["propagated_outputs"]):
            require(got["data_modulus_count"]==want["data_modulus_count"] and
                    got["log2_scale"]==want["log2_scale"] and got["polynomials"]==2,
                    "Observed output differs from propagated state")
