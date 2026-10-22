"""Direct fixed Dacapo RuntimePlan export, without a HEVM converter."""
import hashlib, json, math, re, struct
from hecate_python_env import digest
from poseidon_runtime_paths import SPEC
ENTRY_POLICY = "hecate-private-symbol-dce-v1"

def command(waterline=45, *, entry_policy=None):
    if waterline not in (40,45):raise ValueError("Unqualified Poseidon waterline")
    if entry_policy not in (None, ENTRY_POLICY):raise ValueError("Unknown RuntimePlan entry policy")
    trace = "/out/candidate_trace.entry.mlir" if entry_policy else "/out/candidate_trace.mlir"
    cleanup = ["--symbol-dce"] if entry_policy else []
    return ["/hecate-opt",trace,*cleanup,"--eva","--ckks-config=/profile.json",
        f"--waterline={waterline}","--enable-debug-printer","--mlir-disable-threading","--verify-each",
        "--runtime-plan-id=1","--runtime-plan-target-id=poseidon-ckks-cpu",
        "--runtime-plan-capability-version=1","--runtime-plan-operator-spec-id=poseidon-agent-cpu-v1",
        "--runtime-plan-operator-spec-version=1","--runtime-plan-operator-spec-sha256=sha256:"+digest(SPEC),
        "--runtime-plan-context-id=poseidon-agent-n15-q13-p1-tc128-v1",
        "--runtime-plan-device-count=0","--runtime-plan-ntt=true",
        "--runtime-plan-boot-implementation=native","--runtime-plan-inline-payload-max-bytes=4096",
        "-o","/out/lowered.mlir"]


def entrypoint_trace(source):
    """Set visibility on trusted, fully inlined Hecate function definitions.

    Keep the original frontend IR and all trace events. MLIR's existing symbol
    DCE removes only unreferenced private definitions before EVA. A symbol use
    requires an independently reviewed dependency policy and is rejected here.
    """
    if type(source) is not str or len(source.encode()) > 8*1024**2:
        raise ValueError("RuntimePlan entry trace size")
    if "@" in source or "func.call" in source or "earth.call" in source:
        raise ValueError("RuntimePlan entry policy requires fully inlined Hecate IR")
    pattern = re.compile(r'^  "func\.func"\(\) <\{[^\n]*?sym_name = "([A-Za-z_][A-Za-z_0-9]*)"', re.M)
    names = [m.group(1) for m in pattern.finditer(source)]
    if names.count("_hecate_golden") != 1 or len(names) != len(set(names)):
        raise ValueError("RuntimePlan entry function missing or duplicated")
    module = re.match(r'^"builtin\.module"\(\) <\{sym_name = "(/out/candidate_trace\.mlir)"\}> \(\{', source)
    if len(re.findall(r'\bsym_name\s*=', source)) != len(names)+int(module is not None) or "sym_visibility" in source:
        raise ValueError("Unsupported Hecate function declaration")
    def private(match):
        return match.group(0) if match.group(1) == "_hecate_golden" else match.group(0)+', sym_visibility = "private"'
    selected = pattern.sub(private, source)
    if len(selected.encode()) > 8*1024**2:
        raise ValueError("RuntimePlan selected entry trace size")
    if selected.replace(', sym_visibility = "private"', '') != source:
        raise ValueError("RuntimePlan entry trace body changed")
    receipt = dict(format=ENTRY_POLICY, entry="_hecate_golden",
                   private_functions=[n for n in names if n != "_hecate_golden"],
                   original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   selected_sha256=hashlib.sha256(selected.encode()).hexdigest(),
                   compiler_cleanup="symbol-dce", function_bodies_unchanged=True)
    return selected, receipt

def compact_f64_trace(source):
    """Re-encode frontend f64 buffers without changing their IEEE-754 bits.

    LLVM 18.1.2 emits large dense buffers as little-endian hex. Its f64 decimal
    parser reads binary64, so shortest round-trip literals preserve each value.
    The raw trace keeps its existing 16 MiB frontend file budget; entrypoint_trace
    still checks the independently bounded 8 MiB compiler input.
    """
    from compiler_configuration import TRACE_SERIALIZATION_POLICY
    if type(source) is not str or len(source.encode()) > 16*1024**2:
        raise ValueError("Frontend trace serialization size")
    pattern = re.compile(r'dense<"0x([0-9A-Fa-f]+)"> : tensor<([0-9]+)xf64>')
    records = []
    def replace(match):
        raw = bytes.fromhex(match[1])
        count = int(match[2])
        if count < 1 or len(raw) != count*8:
            raise ValueError("Dense f64 shape/buffer mismatch")
        tokens = []
        for offset in range(0,len(raw),8):
            bits = raw[offset:offset+8]
            value = struct.unpack("<d",bits)[0]
            if not math.isfinite(value):
                raise ValueError("Nonfinite dense f64 constant")
            token = repr(value)
            # MLIR's float grammar requires a decimal point before an exponent.
            if "e" in token and "." not in token.split("e")[0]:
                token = token.replace("e",".0e")
            if struct.pack("<d",float(token)) != bits:
                raise ValueError("Dense f64 serialization changed bits")
            tokens.append(token)
        decimal = "dense<["+", ".join(tokens)+"]> : tensor<"+str(count)+"xf64>"
        shorter = len(decimal) < len(match[0])
        records.append(dict(index=len(records),elements=count,
                            ieee754_sha256=hashlib.sha256(raw).hexdigest(),
                            converted=shorter))
        return decimal if shorter else match[0]
    selected = pattern.sub(replace,source)
    receipt = dict(format=TRACE_SERIALIZATION_POLICY,constants=records,
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        serialized_sha256=hashlib.sha256(selected.encode()).hexdigest(),
        original_bytes=len(source.encode()),serialized_bytes=len(selected.encode()),
        values_bitwise_unchanged=True)
    return selected,receipt


def _entrypoint(source,config):
    serialization = None
    compact = None
    if config.get("schema") == 4:
        from compiler_configuration import TRACE_SERIALIZATION_POLICY
        if config.get("trace_serialization_policy") != TRACE_SERIALIZATION_POLICY:
            raise ValueError("Changed trace serialization policy")
        compact,serialization = compact_f64_trace(source)
        source = compact
    selected,receipt = entrypoint_trace(source)
    if serialization is not None:
        receipt["trace_serialization"] = serialization
    return selected,receipt,compact


def prepare_entrypoint(output, config):
    if config.get("schema") not in (3,4):
        return None
    if config.get("runtime_plan_entry_policy") != ENTRY_POLICY:
        raise ValueError("Changed RuntimePlan entry policy")
    selected, receipt, compact = _entrypoint((output/"candidate_trace.mlir").read_text(),config)
    if compact is not None:
        (output/"candidate_trace.compact.mlir").write_text(compact)
    (output/"candidate_trace.entry.mlir").write_text(selected)
    (output/"entry-selection.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def audit_entrypoint(output, request, receipt, compile_command):
    config=request.get("compiler_configuration",{})
    if config.get("schema") not in (3,4):
        if receipt is not None:
            raise ValueError("Unexpected entry policy for historical compiler configuration")
        return
    selected, expected, compact = _entrypoint((output/"candidate_trace.mlir").read_text(),config)
    if compact is not None and (output/"candidate_trace.compact.mlir").read_text() != compact:
        raise ValueError("Trace serialization provenance mismatch")
    if (receipt != expected or
        json.loads((output/"entry-selection.json").read_text()) != expected or
        (output/"candidate_trace.entry.mlir").read_text() != selected or
        compile_command != command(config["waterline"],entry_policy=config["runtime_plan_entry_policy"])):
        raise ValueError("RuntimePlan entry provenance mismatch")
