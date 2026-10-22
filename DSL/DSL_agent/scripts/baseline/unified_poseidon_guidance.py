"""Deterministic public Poseidon context, independent of experiments and answers."""
import ast
import math

VERSION = "explicit-v10"
REDUCTION = (
    "Reduce the declared logical axes and elements. For periodic packing, admitted rotations "
    "can form a sum without extracting every scalar first. A sum valid only in slot zero "
    "is not a replicated ciphertext scalar. Exclude padded tails and unrelated rows; object "
    "reshape/transpose does not rotate or replicate encrypted slots.")
BROADCAST = (
    "A one-element encrypted tensor must occupy every required destination slot before a "
    "lane-wise broadcast. Distinguish replicated scalars, valid prefixes and padding. Equal "
    "object-array shapes do not establish equal ciphertext layouts. Preserve each named output binding.")
PUBLIC_FACTORS = (
    "For a public factor a, (u*v)*a equals (u*a)*v. Applying independent public weights or "
    "masks to a shallower operand can avoid a late encoded multiplication; prove the same "
    "slot mapping and retain all nonzero factors. This does not justify commuting a factor "
    "through addition, reduction or nonlinear composition. Only admitted constant bindings, "
    "public construction and finite public literals are available; Plain*Plain is not a "
    "ciphertext operation.")
POLYNOMIAL = (
    "Balanced powers and exact polynomial regrouping may reduce serial multiplication depth. "
    "Preserve every coefficient, iteration and approximation domain. Evaluate the declared "
    "approximation, not a newly selected approximation or the ideal nonlinear function.")
VALUE_BOUNDARY = (
    "Public tensor coefficients need the frozen C-order coordinate mapping. Apply each "
    "coefficient to its own logical element, including signs and zero entries. Reusing one "
    "scalar coefficient or mask across unrelated positions is not vector broadcasting.")
BACKEND = (
    "The target is the bound Poseidon CPU RuntimePlan backend. Earth compiler levels count "
    "consumed depth; RuntimePlan/Poseidon levels are zero-based data-modulus indices. "
    "Compiler admission, artifact admission and numerical acceptance are distinct. Keep "
    "the fixed context and requested waterline; bootstrap is unavailable in this context. "
    "A compiler rejection does not prove that all equivalent expressions are impossible.")
SCALE = (
    "RuntimePlan declarations are checked against the pinned Poseidon modulus chain and "
    "instruction propagation. Multiplication changes scale; rescale drops a real modulus. "
    "Modswitch is not a scale conversion. Do not set scale/level, change parameters or use "
    "decrypt-reencrypt. Artifact admission does not establish numerical correctness.")
ZERO = (
    "zero_ct is a genuine client-encrypted zero. Preserve exact zero semantics without "
    "depending on transparent self-cancellation or decoding any intermediate. Do not "
    "change encryption, runtime checks or keys.")

def specification(request):
    from benchmark_graph import validate,require
    require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
            "Poseidon guidance requires an explicit Poseidon CPU configuration")
    g=request["model"];check=validate(g);constants=g["constants"];groups={}
    periodic=request["layout"]["execution_abi"]=="unified-periodic-inputs-v1"
    for node in g["nodes"]:
        op=node["op"];refs=node["inputs"];out=check["shapes"][node["outputs"][0]]
        if periodic and op in ("sum","mean"):groups["packed_reduction"]=REDUCTION
        if op in ("add","subtract","multiply"):
            if any(math.prod(check["shapes"][ref])==1 for ref in refs) and math.prod(out)>1:
                groups["scalar_broadcast"]=BROADCAST
            if op=="multiply" and any(ref in constants for ref in refs):
                groups["public_factor_placement"]=PUBLIC_FACTORS
                if any(ref in constants and math.prod(check["shapes"][ref])>1 for ref in refs):
                    groups["public_tensor_coordinates"]=VALUE_BOUNDARY
        if op=="polynomial":groups["polynomial_grouping"]=POLYNOMIAL
    return dict(version=VERSION,backend="poseidon_cpu",selection=sorted(groups),
                rules={k:groups[k] for k in sorted(groups)},backend_constraints=BACKEND,
                scale_semantics=SCALE,encrypted_zero_rule=ZERO,
                authority="Public graph features and unchanged contracts only; no DSL answer, private input, reference or extra capability.")

def relevant(request):
    return bool(specification(request)["selection"])

def repair_hint(request,source,diagnostic="",stage="static_check"):
    if request.get("generation_guidance",{}).get("version")!=VERSION:return ""
    if not isinstance(source,str) or len(source.encode())>65536:return ""
    try:
        tree=ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):return ""
    if sum(1 for _ in ast.walk(tree))>4096:return ""
    result=[]
    if stage in ("compiler","artifact_gate"):
        rules=specification(request)
        result.extend(rules["rules"].values());result.append(BACKEND);result.append(SCALE)
    # Fixed legacy messages are reused for syntax/type errors without echoing logs.
    from unified_expression_guidance import repair_hint as native_hint
    legacy=dict(request,generation_guidance=dict(request["generation_guidance"],version="explicit-v9"))
    base=native_hint(legacy,source,diagnostic,stage)
    if base:result.append(base)
    return "\n".join(dict.fromkeys(result))[:5000]
