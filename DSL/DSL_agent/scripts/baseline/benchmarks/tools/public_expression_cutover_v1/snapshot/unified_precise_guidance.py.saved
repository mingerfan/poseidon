"""Opt-in v7 clarifications. Public semantics only, no model solution or probes."""
VERSION = "explicit-v7"
IDENTIFIERS = (
    "Bindings and helper parameters must match [A-Za-z][A-Za-z0-9_]{0,63}. "
    "Leading underscore, including the common discard name _, is rejected. "
    "Use distinct admissible names for unpacking and loop targets. "
    "Do not overwrite supplied constants, zero_ct, runtime names or built-ins.")
FEATURE_RULES = {
 "object.named_numeric_constructor":
    "This feature specifically converts a directly supplied named public constant with np.array(name, dtype=object). "
    "The special conversion reads its public numeric value and shape. "
    "A list of Plain expressions, np.empty, np.full, or an array of ciphertexts does not satisfy it. "
    "Consume the converted storage or its extracted public numeric value in a named output dependency.",
 "zero_dim":
    "Execute binary arithmetic with a rank-zero object array as an operand. "
    "Extracting both cells before the arithmetic performs scalar Expr arithmetic and does not qualify. "
    "The admitted rank-zero object-array binary result is a scalar Expr; consume that result.",
 "event.for_else":
    "The public for-else block must execute and affect a named output dependency. "
    "Assigning the same value in both the loop body and else leaves no observable contribution. "
    "The ordinary execution must still compute the frozen model exactly.",
 "call.Chebyshev":
    "The constructor is np.polynomial.Chebyshev with public coefficients and admitted public domain/window. "
    "It constructs public polynomial data; it is not hc.call.Chebyshev, a ciphertext evaluator, "
    "or a new upstream helper permission. Consume its admitted public data in construction.",
 "unary.Not":
    "Apply not to admitted public data, never a ciphertext Expr or its encrypted value. "
    "Use the resulting public truth value in the returned computation without changing the model.",
}
POLYNOMIAL = (
    "Polynomial nodes specify exact coefficients and basis. Equivalent balanced evaluation trees are allowed; "
    "all nonzero coefficients, including tiny ones, must remain. "
    "The Chebyshev identities T_(2k)=2*T_k*T_k-1 and "
    "T_(2k+1)=2*T_k*T_(k+1)-x permit logarithmic-depth basis construction; "
    "doubling a ciphertext with addition has the same mathematical value as multiplication by two. "

    "Lane-local operations may retain the frozen periodic packing when shapes and output selectors agree. "
    "Do not treat padding as logical data during a later reduction or cross-slot operation. "
    "No truncation, rescaling of the mathematical model, extra modulus levels or bootstrap is authorized.")
def specification(request):
    features=request.get("construction_exercise",{}).get("required_features",[])
    return dict(identifier_rule=IDENTIFIERS,
        directed_acceptance=[dict(feature=f,acceptance=FEATURE_RULES[f])
                             for f in features if f in FEATURE_RULES],
        polynomial_rule=POLYNOMIAL,
        authority="Clarification only; unchanged validator, tracing, contribution and numerical gates.")
def repair_hint(request,diagnostic):
    if type(diagnostic) is not str:return ""
    parts=[]
    if any(s in diagnostic for s in ("Read-only or invalid binding","colliding helper","parameters")):
        parts.append(IDENTIFIERS)
    if any(s in diagnostic for s in ("Missing contributing","Unsupported construction call",
                                    "Call target","object-array","unary","public truth")):
        parts.extend(v["acceptance"] for v in specification(request)["directed_acceptance"])
    return " ".join(parts)[:6000]
