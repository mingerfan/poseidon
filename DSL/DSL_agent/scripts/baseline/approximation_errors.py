"""Explicit model-approximation/CKKS error decomposition; never rewrites a model."""
import math


def vector(values):
    if type(values) not in (tuple, list) or not 1 <= len(values) <= 4096:
        raise ValueError("Expected a bounded nonempty output vector")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Expected finite real outputs")
    return list(values)


def metrics(actual, target):
    errors = [a-t for a, t in zip(actual, target)]
    absolute = [abs(e) for e in errors]
    relative = [abs(e/t) if t else None for e, t in zip(errors, target)]
    norm_a, norm_t = math.hypot(*actual), math.hypot(*target)
    cosine = math.fsum((a/norm_a)*(t/norm_t) for a, t in zip(actual, target)) if norm_a and norm_t else None
    return dict(signed_error=errors, absolute_error=absolute, relative_error=relative,
                mae=math.fsum(absolute)/len(errors), max_absolute_error=max(absolute),
                max_nonzero_reference_relative_error=max((v for v in relative if v is not None), default=None),
                cosine_similarity=cosine)


def decompose(original, polynomial, decrypted, *, atol=1e-5, rtol=1e-4):
    """All three outputs must be computed independently from the SAME input.

    original = f(x); polynomial = p(x), evaluated in plaintext;
    decrypted = Dec(Eval(p, Enc(x))). No original-model acceptance is inferred
    from a CKKS pass, and no approximation tolerance is silently selected.
    """
    original, polynomial, decrypted = map(vector, (original, polynomial, decrypted))
    if len({len(v) for v in (original, polynomial, decrypted)}) != 1:
        raise ValueError("Output lengths differ")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("Invalid comparison tolerance")
    approx = metrics(polynomial, original)
    ckks = metrics(decrypted, polynomial)
    total = metrics(decrypted, original)
    residual = [t-a-c for t, a, c in zip(total["signed_error"], approx["signed_error"], ckks["signed_error"])]
    ckks_pass = [e <= atol + rtol*abs(p) for e, p in zip(ckks["absolute_error"], polynomial)]
    original_pass = [e <= atol + rtol*abs(f) for e, f in zip(total["absolute_error"], original)]
    return dict(original=original, polynomial=polynomial, decrypted=decrypted,
                approximation_error=approx, ckks_execution_error=ckks, total_error=total,
                execution_error_interpretation="Residual vs polynomial includes semantic/compiler/runtime errors if the implementation is wrong; not automatically CKKS noise",
                compared_values=len(original), atol=atol, rtol=rtol,
                decomposition_residual=residual, ckks_elementwise_pass=ckks_pass,
                ckks_passed=all(ckks_pass), original_target_elementwise_pass=original_pass,
                original_target_threshold_passed=all(original_pass),
                approximation_acceptance_budget=None, original_semantic_equivalence_verified=False)


def relu_and_quadratic(inputs):
    """Deliberately coarse example, NOT the production activation approximation.

    p(x)=(x+x*x)/2 on [-1,1]. With u=abs(x), p(x)-ReLU(x)
    = (u*u-u)/2, hence max abs error 1/8 at abs(x)=1/2.
    This local illustrative polynomial is not an upstream HE_ReLU helper.
    """
    inputs = vector(inputs)
    if any(not -1 <= x <= 1 for x in inputs):
        raise ValueError("Approximation input outside declared domain [-1,1]")
    return [max(0.0, x) for x in inputs], [(x+x*x)/2 for x in inputs]
