"""Independent original GELU scalar formulas; never call the graph planner.

Reference math is compared with the pinned Torch implementation in the audit.
"""
import math
from model_decomposition import polynomial_profile
from gelu_source_plan import fail

REFERENCE_VARIANTS={"none","tanh"}

def original_scalar(x,variant):
    if type(variant) is not str or variant not in REFERENCE_VARIANTS:
        fail("gelu_source_variant_rejected","approximate")
    if variant=="none":
        return .5*x*(1+math.erf(x/math.sqrt(2)))
    return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))

def check_domain(values, approximation):
    polynomial_profile(approximation)
    lo,hi=approximation["domain"]
    if any(type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi for x in values):
        fail("gelu_source_domain_rejected","input")

