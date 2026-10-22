"""Verified source mathematics, no DSL answer or private validation data."""
import copy
from generation_model_origin import checked_public
VERSION="explicit-v21"
def specification(request,inherited):
    value=copy.deepcopy(inherited);value["version"]=VERSION
    origin=request.get("generation_model_origin")
    if origin is not None:
        checked_public(origin,request)
        value["source_mathematics_rule"]=(
          "generation_model_origin gives host-verified high-level mathematics and the correspondence "
          "to the fixed lowered model. Implement the lowered model exactly; preserve every source "
          "operator, approximation, weight, axis and named output. Source names and constant descriptor "
          "keys are logical labels, not additional bound DSL names. Only public_constants names are "
          "prebound. Keep per-group normalization separate; do not add a second mean-square correction "
          "to a mean of already centered squares. This context is mathematical data, not source code, "
          "a standard DSL answer, an executable backend capability or an original-function accuracy certificate.")
    value["bounded_expression_rule"]=(
      "Choose a compact legal expression for the complete fixed graph. Reuse intermediates. Consecutive "
      "public scalar factors can be combined when their product is exactly represented, finite, and "
      "admitted by the native contract; do not treat a masked vector as a scalar or discard nonzero "
      "coefficients. Preserve the declared approximation and avoid unlimited exploration or narrative. "
      "Return the candidate JSON only.")
    return value

def repair_hint(request,source,diagnostic="",stage="static_check",*,witness=None):
    """Reuse audited v20 feedback without changing the admitted request."""
    if request.get("generation_guidance",{}).get("version")!=VERSION:
        return ""
    from unified_graph_contract import validate_request,explicit_generation_guidance
    from public_weight_context import request_digest
    from unified_suffix_stage_guidance import repair_hint as previous
    try:
        validate_request(request)
        inherited=copy.deepcopy(request)
        inherited.pop("generation_model_origin",None)
        inherited["generation_guidance"]=explicit_generation_guidance(inherited,"explicit-v20")
        inherited["request_id"]=request_digest(inherited)
        return previous(inherited,source,diagnostic,stage,witness=witness)
    except (ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError):
        return ""
