"""Optional physical output alias guidance and bounded repair clarification."""
import copy
from unified_output_aliases import POLICY, policy, rules

VERSION = "explicit-v15"

def specification(request):
    from unified_output_binding_guidance import specification as previous
    from benchmark_graph import require
    require(policy(request["layout"]) == POLICY, "Alias guidance requires exact output binding")
    context = previous(request)
    context["version"] = VERSION
    context["final_output_binding"] = dict(
        version="exact-graph-values-v1",
        physical_outputs=request["layout"]["output_ciphertexts"],
        logical_outputs=copy.deepcopy(request["layout"]["outputs"]),
        rule=rules(request["layout"]))
    return context

def repair_hint(request, source, diagnostic="", stage="static_check", *, witness=None):
    if request.get("generation_guidance", {}).get("version") != VERSION:
        return ""
    if type(source) is not str or len(source.encode()) > 65536:
        return ""
    from unified_graph_contract import validate_request, explicit_generation_guidance
    from unified_output_binding_guidance import repair_hint as previous
    from public_weight_context import request_digest
    try:
        validate_request(request)
        legacy = dict(request, generation_guidance=explicit_generation_guidance(request, "explicit-v14"))
        legacy["request_id"] = request_digest(legacy)
        prior = previous(legacy, source, diagnostic, stage, witness=witness)
        return (rules(request["layout"]) + ("\n" + prior if prior else ""))[:6500]
    except (ValueError, TypeError, KeyError, IndexError, RecursionError, MemoryError):
        return ""
