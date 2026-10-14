"""Opt-in final-output representation guidance for the fixed Poseidon Runtime."""
import json
VERSION="explicit-v14"
RULE=("Every physical final output must have a distinct compiled Value ID. A new Python "
      "name, list cell or object-array cell alone does not copy an expression. When equal "
      "logical values occupy separate physical outputs, form distinct additive expressions "
      "with the trusted zero_ct; successive zero_ct additions keep representations distinct. "
      "Preserve all frozen named outputs, selectors and order. This does not alter the model, "
      "constants, numerical tolerance, allowed syntax or Runtime.")
def specification(request):
    from unified_construction_guidance import specification as previous
    from unified_tensor_axis_guidance import MAX_BYTES
    context=previous(request);context["version"]=VERSION
    context["final_output_binding"]=dict(version="distinct-runtime-output-values-v1",
        physical_outputs=request["layout"]["output_ciphertexts"],rule=RULE)
    lists=[context["tensor_axis_annotations"][k] for k in
           ("transforms","linears","broadcasts","input_chunks","reductions")]
    lists.extend(context["shape_examples"][k] for k in ("reductions","broadcasts"))
    lists.extend((context["logical_constant_scalar_bindings"]["bindings"],
                  context["constant_role_examples"]["bindings"]))
    while len(json.dumps(context,sort_keys=True).encode())>MAX_BYTES:
        candidates=[row for row in lists if row]
        if not candidates:raise ValueError("Output-binding required guidance byte bound")
        max(candidates,key=lambda row:len(json.dumps(row,sort_keys=True))).pop()
    return context
def repair_hint(request,source,diagnostic="",stage="static_check",*,witness=None):
    if request.get("generation_guidance",{}).get("version")!=VERSION:return ""
    if type(source)is not str or len(source.encode())>65536:return ""
    from unified_graph_contract import validate_request,explicit_generation_guidance
    from unified_construction_guidance import repair_hint as previous
    from public_weight_context import request_digest
    try:
        validate_request(request)
        legacy=dict(request,generation_guidance=explicit_generation_guidance(request,"explicit-v13"))
        legacy["request_id"]=request_digest(legacy)
        validate_request(legacy)
        prior=previous(legacy,source,diagnostic,stage,witness=witness)
        if stage=="artifact_gate":prior=RULE+("\n"+prior if prior else "")
        return prior[:6500]
    except (ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError):
        return ""
