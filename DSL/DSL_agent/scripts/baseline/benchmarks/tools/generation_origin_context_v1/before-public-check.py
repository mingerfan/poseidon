"""Public high-level model context from host-verified decomposition provenance.

No source code, private samples, reference answers or deterministic DSL is emitted.
The fixed lowered graph remains authoritative for candidate qualification.
"""
import copy
from benchmark_graph import canonical,digest,require
from public_weight_context import descriptor

FORMAT="poseidon-generation-model-origin-v1"
MAX_BYTES=65536

def build(provenance,request):
    from provenance_json_transport import unpack
    from unified_graph_contract import validate_request
    validate_request(request)
    from candidate_bundle import check_provenance
    value=unpack(provenance,purpose="model-provenance")
    binding=check_provenance(value,request["model"])
    original=value["original"]
    constants={name:descriptor(data)for name,data in original["constants"].items()}
    # Keep mathematical attributes, source locations and logical port order only.
    nodes=[dict(id=node["id"],op=node["op"],inputs=list(node["inputs"]),
                attrs=copy.deepcopy(node["attrs"]),outputs=list(node["outputs"]))
           for node in original["nodes"]]
    result=dict(format=FORMAT,authority="host-verified mathematical source description",
        target_model_sha256=digest(request["model"]),original_model_sha256=digest(original),
        decomposition_binding_sha256=binding["binding_sha256"],
        registry_version=binding["registry_version"],
        inputs=copy.deepcopy(original["inputs"]),outputs=copy.deepcopy(original["outputs"]),
        constants=constants,nodes=nodes,
        source_map=copy.deepcopy(binding["source_map"]),
        original_function_accuracy_certified=False,
        source_code_included=False,reference_answers_included=False,rule_dsl_included=False)
    result["binding"]=digest(result)
    require(len(canonical(result))<=MAX_BYTES,"Public model origin context budget")
    return result

def verify(value,provenance,request):
    """Full source consistency is checked with retained host provenance."""
    require(type(value)is dict and value==build(provenance,request),"Public model origin context changed")
    return value
