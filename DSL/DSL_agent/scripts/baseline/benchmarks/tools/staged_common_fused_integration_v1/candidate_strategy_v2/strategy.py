"""Independent v2 rule draft and data-only candidate checks.

Only the existing generic candidate dispatcher selects an expression. Case names
are report labels. No model/request/backend rewrite, compilation, execution,
provider call, or prior-result mutation occurs here.
"""
import ast
import copy
import hashlib
import json

VERSION = "poseidon-generic-candidate-dispatch-v2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def candidate_source(request):
    """Preserve the existing ABI, public-vector, packing and validation dispatch."""
    from unified_graph_lowering import candidate_source as existing_candidate_source
    return existing_candidate_source(request)


def check_request(request, *, include_source=False):
    """Generate and statically validate one copy of a frozen public request."""
    from unified_graph_contract import validate_candidate
    require(type(include_source) is bool, "Explicit source inclusion")
    before = canonical(request)
    work = copy.deepcopy(request)
    result = candidate_source(work)
    if type(result) is tuple:
        require(len(result) == 2 and type(result[1]) is dict, "Rule strategy record")
        source, record = result
    else:
        source, record = result, {}
    require(type(source) is str and source, "Rule candidate source")
    validate_candidate(dict(schema=1, request_id=work["request_id"], hecate_source=source), work)
    require(canonical(request) == before, "Original frozen request changed")
    require(work["model"] == request["model"] and work["layout"] == request["layout"] and
            work.get("compiler_configuration") == request.get("compiler_configuration"),
            "Graph, physical ABI or compiler parameters changed")
    tree = ast.parse(source)
    golden = [n for n in tree.body if type(n) is ast.FunctionDef and n.name == "golden"]
    require(len(golden) == 1 and golden[0].body and type(golden[0].body[-1]) is ast.Return and
            type(golden[0].body[-1].value) is ast.List, "Generated literal physical output list")
    physical = len(golden[0].body[-1].value.elts)
    expected = work["layout"]["output_ciphertexts"]
    require(type(expected) is int and physical == expected, "Exact physical output count")
    loaded = {n.id for n in ast.walk(tree) if type(n) is ast.Name and type(n.ctx) is ast.Load}
    vectors = {k for k, v in work["public_constants"].items() if type(v) is list}
    descriptor = work.get("constant_origins", {}).get("capacity_tensor_packing", {})
    bindings = descriptor.get("bindings", []) if type(descriptor) is dict else []
    tensor_vectors = {b["key"] for b in bindings if type(b) is dict and b.get("key") in vectors}
    row = dict(format=VERSION, status="passed", request_id=request["request_id"],
                request_canonical_sha256=hashlib.sha256(before).hexdigest(),
                model_sha256=digest(request["model"]), public_constants_sha256=digest(request["public_constants"]),
                execution_abi=work["layout"]["execution_abi"],
                physical_output_count=physical, expected_physical_output_count=expected,
                public_vector_bindings=len(vectors), public_vector_keys_reused=sorted(vectors & loaded),
                public_tensor_vector_bindings=len(tensor_vectors),
                public_tensor_vector_keys_reused=sorted(tensor_vectors & loaded),
                candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                candidate_bytes=len(source.encode()), rule_record=record,
                new_agent_generation=False, provider_calls=0, compiler_calls=0,
                encrypted_executions=0, production_applied=False, scale_evidence="not_run")
    if include_source:
        row["candidate_source"] = source
    return row


def check_requests(requests, *, include_source=False):
    """Names identify rows only; errors never cause a different expression policy."""
    require(type(requests) is dict and 1 <= len(requests) <= 64, "Bounded public request map")
    rows = []
    for name, request in requests.items():
        require(type(name) is str and name and len(name) <= 128, "Report label")
        try:
            row = check_request(request, include_source=include_source)
        except Exception as error:
            row = dict(format=VERSION, status="failed", error_type=type(error).__name__,
                       error=str(error), request_canonical_sha256=digest(request),
                       new_agent_generation=False, provider_calls=0, compiler_calls=0,
                       encrypted_executions=0, production_applied=False, scale_evidence="not_run")
        rows.append(dict(case=name, **row))
    return dict(format=VERSION, status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
                rows=rows, new_agent_generation=False, provider_calls=0, compiler_calls=0,
                encrypted_executions=0, production_applied=False, scale_evidence="not_run")
