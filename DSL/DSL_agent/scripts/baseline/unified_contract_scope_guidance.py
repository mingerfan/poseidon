"""Explicit contract-scoped generation guidance; validators remain authoritative.

The caller must first validate the existing request. This module only selects
fixed explanations for that request's construction contract. It never changes
the syntax checker, candidate source, layout, model, or resource limits.
"""
import ast
import copy

VERSION = "explicit-v16"
BASE_VERSION = "explicit-v15"
BASE_VERSIONS = ("explicit-v14", "explicit-v15")
PUBLIC_PROFILE = "hecate-unified-public-v1"

NATIVE_HELPERS = (
    "Each additional helper is a native Hecate function with a literal @hc.func "
    "signature and unannotated positional c/p parameters. c is a ciphertext Expr "
    "and p is a Plain Expr, not a public Python number or a static integer parameter. "
    "The signature has no integer/index/control kind. No defaults, keywords, "
    "varargs, nested functions or recursion are admitted. Follow the existing "
    "helper parameter and expanded-work limits.")
NATIVE_INTEGER = (
    "Integer operands used for rotation displacements, range arguments, indices "
    "and storage dimensions must be literal integers after the trusted bounded "
    "range expansion. The expander "
    "substitutes range induction variables and folds literal integer +, -, *, //, "
    "% and unary signs within its existing bounds. It does not specialize native "
    "helper p parameters or ordinary Expr bindings into Python integers. A named "
    "bound Plain, even with a known integral value, cannot serve as a static index "
    "or rotation displacement. Use only the rotation steps admitted by this request.")
NATIVE_CONTROL = (
    "This native contract admits bounded for ... in range(...) expansion. It "
    "does not admit if, conditional expressions, comparisons, Boolean operators, "
    "while, break, continue, comprehensions or for-else, including conditions "
    "written entirely with public literals. Each range has at most 128 iterations; "
    "module expansion, AST and work budgets also apply. Loop targets are public "
    "induction variables; nested loops use distinct targets, and those targets "
    "cannot be assigned in the body. No return "
    "inside a loop. Callable native c/p helpers do not add public control flow.")
NATIVE_DTYPE = (
    "The native array constructor is np.array(data, dtype=object), with exactly "
    "one positional data argument. Cells are complete c/p Expr values in admitted "
    "rectangular storage, at most 16 cells and rank 4. Numeric np.array/asarray, "
    "np.empty/full/concatenate, Empty(), float() and int() are not constructors "
    "or conversion calls in this contract. Ciphertext slots are not array cells.")
NATIVE_SHAPE = (
    "Prebound public constants are immutable Plain Expr bindings with scalar or "
    "length-1/P encoding data; their Python numeric values are not exposed by "
    "indexing a Plain Expr. Native object-array reshape/flatten/transpose/T/copy/"
    "item operate on whole Expr cells with the admitted literal arguments. These "
    "operations neither reshape encoded Plain data nor permute ciphertext slots.")
NATIVE_UNARY = (
    "Unary minus on an Expr requires a ciphertext operand; native object-array "
    "negation must satisfy the existing ciphertext-cell arithmetic check. Bound "
    "Plain-only negation/arithmetic, unary plus on Expr or array storage, and "
    "Empty/None construction cells are not admitted. Unary integer signs can be "
    "folded in static literal syntax, which is a separate construction operation.")
NATIVE_RETURNS = (
    "Native helper results are c/p Expr values or admitted containers of Expr "
    "cells. They are not public Python control values. Golden must return exactly "
    "the frozen number and order of complete ciphertext expressions; a Plain "
    "helper result is not an additional encrypted output.")
NATIVE_BROADCAST = (
    "Native object-array arithmetic uses the admitted array-left +, -, * forms "
    "and whole-Expr cell broadcasting. Scalar Expr arithmetic requires a "
    "ciphertext operand. Storage broadcasting does not establish matching "
    "logical tensor coordinates or packed slots; preserve the request layout.")
NATIVE_CALLS = (
    "Use only declared decorated native helpers, the explicitly bound upstream "
    "helper profile when present, bounded range expansion, and the admitted "
    "object-storage/rotate calls. Starred arguments expand admitted storage into "
    "matching scalar c/p parameters. They do not convert Plain Exprs into Python "
    "integers. Arbitrary Python/NumPy builtins, kwargs and conversion calls are "
    "not added by generation guidance.")

# Replaces explanations in the existing type_rules structure, not a capability
# registry. The unchanged validators remain authoritative.
NATIVE_TYPE_RULES = {
    "dtype": NATIVE_DTYPE,
    "shape": NATIVE_SHAPE,
    "unary": NATIVE_UNARY,
    "compare": NATIVE_CONTROL,
    "returns": NATIVE_RETURNS,
    "broadcast": NATIVE_BROADCAST,
    "calls": NATIVE_CALLS,
}


def effective_contract(request):
    """Resolve only the two existing validated contracts; reject a new profile."""
    if type(request) is not dict:
        raise ValueError("Guidance request must be a dictionary")
    if "construction_profile" not in request:
        return "native"
    if request["construction_profile"] != PUBLIC_PROFILE:
        raise ValueError("Unknown construction profile")
    return "public"



def base_version(request):
    """Use output-alias instructions only when the request actually selects them."""
    from unified_output_aliases import POLICY, policy
    return "explicit-v15" if policy(request["layout"]) == POLICY else "explicit-v14"

ROUTE_MEAN = (
    "A tensor_route binding implements its complete declared operation. For a "
    "mean node, the bound route vector already includes the reciprocal of the "
    "full logical reduction member count. Sum all of that node's route "
    "contributions without applying that reciprocal again. The divisor uses "
    "the declared logical axes, not P or padded slots. A sum route has unit "
    "aggregation coefficients. Shape/permutation/slice/join routes carry "
    "their selection and placement coefficients. Use node_id and source to "
    "identify the operation; bound vector names have no numerical meaning.")
ROUTE_COORDINATES = (
    "Each tensor_route selects input_chunk, output_chunk and step in the frozen "
    "packing descriptor. With no giant_step, the coefficient vector is in "
    "destination coordinates: destination j receives vector[j] times source "
    "[(j+step) mod P]. When baby_step/giant_step are present, the registered "
    "vector is pre-rotated as specified by the packing rules. Applying an "
    "additional scaling, moving the mask to another coordinate frame, or "
    "ignoring a declared route changes the mathematical operation.")

def route_semantics(request):
    descriptor = request.get("constant_origins", {}).get("capacity_tensor_packing", {})
    if not any(b.get("role") == "tensor_route" for b in descriptor.get("bindings", [])):
        return None
    return dict(version="bound-routes-mathematics-v1",
                selected_roles=["tensor_route"],
                normalization=ROUTE_MEAN, coordinates=ROUTE_COORDINATES)


def specification(request, inherited):
    """Return a fresh v16 view of existing v15 guidance, with native type scope.

    Public-v1 keeps the explanations for its own admitted public AST operations.
    The caller chooses v14/v15 by the actual output binding. Previous request
    preparation and every earlier guidance version remain unchanged.
    """
    contract = effective_contract(request)
    if type(inherited) is not dict or inherited.get("version") not in BASE_VERSIONS:
        raise ValueError("Contract scope guidance requires explicit-v14/v15 context")
    result = copy.deepcopy(inherited)
    result["version"] = VERSION
    routes = route_semantics(request)
    if routes is not None:
        result["bound_route_semantics"] = routes
    if contract == "public":
        return result
    typed = result["typed_request_bindings"]
    if type(typed.get("type_rules")) is not dict:
        raise ValueError("Missing inherited type rules")
    if not set(NATIVE_TYPE_RULES).issubset(typed["type_rules"]):
        raise ValueError("Incomplete inherited type rules")
    typed["type_rules"].update(NATIVE_TYPE_RULES)
    # Preserve bindings, constant origins, output order and directed requirements.
    result["helper_rule"] = NATIVE_HELPERS
    result["typed_construction_semantics"]["typed_values"] = [
        NATIVE_DTYPE, NATIVE_SHAPE, NATIVE_INTEGER, NATIVE_UNARY,
        NATIVE_RETURNS, NATIVE_CONTROL,
    ]
    expression = result["expression_construction_rules"]
    expression["native_helpers"] = NATIVE_HELPERS
    expression["type_boundary"] = NATIVE_INTEGER
    expression["native_loops"] = NATIVE_CONTROL
    return result


def type_repair_hint(request, diagnostic="", *, reason_code=None):
    """Reuse fixed v6 diagnostic categories, substituting their native wording.

    No diagnostic substring is returned. reason_code must come from the trusted
    structured diagnostic, never from the provider's claim about its program.
    """
    contract = effective_contract(request)
    from unified_typed_guidance import TYPE_RULES, repair_hint as previous
    text = previous(request, diagnostic)
    if contract == "public":
        return text
    for name, rule in NATIVE_TYPE_RULES.items():
        text = text.replace(TYPE_RULES[name], rule)
    extra = []
    if reason_code == "native_control_flow_not_enabled":
        extra.extend((NATIVE_CONTROL, NATIVE_INTEGER))
    elif reason_code == "native_array_constructor":
        extra.append(NATIVE_DTYPE)
    if type(diagnostic) is str and any(
        marker in diagnostic for marker in (
            "Rotation requires", "Native loop iterable", "Native range integer",
            "Native storage integer", "Native negation", "Native core arithmetic")):
        extra.append(NATIVE_INTEGER)
        if "Native negation" in diagnostic:
            extra.append(NATIVE_UNARY)
    return "\n".join(dict.fromkeys([part for part in [text, *extra] if part]))[:6000]


def repair_hint(request, source, diagnostic="", stage="static_check", *, witness=None):
    """Versioned v16 routing for the existing bounded feedback entry points.

    The selected existing output adapter sees a transient correctly hashed view; neither
    the request nor any historical evidence is rewritten. The caller validates the registered v16 request before invoking this wrapper.
    """
    if request.get("generation_guidance", {}).get("version") == "explicit-v21":
        from unified_origin_guidance import repair_hint as origin_hint
        return origin_hint(request, source, diagnostic, stage, witness=witness)
    if request.get("generation_guidance", {}).get("version") == "explicit-v20":
        from unified_suffix_stage_guidance import repair_hint as suffix_hint
        return suffix_hint(request, source, diagnostic, stage, witness=witness)
    if request.get("generation_guidance", {}).get("version") == "explicit-v19":
        from unified_binding_coordinate_guidance import repair_hint as binding_hint
        return binding_hint(request, source, diagnostic, stage, witness=witness)
    if request.get("generation_guidance", {}).get("version") not in (VERSION, "explicit-v17", "explicit-v18"):
        return ""
    if type(source) is not str or len(source.encode()) > 65536:
        return ""
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 4096:
            return ""
        from unified_graph_contract import validate_request, explicit_generation_guidance
        from public_weight_context import request_digest
        validate_request(request)
        reason = None
        if witness is not None:
            from application_diagnostics import checked
            reason = checked(witness)["reason_code"]
        if (reason is None and stage == "static_check" and effective_contract(request) == "native"
                and any(type(n) in (ast.If, ast.IfExp, ast.While, ast.Compare, ast.BoolOp) for n in ast.walk(tree))):
            reason = "native_control_flow_not_enabled"
        parts = []
        if stage == "static_check":
            scoped = type_repair_hint(request, diagnostic, reason_code=reason)
            if scoped:
                parts.append(scoped)
        selected = base_version(request)
        legacy = dict(
            request,
            generation_guidance=explicit_generation_guidance(request, selected))
        legacy["request_id"] = request_digest(legacy)
        if stage == "numerical_comparison":
            coordinates = ""
            if request["generation_guidance"]["version"] in ("explicit-v17","explicit-v18"):
                from unified_route_coordinate_guidance import coordinate_hint
                coordinates = coordinate_hint(request)
            routes = route_semantics(request)
            if routes:
                parts.append(ROUTE_MEAN + "\n" + (coordinates or ROUTE_COORDINATES))
            elif coordinates:
                parts.append(coordinates)
            if request["generation_guidance"]["version"] == "explicit-v18":
                from unified_layout_phase_guidance import coordinate_hint as tile_hint
                tile_coordinates=tile_hint(request)
                if tile_coordinates:parts.append(tile_coordinates)
        if stage != "static_check":
            if selected == "explicit-v15":
                from unified_output_alias_guidance import repair_hint as previous
            else:
                from unified_output_binding_guidance import repair_hint as previous
            prior = previous(legacy, source, diagnostic, stage, witness=witness)
            if effective_contract(request) == "native":
                from unified_typed_guidance import TYPE_RULES
                for name, rule in NATIVE_TYPE_RULES.items():
                    prior = prior.replace(TYPE_RULES[name], rule)
            if prior:
                parts.append(prior)
        return "\n".join(dict.fromkeys(parts))[:6500]
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError,
            RecursionError, MemoryError):
        return ""
