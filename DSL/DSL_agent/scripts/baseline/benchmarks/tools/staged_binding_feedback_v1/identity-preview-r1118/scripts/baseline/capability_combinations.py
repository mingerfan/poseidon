"""Explicit opt-in compositions; absence preserves all legacy request bytes."""
from seal_artifact_gate import require
NAME = "native-bn-directed-v1"
HELPER_PROFILE = "upstream-poly-bn-silu-v2"
CONSTRUCTIONS = ("unified-native-call-scalar_cipher", "unified-native-copy", "unified-scalar")
RULES = """
Composition native-bn-directed-v1: native construction and bound HE_BN helper
requirements are independent and must both contribute to named outputs.
Only HE_BN-prefixed upstream callees may be used in this composition. Retain all
native type/layout/resource limits and both real frontend witness checks.
This composition does not enable public AST helpers, chunked layouts or bootstrap.
"""
def validate_selection(name, construction_profile, construction, helper_profile, helper_exercise, chunk_period):
    require(name == NAME, "Unknown capability composition")
    require(construction_profile is None and construction in CONSTRUCTIONS,
            "Composition requires an explicitly supported native construction")
    require(helper_profile == HELPER_PROFILE and chunk_period is None,
            "Composition requires unchunked bound BN profile")
    require(type(helper_exercise) is list and bool(helper_exercise) and
            all(type(n) is str and n.startswith("HE_BN") for n in helper_exercise),
            "Composition requires directed actual BN helper evidence")
def validate_calls(source, request):
    import ast
    if "capability_composition" not in request:return
    helpers=request["upstream_helpers"]["helpers"]
    for node in ast.walk(ast.parse(source)):
        if type(node) is ast.Call and type(node.func) is ast.Name and node.func.id in helpers:
            require(node.func.id.startswith("HE_BN"), "Upstream callee outside BN composition")
