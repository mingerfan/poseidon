"""Explicit deterministic layout strategies; generation requests stay immutable."""
import copy
from benchmark_graph import require
VERSION = "poseidon-rule-layout-strategies-v1"
MIXED = "mixed-packed-cells-v1"
OPERATORS = ("linear", "add", "subtract", "multiply", "negate", "square", "power",
             "polynomial", "reshape", "flatten", "transpose", "permute", "rotate",
             "concat", "stack", "slice", "split", "sum", "mean")

def describe():
    return {MIXED: dict(version=VERSION, source_kind="deterministic_rule_baseline",
        backend="poseidon_cpu", operators=list(OPERATORS),
        layout="original capacity ABI, canonical C-order input/output chunks",
        rule="packed Linear and same-coordinate arithmetic; lazy replicated cells for routed arithmetic",
        resource_limits="unchanged candidate AST, primitive work and source limits",
        compiler_status="unverified", encrypted_status="unverified", agent_status="unverified",
        scope="generic declared subset; selected configuration requires qualification",
        candidate_rewriting=False)}

def lower(request, strategy, *, source_kind="deterministic_rule_baseline"):
    require(type(strategy) is str and strategy in describe(), "Unregistered rule layout strategy")
    require(type(source_kind) is str and source_kind == "deterministic_rule_baseline",
            "Rule-only layout strategy")
    from unified_mixed_layout import lower as selected
    return selected(request, source_kind=source_kind)
