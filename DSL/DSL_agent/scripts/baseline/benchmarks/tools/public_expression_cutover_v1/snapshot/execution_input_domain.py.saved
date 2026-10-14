"""Pre-encryption checks of hash-bound, explicitly declared approximation domains.

Only trusted host code supplies the original graph and normalized logical inputs.
Expected outputs and observed intermediate values never enter diagnostics.
"""
import copy
import math
from benchmark_graph import NAME, digest, require

FORMAT = "poseidon-execution-domain-check-v1"
DIAGNOSTIC = "poseidon-execution-input-diagnostic-v1"
PROFILE_KEYS = ("approximation", "exp", "reciprocal", "log")
DOMAIN_FAILURE = "Reference input outside explicit approximation domain"


def declared_profiles(node):
    """Return public interval declarations, without weights or coefficients."""
    profiles = {}
    for key in PROFILE_KEYS:
        value = node["attrs"].get(key)
        if type(value) is not dict or "domain" not in value:
            continue
        interval = value["domain"]
        require(type(interval) is list and len(interval) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) for v in interval) and
                -1024 <= interval[0] < interval[1] <= 1024,
                "Execution approximation interval")
        profiles[key] = list(interval)
    return profiles


class ApproximationDomainRejected(ValueError):
    """A controlled public diagnostic; contains no observed values."""

    def __init__(self, index, node, profiles, sample_index):
        require(type(index) is int and 0 <= index < 4096 and
                type(sample_index) is int and 0 <= sample_index < 16,
                "Execution domain location")
        require(type(node["id"]) is str and NAME.fullmatch(node["id"]) and
                type(node["op"]) is str and NAME.fullmatch(node["op"]),
                "Execution domain node")
        require(profiles and profiles == declared_profiles(node),
                "Execution domain diagnostic declarations")
        self._public = dict(
            format=DIAGNOSTIC, code="approximation_domain_rejected", layer="input",
            location=dict(kind="model_node", index=index, id=node["id"], operator=node["op"]),
            context=dict(sample_index=sample_index, declared_domains=copy.deepcopy(profiles)),
            retry_policy="supply_inputs_in_declared_domain")
        super().__init__("Input is outside a declared approximation domain")

    @property
    def public_failure(self):
        return copy.deepcopy(self._public)


def _evaluate_prefix(model, inputs, count):
    from operator_reference import evaluate
    if model.get("format") in ("poseidon-operator-graph-v2","poseidon-operator-graph-v3"):
        from operator_arithmetic_graph import evaluate_reference as evaluate
    prefix = dict(model, nodes=model["nodes"][:count])
    prefix["outputs"] = [dict(name="domain_probe", value=prefix["nodes"][-1]["outputs"][0])]
    # This uses the existing independent reference, with approximation semantics.
    # It never invokes lowering, generated DSL, a compiler or a runtime.
    evaluate(prefix, inputs, ideal=False)


def _first_violation(model, inputs, last):
    """A domain failure remains in every later prefix; search only on rejection."""
    low, high = 0, last
    while low < high:
        middle = (low + high) // 2
        try:
            _evaluate_prefix(model, inputs, middle + 1)
        except ValueError as exc:
            if str(exc) != DOMAIN_FAILURE:
                raise ValueError("Execution domain preflight could not evaluate inputs") from None
            high = middle
        else:
            low = middle + 1
    return low


def check_approximation_domains(original, logical_samples):
    """Check normalized float64 samples against the bound original graph.

    This is an input-domain preflight, not CKKS numerical or application-accuracy
    certification. Basic graphs without explicit declarations are labeled as such.
    """
    require(type(logical_samples) is list and 1 <= len(logical_samples) <= 16,
            "Execution domain sample count")
    entries = [(i, declared_profiles(node)) for i, node in enumerate(original["nodes"])]
    entries = [(i, p) for i, p in entries if p]
    report = dict(format=FORMAT, status="not_declared" if not entries else "passed",
                  source_sha256=digest(original), samples=len(logical_samples),
                  profile_count=sum(len(p) for _, p in entries), node_count=len(entries),
                  scope="Declared original-graph approximation intervals only")
    if not entries:
        return report
    last = entries[-1][0]
    for sample_index, inputs in enumerate(logical_samples):
        try:
            _evaluate_prefix(original, inputs, last + 1)
        except ValueError as exc:
            if str(exc) != DOMAIN_FAILURE:
                raise ValueError("Execution domain preflight could not evaluate inputs") from None
            index = _first_violation(original, inputs, last)
            node = original["nodes"][index]
            profiles = declared_profiles(node)
            require(profiles, "Execution domain failure lacks a declaration")
            raise ApproximationDomainRejected(index, node, profiles, sample_index) from None
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            raise ValueError("Execution domain preflight could not evaluate inputs") from None
    return report
