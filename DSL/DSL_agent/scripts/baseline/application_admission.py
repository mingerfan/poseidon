"""Opt-in application admission diagnostics derived only from public witnesses.

These checks do not replace graph/source validation, capability assessment or
backend qualification. Unknown validator failures remain deliberately generic.
"""
import copy
import math
from benchmark_graph import require
from model_resources import resolve

LEGACY = "legacy-v1"
STRUCTURED = "structured-v1"
POLICIES = (LEGACY, STRUCTURED)
FORMAT = "poseidon-admission-diagnostic-v1"
REASONS = {
    "admission_policy": ("input", "invalid_admission_policy"),
    "backend": ("capability", "backend_not_supported"),
    "validation_level": ("input", "validation_level_not_supported"),
    "generation_budget": ("input", "generation_budget_invalid"),
    "operational_limits": ("input", "operational_limits_invalid"),
    "resource_profile": ("input", "resource_profile_not_supported"),
    "input_count": ("input", "input_count_exceeded"),
    "input_rank": ("input", "input_rank_exceeded"),
    "input_elements": ("input", "input_elements_exceeded"),
    "node_count": ("input", "graph_node_count_exceeded"),
    "source_fields": ("input", "model_source_fields_invalid"),
    "trusted_source": ("input", "local_code_trust_required"),
    "unknown": ("input", "input_or_capability_unsupported"),
    "source_unknown": ("input", "model_source_unsupported"),
}

def diagnostic(reason, *, location=None, context=None):
    require(reason in REASONS, "Admission reason")
    layer, code = REASONS[reason]
    # Public counts and limits only. Locations are controlled, not copied from
    # exceptions, file paths, model identifiers or arbitrary user values.
    require(location in (None, "backend", "validation_level", "limits",
                         "operational_limits", "resource_profile", "inputs",
                         "nodes", "model_source", "trust_local_code"), "Admission location")
    context = {} if context is None else context
    require(type(context) is dict and set(context) <= {"actual", "limit"}, "Admission context")
    require(all(type(v) is int and 0 <= v <= 2**63-1 for v in context.values()), "Admission counts")
    return dict(format=FORMAT, layer=layer, code=code, reason_code=reason,
                location=location, context=copy.deepcopy(context), owner="request",
                retry_policy="never_automatic")

class AdmissionRejected(ValueError):
    def __init__(self, reason, **kwargs):
        self.public_failure = diagnostic(reason, **kwargs)
        super().__init__(self.public_failure["code"])

def policy(value):
    if type(value) is not str or value not in POLICIES:
        raise AdmissionRejected("admission_policy")
    return value

def precheck(model, *, backend, validation_level, limits):
    from application_policy import BACKEND, LEVELS, budget
    if backend != BACKEND:
        raise AdmissionRejected("backend", location="backend")
    if type(validation_level) is not str or validation_level not in LEVELS:
        raise AdmissionRejected("validation_level", location="validation_level")
    try:
        budget(limits)
    except (ValueError, TypeError, KeyError):
        raise AdmissionRejected("generation_budget", location="limits") from None
    if type(model) is not dict:
        return
    # Diagnose only when the graph format/profile and shape witnesses are
    # recognized. The normal validator rejects all other malformed cases.
    if model.get("format") not in ("poseidon-model-graph-v1", "poseidon-model-graph-v2"):
        return
    profile = model.get("resource_profile", "legacy-v1") if model["format"].endswith("v2") else "legacy-v1"
    try:
        resources = resolve(profile)
    except (ValueError, TypeError):
        raise AdmissionRejected("resource_profile", location="resource_profile") from None
    inputs = model.get("inputs")
    if type(inputs) is list:
        if len(inputs) > resources.inputs:
            raise AdmissionRejected("input_count", location="inputs",
                                    context=dict(actual=len(inputs), limit=resources.inputs))
        total = 0
        shapes_valid = True
        for item in inputs:
            shape = item.get("shape") if type(item) is dict else None
            if type(shape) is not list:
                shapes_valid = False
                continue
            if len(shape) > resources.rank:
                raise AdmissionRejected("input_rank", location="inputs",
                                        context=dict(actual=len(shape), limit=resources.rank))
            if not shape or not all(type(x) is int and 0 < x <= resources.axis_size for x in shape):
                shapes_valid = False
                continue
            total += math.prod(shape)
        if shapes_valid and resources.input_elements < total <= 2**63-1:
            raise AdmissionRejected("input_elements", location="inputs",
                                    context=dict(actual=total, limit=resources.input_elements))
    nodes = model.get("nodes")
    if type(nodes) is list and len(nodes) > resources.expanded_nodes:
        raise AdmissionRejected("node_count", location="nodes",
                                context=dict(actual=len(nodes), limit=resources.expanded_nodes))

def source_precheck(source):
    from application_model_source import FORMAT
    if type(source) is not dict or source.get("format") != FORMAT:
        raise AdmissionRejected("source_fields", location="model_source")
    kind = source.get("kind")
    expected = {"graph": {"format", "kind", "model"},
                "restricted_python": {"format", "kind", "manifest"},
                "trusted_pytorch": {"format", "kind", "manifest", "trust_local_code"}}
    if type(kind) is not str or kind not in expected or set(source) != expected[kind]:
        raise AdmissionRejected("source_fields", location="model_source")
    if kind == "trusted_pytorch" and source["trust_local_code"] is not True:
        raise AdmissionRejected("trusted_source", location="trust_local_code")
