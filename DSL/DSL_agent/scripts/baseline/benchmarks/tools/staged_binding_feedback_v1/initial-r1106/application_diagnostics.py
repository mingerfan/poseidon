"""Versioned host diagnostics and repeat decisions; provider wire remains v1.

Only trusted adapters call these functions. Candidate fields are never accepted
as diagnostic metadata. Compiler text and private reference values are not parsed.
"""
import ast
import copy
import hashlib
import json

FORMAT = "poseidon-diagnostic-v2"
LEGACY = "legacy-v1"
PRECISE = "precise-v2"
POLICIES = (LEGACY, PRECISE)

# reason -> layer, public message. No arbitrary diagnostic text is rendered.
MESSAGES = {
    "response_contract": ("response_parse", "Return the fixed JSON response schema and request binding."),
    "source_syntax": ("source_parse", "Candidate source must be valid Python syntax."),
    "entry_missing": ("static_check", "Define exactly one top-level function named golden."),
    "source_size": ("static_check", "Candidate source exceeds the fixed source byte limit."),
    "contract_rejected": ("static_check", "Candidate violates the fixed DSL contract."),
    "undefined_value": ("static_check", "The name at this source location is not bound. Use function parameters, earlier local assignments or approved public_constants names; model.constants names are not implicit DSL globals."),
    "native_array_constructor": ("static_check", "Native Expr containers require exactly np.array(data, dtype=object). Numeric dtype arrays are not admitted public weights or ciphertext slots."),
    "construct_not_executed": ("static_check", "The required construction was not executed."),
    "construction_unresolved": ("static_check", "Required construction contribution could not be established."),
    "frontend_rejected": ("dsl_trace", "The real frontend rejected the candidate."),
    "compiler_rejected": ("compiler", "Compilation rejected the candidate; inspect the approved compiler feedback."),
    "artifact_rejected": ("artifact_gate", "Compiled artifacts violate the fixed backend contract."),
    "compiler_scale_capacity": ("compiler", "Accumulated scale exceeds the fixed compiler capacity."),
    "artifact_scale_capacity": ("artifact_gate", "Scale exceeds the capacity at the identified instruction."),
    "artifact_level_exhausted": ("artifact_gate", "The identified instruction exhausts the modulus chain."),
    "runtime_rejected": ("poseidon_runtime", "Encrypted execution failed."),
    "numerical_mismatch": ("numerical_comparison", "Encrypted outputs failed the fixed numerical comparison."),
    "integrity_rejected": ("integrity", "Trusted input, dependency or artifact integrity failed."),
    "environment_failure": ("pipeline", "The execution service could not complete validation."),
    "provider_failure": ("provider", "The provider could not complete generation."),
}
EXACT = {
    "entry_missing": {"expected", "actual"},
    "source_size": {"limit", "observed"},
    "compiler_scale_capacity": {"compiler_accumulated_scale", "compiler_budget"},
    "artifact_scale_capacity": {"scale_bits", "capacity_bits"},
    "artifact_level_exhausted": {"input_level", "required_levels"},
}
CONSTRAINTS = {
    "expected", "actual", "limit", "observed", "compiler_accumulated_scale",
    "compiler_budget", "scale_bits", "capacity_bits", "input_level", "required_levels",
}
CODES = {
    "response_parse": ("candidate_response_rejected", "candidate", "repair_within_budget"),
    "source_parse": ("candidate_response_rejected", "candidate", "repair_within_budget"),
    "static_check": ("candidate_contract_rejected", "candidate", "repair_within_budget"),
    "dsl_trace": ("backend_rejected", "candidate_or_backend", "diagnose"),
    "compiler": ("backend_rejected", "candidate_or_backend", "diagnose"),
    "artifact_gate": ("backend_rejected", "candidate_or_backend", "diagnose"),
    "poseidon_runtime": ("backend_rejected", "candidate_or_backend", "diagnose"),
    "numerical_comparison": ("numerical_mismatch", "candidate_or_backend", "diagnose"),
    "integrity": ("integrity_rejected", "host", "never_automatic"),
    "pipeline": ("pipeline_failure", "host", "review"),
    "provider": ("pipeline_failure", "host", "review"),
}

def _location(value):
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Diagnostic location")
    kind = value.get("kind")
    if kind == "entry" and value == {"kind": "entry", "name": "golden"}:
        return dict(value)
    if kind == "candidate" and set(value) == {"kind"}:
        return dict(value)
    if kind == "source" and set(value) == {"kind", "line", "column"}:
        if all(type(value[k]) is int and 0 <= value[k] <= 65536 for k in ("line", "column")):
            return dict(value)
    if kind == "instruction" and set(value) == {"kind", "index"}:
        if type(value["index"]) is int and 0 <= value["index"] <= 1000000:
            return dict(value)
    raise ValueError("Diagnostic location")

def make(reason, *, location=None, context=None):
    if reason not in MESSAGES:
        raise ValueError("Diagnostic reason")
    context = {} if context is None else context
    if type(context) is not dict or not set(context) <= CONSTRAINTS:
        raise ValueError("Diagnostic constraints")
    if any(type(v) is not int or not 0 <= v <= 2**53 for v in context.values()):
        raise ValueError("Diagnostic public integer constraint")
    layer = MESSAGES[reason][0]
    code, owner, retry = CODES[layer]
    if reason == "construct_not_executed":
        code = "required_construct_not_executed"
    elif reason == "construction_unresolved":
        code, owner, retry = "construction_evidence_unresolved", "coverage_checker_or_candidate", "review"
    location = _location(location)
    exact = reason in EXACT and set(context) == EXACT[reason]
    expected_location = {"entry_missing": "entry", "source_size": "candidate",
        "compiler_scale_capacity": "instruction", "artifact_scale_capacity": "instruction",
        "artifact_level_exhausted": "instruction"}.get(reason)
    exact = exact and location is not None and location["kind"] == expected_location
    # Validate the exact predicates instead of trusting a supplied repeat flag.
    if exact:
        if reason == "entry_missing":
            exact = context == {"expected": 1, "actual": 0}
        elif reason == "source_size":
            exact = context["limit"] == 65536 and context["observed"] > context["limit"]
        elif reason == "compiler_scale_capacity":
            exact = context["compiler_accumulated_scale"] > context["compiler_budget"]
        elif reason == "artifact_scale_capacity":
            exact = context["scale_bits"] >= context["capacity_bits"]
        elif reason == "artifact_level_exhausted":
            exact = context["required_levels"] > context["input_level"]
    return dict(format=FORMAT, layer=layer, code=code, reason_code=reason, owner=owner,
                retry_policy=retry, location=location, context=dict(context),
                repeat_key_available=bool(exact))

def checked(value):
    if type(value) is not dict:
        raise ValueError("Diagnostic object")
    fields = {"format", "layer", "code", "reason_code", "owner", "retry_policy",
              "location", "context", "repeat_key_available"}
    if set(value) != fields or value.get("format") != FORMAT:
        raise ValueError("Diagnostic fields")
    rebuilt = make(value.get("reason_code"), location=value.get("location"), context=value.get("context"))
    if value != rebuilt:
        raise ValueError("Diagnostic mismatch")
    return rebuilt

def from_result(result, *, candidate=None):
    raw = result.get("failure") or {}
    public = result.get("public_feedback") or {}
    # Terminal trust failures cannot be overridden by a conflicting typed hint.
    if raw.get("layer") == "integrity" or public.get("category") == "integrity":
        return make("integrity_rejected")
    if public.get("category") == "infrastructure":
        return make("environment_failure")
    supplied = result.get("structured_diagnostic")
    if supplied is None and raw.get("format") == FORMAT:
        supplied = raw
    if supplied is not None:
        try:
            return checked(supplied)
        except (ValueError, TypeError):
            return make("environment_failure")
    layer = raw.get("layer", public.get("layer", "pipeline"))
    reason = {
        "response_parse": "response_contract", "source_parse": "source_syntax",
        "static_check": "contract_rejected", "dsl_trace": "frontend_rejected",
        "compiler": "compiler_rejected", "artifact_gate": "artifact_rejected",
        "poseidon_runtime": "runtime_rejected", "numerical_comparison": "numerical_mismatch",
        "integrity": "integrity_rejected", "provider": "provider_failure",
    }.get(layer, "environment_failure")
    if layer == "static_check":
        if raw.get("code") == "required_construct_not_executed":
            return make("construct_not_executed")
        if raw.get("code") == "construction_evidence_unresolved":
            return make("construction_unresolved")
    # Environment/integrity classification takes priority over source inspection.
    source = candidate.get("hecate_source") if type(candidate) is dict else None
    if layer in ("static_check", "source_parse") and type(source) is str:
        size = len(source.encode())
        if size > 65536:
            return make("source_size", location={"kind": "candidate"},
                        context={"limit": 65536, "observed": size})
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return make("source_syntax", location={"kind": "source",
                "line": min(error.lineno or 0, 65536), "column": min(error.offset or 0, 65536)})
        if layer == "static_check" and not any(type(n) is ast.FunctionDef and n.name == "golden" for n in tree.body):
            return make("entry_missing", location={"kind": "entry", "name": "golden"},
                        context={"expected": 1, "actual": 0})
    # Coarse compiler diagnostics lack a stable instruction location. Keep their
    # public constraints for inspection, but they cannot trigger repeat stopping.
    earth = result.get("compiler_diagnosis")
    context = {}
    if layer == "compiler" and type(earth) is dict and earth.get("format") == "fixed-earth-mul-diagnosis-v1":
        context = {k: earth[k] for k in ("compiler_accumulated_scale", "compiler_budget")
                   if type(earth.get(k)) is int and 0 <= earth[k] <= 2**53}
    return make(reason, context=context)

def repeat_key(value):
    value = checked(value)
    if not value["repeat_key_available"]:
        return None
    content = {k: value[k] for k in ("format", "layer", "reason_code", "location", "context")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def render(value, prior=None):
    """Render only approved values into the unchanged provider feedback schema."""
    from public_feedback_contract import public_feedback
    value = checked(value)
    category = "integrity" if value["layer"] == "integrity" else "infrastructure" if value["owner"] == "host" else "candidate"
    public = dict(status="failed", layer=value["layer"], category=category,
                  diagnostic=MESSAGES[value["reason_code"]][1])
    # Already approved backend hints remain useful. Only preserve the narrow
    # provider-approved tool/aggregate fields, never arbitrary local diagnostics.
    if prior is not None:
        try:
            safe = public_feedback(prior)
        except (ValueError, TypeError):
            safe = {}
        for name in ("tool_diagnostic", "numerical_summary"):
            if name in safe:
                public[name] = copy.deepcopy(safe[name])
    # The reason and context themselves have already passed the strict schema.
    if value["location"] is not None:
        public["diagnostic"] += " Location: " + json.dumps(value["location"], sort_keys=True)
    if value["context"]:
        public["diagnostic"] += " Constraints: " + json.dumps(value["context"], sort_keys=True)
    public["diagnostic"] = public["diagnostic"].encode()[:7800].decode("utf-8", errors="ignore")
    return public_feedback(public)

def attach(result, *, candidate=None):
    result = copy.deepcopy(result)
    if result.get("status") == "passed":
        return result
    diagnostic = from_result(result, candidate=candidate)
    result["structured_diagnostic"] = diagnostic
    result["failure"] = diagnostic
    result["public_feedback"] = render(diagnostic, result.get("public_feedback"))
    return result


class RepairDecision:
    """Shared precise policy state for application and CLI control adapters."""
    def __init__(self):
        self.seen = set()
        self.last_failure = None
        self.last_key = None
        self.repeated = 0

    def before_candidate(self, raw):
        try:
            from candidate_contract import strict_json
            candidate = strict_json(raw)
            source = candidate["hecate_source"]
            if type(source) is not str:
                raise TypeError("Source")
            identity=source.encode()
            if len(identity)<=65536:
                try:
                    tree=ast.parse(source)
                    if sum(1 for _ in ast.walk(tree))<=4096:
                        identity=ast.dump(tree,include_attributes=False).encode()
                except (SyntaxError,ValueError,RecursionError):
                    pass
            fingerprint = hashlib.sha256(identity).hexdigest()
        except (ValueError, TypeError, KeyError, AttributeError):
            fingerprint = hashlib.sha256(str(raw).encode()).hexdigest()
        if fingerprint in self.seen:
            self.last_failure = dict(code="repeated_candidate", layer="generation", retry_policy="stop")
            return "validation_failed"
        self.seen.add(fingerprint)

    def after_failure(self, diagnostic):
        from application_policy import decision
        diagnostic = checked(diagnostic)
        self.last_failure = diagnostic
        route = decision(diagnostic)
        if route != "repair":
            return route
        key = repeat_key(diagnostic)
        self.repeated = 0 if key is None else self.repeated + 1 if key == self.last_key else 1
        self.last_key = key
        if self.repeated >= 2:
            self.last_failure = dict(diagnostic, stop_reason="repeated_failure")
            return "validation_failed"
