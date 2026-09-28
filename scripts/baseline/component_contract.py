"""Stable data-only component contracts; no environment, credentials or execution."""
import copy
from benchmark_graph import canonical, digest, require, BOUNDS
from unified_graph_contract import prepare, validate_request, validate_candidate, COMPACT
from compiler_configuration import configuration, PROFILE_SHA256

VERSION = "poseidon-agent-component-v1"
OPTION_KEYS = {"compiler_configuration", "construction_profile", "construction",
               "helper_profile", "helper_exercise", "chunk_period", "generation_guidance", "capability_composition"}

def prepare_task(model, options=None):
    options = {} if options is None else copy.deepcopy(options)
    require(type(options) is dict and set(options) <= OPTION_KEYS, "Component option fields")
    name = options.pop("compiler_configuration", "seal-cpu-eva-w45-v1")
    config = configuration(name) if name is not None else None
    return prepare(copy.deepcopy(model), PROFILE_SHA256, config, **options)

def request_options(request):
    """Recover all semantic switches, including explicit absence of a compiler extension."""
    validate_request(request)
    return dict(
        compiler_configuration=request.get("compiler_configuration", {}).get("name"),
        construction_profile=request.get("construction_profile"),
        construction=request.get("construction_exercise", {}).get("id"),
        helper_profile=request.get("upstream_helpers", {}).get("profile"),
        helper_exercise=request.get("upstream_exercise", {}).get("required_helpers"),
        chunk_period=request["layout"]["input_slot_period"] if "logical_inputs" in request["layout"] else None,
        generation_guidance=request.get("generation_guidance", {}).get("version"),
        capability_composition=request.get("capability_composition"))

def reconstruct_request(request):
    options = request_options(request)
    name = options.pop("compiler_configuration")
    policy = request["constant_origins"].get("policy")
    reconstructed = prepare(request["model"], request["compiler_profile_sha256"],
                            configuration(name) if name else None, constant_policy=policy, **options)
    require(canonical(reconstructed) == canonical(request), "Component request reconstruction mismatch")
    return reconstructed

def runner_options(request):
    options = request_options(request)
    args = []
    for name, flag in (("compiler_configuration", "--compiler-configuration"),
                       ("construction", "--unified-exercise"),
                       ("helper_profile", "--unified-helpers"),
                       ("chunk_period", "--unified-chunk-period"),
                       ("generation_guidance", "--unified-guidance"),
                       ("capability_composition", "--capability-composition")):
        if options[name] is not None:
            args += [flag, str(options[name])]
    if options["construction_profile"] is not None:
        args += ["--unified-profile", "public-v1"]
    for helper in options["helper_exercise"] or []:
        args += ["--unified-helper-exercise", helper]
    return args

def failure(layer, diagnostic):
    """Conservative routing: absence of a bounded witness is not proof of no influence."""
    text = str(diagnostic)
    evidence = getattr(diagnostic, "evidence", None)
    if layer == "static_check" and evidence and evidence["reason"] == "not_executed":
        code, owner, retry = "required_construct_not_executed", "candidate", "repair_within_budget"
    elif layer == "static_check" and ("Missing contributing" in text or "influence work bound" in text):
        code, owner, retry = "construction_evidence_unresolved", "coverage_checker_or_candidate", "review"
    elif layer == "static_check":
        code, owner, retry = "candidate_contract_rejected", "candidate", "repair_within_budget"
    elif layer == "numerical_comparison":
        code, owner, retry = "numerical_mismatch", "candidate_or_backend", "diagnose"
    elif layer in ("compiler", "artifact_gate"):
        code, owner, retry = "backend_rejected", "candidate_or_backend", "diagnose"
    elif layer == "integrity":
        code, owner, retry = "integrity_rejected", "host", "never_automatic"
    else:
        code, owner, retry = "pipeline_failure", "host", "review"
    outcome = dict(layer=layer, code=code, owner=owner, retry_policy=retry, diagnostic=text[:2000])
    if evidence is not None:outcome["construction_evidence"] = evidence
    return outcome

def validate_program(candidate, request):
    """Static legality and directed evidence are reported independently; neither proves FHE."""
    validate_request(request)
    outcome = dict(format=VERSION, request_id=request["request_id"], status="rejected",
                   semantic_validation="not_checked", construction_coverage="not_requested",
                   encrypted_execution=False, numerically_validated=False, paid_calls=0)
    try:
        validate_candidate(candidate, request, check_construction=False)
        outcome["semantic_validation"] = "accepted"
    except (ValueError, TypeError, KeyError, IndexError) as error:
        outcome["failure"] = failure("static_check", error)
        return outcome
    directed = "construction_exercise" in request or "upstream_exercise" in request
    if directed:
        try:
            validate_candidate(candidate, request)
            outcome["construction_coverage"] = "static_witness_obtained"
        except (ValueError, TypeError, KeyError, IndexError) as error:
            outcome["construction_coverage"] = "not_verified"
            outcome["failure"] = failure("static_check", error)
            return outcome
    outcome["status"] = "static_accepted_not_executed"
    return outcome

def synthesize(request, provider, evaluator, record, *, max_repairs=3, validation_level="numerical", control=None):
    """Trusted injected capabilities, with the existing bounded loop; no plugin loading."""
    from candidate_contract import run_feedback_loop
    from deepseek_provider import public_feedback
    validate_request(request)
    require(validation_level in ("compiled", "numerical"), "Explicit validation level")
    def checked_evaluate(raw, index):
        result = evaluator(raw, index)
        if result.get("status") == "passed":
            if validation_level == "numerical":
                require(result.get("encrypted_execution") is True and result.get("numerically_validated") is True,
                        "Component success requires encrypted numerical validation")
            else:
                require(result.get("compiled_validated") is True and result.get("validation_level") == "compiled",
                        "Component compiled success requires audited artifacts")
            return dict(status="passed", layer="complete")
        # Qualification reports contain private local evidence paths. Never forward
        # them, arbitrary diagnostic text, or reference arrays to the provider.
        if control is not None:
            control.observe_failure(result)
        feedback = result.get("public_feedback")
        if feedback is None:
            feedback = dict(status="failed", layer="component", category="infrastructure",
                            diagnostic="Qualification failed without approved public feedback")
        return public_feedback(feedback)
    return run_feedback_loop(copy.deepcopy(request), provider, checked_evaluate, record, max_repairs, control=control)

def capabilities():
    from upstream_candidate_helpers import (PROFILE, BN_PROFILE, CONCAT_PROFILE, SPATIAL_PROFILE,
        MAPPED_PROFILE, FUSED_PROFILE, DS_PROFILE, VR_PROFILE, CHUNK_PROFILE, POLYNOMIAL_PROFILE)
    return dict(format=VERSION, logical_graph="poseidon-model-graph-v1", bounds=dict(BOUNDS),
        execution_backend="upstream_SEAL_HEVM_CPU", platforms=["x86_64-linux", "aarch64-linux"],
        construction_profiles=["hecate-unified-native-v1", "hecate-unified-public-v1"],
        helper_profiles=[PROFILE, BN_PROFILE, CONCAT_PROFILE, SPATIAL_PROFILE, MAPPED_PROFILE,
                         FUSED_PROFILE, DS_PROFILE, VR_PROFILE, CHUNK_PROFILE, POLYNOMIAL_PROFILE],
        compatibility_rules=["helpers require native construction", "chunked helpers require their dedicated profile"],
        capability_compositions=["native-bn-directed-v1"], bootstrap=False, paid_worker_actions=False, dynamic_plugins=False,
        evidence_scope="Declared contracts only; not a current all-partition execution score")
