"""Application capability preflight over the existing immutable contracts.

This module checks declarations and prepares bindings. It does not execute a
candidate, consult experiment reports, or certify compiler/numerical success.
Physical resource costs without a compiled candidate remain unknown.
"""
import copy
import math

from benchmark_graph import (FORMAT, CAPACITY_FORMAT, OPS, BOUNDS, canonical, digest,
                             graph_resource_profile, validate)
from compiler_configuration import CPU_CONFIGURATIONS
from model_decomposition import FORMAT as OPERATOR_FORMAT, REGISTRY, REGISTRY_VERSION
from operator_arithmetic_graph import FORMATS as OPERATOR_FORMATS, decompose, preparation_options
from model_resources import PROFILES, identity as resource_identity
from unified_graph_contract import ABI, CONTRACT, validate_request
from unified_chunk_layout import ABI as CHUNK_ABI, PERIODS
from unified_capacity_layout import (ABI as CAPACITY_ABI, EXTENDED_ABI, WIDE_ABI,
                                     WIDE_PERIODS, physical_limits, EXTENDED_PROFILES)

VERSION = "poseidon-application-capabilities-v1"
REPORT = "poseidon-application-capability-report-v1"
BACKEND = "poseidon_cpu"
RESOURCE_CERTAINTIES = ("exact", "conservative", "unknown")
STAGES = ("input", "decomposition", "layout", "compiler", "encrypted", "agent")


def input_sources():
    from pytorch_source_contract import LEGACY,FORMAT as PYTORCH_V2,WEIGHT_FORMAT as PYTORCH_V3,STATIC_FORMAT as PYTORCH_V4,APPROXIMATION_FORMAT as PYTORCH_V5,INDEX_FORMAT as PYTORCH_V6,MATRIX_FORMAT as PYTORCH_V7,FLOAT_POLICIES
    from pytorch_model_intake import SOURCE_FORMATS,SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT,COMMON_FORMAT,ORDERED_FORMAT,ORDERED_COMMON_FORMAT,ORDERED_FUSED_FORMAT
    from model_weight_package import FORMAT as WEIGHT_FORMAT
    from pytorch_model_capture import functional_input_rules
    return dict(
        graph=dict(formats=[FORMAT,CAPACITY_FORMAT,*OPERATOR_FORMATS],input_status="registered",
            arithmetic_normalization=dict(format=OPERATOR_FORMATS[2],
                policy="operator-graph-public-arithmetic-v1",operators=["add","subtract","multiply"],
                public_side=["left","right"],bounds="selected resource profile",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified"),
            matrix_normalization=dict(format=OPERATOR_FORMATS[1],policy="static-tensor-matmul-v1",
                operands=["vectors","matrices","static broadcast batches"],
                public_weights="immutable vectors/matrices/batched tensors",
                scalar_carrier_shape=[1],bounds="selected resource profile",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified")) ,
        restricted_python=dict(format="poseidon-python-model-v1",mode="static_parser",
                               resource_profiles=["legacy-v1"],input_status="registered"),
        trusted_pytorch=dict(formats=[LEGACY,PYTORCH_V2,PYTORCH_V3,PYTORCH_V4,PYTORCH_V5,PYTORCH_V6,PYTORCH_V7,*SOURCE_FORMATS],mode="isolated_explicit_trust",
            factory_parameters=dict(version=PYTORCH_V2,max_bytes=8192,max_depth=8,max_values=1024),
            dtype_policies=list(FLOAT_POLICIES),input_dtype="float64",
            source_files=8,source_bytes=65536,
            output_containers=["tensor","tuple","list","string_key_dict"],output_tensor_leaves=4,
            static_shape_policy=dict(version=PYTORCH_V4,queries=["shape","size","ndim","dim","numel","len"],
                control="public shape only; bounded FX expansion",scalar_outputs="restored rank zero"),
            approximation_selection=dict(version=PYTORCH_V5,configuration="poseidon-pytorch-approximations-v1",
                precedence=["node","module","default"],module_paths="exact FX owner paths",
                conflicts="rejected",unmatched="rejected"),
            static_indexing=dict(version=PYTORCH_V6,basic=["integer","positive_slice","ellipsis","newaxis"],
                public_vector="one axis; surrounding slices/newaxis",calls=["Tensor.__getitem__","Tensor.select","torch.select","Tensor.narrow","torch.narrow","Tensor.index_select","torch.index_select"],
                dynamic=False,empty_tensors=False),
            static_matrix=dict(version=PYTORCH_V7,policy="static-tensor-matmul-v1",
                calls=["Tensor.matmul","torch.matmul","operator.matmul","Tensor.dot","torch.dot","Tensor.mm","torch.mm","Tensor.bmm","torch.bmm"],
                operands=["vectors","matrices","static broadcast batches"],public_weights="fixed vectors/matrices/batched tensors",
                bounds="selected resource profile; materialized product tensor checked before normalization",
                output="PyTorch vector/scalar promotion restored"),
            arithmetic_normalization=dict(version="poseidon-trusted-pytorch-v8",
                policy="poseidon-fx-public-arithmetic-v1",operators=["add","subtract","multiply"],
                calls=["operator","torch function","Tensor method"],public_side=["left","right"],
                input_status="registered",compiler_status="unverified",encrypted_status="unverified",
                agent_status="unverified"),
            application_entry="submit_import",input_status="registered",
            arithmetic_result_shape=dict(version=SHAPE_FORMAT,
                policy="signature-static-arithmetic-shape-v1",
                calls=["Tensor.add","Tensor.sub","Tensor.subtract","Tensor.mul","Tensor.multiply"],
                queries=["shape","size","ndim","dim","numel","len"],
                source="declared signature and registered metadata rules",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified"),
            operator_result_shape=dict(version=OPERATOR_SHAPE_FORMAT,
                policy="signature-static-operator-shape-v1",
                calls=["nn.InstanceNorm1d","nn.InstanceNorm2d","F.instance_norm",
                       "nn.AdaptiveAvgPool1d","nn.AdaptiveAvgPool2d","F.adaptive_avg_pool1d","F.adaptive_avg_pool2d",
                       "nn.ConstantPad1d","nn.ConstantPad2d","nn.ZeroPad2d",
                       "nn.ReflectionPad1d","nn.ReflectionPad2d","nn.ReplicationPad1d","nn.ReplicationPad2d"],
                queries=["shape","size","ndim","dim","numel","len"],
                source="declared signature and registered metadata rules",
                input_status="registered",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified"),
            functional_adaptive_average=dict(version=OPERATOR_SHAPE_FORMAT,policy="static-adaptive-average-v1",
                calls=["F.adaptive_avg_pool1d","F.adaptive_avg_pool2d"],ranks=[3,4],
                output_size="positive static spatial dimensions, each no larger than its input",
                input_status="registered",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified"),
            common_calls=dict(version=COMMON_FORMAT,policy="poseidon-fx-common-calls-v1",
                static_shape_policy="signature-static-common-call-shape-v1",
                arithmetic_alpha="finite static numeric alpha; no out or in-place mutation",
                dropout="eval inference; no in-place mutation",
                sdpa="static Q/K/V, including projections; public boolean, finite additive or causal masks",
                sdpa_dropout=0,approximations="explicit exp and reciprocal configurations",
                bounds="selected resource profile",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified"),
            ordered_tensor_effects=dict(version=ORDERED_FORMAT,
                policy="signature-static-ordered-alias-staging-v1",
                views=["reshape","view","transpose","permute","basic_index","split"],
                copies=["clone","contiguous","detach","copy_"],
                mutations=["add_","sub_","mul_","square_","explicit_approximate_relu"],
                overlap="repeated/partial overlapping writes rejected",
                public_state="immutable; original NPY and derived views bound",
                input_storage="distinct contiguous tensors",reference_inputs="separate clones",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified"),
            ordered_common_calls=dict(version=ORDERED_COMMON_FORMAT,
                policy="poseidon-fx-ordered-common-calls-draft-v1",
                calls=["static_alpha","inference_dropout","scaled_dot_product_attention"],
                composition="registered ordered tensor effects",
                public_state="declared NPY plus independently derived constants",
                approximations="explicit finite domains",
                input_status="registered",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified"),
            ordered_fused_matrix_calls=dict(version=ORDERED_FUSED_FORMAT,
                policy="poseidon-fx-fused-matrix-draft-v1",calls=["addmm","baddbmm","addbmm"],
                call_styles=["function","Tensor method","keyword"],
                matrix_ranks=dict(addmm=2,baddbmm=3,addbmm=3),bias="result-shape broadcast",
                scalars="finite public alpha/beta, absolute value <= 1024",
                matrix_dependency="at least one encrypted operand",ordered_effects="registered",
                input_status="registered",decomposition_status="registered",
                compiler_status="unverified",encrypted_status="unverified",agent_status="unverified"),
            source_application_entry="submit_source",
            functional_input_rules=functional_input_rules()),
        public_weight_package=dict(format=WEIGHT_FORMAT,dtype="<f8",pickle=False,
            generation_policy="descriptors-v1",generation_context_max_bytes=131072,
            trusted_request_max_bytes=4*1024**2,delivery_format="poseidon-dsl-bundle-v6",
            application_entry="submit_import",input_status="registered"),
    )

def describe():
    """Return declarations derived from active registries, not execution scores."""
    import upstream_candidate_helpers as helpers
    from unified_public_contract import CONTRACT as PUBLIC
    helper_profiles = sorted(value for name, value in vars(helpers).items()
                             if (name == "PROFILE" or name.endswith("_PROFILE"))
                             and type(value) is str and value.startswith("upstream-"))
    layouts = [
        dict(abi=ABI, periods=list(PERIODS), resource_profiles=["legacy-v1"],
             input_ciphertexts=BOUNDS["inputs"], output_ciphertexts=BOUNDS["outputs"]),
        dict(abi=CHUNK_ABI, periods=list(PERIODS), resource_profiles=["legacy-v1"],
             input_ciphertexts=BOUNDS["inputs"], output_ciphertexts=BOUNDS["outputs"]),
    ]
    for abi in (CAPACITY_ABI, EXTENDED_ABI, WIDE_ABI):
        profiles = [p for p in PROFILES if abi == CAPACITY_ABI or p in EXTENDED_PROFILES]
        _, inputs, outputs = physical_limits(profiles[0], abi)
        layouts.append(dict(abi=abi, periods=list(WIDE_PERIODS if abi == WIDE_ABI else PERIODS),
                            resource_profiles=profiles, input_ciphertexts=inputs,
                            output_ciphertexts=outputs))
    result = dict(
        format=VERSION, execution_backend=BACKEND,
        graph_formats=[FORMAT, CAPACITY_FORMAT, *OPERATOR_FORMATS], input_sources=input_sources(),
        core_operators={name: dict(attributes=sorted(attrs), input_status="registered",
                                  compiler_status="unverified", encrypted_status="unverified",
                                  agent_status="unverified")
                        for name, attrs in sorted(OPS.items())},
        decomposition_registry=dict(
            version=REGISTRY_VERSION,
            operators={name: dict(attributes=sorted(op.attributes), reference=op.reference,
                                  rule=op.rule, required_capabilities=list(op.capabilities),
                                  input_contract=op.input_contract,
                                  approximation=op.approximation,
                                  decomposition_status="registered",
                                  compiler_status="unverified", encrypted_status="unverified",
                                  agent_status="unverified")
                       for name, op in sorted(REGISTRY.items())}),
        resource_profiles={name: resource_identity(name) for name in PROFILES},
        layouts=layouts, auxiliary_ciphertexts="one trusted encrypted zero per request",
        input_dtype="float64", compiler_configurations=list(CPU_CONFIGURATIONS),
        construction_profiles=[CONTRACT, PUBLIC], helper_profiles=helper_profiles,
        restrictions=[
            dict(code="bootstrap_unavailable", scope="active application backend",
                 reason="The active contract does not expose bootstrap."),
            dict(code="capacity_helper_unverified", scope="capacity graph and upstream helper",
                 reason="Capacity layouts cannot select an upstream helper profile."),
            dict(code="dtype_not_supported", scope="model input",
                 reason="The numerical contract uses float64 real values."),
            dict(code="layout_requires_binding", scope="physical layout",
                 reason="Only the canonical request layout authorizes periods and ciphertext arity."),
        ],
        resource_certainties=list(RESOURCE_CERTAINTIES),
        evidence_scope="contract declarations; no experiment report is consulted",
        acceptance="Registered operators and accepted inputs do not certify a compiled or executable composition.",
    )
    result["identity_sha256"] = digest(result)
    return result


def _stage(status, code):
    return dict(status=status, code=code)


def _resource(value, certainty, basis, bound=None):
    result = dict(value=value, certainty=certainty, basis=basis)
    if bound is not None:
        result["bound"] = bound
    return result


def _report(backend):
    return dict(
        format=REPORT, capability_version=VERSION, backend=backend,
        admission="rejected", execution_readiness="unknown",
        stages={name: _stage("unverified", "not_checked") for name in STAGES},
        resources={}, diagnostics=[], identity={},
        evidence_scope="request preflight only",
        encrypted_execution=False, numerically_validated=False, paid_calls=0,
    )


def _finish(report):
    # This compact summary deliberately excludes graph labels, model/request hashes,
    # arbitrary diagnostics and public values. Context selection may depend on it.
    report["summary"] = dict(
        admission=report["admission"], execution_readiness=report["execution_readiness"],
        stages={name: copy.deepcopy(report["stages"][name]) for name in STAGES},
        resource_profile=report.get("resource_profile"),
        resources={name: dict(value=item["value"], certainty=item["certainty"])
                   for name, item in report["resources"].items() if name != "model_json_bytes"},
    )
    return report


def _reject(report, stage, code, message, *, status="rejected"):
    report["stages"][stage] = _stage(status, code)
    report["diagnostics"].append(dict(stage=stage, code=code, message=message))
    return _finish(report)


def _check_requirements(report, requirements):
    if requirements is None:
        return None
    if (type(requirements) is not dict or set(requirements) - {"bootstrap"} or
            ("bootstrap" in requirements and type(requirements["bootstrap"]) is not bool)):
        return _reject(report, "input", "requirements_invalid",
                       "Requirements accept only an explicit bootstrap boolean.")
    if requirements.get("bootstrap"):
        return _reject(report, "encrypted", "bootstrap_unavailable",
                       "The active Poseidon application contract does not expose bootstrap.",
                       status="blocked")
    return None


def _input_shape_rejection(report, model, profile):
    """Give stable public diagnostics before the detailed trusted graph validator."""
    specs = model.get("inputs")
    if type(specs) is list:
        for spec in specs:
            if type(spec) is not dict:
                continue
            # Model graph declarations have no dtype field. A typed source frontend
            # must normalize to the existing fixed float64 graph contract first.
            if "dtype" in spec:
                return _reject(report, "input", "dtype_declaration_unsupported",
                               "Graph input declarations use name and shape; dtype is fixed to float64.")
            shape = spec.get("shape")
            if type(shape) is list and not 1 <= len(shape) <= PROFILES[profile].rank:
                return _reject(report, "input", "rank_not_supported",
                               "Input tensor rank is outside the selected resource profile.")
    return None


def _logical_resources(report, original, lowered, checked, profile):
    shapes = checked["shapes"]
    elements = lambda name: math.prod(shapes[name])
    output_elements = sum(elements(o["value"]) for o in lowered["outputs"])
    report["resource_profile"] = profile
    report["resources"] = {
        "logical_inputs": _resource(len(lowered["inputs"]), "exact", "validated declarations"),
        "logical_outputs": _resource(len(lowered["outputs"]), "exact", "validated declarations"),
        "input_elements": _resource(sum(elements(i["name"]) for i in lowered["inputs"]),
                                    "exact", "validated shapes"),
        "output_elements": _resource(output_elements, "exact", "validated named output shapes"),
        "original_nodes": _resource(len(original["nodes"]), "exact", "submitted graph"),
        "expanded_nodes": _resource(len(lowered["nodes"]), "exact", "validated core graph"),
        "largest_logical_tensor": _resource(
            max((elements(name) for name in shapes if checked["kinds"][name] == "cipher"), default=0),
            "exact", "validated encrypted tensor shapes"),
        "public_constant_elements": _resource(
            sum(elements(name) for name in lowered["constants"]), "exact", "validated public constants"),
        "model_json_bytes": _resource(len(canonical(lowered)), "exact", "canonical lowered model"),
        # The request binds I/O only. Internal layouts, operation ordering, live
        # ciphertexts, key storage and multiplicative depth require the candidate.
        "intermediate_ciphertexts": _resource(None, "unknown", "requires compiled candidate"),
        "multiplicative_depth": _resource(None, "unknown", "requires compiled candidate"),
        "peak_memory_bytes": _resource(None, "unknown", "requires compiled candidate and runtime"),
        "execution_seconds": _resource(None, "unknown", "requires measured execution"),
    }


def _bind_request(report, request):
    layout = request["layout"]
    model_inputs = len(layout["inputs"])
    auxiliary = len(layout["auxiliary_ciphertexts"])
    outputs = layout["output_ciphertexts"]
    report["resources"].update({
        "input_ciphertexts": _resource(model_inputs, "exact", "canonical bound model inputs"),
        "auxiliary_ciphertexts": _resource(auxiliary, "exact", "trusted encrypted zero binding"),
        "output_ciphertexts": _resource(outputs, "exact", "canonical bound output selectors"),
        "slot_period": _resource(layout["input_slot_period"], "exact", "canonical layout"),
        # This lower bound counts separately encrypted inputs only. Outputs can
        # alias, so their count cannot raise the bound. It is not a peak memory
        # estimate or a budget sufficiency claim.
        "ciphertext_objects_lower_bound": _resource(
            model_inputs + auxiliary, "conservative",
            "trusted input preparation encrypts each physical input and the auxiliary zero separately",
            bound="lower"),
    })
    report["layout"] = dict(execution_abi=layout["execution_abi"], sha256=digest(layout))
    report["identity"].update(request_id=request["request_id"],
                              compiler_configuration_sha256=request["compiler_configuration"]["identity_sha256"],
                              lowered_model_sha256=digest(request["model"]))
    report["stages"]["layout"] = _stage("accepted", "canonical_binding_prepared")
    report["stages"]["compiler"] = _stage("unverified", "candidate_not_compiled")
    report["stages"]["encrypted"] = _stage("unverified", "candidate_not_executed")
    report["stages"]["agent"] = _stage("unverified", "candidate_not_generated")
    report["admission"] = "accepted_pending_validation"
    return _finish(report)


def _preparation_error(report, error):
    # Map a bounded set of trusted contract diagnostics to public codes. Never
    # forward arbitrary exception text (which can contain local dependency paths).
    text = str(error)
    if "Capacity helper profile" in text:
        code, message = "capacity_helper_unverified", "Capacity layouts cannot select an upstream helper profile."
    elif any(part in text for part in ("period", "ciphertext budget", "execution ABI", "layout binding")):
        code, message = "layout_not_supported", "The requested layout is outside the canonical layout contract."
    elif any(part in text for part in ("helper", "Helper", "composition", "Composition")):
        code, message = "construction_combination_unsupported", "The selected helper or construction combination is not supported."
    elif "Component option fields" in text:
        code, message = "options_invalid", "Unknown or malformed component options."
    else:
        code, message = "preparation_rejected", "The model or selected options fail request preparation."
    return _reject(report, "layout", code, message)


def assess_model(model, *, options=None, backend=BACKEND, resource_profile=None,
                 dtype="float64", requirements=None):
    """Validate a graph and its request binding without compiling or generating DSL.

    High-level operator graphs are checked through the existing recursive
    decomposition. This is an assessment API, not a new task submission path.
    It deliberately does not return lowered code or a deterministic DSL answer.
    """
    report = _report(backend)
    if backend != BACKEND:
        return _reject(report, "input", "backend_not_supported",
                       "This application entry accepts the Poseidon CPU backend.")
    if type(dtype) is not str or dtype != "float64":
        return _reject(report, "input", "dtype_not_supported", "Input dtype must be float64.")
    rejected = _check_requirements(report, requirements)
    if rejected is not None:
        return rejected
    if type(model) is not dict or model.get("format") not in (FORMAT, CAPACITY_FORMAT, *OPERATOR_FORMATS):
        return _reject(report, "input", "graph_format_unsupported", "Select a versioned model or operator graph.")
    if options is not None and type(options) is not dict:
        return _reject(report, "input", "options_invalid", "Component options must be an object.")
    options = copy.deepcopy(options or {})
    if options.get("compiler_configuration", "poseidon-cpu-eva-w45-v1") not in CPU_CONFIGURATIONS:
        return _reject(report, "compiler", "compiler_backend_not_supported",
                       "New application requests require an active Poseidon CPU configuration.", status="blocked")
    high = model["format"] in OPERATOR_FORMATS
    try:
        profile = (resource_profile or "legacy-v1") if high else graph_resource_profile(model)
        resource_identity(profile)
        if not high and resource_profile is not None and resource_profile != profile:
            return _reject(report, "input", "resource_profile_mismatch",
                           "The selected profile differs from the model's bound resource profile.")
        rejected = _input_shape_rejection(report, model, profile)
        if rejected is not None:
            return rejected
        report["identity"]["original_model_sha256"] = digest(model)
        if high:
            lowered, binding = decompose(copy.deepcopy(model), resource_profile=profile)
            report["decomposition"] = dict(
                binding_sha256=binding["binding_sha256"],
                registry_version=binding["registry_version"],
                implementation_sha256=binding["implementation_sha256"],
                original_sha256=binding["original_sha256"],
                lowered_sha256=binding["lowered_sha256"])
            report["stages"]["decomposition"] = _stage("accepted", "bounded_decomposition_validated")
            options=preparation_options(model,lowered,options)
        else:
            lowered = copy.deepcopy(model)
            report["stages"]["decomposition"] = _stage("not_required", "core_graph_submitted")
        checked = validate(lowered)
        report["stages"]["input"] = _stage("accepted", "logical_graph_validated")
        _logical_resources(report, model, lowered, checked, profile)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return _reject(report, "decomposition" if high else "input",
                       "decomposition_rejected" if high else "graph_invalid",
                       "The graph violates registered operator, type, shape, approximation or resource constraints.")
    if lowered["format"] == CAPACITY_FORMAT and options.get("helper_profile") is not None:
        return _reject(report, "layout", "capacity_helper_unverified",
                       "Capacity layouts cannot select an upstream helper profile.", status="blocked")
    try:
        # Lazy import avoids a dependency cycle if the old component facade later
        # delegates its declarations to this module.
        from component_contract import prepare_task
        request = prepare_task(lowered, options)
        validate_request(request)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as error:
        return _preparation_error(report, error)
    return _bind_request(report, request)


def assess_request(request, *, requirements=None):
    """Assess an existing bound request; preserve its bytes and historical hash."""
    report = _report(BACKEND)
    rejected = _check_requirements(report, requirements)
    if rejected is not None:
        return rejected
    try:
        validate_request(request)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return _reject(report, "input", "request_binding_invalid", "The request fails canonical integrity validation.")
    config = request.get("compiler_configuration", {})
    if config.get("name") not in CPU_CONFIGURATIONS or config.get("backend") != BACKEND:
        return _reject(report, "compiler", "compiler_backend_not_supported",
                       "New application execution requires an active Poseidon CPU configuration.", status="blocked")
    model = request["model"]
    checked = validate(model)
    report["stages"]["input"] = _stage("accepted", "logical_graph_validated")
    # A request contains a core graph, not trusted provenance for the high-level
    # input that may have produced it. Do not invent decomposition evidence.
    report["stages"]["decomposition"] = _stage("not_required", "core_graph_submitted")
    report["identity"]["original_model_sha256"] = digest(model)
    _logical_resources(report, model, model, checked, graph_resource_profile(model))
    return _bind_request(report, request)
