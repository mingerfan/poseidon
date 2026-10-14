"""Capability preflight tests; no SDK, Torch, compiler, provider or FHE execution."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import application_capabilities as caps
from benchmark_graph import FORMAT, CAPACITY_FORMAT, OPS, canonical
from component_contract import prepare_task
from model_decomposition import FORMAT as OPERATOR_FORMAT, REGISTRY, REGISTRY_VERSION
from model_resources import PROFILES, identity
from unified_capacity_layout import WIDE_ABI, EXTENDED_ABI, ABI as CAPACITY_ABI
from upstream_candidate_helpers import PROFILE as HELPER_PROFILE


def graph(size=4, *, profile=None):
    result = dict(
        format=FORMAT, id="cap_example", inputs=[dict(name="a", shape=[size])],
        constants={},
        nodes=[dict(id="n", op="square", inputs=["a"], attrs={}, outputs=["z"])],
        outputs=[dict(name="out", value="z")])
    if profile is not None:
        result.update(format=CAPACITY_FORMAT, resource_profile=profile)
    return result


class CapabilityDeclarations(unittest.TestCase):
    def test_registry_and_resources_are_derived(self):
        report = caps.describe()
        self.assertEqual(set(report["core_operators"]), set(OPS))
        self.assertEqual(set(report["decomposition_registry"]["operators"]), set(REGISTRY))
        self.assertEqual(report["decomposition_registry"]["version"], REGISTRY_VERSION)
        for name, op in REGISTRY.items():
            self.assertEqual(report["decomposition_registry"]["operators"][name]["attributes"],
                             sorted(op.attributes))
        for name in PROFILES:
            self.assertEqual(report["resource_profiles"][name], identity(name))
        self.assertEqual(report["resource_certainties"], ["exact", "conservative", "unknown"])
        json.dumps(report, allow_nan=False)

    def test_registry_is_not_an_execution_score(self):
        report = caps.describe()
        for item in list(report["core_operators"].values()) + list(report["decomposition_registry"]["operators"].values()):
            for key in ("compiler_status", "encrypted_status", "agent_status"):
                self.assertEqual(item[key], "unverified")

    def test_layout_profiles_and_periods_are_separate(self):
        layouts = {item["abi"]: item for item in caps.describe()["layouts"]}
        self.assertEqual(layouts[WIDE_ABI]["periods"], [512, 1024, 2048, 4096])
        self.assertEqual(layouts[WIDE_ABI]["resource_profiles"], ["resnet20-v1","general-tensors-v1"])
        self.assertNotIn(512, layouts[EXTENDED_ABI]["periods"])
        self.assertEqual(layouts[CAPACITY_ABI]["input_ciphertexts"], 4)
        self.assertEqual(layouts[EXTENDED_ABI]["input_ciphertexts"], 16)

    def test_describe_returns_detached_values(self):
        first = caps.describe()
        first["resource_profiles"]["legacy-v1"]["input_elements"] = -1
        first["core_operators"]["square"]["attributes"].append("bad")
        self.assertEqual(caps.describe()["resource_profiles"]["legacy-v1"], identity())
        self.assertEqual(caps.describe()["core_operators"]["square"]["attributes"], [])


class CapabilityAssessment(unittest.TestCase):
    def test_core_admission_does_not_claim_execution(self):
        result = caps.assess_model(graph())
        self.assertEqual(result["admission"], "accepted_pending_validation")
        self.assertEqual(result["execution_readiness"], "unknown")
        self.assertEqual(result["stages"]["input"]["status"], "accepted")
        self.assertEqual(result["stages"]["decomposition"]["status"], "not_required")
        for stage in ("compiler", "encrypted", "agent"):
            self.assertEqual(result["stages"][stage]["status"], "unverified")
        self.assertFalse(result["encrypted_execution"])
        self.assertFalse(result["numerically_validated"])
        self.assertEqual(result["paid_calls"], 0)

    def test_exact_resources_and_unknown_physical_costs(self):
        result = caps.assess_model(graph())
        resources = result["resources"]
        for key, value in {"input_elements": 4, "output_elements": 4, "original_nodes": 1,
                           "expanded_nodes": 1, "input_ciphertexts": 1,
                           "auxiliary_ciphertexts": 1, "output_ciphertexts": 1,
                           "slot_period": 4}.items():
            self.assertEqual(resources[key]["value"], value)
            self.assertEqual(resources[key]["certainty"], "exact")
        for key in ("intermediate_ciphertexts", "multiplicative_depth",
                    "peak_memory_bytes", "execution_seconds"):
            self.assertIsNone(resources[key]["value"])
            self.assertEqual(resources[key]["certainty"], "unknown")
        lower = resources["ciphertext_objects_lower_bound"]
        self.assertEqual((lower["value"], lower["certainty"], lower["bound"]),
                         (2, "conservative", "lower"))

    def test_output_aliases_do_not_inflate_object_lower_bound(self):
        model = graph()
        model["outputs"] = [dict(name="out" + str(i), value="z") for i in range(4)]
        resources = caps.assess_model(model)["resources"]
        self.assertEqual(resources["output_ciphertexts"]["value"], 4)
        self.assertEqual(resources["ciphertext_objects_lower_bound"]["value"], 2)

    def test_old_request_hash_is_unchanged(self):
        model = graph()
        before = canonical(model)
        expected = prepare_task(model)
        self.assertEqual(expected["request_id"],
                         "6568de531d0133b74d1d3642e14a13a03d2cd0408dbe3ca177a33dabc943d023")
        result = caps.assess_model(model)
        self.assertEqual(result["identity"]["request_id"], expected["request_id"])
        self.assertEqual(canonical(model), before)
        self.assertEqual(canonical(prepare_task(model)), canonical(expected))

    def test_existing_request_assessment_preserves_bytes(self):
        request = prepare_task(graph())
        before = canonical(request)
        result = caps.assess_request(request)
        self.assertEqual(result["admission"], "accepted_pending_validation")
        self.assertEqual(result["identity"]["request_id"], request["request_id"])
        self.assertEqual(canonical(request), before)
        self.assertEqual(result["summary"], caps.assess_model(graph())["summary"])

    def test_summary_does_not_depend_on_model_or_node_names(self):
        original = graph()
        renamed = graph()
        renamed["id"] = "renamed"
        renamed["nodes"][0]["id"] = "renamed_node"
        left, right = caps.assess_model(original), caps.assess_model(renamed)
        self.assertEqual(left["summary"], right["summary"])
        self.assertNotEqual(left["identity"]["request_id"], right["identity"]["request_id"])

    def test_explicit_decomposition_is_separate_from_runtime(self):
        model = graph()
        model["format"] = OPERATOR_FORMAT
        model["nodes"][0].update(op="dropout", attrs=dict(training=False))
        result = caps.assess_model(model)
        self.assertEqual(result["admission"], "accepted_pending_validation")
        self.assertEqual(result["stages"]["decomposition"]["status"], "accepted")
        self.assertEqual(result["decomposition"]["registry_version"], REGISTRY_VERSION)
        self.assertEqual(result["stages"]["compiler"]["status"], "unverified")

    def test_approximation_without_profile_is_rejected(self):
        model = graph()
        model["format"] = OPERATOR_FORMAT
        model["nodes"][0].update(op="relu", attrs={})
        result = caps.assess_model(model)
        self.assertEqual(result["admission"], "rejected")
        self.assertEqual(result["stages"]["decomposition"]["status"], "rejected")

    def test_capacity_v4_uses_exact_bound_arity(self):
        model = graph(512, profile="resnet20-v1")
        options = dict(chunk_period=512, capacity_packing_version="capacity-public-tensors-v8")
        result = caps.assess_model(model, options=options)
        self.assertEqual(result["admission"], "accepted_pending_validation")
        self.assertEqual(result["layout"]["execution_abi"], WIDE_ABI)
        self.assertEqual(result["resources"]["input_ciphertexts"]["value"], 1)
        self.assertEqual(result["resources"]["output_elements"]["value"], 512)
        self.assertEqual(result["resources"]["slot_period"]["value"], 512)

    def test_capacity_helper_is_explicitly_blocked(self):
        result = caps.assess_model(graph(8, profile="medium-v1"),
                                   options=dict(helper_profile=HELPER_PROFILE))
        self.assertEqual(result["admission"], "rejected")
        self.assertEqual(result["stages"]["layout"],
                         dict(status="blocked", code="capacity_helper_unverified"))

    def test_bootstrap_is_explicitly_blocked(self):
        for evaluate, value in ((caps.assess_model, graph()),
                                (caps.assess_request, prepare_task(graph()))):
            result = evaluate(value, requirements=dict(bootstrap=True))
            self.assertEqual(result["admission"], "rejected")
            self.assertEqual(result["stages"]["encrypted"],
                             dict(status="blocked", code="bootstrap_unavailable"))
        self.assertEqual(caps.assess_model(graph(), requirements=dict(bootstrap=False))["admission"],
                         "accepted_pending_validation")

    def test_requirements_and_dtype_are_strict(self):
        for requirements in (dict(bootstrap=1), dict(bootstrap="false"), dict(arbitrary=True), []):
            result = caps.assess_model(graph(), requirements=requirements)
            self.assertEqual(result["diagnostics"][0]["code"], "requirements_invalid")
        for dtype in ("float32", "complex128", "int64", None):
            result = caps.assess_model(graph(), dtype=dtype)
            self.assertEqual(result["diagnostics"][0]["code"], "dtype_not_supported")
        model = graph()
        model["inputs"][0]["dtype"] = "float64"
        self.assertEqual(caps.assess_model(model)["diagnostics"][0]["code"],
                         "dtype_declaration_unsupported")

    def test_rank_and_invalid_operator_rejected(self):
        model = graph()
        model["inputs"][0]["shape"] = [1] * 5
        self.assertEqual(caps.assess_model(model)["diagnostics"][0]["code"], "rank_not_supported")
        model = graph()
        model["nodes"][0]["op"] = "arbitrary_op"
        self.assertEqual(caps.assess_model(model)["admission"], "rejected")

    def test_profiles_and_layout_periods_do_not_silently_fallback(self):
        for period in (3, 512, True):
            result = caps.assess_model(graph(), options=dict(chunk_period=period))
            self.assertEqual(result["admission"], "rejected")
        result = caps.assess_model(graph(512, profile="medium-v1"), options=dict(chunk_period=512))
        self.assertEqual(result["admission"], "rejected")
        result = caps.assess_model(graph(), resource_profile="medium-v1")
        self.assertEqual(result["diagnostics"][0]["code"], "resource_profile_mismatch")

    def test_unsupported_backend_never_falls_back(self):
        self.assertEqual(caps.assess_model(graph(), backend="poseidon_cuda")["admission"], "rejected")
        for name in ("seal-cpu-eva-w45-v1", None, "unregistered"):
            result = caps.assess_model(graph(), options=dict(compiler_configuration=name))
            self.assertEqual(result["stages"]["compiler"],
                             dict(status="blocked", code="compiler_backend_not_supported"))

    def test_tampered_request_rejected(self):
        request = prepare_task(graph())
        request["layout"]["input_slot_period"] = 512
        result = caps.assess_request(request)
        self.assertEqual(result["diagnostics"][0]["code"], "request_binding_invalid")
        self.assertEqual(result["admission"], "rejected")

    def test_errors_do_not_expose_dependency_paths(self):
        with patch("component_contract.prepare_task", side_effect=ValueError("/private/credential/location")):
            result = caps.assess_model(graph())
        self.assertEqual(result["admission"], "rejected")
        self.assertNotIn("credential", json.dumps(result))

    def test_does_not_read_development_reports(self):
        # Runtime preflight has no historical-evidence query. The high-level
        # decomposer may hash its own implementation, so this covers core graphs.
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected report read")):
            caps.describe()
            result = caps.assess_model(graph())
        self.assertEqual(result["admission"], "accepted_pending_validation")

    def test_malformed_data_fails_closed(self):
        for value in (None, [], {"format": FORMAT}, dict(graph(), constants=float("nan"))):
            result = caps.assess_model(value)
            self.assertEqual(result["admission"], "rejected")
        for value in (None, [], {}):
            self.assertEqual(caps.assess_request(value)["admission"], "rejected")


if __name__ == "__main__":
    unittest.main()
