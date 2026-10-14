"""Draft tests: source syntax checked only; not yet run or production-wired."""
import copy
import json
import unittest

from unified_contract_scope_guidance import (
    BASE_VERSION, VERSION, NATIVE_CONTROL, NATIVE_DTYPE, NATIVE_INTEGER,
    NATIVE_TYPE_RULES, effective_contract, specification, type_repair_hint)


def inherited_fixture():
    return {
        "version": BASE_VERSION,
        "helper_rule": "existing helper rule",
        "typed_request_bindings": {
            "type_rules": {name: "existing " + name for name in
                ("names", "dtype", "shape", "unary", "compare",
                 "returns", "broadcast", "calls")},
            "constant_bindings": [{"name": "weight", "immutable": True}],
            "rotation_steps": [1, 2],
            "output_ciphertexts": 1,
        },
        "typed_construction_semantics": {
            "typed_values": ["shared numeric constructors"],
            "contribution": ["must influence the output"],
            "backend": ["compiler and numerical checks remain required"],
        },
        "expression_construction_rules": {
            "native_helpers": "inherited helpers",
            "type_boundary": "inherited types",
            "native_loops": "inherited loops",
            "depth": "unchanged depth semantics",
        },
        "poseidon_expression_rules": {
            "version": BASE_VERSION,
            "construction_resources": {"golden_work_limit": 1024},
            "planning_costs": {
                "compiler_feasibility": "unknown",
                "execution_feasibility": "unknown",
            },
        },
    }


def request_fixture(public=False):
    request = {
        "layout": {"input_slot_period": 4, "output_ciphertexts": 1},
        "public_constants": {"weight": [1.0, 2.0, 3.0, 4.0]},
    }
    if public:
        request["construction_profile"] = "hecate-unified-public-v1"
    return request


class ContractScopeDraftTests(unittest.TestCase):
    def test_native_contract_resolution(self):
        self.assertEqual(effective_contract(request_fixture()), "native")

    def test_public_contract_resolution(self):
        self.assertEqual(effective_contract(request_fixture(True)), "public")

    def test_unknown_and_null_profiles_rejected(self):
        for value in (None, "", "hecate-unified-public-v2", {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                effective_contract({"construction_profile": value})

    def test_wrong_inherited_version_rejected(self):
        for version in ("explicit-v14", VERSION, None):
            base = inherited_fixture()
            base["version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                specification(request_fixture(), base)

    def test_native_constructor_and_control_are_scoped(self):
        actual = specification(request_fixture(), inherited_fixture())
        rules = actual["typed_request_bindings"]["type_rules"]
        self.assertEqual(rules["dtype"], NATIVE_DTYPE)
        self.assertEqual(rules["compare"], NATIVE_CONTROL)
        self.assertIn("not admit if", rules["compare"])
        self.assertIn("np.array(data, dtype=object)", rules["dtype"])

    def test_plain_formals_are_not_static_integers(self):
        actual = specification(request_fixture(), inherited_fixture())
        self.assertIn("p is a Plain Expr", actual["helper_rule"])
        self.assertIn(NATIVE_INTEGER,
                      actual["typed_construction_semantics"]["typed_values"])
        self.assertEqual(actual["expression_construction_rules"]["type_boundary"],
                         NATIVE_INTEGER)

    def test_public_explanations_are_preserved(self):
        base = inherited_fixture()
        expected = copy.deepcopy(base)
        expected["version"] = VERSION
        self.assertEqual(specification(request_fixture(True), base), expected)

    def test_request_and_legacy_context_not_mutated(self):
        request = request_fixture()
        base = inherited_fixture()
        before = json.dumps([request, base], sort_keys=True)
        actual = specification(request, base)
        actual["typed_request_bindings"]["constant_bindings"][0]["name"] = "changed"
        self.assertEqual(json.dumps([request, base], sort_keys=True), before)

    def test_no_model_id_or_family_dispatch(self):
        base = inherited_fixture()
        a = request_fixture()
        b = request_fixture()
        a.update(id="alpha", family="one", model={"id": "ignored"})
        b.update(id="beta", family="other", model={"id": "also_ignored"})
        self.assertEqual(specification(a, base), specification(b, base))

    def test_no_request_private_fields_copied(self):
        request = request_fixture()
        request.update(private_reference="sentinel-answer",
                       private_inputs="sentinel-input",
                       hecate_source="sentinel-dsl")
        actual = json.dumps(specification(request, inherited_fixture()))
        for sentinel in ("sentinel-answer", "sentinel-input", "sentinel-dsl"):
            self.assertNotIn(sentinel, actual)

    def test_planning_unknown_and_resources_preserved(self):
        base = inherited_fixture()
        actual = specification(request_fixture(), base)
        self.assertEqual(actual["poseidon_expression_rules"],
                         base["poseidon_expression_rules"])
        self.assertEqual(actual["typed_request_bindings"]["rotation_steps"], [1, 2])
        self.assertEqual(actual["typed_request_bindings"]["constant_bindings"],
                         base["typed_request_bindings"]["constant_bindings"])

    def test_native_feedback_uses_same_scoped_rules(self):
        actual = type_repair_hint(request_fixture(), "dtype failure")
        self.assertEqual(actual, NATIVE_TYPE_RULES["dtype"])

    def test_control_feedback_is_fixed_and_does_not_echo(self):
        actual = type_repair_hint(
            request_fixture(), "untrusted sentinel-secret",
            reason_code="native_control_flow_not_enabled")
        self.assertEqual(actual, NATIVE_CONTROL + "\n" + NATIVE_INTEGER)
        self.assertNotIn("sentinel-secret", actual)

    def test_public_feedback_preserves_existing_path(self):
        from unified_typed_guidance import repair_hint
        request = request_fixture(True)
        self.assertEqual(type_repair_hint(request, "dtype failure"),
                         repair_hint(request, "dtype failure"))


if __name__ == "__main__":
    unittest.main()
