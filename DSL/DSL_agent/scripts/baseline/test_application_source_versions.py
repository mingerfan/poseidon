"""Public source capability compatibility; no capture, Provider or FHE."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import application_capabilities as caps
from component_contract import capabilities
from application_policy import describe
from pytorch_model_intake import SOURCE_FORMATS
from pytorch_spatial_policy import INTERNAL_FORMATS, supports_input_call, supports_spatial
from pytorch_static_broadcast import SOURCE_FORMAT, POLICY, METHODS


class SourceVersionDeclarations(unittest.TestCase):
    def trusted(self):
        return caps.input_sources()["trusted_pytorch"]

    def test_structured_call_versions_match_all_public_contracts(self):
        report = self.trusted()
        expected = ["poseidon-trusted-pytorch-v26",
                    "poseidon-trusted-pytorch-v27",
                    "poseidon-trusted-pytorch-v28", "poseidon-trusted-pytorch-v29"]
        self.assertEqual(report["input_call"]["versions"], expected)
        for version in SOURCE_FORMATS:
            self.assertEqual(version in expected, supports_input_call({"format": version}))

    def test_spatial_inheritance_is_visible_to_applications(self):
        values = self.trusted()["spatial_calls"]["versions"]
        self.assertEqual(values, ["poseidon-trusted-pytorch-v27", SOURCE_FORMAT, "poseidon-trusted-pytorch-v29"])
        for version in SOURCE_FORMATS:
            self.assertEqual(version in values, supports_spatial({"format": version}))

    def test_private_transports_and_unknown_versions_are_not_advertised(self):
        report = self.trusted()
        for values in (report["formats"], report["input_call"]["versions"],
                       report["spatial_calls"]["versions"], report["static_broadcast"]["versions"]):
            self.assertEqual(len(values), len(set(values)))
            for private in (*INTERNAL_FORMATS, "poseidon-trusted-pytorch-v30",
                            "poseidon-trusted-pytorch-v999", "latest"):
                self.assertNotIn(private, values)
        for key in ("input_call", "spatial_calls", "static_broadcast"):
            self.assertTrue(set(report[key]["versions"]).issubset(report["formats"]))

    def test_broadcast_declaration_keeps_real_static_constraints(self):
        value = self.trusted()["static_broadcast"]
        self.assertEqual(value["version"], SOURCE_FORMAT)
        self.assertEqual(value["versions"], [SOURCE_FORMAT, "poseidon-trusted-pytorch-v29"])
        self.assertEqual(value["policy"], POLICY)
        self.assertEqual(value["calls"], ["Tensor.broadcast_to", "Tensor.expand_as"])
        self.assertEqual(value["calls"], ["Tensor." + name for name in sorted(METHODS)])
        self.assertEqual(value["size_containers"], ["tuple", "list", "torch.Size"])
        self.assertIs(value["dynamic_shape"], False)
        self.assertIs(value["scalar_intermediates"], True)
        self.assertEqual(value["source_dtype"], "preserved")
        self.assertNotIn("torch.broadcast_to", value["calls"])

    def test_registration_does_not_promote_qualification(self):
        report = self.trusted()
        for key in ("input_call", "spatial_calls", "static_broadcast"):
            self.assertEqual(report[key]["input_status"], "registered")
            for stage in ("compiler", "encrypted", "agent"):
                self.assertEqual(report[key][stage + "_status"], "unverified")
        self.assertEqual(report["static_broadcast"]["decomposition_status"], "registered")

    def test_component_and_application_expose_the_same_source_rules(self):
        source = caps.input_sources()
        self.assertEqual(capabilities()["input_sources"], source)
        public = describe()
        self.assertEqual(public["input_sources"], source)
        self.assertEqual(public["capability_matrix"]["input_sources"], source)

    def test_capability_mutation_does_not_change_next_request(self):
        first = self.trusted()
        expected = copy.deepcopy(first)
        first["input_call"]["versions"].clear()
        first["spatial_calls"]["versions"].append("invented")
        first["static_broadcast"]["calls"].append("Tensor.unsafe")
        first["static_broadcast"]["encrypted_status"] = "passed"
        self.assertEqual(self.trusted(), expected)
        json.dumps(self.trusted(), allow_nan=False)

    def test_public_descriptions_do_not_read_development_reports(self):
        self.trusted()
        with patch.object(Path, "read_text", side_effect=AssertionError("report dependency")):
            source = caps.input_sources()
            public = describe()
        self.assertEqual(public["input_sources"], source)

    def test_new_versions_do_not_expand_input_and_source_budgets(self):
        report = self.trusted()
        self.assertEqual(report["input_dtype"], "float64")
        self.assertEqual((report["source_files"], report["source_bytes"]), (8, 65536))
        self.assertEqual((report["input_call"]["tensor_leaves"], report["output_tensor_leaves"]), (4, 4))
        self.assertEqual(report["mode"], "isolated_explicit_trust")
        self.assertEqual(report["source_application_entry"], "submit_source")


if __name__ == "__main__":
    unittest.main()
