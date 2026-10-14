"""Standard-library controls; mock rule generation never claims real candidates."""
import copy
import hashlib
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch
import strategy as s


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(request_id="frozen-request", model=dict(id="arbitrary-name", nodes=[]),
                            layout=dict(execution_abi="unified-capacity-chunked-inputs-v3",
                                        output_ciphertexts=2),
                            compiler_configuration=dict(backend="poseidon_cpu", name="fixed"),
                            public_constants=dict(pv_bound=[1., 2., 3., 4.], mask0=[1., 0., 0., 0.]),
                            constant_origins=dict(capacity_tensor_packing=dict(bindings=[dict(key="pv_bound")])))
        self.source = "@hc.func('c,c')\ndef golden(x, zero_ct):\n    value = x * pv_bound\n    return [value, x]\n"
        self.lowering = types.ModuleType("unified_graph_lowering")
        self.lowering.candidate_source = Mock(return_value=(self.source, dict(strategy="capacity-chunks-v10")))
        self.contract = types.ModuleType("unified_graph_contract")
        self.contract.validate_candidate = Mock(return_value=dict(checked=True))
        self.modules = patch.dict(sys.modules, unified_graph_lowering=self.lowering,
                                  unified_graph_contract=self.contract)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_delegates_exact_request_without_extra_options(self):
        result = s.candidate_source(self.request)
        self.assertEqual(result, self.lowering.candidate_source.return_value)
        self.lowering.candidate_source.assert_called_once_with(self.request)

    def test_case_and_model_names_do_not_select_expression(self):
        requests = {}
        for name in ("renamed_1", "unrelated_2"):
            request = copy.deepcopy(self.request)
            request["model"]["id"] = name
            requests[name] = request
        report = s.check_requests(requests)
        self.assertEqual(report["status"], "passed")
        self.assertEqual([r["candidate_sha256"] for r in report["rows"]],
                         [hashlib.sha256(self.source.encode()).hexdigest()] * 2)
        self.assertEqual([call.args[0]["model"]["id"] for call in self.lowering.candidate_source.call_args_list],
                         list(requests))

    def test_static_check_preserves_original_and_counts_bound_vectors(self):
        before = s.canonical(self.request)
        row = s.check_request(self.request)
        self.assertEqual(s.canonical(self.request), before)
        self.assertEqual(row["physical_output_count"], 2)
        self.assertEqual(row["public_vector_bindings"], 2)
        self.assertEqual(row["public_vector_keys_reused"], ["pv_bound"])
        self.assertEqual(row["public_tensor_vector_keys_reused"], ["pv_bound"])
        self.assertEqual(row["candidate_sha256"], hashlib.sha256(self.source.encode()).hexdigest())
        self.assertEqual(row["request_canonical_sha256"], hashlib.sha256(before).hexdigest())
        self.contract.validate_candidate.assert_called_once()
        self.assertEqual(self.contract.validate_candidate.call_args.args[0]["request_id"], "frozen-request")

    def test_vectors_in_comments_or_strings_are_not_reuse(self):
        self.lowering.candidate_source.return_value = (
            "@hc.func('c,c')\ndef golden(x, zero_ct):\n    # pv_bound\n    label = 'pv_bound'\n    return [x, x]\n", {})
        self.assertEqual(s.check_request(self.request)["public_vector_keys_reused"], [])

    def test_immutable_caller_survives_trusted_storage_materialization(self):
        before = s.canonical(self.request)
        def materialize(work):
            work["public_constants"]["extra"] = [0., 0., 0., 0.]
            return self.source, {}
        self.lowering.candidate_source.side_effect = materialize
        s.check_request(self.request)
        self.assertEqual(s.canonical(self.request), before)

    def test_graph_layout_and_parameters_must_remain_unchanged(self):
        for field in ("model", "layout", "compiler_configuration"):
            def change(work, field=field):
                work[field]["changed"] = True
                return self.source, {}
            self.lowering.candidate_source.side_effect = change
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Graph, physical ABI"):
                s.check_request(self.request)

    def test_explicit_source_copy_matches_the_checked_hash(self):
        row = s.check_request(self.request, include_source=True)
        self.assertEqual(row["candidate_source"], self.source)
        self.assertEqual(row["candidate_sha256"],
                         hashlib.sha256(row["candidate_source"].encode()).hexdigest())

    def test_wrong_physical_output_count_is_failure(self):
        self.request["layout"]["output_ciphertexts"] = 3
        with self.assertRaisesRegex(ValueError, "Exact physical output"):
            s.check_request(self.request)

    def test_existing_validator_rejections_are_preserved_without_fallback(self):
        self.contract.validate_candidate.side_effect = ValueError("original static rejection")
        report = s.check_requests({"report-label": self.request})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["rows"][0]["error"], "original static rejection")
        self.lowering.candidate_source.assert_called_once()
        self.assertNotIn("candidate_sha256", report["rows"][0])

    def test_string_source_return_remains_supported(self):
        self.lowering.candidate_source.return_value = self.source
        self.assertEqual(s.check_request(self.request)["rule_record"], {})

    def test_boolean_row_carries_no_new_scale_or_backend_claim(self):
        report = s.check_requests({"Boolean-label-is-only-a-label": self.request})
        for value in (report, report["rows"][0]):
            self.assertEqual(value["scale_evidence"], "not_run")
            for key in ("provider_calls", "compiler_calls", "encrypted_executions"):
                self.assertEqual(value[key], 0)
            self.assertFalse(value["new_agent_generation"])
            self.assertFalse(value["production_applied"])

    def test_import_needs_no_numpy_sdk_or_backend(self):
        # The real draft's eager imports are all standard-library modules.
        import ast
        from pathlib import Path
        tree = ast.parse(Path(s.__file__).read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        modules = {name.name for n in imports if isinstance(n, ast.Import) for name in n.names}
        self.assertEqual(modules, {"ast", "copy", "hashlib", "json"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
