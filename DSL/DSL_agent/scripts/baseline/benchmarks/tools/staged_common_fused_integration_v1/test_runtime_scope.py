"""Pure control regressions for the frozen joint scope; no backend execution.

Synthetic approvals remain in temporary directories and never create a project
approval. SDK identity and candidate lowering are mocked; all other plan facts
come from the retained public source, requests and evidence.
"""
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import runtime_scope as s


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joint = s.read(s.HERE / "plan.json")
        cls.backend = s.read(s.RESULTS / "candidate-replay-rsh9wadc/report.json")["backend_identity"]
        identity_module = types.ModuleType("poseidon_backend_identity")
        identity_module.identity = Mock(return_value=cls.backend)
        guard = s.guard()

        def git_snapshot(command, **kwargs):
            if command == ["git", "branch", "--show-current"]:
                return cls.joint["branch"] + "\n"
            if command == ["git", "rev-parse", "HEAD"]:
                return cls.joint["head"] + "\n"
            raise AssertionError("Unexpected subprocess in pure plan test")

        with tempfile.TemporaryDirectory(prefix="poseidon-joint-plan-test-") as temp:
            with patch.object(s, "APPROVAL", Path(temp) / "missing-approval.json"), \
                 patch.object(s, "guard", return_value=guard), \
                 patch.object(guard, "active_processes", return_value=[]), \
                 patch.object(guard.subprocess, "check_output", side_effect=git_snapshot), \
                 patch.dict(sys.modules, {"poseidon_backend_identity": identity_module}):
                cls.p = s.plan()
        identity_module.identity.assert_called_once()
        cls.tools = s.driver_hashes()
        cls.requests = {name: s.request_value(name) for name in s.CASES}
        spec = importlib.util.spec_from_file_location("_joint_control_runtime_driver", s.COMMON / "runtime_driver.py")
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)

    def rebind(self, value):
        value["binding"] = s.digest({k: v for k, v in value.items() if k != "binding"})
        return value

    def proposal(self):
        return copy.deepcopy(self.p)

    def receipt(self, proposal=None):
        proposal = self.p if proposal is None else proposal
        return dict(format="poseidon-common-fused-offline-approval-v1", approved=True,
                    proposal_binding=proposal["binding"], proposal_sha256="f" * 64,
                    driver_hashes=copy.deepcopy(self.tools), human_message="Synthetic control test only.")

    def runtime(self):
        value = self.proposal()
        value.update(started_wall=1800000000.0, deadline_wall=1800000000.0 + s.SCOPE["wall_seconds"],
                     initial_result_directories=["old-a", "old-b"],
                     readonly_result_entries=["old-a", "old-b"])
        return self.rebind(value)

    def check_runtime(self, value):
        frozen = self.p["bindings"]["files"]
        real_sha = s.sha
        with patch.object(s, "before_launch") as approval, \
             patch.object(s, "read", return_value=self.p), \
             patch.object(s, "sha", side_effect=lambda p: frozen.get(str(p), None) or real_sha(p)):
            result = s.check_plan(value)
            approval.assert_called_once()
            return result

    def test_exact_original_joint_and_protected_identities(self):
        bindings = self.p["bindings"]
        self.assertEqual(len(bindings["original_production"]), 704)
        self.assertEqual(len(bindings["inside_sources"]), 706)
        self.assertEqual(len(bindings["protected_sources"]), 654)
        self.assertEqual(bindings["original_production"], self.joint["original_production_hashes"])
        self.assertEqual(bindings["inside_sources"], self.joint["inside_source_hashes"])
        self.assertEqual(bindings["combined_plan_binding"], self.joint["binding"])

    def test_exact_twelve_cases_and_thirty_six_phases(self):
        self.assertEqual(len(s.CASES), 12)
        self.assertEqual(len(set(s.CASES)), 12)
        self.assertEqual(self.p["cases"], list(s.CASES))
        self.assertEqual(self.p["phases"], ["qualification", "new_inputs", "cache"])
        self.assertEqual(len(s.CASES) * len(s.PHASES), 36)
        self.assertEqual(set(self.p["bindings"]["requests"]), set(s.CASES))
        self.assertEqual(set(self.p["bindings"]["source_reference"]), set(s.CASES))

    def test_plan_is_offline_without_activation_or_feasibility_claim(self):
        self.assertTrue(self.p["approval_required"])
        self.assertFalse(self.p["production_applied"])
        self.assertFalse(self.p["new_agent_generation"])
        self.assertFalse(self.p["bootstrap"])
        self.assertFalse(self.p["cuda"])
        self.assertEqual(self.p["candidate_strategy"], "lower-balanced-scalar-v1")
        self.assertEqual(self.p["candidate_origin"], s.CANDIDATE_ORIGIN)
        for field in ("provider_calls", "http_attempts", "installations", "downloads", "backend_changes", "parameter_changes"):
            self.assertEqual(self.p["scope"][field], 0)
        self.assertEqual(self.p["scope"]["compiler_attempts_max"], 12)
        self.assertEqual(self.p["scope"]["phase_seconds"], 180)
        self.assertEqual(self.p["scope"]["wall_seconds"], 3600)

    def test_driver_hashes_include_both_control_tests(self):
        expected = {s.HERE / "runtime_scope.py", s.HERE / "runtime_acceptance.py",
                    s.COMMON / "runtime_driver.py", s.HERE / "test_runtime_scope.py",
                    s.COMMON / "test_runtime_driver.py"}
        self.assertEqual(set(s.driver_hashes()), {str(p.relative_to(s.ROOT)) for p in expected})
        self.assertEqual(s.driver_hashes()[str(Path(__file__).resolve().relative_to(s.ROOT))], s.sha(__file__))

    def test_matching_plan_passes_current_file_check(self):
        s.check_plan(self.p)

    def test_rehashed_budget_expansion_and_type_change_rejected(self):
        for field in ("wall_seconds", "phase_seconds", "artifact_bytes", "compiler_attempts_max"):
            with self.subTest(field=field):
                changed = self.proposal()
                changed["scope"][field] += 1
                with self.assertRaisesRegex(ValueError, "Exact finite scope"):
                    s.check_plan(self.rebind(changed))
        changed = self.proposal()
        changed["scope"]["native_workers"] = True
        with self.assertRaisesRegex(ValueError, "Exact finite scope"):
            s.check_plan(self.rebind(changed))

    def test_case_or_phase_substitution_rejected_even_after_rehash(self):
        for field in ("cases", "phases"):
            changed = self.proposal()
            changed[field][0] = "replacement"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Exact reviewed"):
                s.check_plan(self.rebind(changed))

    def test_invalid_plan_binding_rejected(self):
        changed = self.proposal()
        changed["binding"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Runtime plan identity"):
            s.check_plan(changed)

    def test_changed_driver_hash_rejected(self):
        path = s.HERE / "runtime_acceptance.py"
        actual = s.sha
        with patch.object(s, "sha", side_effect=lambda p: "0" * 64 if Path(p) == path else actual(p)):
            with self.assertRaisesRegex(ValueError, "Frozen tool/fixture/reference changed"):
                s.check_plan(self.p)

    def test_missing_approval_rejected_before_record_reads(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-joint-approval-test-") as temp:
            with patch.object(s, "APPROVAL", Path(temp) / "missing.json"), patch.object(s, "read") as read:
                with self.assertRaisesRegex(ValueError, "Pending independent"):
                    s.before_launch()
                read.assert_not_called()

    def test_valid_synthetic_approval_data_passes(self):
        s.check_approval(self.p, self.receipt(), "f" * 64, self.tools)

    def test_unapproved_or_nonboolean_receipt_rejected(self):
        for approved in (False, 1, "yes"):
            receipt = self.receipt()
            receipt["approved"] = approved
            with self.subTest(approved=approved), self.assertRaisesRegex(ValueError, "Separate human approval"):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_receipt_field_set_is_exact(self):
        for extra in (False, True):
            receipt = self.receipt()
            if extra:
                receipt["extra"] = 1
            else:
                receipt.pop("human_message")
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Separate human approval"):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_receipt_proposal_binding_and_bytes_are_exact(self):
        for field in ("proposal_binding", "proposal_sha256"):
            receipt = self.receipt()
            receipt[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Exact approved proposal"):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_receipt_tool_hash_and_human_message_are_exact(self):
        for field, value in (("driver_hashes", {}), ("human_message", ""), ("human_message", "x" * 1025)):
            receipt = self.receipt()
            receipt[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Approved tools"):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_approval_cannot_expand_budget_or_replace_cases(self):
        for field in ("scope", "cases"):
            changed = self.proposal()
            if field == "scope":
                changed[field]["wall_seconds"] += 1
            else:
                changed[field][0] = "replacement"
            self.rebind(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Exact approved"):
                s.check_approval(changed, self.receipt(changed), "f" * 64, self.tools)

    def test_approval_cannot_enable_generation_or_change_strategy(self):
        for field, value in (("new_agent_generation", True), ("production_applied", True),
                             ("candidate_strategy", "other")):
            changed = self.proposal()
            changed[field] = value
            self.rebind(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Offline rule scope only"):
                s.check_approval(changed, self.receipt(changed), "f" * 64, self.tools)

    def test_before_launch_checks_temporary_receipt_and_toolhash(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-joint-receipt-test-") as temp:
            proposal_path = Path(temp) / "proposal.json"
            approval_path = Path(temp) / "approval.json"
            proposal_path.write_bytes(s.canonical(self.p))
            receipt = self.receipt()
            receipt["proposal_sha256"] = s.sha(proposal_path)
            approval_path.write_bytes(s.canonical(receipt))
            with patch.object(s, "PROPOSAL", proposal_path), patch.object(s, "APPROVAL", approval_path), \
                 patch.object(s, "driver_hashes", return_value=self.tools):
                s.before_launch()
                receipt["driver_hashes"] = {}
                approval_path.write_bytes(s.canonical(receipt))
                with self.assertRaisesRegex(ValueError, "Approved tools"):
                    s.before_launch()

    def test_all_twelve_candidates_bind_complete_original_requests(self):
        program = "@hc.func('c')\ndef golden(x):\n    return [x]\n"
        with patch.object(s, "candidate_source", return_value=program) as lowering:
            for name, request in self.requests.items():
                with self.subTest(case=name):
                    s.validate_candidate(name, request, program, self.p)
            self.assertEqual(lowering.call_count, 12)
        self.assertNotIn("numpy", sys.modules)

    def test_full_request_changes_fail_before_lowering(self):
        name = s.CASES[0]
        for field in ("request_id", "generation_guidance", "layout", "model", "public_constants"):
            request = copy.deepcopy(self.requests[name])
            if field == "request_id":
                request[field] = "0" * 64
            else:
                request[field]["synthetic_change"] = True
            with patch.object(s, "candidate_source") as lowering, self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Original frozen generation request"):
                    s.validate_candidate(name, request, "program", self.p)
                lowering.assert_not_called()

    def test_graph_and_public_weight_hashes_are_separate(self):
        name = s.CASES[0]
        for field in ("model_sha256", "public_constants_sha256"):
            changed = self.proposal()
            changed["bindings"]["requests"][name][field] = "0" * 64
            with patch.object(s, "candidate_source") as lowering, self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Original graph/weight semantics"):
                    s.validate_candidate(name, self.requests[name], "program", changed)
                lowering.assert_not_called()

    def test_selected_candidate_bytes_and_tuple_return_are_exact(self):
        name = s.CASES[0]
        with patch.object(s, "candidate_source", return_value=("selected", {"strategy": "test"})):
            s.validate_candidate(name, self.requests[name], "selected", self.p)
            with self.assertRaisesRegex(ValueError, "Exact deterministic selected expression"):
                s.validate_candidate(name, self.requests[name], "different", self.p)

    def test_empty_oversize_and_nontext_candidates_rejected(self):
        name = s.CASES[0]
        for program in ("", "x" * 131073, None, 1):
            with patch.object(s, "candidate_source") as lowering, self.subTest(kind=type(program).__name__):
                with self.assertRaisesRegex(ValueError, "Bounded rule candidate"):
                    s.validate_candidate(name, self.requests[name], program, self.p)
                lowering.assert_not_called()

    def test_runtime_plan_matches_proposal_and_exact_deadline(self):
        self.check_runtime(self.runtime())

    def test_runtime_metadata_cannot_replace_proposal_fields(self):
        changed = self.runtime()
        changed["candidate_strategy"] = "replacement"
        with self.assertRaisesRegex(ValueError, "Runtime preserves approved plan"):
            self.check_runtime(self.rebind(changed))

    def test_runtime_started_and_deadline_must_preserve_original_budget(self):
        for mutation in ("later_deadline", "missing_start"):
            changed = self.runtime()
            if mutation == "later_deadline":
                changed["deadline_wall"] += 1
            else:
                changed.pop("started_wall")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Original finite deadline"):
                self.check_runtime(self.rebind(changed))

    def test_runtime_readonly_inventory_is_complete_and_safe(self):
        for mutation in ("duplicate", "outside", "incomplete"):
            changed = self.runtime()
            if mutation == "duplicate":
                changed["initial_result_directories"] = ["old-a", "old-a"]
            elif mutation == "outside":
                changed["initial_result_directories"] = ["../outside"]
            else:
                changed["readonly_result_entries"] = ["old-a"]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "historical result inventory|read-only history"):
                self.check_runtime(self.rebind(changed))

    def test_fixed_run_freshness_and_runtime_directory(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-joint-output-test-") as temp:
            run = Path(temp).resolve() / "fixed"
            with patch.object(s, "RUN", run):
                s.validate_output(run)
                with self.assertRaisesRegex(ValueError, "Single-use approved batch"):
                    s.validate_output(run.with_name("other"))
                run.mkdir()
                s.validate_runtime_output(run)
                with self.assertRaisesRegex(ValueError, "Single-use approved batch"):
                    s.validate_output(run)
                alias = run.with_name("alias")
                alias.symlink_to(run)
                with self.assertRaisesRegex(ValueError, "Original single-use runtime path"):
                    s.validate_runtime_output(alias)

    def test_continuation_rejected_before_plan_locks_or_output(self):
        self.assertFalse(s.ALLOW_CONTINUATION)
        self.assertTrue(s.UNIQUE_PHASES)
        driver = self.core.Driver(s, s.HERE / "runtime_acceptance.py")
        with patch.object(s, "before_launch"), patch.object(s, "plan") as plan, \
             patch.object(Path, "open", side_effect=AssertionError("No lock or output access allowed")):
            with self.assertRaisesRegex(ValueError, "forbids budget continuation"):
                driver.outer(s.RUN, s.RESULTS / "previous")
            plan.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
