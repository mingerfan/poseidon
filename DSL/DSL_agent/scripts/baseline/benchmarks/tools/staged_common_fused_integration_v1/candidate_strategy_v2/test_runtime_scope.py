"""Stdlib control tests for r1301; no SDK, compiler, FHE or API is started.

The parent source_binding SDK query is replaced by its immutable r1296 data.
Synthetic approvals and runtime paths exist only inside TemporaryDirectory.
"""
import copy
import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_poseidon_capacity_scope_tests_r1301", HERE / "capacity_scope.py")
s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s)


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_plan = s.read(s.PRIOR_RUN / "plan.json")
        cls.old_bindings = cls.old_plan["bindings"]
        with tempfile.TemporaryDirectory(prefix="poseidon-capacity-control-") as temp:
            with patch.object(s.parent_scope, "source_binding", side_effect=lambda: copy.deepcopy(cls.old_bindings)), \
                 patch.object(s, "APPROVAL", Path(temp) / "absent.json"):
                cls.p = s.plan()
        cls.tools = s.driver_hashes()
        cls.requests = {name: s.request_value(name) for name in s.CASES}
        spec = importlib.util.spec_from_file_location("_poseidon_capacity_shared_driver_tests", s.COMMON / "runtime_driver.py")
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)

    def rebind(self, value):
        value["binding"] = s.digest({k: v for k, v in value.items() if k != "binding"})
        return value

    def proposal(self):
        return copy.deepcopy(self.p)

    def candidate_plan(self, source):
        value = self.proposal()
        value["bindings"]["candidate_hashes"] = {name: hashlib.sha256(source.encode()).hexdigest() for name in s.CASES}
        return value

    def receipt(self, proposal=None):
        proposal = self.p if proposal is None else proposal
        return dict(format="poseidon-common-fused-capacity-offline-approval-v1", approved=True,
                    proposal_binding=proposal["binding"], proposal_sha256="f" * 64,
                    driver_hashes=copy.deepcopy(self.tools), human_message="Synthetic control test only.")

    def runtime(self):
        value = self.proposal()
        value.update(started_wall=1800000000.0, deadline_wall=1800003600.0,
                     initial_result_directories=["old-a"], readonly_result_entries=["old-a"])
        return self.rebind(value)

    def check_runtime(self, value):
        actual = s.read
        with patch.object(s, "before_launch") as launch, patch.object(s, "read", side_effect=lambda path: self.p if Path(path) == s.PROPOSAL else actual(path)):
            s.check_plan(value)
            launch.assert_called_once()

    def prior_report_change(self, edit):
        actual = s.read
        report_path = s.PRIOR_RUN / "report.json"
        def altered(path):
            value = actual(path)
            if Path(path) == report_path:
                edit(value)
            return value
        with patch.object(s, "read", side_effect=altered):
            return s.prior_evidence(copy.deepcopy(self.old_bindings))

    def test_eleven_original_failures_and_thirty_three_new_phases(self):
        self.assertEqual(s.CASES, tuple(name for name in s.parent_scope.CASES if name != "common_boolean"))
        self.assertEqual(len(s.CASES), 11)
        self.assertEqual(len(set(s.CASES)), 11)
        self.assertEqual(len(s.CASES) * len(s.PHASES), 33)
        self.assertEqual(self.p["cases"], list(s.CASES))
        self.assertEqual(self.p["phases"], ["qualification", "new_inputs", "cache"])
        self.assertNotIn("common_boolean", self.p["bindings"]["requests"])

    def test_shared_driver_uses_original_reviewed_overlay_directory(self):
        driver = self.core.Driver(s, HERE / "runtime_acceptance.py")
        self.assertEqual(driver.HERE, s.parent_scope.HERE)
        self.assertEqual(driver.entry_script, HERE / "runtime_acceptance.py")
        self.assertEqual(s.HERE / "plan.json", s.parent_scope.HERE / "plan.json")
        self.assertEqual(s.TOOL_HERE, HERE)
        self.assertFalse((HERE / "runtime_scope.py").exists())

    def test_original_704_joint_706_and_backend_reference_654_preserved(self):
        bindings = self.p["bindings"]
        for field, count in (("original_production", 704), ("inside_sources", 706), ("protected_sources", 654)):
            self.assertEqual(len(bindings[field]), count)
            self.assertEqual(bindings[field], self.old_bindings[field])
        self.assertEqual(bindings["backend_identity"], self.old_bindings["backend_identity"])
        self.assertEqual(bindings["combined_plan_binding"], self.old_bindings["combined_plan_binding"])

    def test_prior_failed_report_and_boolean_reuse_have_independent_identity(self):
        history = self.p["bindings"]["prior_attempt"]
        reuse = self.p["bindings"]["reused_success"]
        self.assertEqual(history["report_sha256"], s.PRIOR_REPORT_SHA256)
        self.assertEqual(history["runtime_binding"], s.PRIOR_RUNTIME_BINDING)
        self.assertEqual([row["case"] for row in history["failed_cases"]], list(s.CASES))
        self.assertEqual(reuse["case"], "common_boolean")
        self.assertEqual(reuse["status"], "historical_passed")
        self.assertEqual(set(reuse["phase_receipts"]), set(s.PHASES))
        self.assertFalse(reuse["new_execution"])
        self.assertFalse(reuse["counted_in_current_rows"])
        self.assertNotEqual(s.RUN, s.PRIOR_RUN)

    def test_scope_has_exact_finite_caps_and_zero_external_actions(self):
        for key in ("provider_calls", "http_attempts", "installations", "downloads", "backend_changes", "parameter_changes"):
            self.assertEqual(s.SCOPE[key], 0)
        for key in ("compiler_attempts_max", "qualification_encrypted_phases_max", "new_input_encrypted_phases_max"):
            self.assertEqual(s.SCOPE[key], 11)
        self.assertEqual(s.SCOPE["wall_seconds"], 3600)
        self.assertEqual(s.SCOPE["phase_seconds"], 180)
        self.assertEqual(s.SCOPE["artifact_bytes"], 4 * 1024**3)
        self.assertEqual(s.SCOPE["global_bytes"], 32 * 1024**3)
        self.assertEqual(s.SCOPE["reserve_bytes"], 4 * 1024**3)
        self.assertEqual((s.SCOPE["native_workers"], s.SCOPE["compile_jobs"], s.SCOPE["link_jobs"]), (1, 2, 1))

    def test_new_strategy_and_origin_do_not_claim_agent_or_production(self):
        self.assertEqual(self.p["candidate_strategy"], "poseidon-generic-candidate-dispatch-v2")
        self.assertEqual(self.p["candidate_origin"], "rule_baseline_layout_aware_capacity")
        self.assertTrue(self.p["approval_required"])
        for field in ("new_agent_generation", "production_applied", "bootstrap", "cuda"):
            self.assertIs(self.p[field], False)

    def test_tool_inventory_binds_new_strategy_tests_and_immutable_driver(self):
        self.assertEqual(len(self.tools), 10)
        expected = {HERE / "capacity_scope.py", HERE / "runtime_acceptance.py", HERE / "test_runtime_scope.py",
                    HERE / "strategy.py", HERE / "test_strategy.py", s.HERE / "runtime_scope.py",
                    s.HERE / "runtime_acceptance.py", s.HERE / "test_runtime_scope.py",
                    s.COMMON / "runtime_driver.py", s.COMMON / "test_runtime_driver.py"}
        self.assertEqual(set(self.tools), {str(path.relative_to(s.ROOT)) for path in expected})
        for path in expected - {p for p in expected if p.parent == HERE}:
            self.assertEqual(s.sha(path), self.old_bindings["files"][str(path)])

    def test_matching_proposal_passes_without_sdk(self):
        s.check_plan(self.p)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("poseidon_backend_identity", sys.modules)

    def test_changed_prior_report_bytes_rejected_before_record_reads(self):
        actual = s.sha
        with patch.object(s, "sha", side_effect=lambda p: "0" * 64 if Path(p) == s.PRIOR_RUN / "report.json" else actual(p)), \
             patch.object(s, "read") as reader:
            with self.assertRaisesRegex(ValueError, "Original r1296 report bytes"):
                s.prior_evidence(self.old_bindings)
            reader.assert_not_called()

    def test_prior_pending_duplicate_or_passed_failure_is_rejected(self):
        mutations = [
            (lambda r: r["pending"].pop(), "not-run accounting"),
            (lambda r: r["rows"].append(r["rows"][0]), "case-phase inventory"),
            (lambda r: r["rows"][0].update(status="passed"), "uncompiled failures"),
            (lambda r: r["rows"][0].update(compiled=True), "uncompiled failures"),
            (lambda r: r["rows"][0].update(encrypted_execution=0), "uncompiled failures"),
            (lambda r: r["rows"][0].update(failure_layer="resources"), "uncompiled failures"),
            (lambda r: r.update(compilations=True), "not-run accounting"),
        ]
        for mutation, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.prior_report_change(mutation)

    def test_prior_boolean_success_cannot_be_failed_or_new_compile(self):
        for phase, field, value in (("qualification", "status", "failed"),
                                    ("new_inputs", "compiled", True), ("cache", "encrypted_execution", True)):
            def change(report):
                row = next(row for row in report["rows"] if row["case"] == "common_boolean" and row["phase"] == phase)
                row[field] = value
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "Preserved"):
                self.prior_report_change(change)

    def test_prior_binding_model_or_sdk_changes_rejected(self):
        for field in ("requests", "source_reference", "backend_identity", "inside_sources"):
            bindings = copy.deepcopy(self.old_bindings)
            bindings[field]["synthetic_change"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Original source/request/backend identity"):
                s.prior_evidence(bindings)

    def test_original_active_worker_rejected_before_read_or_sdk(self):
        with patch.object(s, "prior_running_processes", return_value=[123]), patch.object(s, "sha") as hashing:
            with self.assertRaisesRegex(ValueError, "execution still active"):
                s.prior_evidence(self.old_bindings)
            hashing.assert_not_called()

    def test_process_matching_uses_exact_old_path_arguments(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-proc-test-") as temp:
            folder = Path(temp)
            prefix = 100000 + os.getpid()
            cases = [(b"/usr/bin/python\0" + os.fsencode(str(s.PRIOR_RUN)) + b"\0", True),
                     (b"/runner\0" + os.fsencode(str(s.PRIOR_RUN / "common_alpha/tasks")) + b"\0", True),
                     (b"python\0" + os.fsencode(str(s.RUN)) + b"\0", False),
                     (b"bash\0-c\0print('" + os.fsencode(str(s.PRIOR_RUN)) + b"')\0", False)]
            expected = []
            for offset, (command, included) in enumerate(cases):
                pid = prefix + offset
                p = folder / str(pid); p.mkdir(); (p / "cmdline").write_bytes(command)
                if included: expected.append(pid)
            self.assertEqual(s.prior_running_processes(folder), expected)

    def test_rehashed_budget_type_and_scope_expansion_rejected(self):
        for field, value in (("wall_seconds", 3601), ("compiler_attempts_max", 12), ("native_workers", True)):
            proposal = self.proposal(); proposal["scope"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Exact finite scope"):
                s.check_plan(self.rebind(proposal))

    def test_case_phase_and_historical_reuse_substitution_rejected(self):
        proposal = self.proposal(); proposal["cases"][0] = "common_boolean"
        with self.assertRaisesRegex(ValueError, "Exact reviewed"):
            s.check_plan(self.rebind(proposal))
        for field in ("new_execution", "counted_in_current_rows"):
            proposal = self.proposal(); proposal["bindings"]["reused_success"][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Historical failure/reuse binding"):
                s.check_plan(self.rebind(proposal))

    def test_changed_tool_or_strategy_cannot_be_hidden_by_plan_rehash(self):
        proposal = self.proposal(); proposal["candidate_strategy"] = "old"
        with self.assertRaisesRegex(ValueError, "Offline rule scope only"):
            s.check_plan(self.rebind(proposal))
        actual = s.sha
        with patch.object(s, "sha", side_effect=lambda p: "0" * 64 if Path(p) == HERE / "strategy.py" else actual(p)):
            with self.assertRaisesRegex(ValueError, "Frozen tool/fixture/reference changed"):
                s.check_plan(self.p)

    def test_invalid_plan_hash_rejected(self):
        proposal = self.proposal(); proposal["binding"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Runtime plan identity"):
            s.check_plan(proposal)

    def test_valid_separate_synthetic_receipt(self):
        s.check_approval(self.p, self.receipt(), "f" * 64, self.tools)

    def test_old_r1296_approval_is_not_inherited(self):
        receipt = self.receipt(); receipt["format"] = "poseidon-common-fused-offline-approval-v1"
        with self.assertRaisesRegex(ValueError, "Separate human approval"):
            s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_nonboolean_extra_missing_or_wrong_receipt_data_rejected(self):
        for field, value in (("approved", 1), ("approved", False), ("proposal_binding", "0" * 64),
                             ("proposal_sha256", "0" * 64), ("driver_hashes", {}), ("human_message", "")):
            receipt = self.receipt(); receipt[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)
        for operation in ("extra", "missing"):
            receipt = self.receipt()
            if operation == "extra": receipt["extra"] = 1
            else: receipt.pop("human_message")
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "Separate human approval"):
                s.check_approval(self.p, receipt, "f" * 64, self.tools)

    def test_approval_cannot_expand_scope_or_enable_generation(self):
        for field in ("scope", "cases", "new_agent_generation", "production_applied"):
            proposal = self.proposal()
            if field == "scope": proposal[field]["phase_seconds"] += 1
            elif field == "cases": proposal[field][0] = "replacement"
            else: proposal[field] = True
            self.rebind(proposal)
            with self.subTest(field=field), self.assertRaises(ValueError):
                s.check_approval(proposal, self.receipt(proposal), "f" * 64, self.tools)

    def test_missing_approval_denies_all_execution_entries_before_sdk_or_output(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-capacity-deny-") as temp:
            missing = Path(temp) / "missing-approval.json"
            output = Path(temp) / "run"
            driver = self.core.Driver(s, HERE / "runtime_acceptance.py")
            calls = [(driver.outer, (output,)), (driver.namespace, (output,)), (driver.inside, (output,)),
                     (driver.mount_layer, (output, 0)), (driver.one, (output, s.CASES[0], "qualification"))]
            with patch.object(s, "APPROVAL", missing), patch.object(s, "read") as reader, \
                 patch.object(s, "plan") as planning, patch.object(self.core.subprocess, "Popen") as spawning:
                for method, args in calls:
                    with self.subTest(entry=method.__name__), self.assertRaisesRegex(ValueError, "Pending independent r1301"):
                        method(*args)
                reader.assert_not_called(); planning.assert_not_called(); spawning.assert_not_called()
            self.assertFalse(output.exists())
            self.assertNotIn("numpy", sys.modules)

    def test_temporary_receipt_binds_exact_proposal_and_checks_old_idle(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-capacity-approval-") as temp:
            proposal = Path(temp) / "proposal.json"; approval = Path(temp) / "approval.json"
            proposal.write_bytes(s.canonical(self.p))
            receipt = self.receipt(); receipt["proposal_sha256"] = s.sha(proposal)
            approval.write_bytes(s.canonical(receipt))
            with patch.object(s, "PROPOSAL", proposal), patch.object(s, "APPROVAL", approval), \
                 patch.object(s, "driver_hashes", return_value=self.tools), \
                 patch.object(s, "prior_running_processes", return_value=[]):
                s.before_launch()
                with patch.object(s, "prior_running_processes", return_value=[999]):
                    with self.assertRaisesRegex(ValueError, "execution still active"):
                        s.before_launch()

    def test_all_original_requests_bind_before_candidate_selection(self):
        with patch.object(s, "candidate_source", return_value=("selected", {})) as selected:
            for name, request in self.requests.items():
                s.validate_candidate(name, request, "selected", self.candidate_plan("selected"))
            self.assertEqual(selected.call_count, 11)

    def test_changed_request_or_public_weights_fail_before_candidate_selection(self):
        name = s.CASES[0]
        for field in ("request_id", "model", "public_constants", "layout", "generation_guidance"):
            request = copy.deepcopy(self.requests[name])
            if field == "request_id": request[field] = "0" * 64
            else: request[field]["synthetic_change"] = True
            with patch.object(s, "candidate_source") as selected, self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Original frozen generation request"):
                    s.validate_candidate(name, request, "selected", self.candidate_plan("selected"))
                selected.assert_not_called()

    def test_invalid_candidate_text_or_selected_bytes_rejected(self):
        name = s.CASES[0]
        for source in ("", None, 1, "x" * 131073):
            with patch.object(s, "candidate_source") as selected, self.subTest(type=type(source).__name__):
                with self.assertRaisesRegex(ValueError, "Bounded rule candidate"):
                    s.validate_candidate(name, self.requests[name], source, self.p)
                selected.assert_not_called()
        with patch.object(s, "candidate_source", return_value=("selected", {})):
            with self.assertRaisesRegex(ValueError, "Exact deterministic selected expression"):
                s.validate_candidate(name, self.requests[name], "other", self.candidate_plan("other"))

    def test_runtime_fields_preserve_approved_proposal_and_deadline(self):
        self.check_runtime(self.runtime())
        for mutation in ("deadline", "start", "readonly", "strategy"):
            value = self.runtime()
            if mutation == "deadline": value["deadline_wall"] += 1
            elif mutation == "start": value.pop("started_wall")
            elif mutation == "readonly": value["readonly_result_entries"] = []
            else: value["candidate_strategy"] = "replacement"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check_runtime(self.rebind(value))

    def test_single_use_path_and_no_budget_continuation(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-capacity-path-") as temp:
            run = Path(temp).resolve() / "fixed"
            with patch.object(s, "RUN", run):
                s.validate_output(run)
                with self.assertRaisesRegex(ValueError, "Single-use approved batch"):
                    s.validate_output(s.PRIOR_RUN)
                run.symlink_to(Path(temp) / "absent")
                with self.assertRaisesRegex(ValueError, "Single-use approved batch"):
                    s.validate_output(run)
                run.unlink(); run.mkdir()
                s.validate_runtime_output(run)
                with self.assertRaisesRegex(ValueError, "Single-use approved batch"):
                    s.validate_output(run)
        driver = self.core.Driver(s, HERE / "runtime_acceptance.py")
        with patch.object(s, "before_launch"), patch.object(s, "plan") as planning:
            with self.assertRaisesRegex(ValueError, "forbids budget continuation"):
                driver.outer(s.RUN, s.PRIOR_RUN)
            planning.assert_not_called()

    def static_report_change(self, edit):
        actual = s.read
        changed = copy.deepcopy(actual(s.STATIC_REPORT))
        edit(changed)
        changed["binding"] = s.digest({key: value for key, value in changed.items() if key != "binding"})
        def altered(path):
            return copy.deepcopy(changed) if Path(path) == s.STATIC_REPORT else actual(path)
        with patch.object(s, "read", side_effect=altered), patch.object(s, "STATIC_REPORT_BINDING", changed["binding"]):
            return s.static_candidates(copy.deepcopy(self.old_bindings))

    def test_actual_static_report_freezes_only_eleven_candidate_bytes(self):
        hashes, evidence, files = s.static_candidates(copy.deepcopy(self.old_bindings))
        self.assertEqual(set(hashes), set(s.CASES))
        self.assertEqual(len(hashes), 11)
        self.assertEqual(len(files), 12)
        self.assertEqual(hashes, self.p["bindings"]["candidate_hashes"])
        self.assertEqual(evidence, self.p["bindings"]["static_candidate_evidence"])
        self.assertEqual(evidence["report_sha256"], s.STATIC_REPORT_SHA256)
        self.assertFalse(evidence["boolean_counted_as_numerical_proof"])
        self.assertNotIn(str(s.STATIC_REPORT.parent / "common_boolean/candidate.py"), files)

    def test_static_report_must_have_original_bytes_and_self_binding(self):
        actual = s.sha
        with patch.object(s, "sha", side_effect=lambda path: "0" * 64 if Path(path) == s.STATIC_REPORT else actual(path)):
            with self.assertRaisesRegex(ValueError, "Frozen static report bytes"):
                s.static_candidates(self.old_bindings)
        with patch.object(s, "STATIC_REPORT_BINDING", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Frozen static report binding"):
                s.static_candidates(self.old_bindings)

    def test_static_row_request_abi_count_and_zero_call_mismatches_rejected(self):
        edits = [
            (lambda r: r["rows"][0].update(model_sha256="0" * 64), "request/model/weight identity"),
            (lambda r: r["rows"][0].update(request_canonical_sha256="0" * 64), "request/model/weight identity"),
            (lambda r: r["rows"][0].update(physical_output_count=3), "physical ABI"),
            (lambda r: r["rows"][0].update(expected_physical_output_count=True), "physical ABI"),
            (lambda r: r["rows"][0].update(execution_abi="legacy"), "physical ABI"),
            (lambda r: r["rows"][0].update(compiler_calls=True), "zero calls"),
            (lambda r: r["rows"][0].update(status="failed"), "repair provenance"),
            (lambda r: r.update(compiler_calls=1), "zero calls"),
            (lambda r: r["rows"].pop(), "twelve-case inventory"),
        ]
        for edit, message in edits:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.static_report_change(edit)

    def test_static_candidate_path_and_content_cannot_be_substituted(self):
        for path in ("/private/tmp/candidate.py", str(s.STATIC_REPORT.parent / "common_boolean/candidate.py")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "confined path"):
                self.static_report_change(lambda r: r["rows"][0].update(candidate_file=path))
        with self.assertRaisesRegex(ValueError, "Frozen static candidate bytes"):
            self.static_report_change(lambda r: r["rows"][0].update(candidate_sha256="0" * 64))

    def test_runtime_candidate_must_match_pregenerated_bytes_before_factory(self):
        name = s.CASES[0]
        with patch.object(s, "candidate_source") as selected:
            with self.assertRaisesRegex(ValueError, "Frozen static candidate bytes"):
                s.validate_candidate(name, self.requests[name], "different", self.p)
            selected.assert_not_called()


    def test_rehashed_candidate_hash_substitution_is_rejected(self):
        value = self.proposal()
        value["bindings"]["candidate_hashes"][s.CASES[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Exact frozen static candidate evidence"):
            s.check_plan(self.rebind(value))


    def test_resource_deadline_and_reserve_are_not_extended(self):
        with patch.object(s.time, "time", return_value=10):
            with self.assertRaisesRegex(ValueError, "Original offline deadline"):
                s.resource_boundary(s.RUN, {"deadline_wall": 10, "initial_result_directories": []})
            with patch.object(s.shutil, "disk_usage", return_value=types.SimpleNamespace(free=s.SCOPE["reserve_bytes"] - 1)):
                with self.assertRaisesRegex(ValueError, "Disk reserve"):
                    s.resource_boundary(s.RUN, {"deadline_wall": 11, "initial_result_directories": []})


if __name__ == "__main__":
    unittest.main(verbosity=2)
