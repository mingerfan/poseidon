"""Storage and identity boundaries; real backend evidence is recorded separately."""
import copy
import hashlib
import json
import unittest
from collections import Counter
from unittest import mock
from application_planning.compiled_resources import (
    _measure_plan, analyze_program, canonical, check_analysis, require_execution_budget,
    FORMAT, POLICY, _load_plan,
)

def descriptor(value, kind="ciphertext", level=12, components=2):
    return dict(id=value, kind=kind, level=level, components=components,
                scale_log2=45, context="context", place=dict(kind="host", rank=0), ntt=True)

def plan():
    # Eight coefficients per RNS polynomial make the independent byte table small.
    values = [descriptor("x"), descriptor("p", "plaintext", components=1),
              descriptor("a"), descriptor("m", components=3),
              descriptor("r"), descriptor("s", level=11)]
    return dict(values=values, external_inputs=["x"], final_outputs=["s"],
        initialization=[dict(kind="encode", ordinal=0, output="p",
                             payload=dict(kind="inline", values=[.5, -.25]))],
        execution=[dict(kind="compute", ordinal=1, op="add_cp", inputs=["x", "p"], output="a"),
                   dict(kind="compute", ordinal=2, op="mul_cc", inputs=["a", "a"], output="m"),
                   dict(kind="compute", ordinal=3, op="relinearize", inputs=["m"], output="r"),
                   dict(kind="compute", ordinal=4, op="rescale", inputs=["r"], output="s",
                        attrs=dict(target_level=11, target_scale_log2=30))], finalization=[])

class CompiledStorageTests(unittest.TestCase):
    def measure(self, p=None, budget=1 << 30, lengths=None):
        return _measure_plan(plan() if p is None else p, 8, budget, lengths or {})

    def test_table_of_coefficients_components_and_levels(self):
        r = self.measure()
        # 64 bytes per polynomial/modulus: x,a,r=1664; p=832; m=2496; s=1536.
        self.assertEqual([x["coefficient_payload_bytes"] for x in r["definitions_in_order"]],
                         [1664, 832, 1664, 2496, 1664, 1536])
        self.assertEqual(r["retained_ciphertext_coefficient_bytes"], 9024)
        self.assertEqual(r["retained_plaintext_coefficient_bytes"], 832)
        self.assertEqual(r["known_retained_payload_lower_bytes"], 9872)
        self.assertEqual(r["definitions"], dict(ciphertext=5, plaintext=1, total=6,
                                               unallocated_declarations=0))
        self.assertEqual(r["schedule"]["final_output_levels"], [11])
        self.assertTrue(r["evaluation_keys"]["relinearization_required"])

    def test_all_definitions_remain_even_when_no_longer_used(self):
        r = self.measure()
        self.assertEqual(r["retained_coefficient_payload_bytes"], 9856)
        self.assertEqual(r["definitions_in_order"][-1]["retained_coefficient_payload_bytes"], 9856)
        self.assertGreater(r["retained_coefficient_payload_bytes"], 1536 + 1664)

    def test_duplicate_final_output_is_one_existing_object(self):
        p = plan();p["final_outputs"] *= 3
        r = self.measure(p)
        self.assertEqual(r["output_objects"], 1)
        self.assertEqual(r["output_declarations"], 3)
        self.assertEqual(r["known_retained_payload_lower_bytes"], 9872)

    def test_unused_descriptors_have_no_allocated_payload(self):
        p = plan();p["values"].append(descriptor("unallocated", components=3))
        r = self.measure(p)
        self.assertEqual(r["definitions"]["unallocated_declarations"], 1)
        self.assertEqual(r["known_retained_payload_lower_bytes"], 9872)

    def test_bundle_slot_content_is_deduplicated_but_encoded_values_are_not(self):
        p = plan();p["values"].append(descriptor("p2", "plaintext", components=1))
        p["initialization"][0]["payload"] = dict(kind="bundle", content="content")
        p["initialization"].append(dict(kind="encode", ordinal=8, output="p2",
            payload=dict(kind="bundle", content="content")))
        r = self.measure(p, lengths={"content": 4})
        self.assertEqual(r["retained_bundle_slot_vectors"], 1)
        self.assertEqual(r["retained_plain_slot_payload_bytes"], 32)
        self.assertEqual(r["retained_plaintext_coefficient_bytes"], 1664)
        self.assertEqual(r["known_retained_payload_lower_bytes"], 10720)

    def test_inline_payloads_in_both_phases_are_retained(self):
        p = plan();p["values"].append(descriptor("p2", "plaintext", components=1))
        p["execution"].append(dict(kind="encode", ordinal=5, output="p2",
                                  payload=dict(kind="inline", values=[.25])))
        self.assertEqual(self.measure(p)["retained_plain_slot_payload_bytes"], 24)

    def test_exact_boundary_and_first_exceeding_definition(self):
        self.assertIsNone(self.measure(budget=9872)["first_exceeding_definition"])
        r = self.measure(budget=9871)
        self.assertEqual(r["first_exceeding_definition"]["ordinal"], 4)
        self.assertEqual(r["first_exceeding_definition"]["phase"], "execution")
        self.assertEqual(self.measure(budget=1)["first_exceeding_definition"]["phase"], "external_inputs")

    def test_requested_rotation_steps_are_unique_and_sorted(self):
        p = plan()
        for i, step in enumerate([1, -2, 1], 5):
            dst = "rot" + str(i);p["values"].append(descriptor(dst, level=11))
            p["execution"].append(dict(kind="compute", ordinal=i, op="rotate", inputs=["s"],
                                      output=dst, attrs=dict(steps=step)))
        self.assertEqual(self.measure(p)["evaluation_keys"]["requested_rotation_steps"], [-2, 1])

    def test_invalid_graph_definitions_are_rejected(self):
        variants = []
        p = plan();p["values"].append(copy.deepcopy(p["values"][0]));variants.append(p)
        p = plan();p["execution"][0]["inputs"] = ["undefined", "p"];variants.append(p)
        p = plan();p["execution"][0]["output"] = "x";variants.append(p)
        p = plan();p["execution"][0]["ordinal"] = 0;variants.append(p)
        p = plan();p["final_outputs"] = ["p"];variants.append(p)
        p = plan();p["finalization"] = [dict(kind="communication")];variants.append(p)
        for p in variants:
            with self.subTest(plan=p):
                with self.assertRaises(ValueError):self.measure(p)

    def test_invalid_metadata_and_parameters_are_rejected(self):
        for field, value in [("level", -1), ("level", 13), ("components", 0),
                             ("components", 4), ("components", True), ("kind", "opaque")]:
            p = plan();p["values"][0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):self.measure(p)
        for n, b in [(0, 1), (True, 1), (8, 0), (8, True)]:
            with self.assertRaises(ValueError):_measure_plan(plan(), n, b, {})

    def test_bundle_length_cannot_be_unknown_or_over_capacity(self):
        p = plan();p["initialization"][0]["payload"] = dict(kind="bundle", content="x")
        for lengths in ({}, {"x": 0}, {"x": 5}, {"x": True}):
            with self.assertRaises(ValueError):self.measure(p, lengths=lengths)

class CompiledAnalysisContractTests(unittest.TestCase):
    def sample(self, status="assessed"):
        r = dict(format=FORMAT, policy=POLICY, status=status,
                 requested_memory_budget_bytes=1048576, execution_readiness="unknown",
                 compilation_performed=False, encrypted_execution=False, paid_calls=0)
        r["binding"] = hashlib.sha256(canonical(r)).hexdigest()
        return r

    def test_matching_report_is_checked_before_host_execution(self):
        r = self.sample()
        with mock.patch("application_planning.compiled_resources.analyze_program", return_value=r) as analyzer:
            self.assertEqual(require_execution_budget(r, "program"), r)
            analyzer.assert_called_once_with("program", memory_budget_bytes=1048576)
        self.assertFalse(r["encrypted_execution"])
        self.assertEqual(r["execution_readiness"], "unknown")

    def test_forged_fields_or_stale_identity_are_not_reused(self):
        expected = self.sample()
        for rebind in (False, True):
            changed = copy.deepcopy(expected);changed["execution_readiness"] = "ready"
            if rebind:
                changed["binding"] = hashlib.sha256(canonical({k:v for k,v in changed.items() if k!="binding"})).hexdigest()
            with mock.patch("application_planning.compiled_resources.analyze_program", return_value=expected):
                with self.assertRaises(ValueError):check_analysis(changed, "program")

    def test_blocked_budget_is_terminal_for_host_gate(self):
        r = self.sample("blocked")
        with mock.patch("application_planning.compiled_resources.analyze_program", return_value=r):
            with self.assertRaisesRegex(ValueError, "compiled_resource_execution_blocked"):
                require_execution_budget(r, "program")

    def test_analysis_does_not_load_unchecked_packages_or_execute(self):
        with mock.patch("candidate_bundle.check_execution_dependencies",
                        side_effect=ValueError("bad package")), \
             mock.patch("application_planning.compiled_resources.source_authority", return_value={}), \
             mock.patch("application_planning.compiled_resources._basis", return_value={}), \
             mock.patch("program_execution.execute_program") as execute:
            with self.assertRaisesRegex(ValueError, "bad package"):analyze_program("bad")
            execute.assert_not_called()

class CompiledPlanProtocolTests(unittest.TestCase):
    def test_runtime_plan_uses_artifact_capacity_not_response_capacity(self):
        import tempfile
        from pathlib import Path
        p = plan()
        p["values"] += [descriptor("unused" + str(i)) for i in range(1024)]
        raw = canonical(p)
        self.assertGreater(len(raw), 128 * 1024)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "plan.json";file.write_bytes(raw)
            received, parsed = _load_plan(file)
        self.assertEqual(received, raw)
        self.assertEqual(parsed, p)

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "plan.json"
            for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
                file.write_bytes(raw)
                with self.assertRaises(ValueError):_load_plan(file)

    def test_artifact_symlink_and_size_bound_are_preserved(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "plan.json";file.write_bytes(canonical(plan()))
            link = Path(tmp) / "alias.json";link.symlink_to(file)
            with self.assertRaises(ValueError):_load_plan(link)
            file.write_bytes(b" " * (16 * 1024**2 + 1))
            with self.assertRaises(ValueError):_load_plan(file)

class CompiledArtifactSnapshotTests(unittest.TestCase):
    def test_parsed_snapshot_must_match_bound_artifact_bytes(self):
        import tempfile
        from pathlib import Path
        from application_planning.compiled_resources import sha
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "plan.json";raw = canonical(plan());file.write_bytes(raw)
            self.assertEqual(_load_plan(file, sha(raw)), (raw, plan()))
            file.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "compiled_resource_artifact_snapshot_changed"):
                _load_plan(file, sha(raw))
