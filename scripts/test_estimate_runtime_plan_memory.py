"""Small accounting examples whose peaks can be calculated by hand."""

import copy
import unittest

from estimate_runtime_plan_memory import build_report, estimate, value_bytes


SPEC = {"spec_id": "test", "version": 1,
        "context": {"context_id": "test", "poly_degree": 8}}


def value(name, kind="device", index=0, components=2, level=1):
    place = {"kind": kind, "rank": 0}
    if kind == "device":
        place["index"] = index
    return {"id": name, "kind": "plaintext" if components == 1 else "ciphertext",
            "place": place, "context": "test", "level": level,
            "scale_log2": 20, "ntt": True, "components": components}


def chain():
    return {"format_version": 1, "target": {"operator_spec": {"id": "test", "version": 1}},
            "values": [value("h", "host"), value("x"), value("a"), value("b")],
            "external_inputs": ["h"], "final_outputs": ["b"],
            "initialization": [{"ordinal": 0, "kind": "transfer", "inputs": ["h"], "outputs": ["x"]}],
            "execution": [
                {"ordinal": 1, "kind": "compute", "op": "negate", "inputs": ["x"], "output": "a"},
                {"ordinal": 2, "kind": "compute", "op": "rotate", "inputs": ["a"], "output": "b"}],
            "finalization": []}


class EstimateTests(unittest.TestCase):
    def test_host_and_gpu_word_size(self):
        self.assertEqual(value_bytes(value("a"), 8), 128)
        self.assertEqual(value_bytes(value("h", "host"), 8), 256)
        self.assertEqual(value_bytes(value("p", components=1), 8), 64)

    def test_peak_counts_output_before_releasing_input(self):
        p = chain()
        report = build_report(p, SPEC, True)
        ordinary, release, reuse = [r["places"]["rank0/gpu0"] for r in report["results"]]
        self.assertEqual(ordinary["serial_peak_bytes"], 384)
        self.assertEqual(release["serial_peak_bytes"], 256)
        self.assertEqual(reuse["all_value_allocations_bytes"], 256)
        self.assertEqual(reuse["serial_peak_bytes"], 256)
        self.assertEqual(report["results"][2]["reuse_count"], 1)
        self.assertEqual(p, chain())  # What-if analysis must not edit the plan.

    def test_explicit_release_and_reuse(self):
        p = chain()
        p["format_version"] = 2
        p["execution"].insert(1, {"ordinal": 2, "kind": "release", "value": "x"})
        p["execution"][-1].update(ordinal=3, reuse_input=0)
        result = estimate(p, SPEC)
        self.assertEqual(result["reuse_count"], 1)
        self.assertEqual(result["places"]["rank0/gpu0"]["serial_peak_bytes"], 256)
        self.assertEqual(result["places"]["rank0/gpu0"]["all_value_allocations_bytes"], 256)

    def test_phase_carry_and_caller_retained_input(self):
        result = estimate(chain(), SPEC, "last_use")
        host = result["places"]["rank0/host"]
        self.assertEqual(host["phase_serial_peaks_bytes"]["finalization"], 256)
        gpu = result["places"]["rank0/gpu0"]
        self.assertEqual(gpu["phase_serial_peaks_bytes"]["initialization"], 128)
        self.assertEqual(gpu["phase_serial_peaks_bytes"]["finalization"], 128)

    def test_replication_has_one_copy_per_device(self):
        p = chain()
        p["values"].append(value("y", index=1))
        p["initialization"][0].update(kind="replicate", outputs=["x", "y"])
        p["final_outputs"].append("y")
        result = estimate(p, SPEC)
        self.assertEqual(result["places"]["rank0/gpu0"]["serial_peak_bytes"], 384)
        self.assertEqual(result["places"]["rank0/gpu1"]["serial_peak_bytes"], 128)
        self.assertEqual(result["places"]["rank0/host"]["serial_peak_bytes"], 256)

    def test_reuse_not_allowed_when_input_has_other_users(self):
        p = chain()
        p["values"].append(value("c"))
        p["execution"].append({"ordinal": 3, "kind": "compute", "op": "add_cc",
                               "inputs": ["a", "a"], "output": "c"})
        self.assertEqual(estimate(p, SPEC, "last_use_and_reuse")["reuse_count"], 0)
        p["format_version"] = 2
        p["execution"][1]["reuse_input"] = 0
        with self.assertRaisesRegex(ValueError, "unsupported reuse"):
            estimate(p, SPEC)

    def test_comparison_does_not_hide_use_after_release(self):
        p = chain()
        p["format_version"] = 2
        p["execution"].insert(0, {"ordinal": 1, "kind": "release", "value": "x"})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            build_report(p, SPEC, True)

    def test_original_v1_rejects_new_fields(self):
        p = chain()
        p["execution"][1]["reuse_input"] = 0
        with self.assertRaisesRegex(ValueError, "require V2"):
            estimate(p, SPEC)

    def test_metadata_mismatch_rejects_declared_reuse(self):
        p = chain()
        p["format_version"] = 2
        p["values"][-1]["level"] = 0
        p["execution"][1]["reuse_input"] = 0
        with self.assertRaisesRegex(ValueError, "unsupported reuse"):
            estimate(p, SPEC)
        spec = copy.deepcopy(SPEC)
        spec["context"]["context_id"] = "other"
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            estimate(chain(), spec)


if __name__ == "__main__":
    unittest.main()
