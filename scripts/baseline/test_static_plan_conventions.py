"""Evidence for preparation only; mock HEVM parsing is not HEVM execution."""
import hashlib
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get("POSEIDON_STATIC_PLAN_CONVENTION_RESULTS"),
                     "Requires native explicit static-plan preparation results")
class StaticPlanConventionEvidenceTests(unittest.TestCase):
    def test_native_plan_output_and_source_snapshots(self):
        root = Path(os.environ["POSEIDON_STATIC_PLAN_CONVENTION_RESULTS"]).resolve()
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["gpu_executed"])
        self.assertFalse(report["hevm_execution"])
        self.assertEqual(report["agent_calls"], 0)
        check = report["checks"]["static_plan_convention"]
        self.assertEqual(check["exit_code"], 0)
        self.assertEqual(len(check["binary_sha256"]), 64)
        for stream in ("stdout", "stderr"):
            raw = (root / ("static_plan_convention."+stream)).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), check[stream+"_sha256"])
        output = (root / "static_plan_convention.stdout").read_text()
        self.assertIn("tc128 explicit static plan: Q=1,2,8, full-slot decode and legacy/invalid checks passed", output)
        self.assertIn("mock HEVM; no encrypted or GPU execution", output)
        self.assertIn("mgpu HEVM static execution plan tests passed", output)
        source_names = []
        for path, expected in report["source_hashes"].items():
            source = (root / path).resolve()
            self.assertTrue(source.is_relative_to(root))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
            source_names.append(source.name)
        for suffix in ("hevm_static_execution_plan.h", "hevm_static_execution_plan.cpp",
                       "hevm_static_execution_plan_test.cpp"):
            self.assertTrue(any(name.endswith("-"+suffix) for name in source_names))


if __name__ == "__main__":
    unittest.main()
