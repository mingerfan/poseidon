"""Opt-in audit of real explicit-physical-Q GPU schedule execution."""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get("POSEIDON_GPU_DROP_SCHEDULE_RESULTS"), "Requires actual GPU static schedule evidence")
class GpuDropScheduleEvidenceTests(unittest.TestCase):
    def test_frozen_sources_outputs_and_all_four_inputs_ten_transitions(self):
        root=Path(os.environ["POSEIDON_GPU_DROP_SCHEDULE_RESULTS"])
        report=json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["gpu_executed"])
        self.assertTrue(report["fhe_executed"])
        self.assertFalse(report["hevm_executed"])
        self.assertEqual(report["agent_calls"], 0)
        for rel, expected in report["source_hashes"].items():
            path=(root / rel).resolve()
            self.assertTrue(path.is_relative_to(root.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        for name, item in report["checks"].items():
            self.assertEqual(item["exit_code"],0)
            for suffix in ("stdout", "stderr"):
                raw=(root / (name+"."+suffix)).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),item[suffix+"_sha256"])
        data=report["result"]
        self.assertEqual(json.loads((root / "primitive.stdout").read_text()),data)
        self.assertEqual(data["backend"],"poseidon_gpu_static_schedule")
        self.assertEqual(data["security_request"],"tc128")
        self.assertEqual((data["degree"],data["slots"],data["scale"]),(16384,8192,2**40))
        self.assertEqual((data["atol"],data["rtol"]),(1e-5,1e-4))
        self.assertEqual(data["input_cases"],4)
        self.assertEqual(data["runtime_mismatch_checks"],4)
        self.assertEqual([(r["input_case"],r["target_q_count"]) for r in data["cases"]],
                         [(i,q) for i in range(4) for q in range(11,1,-1)])
        for row in data["cases"]:
            self.assertTrue(row["cpu_gpu_exact"])
            self.assertTrue(row["source_unchanged"])
            self.assertTrue(math.isfinite(row["max_absolute_error"]))
            self.assertLessEqual(row["max_absolute_error"],1e-5+1e-4*0.5)
            self.assertGreaterEqual(row["mae"],0)
            self.assertLessEqual(row["mae"],row["max_absolute_error"])


if __name__ == "__main__":
    unittest.main()
