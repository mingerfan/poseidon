"""A completed diagnostic run must never be counted as GPU execution success."""
import json
import os
from pathlib import Path
import unittest

from run_poseidon_cohort_preflight import summarize, digest


class CohortPreflightTests(unittest.TestCase):
    def test_diagnostics_completion_is_not_execution_success(self):
        rows = [dict(family="linear", status="diagnosed", unsupported_opcodes=[4], schedule_built=False),
                dict(family="conv1d", status="failed")]
        summary = summarize(rows, 96)
        self.assertEqual(summary["planned"], 96)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["diagnosed"], 1)
        self.assertEqual(summary["unsupported_opcode_case_counts"], {"4": 1})
        self.assertFalse(summary["poseidon_gpu_executed"])
        self.assertFalse(summary["gpu_correctness_validated"])


@unittest.skipUnless(os.environ.get("POSEIDON_COHORT_PREFLIGHT_RESULTS"), "Requires actual native Poseidon diagnostics")
class CohortPreflightEvidenceTests(unittest.TestCase):
    def test_all_96_real_artifacts_blocked_at_modswitch_with_no_schedule(self):
        root = Path(os.environ["POSEIDON_COHORT_PREFLIGHT_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "diagnostics_complete")
        self.assertEqual(digest(Path(report["source_report"])), report["source_sha256"])
        self.assertEqual(digest(Path(report["tool"])), report["tool_sha256"])
        self.assertEqual(report["summary"], summarize(report["cases"], 96))
        self.assertEqual(len(report["cases"]), 96)
        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["summary"]["unsupported_opcode_case_counts"], {"4": 96})
        self.assertFalse(report["poseidon_gpu_executed"])
        for row in report["cases"]:
            native = json.loads((root / row["report_file"]).read_text())
            self.assertEqual(row["exit_code"], 1)
            self.assertEqual(row["execution_gate"], native["execution_gate"])
            self.assertEqual(native["execution_gate"]["status"], "not_ready")
            self.assertFalse(native["execution_gate"]["checks"]["schedule_built"])
            self.assertTrue(any(c["opcode"] == 4 and not c["supported"]
                                for c in native["hevm_opcode_summary"]["opcode_counts"]))
            for key in ("hevm", "constants"):
                path = Path(native["artifacts"][key])
                self.assertEqual(digest(path), row["artifact_sha256"][path.name])
            self.assertFalse(native["execution_config"]["preflight"]["galois_keys"])
            self.assertFalse(native["execution_config"]["preflight"]["relin_keys"])


if __name__ == "__main__":
    unittest.main()
