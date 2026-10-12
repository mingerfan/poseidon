import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "summarize_run", HERE / "summarize_run.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


class RunSummaryTest(unittest.TestCase):
    def make_log(self, complete=True):
        lines = ["Script started on 2026-09-14 01:00:00+08:00"]
        count = 18 if complete else 2
        for index in range(count):
            lines.append("PHASE bootstrap_existing_ciphertext q=6")
            lines.append(
                "BOOTSTRAP_OFFLINE_CACHE matrices="
                + ("miss" if index == 0 else "hit")
            )
            for stage, elapsed in (
                ("S2C", 20),
                ("prepare", 1),
                ("ModRaise", 1),
                ("C2S", 100),
                ("EvalMod_real", 80),
                ("EvalMod_imag", 95),
                ("recombine_project_real", 3),
            ):
                lines.append(
                    f"BOOTSTRAP_GPU_TIMING stage={stage} elapsed_ms={elapsed} "
                    "excludes_offline_setup=true"
                )
            lines.append(
                "BOOTSTRAP_GPU_TIMING total_stages_ms=300 "
                "excludes_offline_setup=true"
            )
        for index in range(9 if complete else 1):
            lines.append(
                f"NETWORK_BLOCK_RESULT block=layer1.{index} "
                f"completed={index + 1} free_GiB=18 q=9 log_scale=40"
            )
        if complete:
            lines.append(
                "NETWORK_RESULT integration=PASS staged=false completed_blocks=9 "
                "bootstraps=18 convolutions=18 residual_adds=9 "
                "input_encryptions=27 intermediate_reencryptions=0 true_label=3 "
                "plain_prediction=3 gpu_prediction=3 max_logit_error=0.01 "
                "max_boundary_error=0.001 first_failure=none "
                "full_network_tested=true"
            )
            lines.append("Script done on 2026-09-14 01:10:00+08:00")
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with temporary:
            temporary.write("\n".join(lines))
        return Path(temporary.name)

    def test_complete_log(self):
        path = self.make_log()
        self.addCleanup(path.unlink)
        result = SUMMARY.summarize(path)
        self.assertTrue(result["complete"])
        self.assertEqual(result["bootstrap_gpu_stages"]["count"], 18)
        self.assertEqual(result["bootstrap_gpu_stages"]["sum_ms"], 5400)
        self.assertEqual(result["validation_wall_seconds"], 600)

    def test_incomplete_log_is_rejected_by_default(self):
        path = self.make_log(complete=False)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "incomplete/failed"):
            SUMMARY.summarize(path)
        self.assertFalse(SUMMARY.summarize(path, require_complete=False)["complete"])


if __name__ == "__main__":
    unittest.main()
