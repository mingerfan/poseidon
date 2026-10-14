"""Mocked orchestration tests; never invoke Nix, download or claim FHE success."""
import json
from pathlib import Path
import unittest

import continue_dacapo_cpp as job


class ContinuationTests(unittest.TestCase):
    def test_cpp_driver_conserves_link_memory(self):
        driver = Path(__file__).with_name("build_hecate_cpp.sh").read_text()
        for flag in ("-DCMAKE_JOB_POOLS=compile=2;link=1", "-DCMAKE_JOB_POOL_COMPILE=compile",
                     "-DCMAKE_JOB_POOL_LINK=link", "--parallel 2"):
            self.assertIn(flag, driver)

    def simulate(self, failure=None, plan="", mutate=False):
        self.stages = []
        self.verifications = 0

        def verify():
            self.verifications += 1
            if mutate and self.verifications == 2:
                raise RuntimeError("recipe changed")

        def run(stage, command, timeout):
            self.stages.append(stage)
            self.assertGreater(timeout, 0)
            self.assertNotIn("toolchain", command)  # Never restart the LLVM build.
            if stage == failure:
                raise RuntimeError(stage)
            if stage == "dependency_metadata":
                return json.dumps({"compiler_outputs": [
                    f"/nix/store/{'a' * 32}-{name}" for name in ("llvm", "clang", "seal")],
                    "shell_bash": f"/nix/store/{'d' * 32}-bash-5.2p26/bin/bash"})
            if stage == "remaining_shell_plan":
                return plan
            if stage == "build_hecate_cpp":
                self.assertIn("--pure", command)
                i = command.index("substitute")
                self.assertEqual(command[i + 1], "false")
                self.assertEqual(command[command.index("--max-jobs") + 1], "0")
                self.assertEqual(command[command.index("builders") + 1], "")
                self.assertIn(f"NIX_BUILD_SHELL=/nix/store/{'d' * 32}-bash-5.2p26/bin/bash", command)
            return ""

        job.continue_cpp(run, verify)

    def test_order_and_checks(self):
        self.simulate()
        self.assertEqual(self.stages, ["wait_existing_build", "dependency_metadata",
            "dependency_validity", "remaining_shell_plan", "realize_shell", "build_hecate_cpp"])
        self.assertEqual(self.verifications, 5)

    def test_failed_toolchain_is_not_restarted(self):
        with self.assertRaisesRegex(RuntimeError, "dependency_validity"):
            self.simulate(failure="dependency_validity")
        self.assertNotIn("realize_shell", self.stages)

    def test_failed_shell_does_not_build_dacapo(self):
        with self.assertRaisesRegex(RuntimeError, "realize_shell"):
            self.simulate(failure="realize_shell")
        self.assertNotIn("build_hecate_cpp", self.stages)

    def test_failed_native_build_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "build_hecate_cpp"):
            self.simulate(failure="build_hecate_cpp")

    def test_recipe_change_during_wait_stops(self):
        with self.assertRaisesRegex(RuntimeError, "recipe changed"):
            self.simulate(mutate=True)
        self.assertEqual(self.stages, ["wait_existing_build"])

    def test_wait_timeout_stops(self):
        with self.assertRaisesRegex(RuntimeError, "wait_existing_build"):
            self.simulate(failure="wait_existing_build")
        self.assertEqual(self.stages, ["wait_existing_build"])

    def test_unexpected_source_build_stops(self):
        with self.assertRaisesRegex(RuntimeError, "source builds"):
            self.simulate(plan=f"  /nix/store/{'b' * 32}-extra-llvm.drv")
        self.assertNotIn("realize_shell", self.stages)

    def test_download_budget_stops(self):
        with self.assertRaisesRegex(RuntimeError, "112 MiB"):
            self.simulate(plan="these 2 paths will be fetched (113.00 MiB download, 450 MiB unpacked):")
        self.assertNotIn("realize_shell", self.stages)

    def test_unrecognized_budget_stops(self):
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            self.simulate(plan="these 2 paths will be fetched (1.2 GiB download, 4 GiB unpacked):")

    def test_expected_small_shell_plan(self):
        plan = (f"these 2 derivations will be built:\n"
                f"  /nix/store/{'b' * 32}-stdenv-linux.drv\n"
                f"  /nix/store/{'c' * 32}-nix-shell.drv\n"
                "these 2 paths will be fetched (91.23 MiB download, 450 MiB unpacked):")
        self.simulate(plan=plan)
        self.assertEqual(self.stages[-1], "build_hecate_cpp")


if __name__ == "__main__":
    unittest.main()
