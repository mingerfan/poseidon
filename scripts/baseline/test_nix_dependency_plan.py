"""Offline Nix checks: evaluate recipes, optionally enter an already-built shell."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS

import check_dacapo_environment as gate

REPO = Path(__file__).resolve().parents[2]
DEPS = REPO / "src/poseidon/tools/dacapo/dacapo-dependencies.nix"
SHELL = DEPS.with_name("dacapo-shell.nix")
LOCK = json.loads(DEPS.with_name("dependency-lock.json").read_text())
LAUNCHER = Path(str(WORK / "deps/nix-portable-v012/nix-portable"))


def evaluate(path, attribute, *extra):
    return subprocess.run(
        ["timeout", "-k", "3s", "90s", "bash",
         str(REPO / "scripts/baseline/nix_portable.sh"), "nix", "eval",
         "--offline", "--option", "allow-import-from-derivation", "false",
         "--json", "--file", str(path), attribute, *extra],
        cwd=REPO, capture_output=True, text=True, timeout=100, check=False,
    )


class DependencyLockTests(unittest.TestCase):
    def test_lock_matches_environment_gate(self):
        self.assertEqual(LOCK["dacapo_commit"], gate.DACAPO_COMMIT)
        self.assertEqual(LOCK["llvm"]["version"], gate.LLVM_VERSION)
        self.assertEqual(LOCK["seal"]["version"], gate.SEAL_VERSION)

    def test_downloads_are_content_pinned(self):
        for name in ("llvm", "seal"):
            self.assertRegex(LOCK[name]["sha256"], r"^[a-f0-9]{64}$")
            self.assertTrue(LOCK[name]["url"].startswith("https://github.com/"))
            self.assertGreater(LOCK[name]["archive_bytes"], 0)
        for name in ("nixpkgs", "msgsl"):
            self.assertRegex(LOCK[name]["nar_hash"], r"^sha256-[A-Za-z0-9+/]{43}=$")

    def test_python_install_is_explicitly_separate(self):
        self.assertFalse(LOCK["python_stage"]["included_in_compiler_install"])
        self.assertTrue(LOCK["python_stage"]["wheels_locked"])
        self.assertTrue(DEPS.with_name(LOCK["python_stage"]["wheel_lock"]).is_file())


@unittest.skipUnless(sys.platform == "linux" and LAUNCHER.is_file(),
                     "requires the separately approved WSL nix-portable bootstrap")
class OfflineNixPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = evaluate(DEPS, "metadata")
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.metadata = json.loads(result.stdout)

    def test_exact_toolchain_and_seal(self):
        versions = self.metadata["versions"]
        for name in ("llvm", "mlir", "clang"):
            self.assertEqual(versions[name], "18.1.2")
        self.assertEqual(versions["seal"], "4.0.0")
        self.assertEqual(versions["msgsl"], "3.1.0")

    def test_no_execution_success_from_evaluation(self):
        self.assertEqual(self.metadata["scope"], "dependency_plan_only")
        self.assertFalse(self.metadata["compiler_build_validated"])
        self.assertFalse(self.metadata["encrypted_execution_validated"])

    def test_small_parallelism_and_preserved_seal_guards(self):
        self.assertEqual(self.metadata["limits"], {"max_jobs": 1, "cores": 2, "link_jobs": 1})
        for flag in ("-DLLVM_PARALLEL_COMPILE_JOBS=2", "-DLLVM_PARALLEL_LINK_JOBS=1"):
            self.assertIn(flag, self.metadata["toolchain_cmake_flags"])
        for flag in ("-DCMAKE_INSTALL_INCLUDEDIR=include",
                     "-DSEAL_BUILD_DEPS=OFF", "-DSEAL_USE_MSGSL=ON",
                     "-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON"):
            self.assertIn(flag, self.metadata["seal_cmake_flags"])

    def test_shell_instantiates_without_build_or_download(self):
        result = evaluate(SHELL, "drvPath")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout).endswith(".drv"))

    def test_realized_shell_needs_no_implicit_download_or_build(self):
        physical = LAUNCHER.parent / ".nix-portable"
        if not all((physical / output.lstrip("/")).exists()
                   for output in self.metadata["compiler_outputs"]):
            self.skipTest("compiler dependencies not built yet")
        result = subprocess.run([
            "timeout", "-k", "3s", "60s", "env",
            f"NIX_BUILD_SHELL={self.metadata['shell_bash']}", "bash",
            str(REPO / "scripts/baseline/nix_portable.sh"), "nix-shell", "--pure",
            "--option", "substitute", "false", "--max-jobs", "0",
            "--option", "builders", "", "--option", "allow-import-from-derivation", "false",
            str(SHELL), "--run",
            'test "$CC" = "$NIX_CC/bin/clang" && test "$CXX" = "$NIX_CC/bin/clang++" && llvm-config --version'],
            cwd=REPO, capture_output=True, text=True, timeout=70, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "18.1.2")
        self.assertNotIn("will be built", result.stderr)
        self.assertNotIn("will be fetched", result.stderr)

    def test_missing_source_is_rejected(self):
        result = evaluate(DEPS, "metadata", "--argstr", "nixpkgsSource",
                          str(REPO / "scripts/baseline/intentionally-missing-nixpkgs"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pinned nixpkgs source is missing", result.stderr)

    def test_unrelated_source_cannot_replace_pinned_snapshot(self):
        # Nix can reuse a store object with the expected hash without copying
        # the candidate. Either reject the candidate or reuse identical pinned
        # metadata; never interpret the unrelated tree as a package set.
        result = evaluate(DEPS, "metadata", "--argstr", "nixpkgsSource", str(REPO / ".agents"))
        if result.returncode:
            self.assertIn("hash mismatch", result.stderr)
        else:
            self.assertEqual(json.loads(result.stdout), self.metadata)


if __name__ == "__main__":
    unittest.main()
