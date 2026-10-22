import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_dacapo_environment as gate


class EnvironmentGateTests(unittest.TestCase):
    def test_exact_llvm_version(self):
        self.assertTrue(gate.version_matches("18.1.2", gate.LLVM_VERSION))
        for version in (None, "16.0.6", "18.1.8", "19.1.2"):
            self.assertFalse(gate.version_matches(version, gate.LLVM_VERSION))

    def test_minimum_version(self):
        self.assertTrue(gate.version_matches("3.27.1", "3.22.1", minimum=True))
        self.assertFalse(gate.version_matches("3.9.9", "3.22.1", minimum=True))

    def test_version_extraction(self):
        self.assertEqual(gate.version_from_output("LLVM version 18.1.2\n"), "18.1.2")
        self.assertEqual(gate.version_from_output("18.1.2\n"), "18.1.2")
        self.assertIsNone(gate.version_from_output("not available"))

    def test_uninitialized_submodule_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            (repo / "third_party/dacapo").mkdir(parents=True)
            output = f"160000 commit {gate.DACAPO_COMMIT}\tthird_party/dacapo"
            with patch.object(gate, "probe", return_value=(0, output)) as probe:
                result = gate.check_submodule(repo)
            self.assertTrue(result["gitlink_ok"])
            self.assertFalse(result["ok"])
            self.assertEqual(probe.call_count, 1)

    def test_missing_explicit_prefix_does_not_fall_back_to_path(self):
        with tempfile.TemporaryDirectory() as folder:
            result = gate.check_tool("llvm-config", gate.LLVM_VERSION, Path(folder))
        self.assertFalse(result["ok"])
        self.assertEqual(result["diagnostic"], "missing executable")

    def test_missing_seal_prefix(self):
        self.assertFalse(gate.check_seal(None)["ok"])

    def test_probe_reports_timeout(self):
        with patch.object(gate.subprocess, "run", side_effect=gate.subprocess.TimeoutExpired("probe", 25)):
            code, _ = gate.probe(["cmake", "--version"])
        self.assertNotEqual(code, 0)

    def test_gate_checks_selected_nix_compiler_without_path_fallback(self):
        for compiler, cxx, version in (("gcc", "g++", "13.2.0"),
                                        ("clang", "clang++", "18.1.2")):
            with self.subTest(compiler=compiler), \
                    patch.dict(gate.os.environ, {"IN_NIX_SHELL": "pure",
                        "NIX_CC": "/nix/store/test-wrapper",
                        "HECATE_C_COMPILER_NAME": compiler}, clear=True), \
                    patch.object(gate.sys, "argv", ["check_dacapo_environment.py"]), \
                    patch.object(gate.sys, "platform", "linux"), \
                    patch("platform.machine", return_value="x86_64"), \
                    patch.object(gate, "check_tool", return_value={"ok": True}) as tool, \
                    patch.object(gate, "check_submodule", return_value={"ok": True}), \
                    patch.object(gate, "check_seal", return_value={"ok": True}), \
                    patch.object(gate, "check_python_package", return_value={"ok": True}), \
                    contextlib.redirect_stdout(io.StringIO()):
                gate.main()
                tool.assert_any_call(compiler, version, prefix=Path("/nix/store/test-wrapper"))
                tool.assert_any_call(cxx, version, prefix=Path("/nix/store/test-wrapper"))
                other = "clang" if compiler == "gcc" else "gcc"
                self.assertNotIn(other, [call.args[0] for call in tool.call_args_list])

    def test_unknown_environment_compiler_is_rejected(self):
        with patch.dict(gate.os.environ, {"HECATE_C_COMPILER_NAME": "unexpected"}), \
                patch.object(gate.sys, "argv", ["check_dacapo_environment.py"]), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            gate.main()
        self.assertEqual(error.exception.code, 2)

    def test_cpu_torch_wheel_matches_release(self):
        with patch.object(gate.importlib.metadata, "version", return_value="2.0.1+cpu"):
            self.assertTrue(gate.check_python_package("torch", "2.0.1+cpu")["ok"])
            self.assertFalse(gate.check_python_package("torch", "2.0.1")["ok"])


if __name__ == "__main__":
    unittest.main()
