"""The toolchain builds its compiler command itself and no longer imports the legacy SDK modules."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from pt_agent import paths, toolchain
from pt_agent.tools import nixrun

FIXTURE = Path(__file__).resolve().parent / "fixtures/toolchain_argv_legacy.json"
LEGACY = ("poseidon_runtime_paths", "poseidon_runtime_compile", "compiler_configuration", "native_sdk_manifest")


class DecoupledToolchainTests(unittest.TestCase):
    def test_compile_argv_matches_legacy_command(self):
        expected = json.loads(FIXTURE.read_text())["argv"]
        for key, argv in expected.items():
            mode, boot = key.split("|")
            with self.subTest(key=key):
                self.assertEqual(toolchain.compile_argv(mode, None if boot == "None" else boot, Path("/RUN")), argv)

    def test_no_legacy_sdk_modules_imported(self):
        code = ("import sys; from pathlib import Path; from pt_agent import toolchain, cost; "
                "toolchain.setup_env(); toolchain.host_paths(); "
                "[toolchain.compile_argv(m, b, Path('/RUN')) for m in toolchain.COMPILE_MODES for b in toolchain.BOOT_PROFILES "
                "if not (b and m == 'eva')]; "
                f"print([m for m in {LEGACY!r} if m in sys.modules])")
        scripts = Path(toolchain.__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items() if k != "POSEIDON_SDK_MANIFEST"}
        out = subprocess.run([sys.executable, "-c", code], cwd=scripts, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")
        self.assertNotIn("POSEIDON_SDK_MANIFEST", env)

    def test_profiles_shipped_with_pt_agent(self):
        hp = toolchain.host_paths()
        for key in ("profile", "spec"):
            self.assertTrue(hp[key].is_file())
            self.assertTrue(hp[key].resolve().is_relative_to(paths.PROFILES_DIR))
        for name in ("compiler-config.json", "operator-spec.json"):
            self.assertTrue((toolchain.BOOT_CONFIG_DIR / name).is_file())
        self.assertEqual(toolchain.BOOT_CONFIG_DIR, paths.PROFILES_DIR / "qwen48")

    def test_location_overrides(self):
        code = ("from pt_agent import paths; print(paths.RESULTS_DIR, paths.HECATE_BUILD, paths.DACAPO_ROOT, sep='|')")
        scripts = Path(toolchain.__file__).resolve().parents[1]
        env = dict(os.environ, PT_AGENT_RESULTS_ROOT="/x/results", PT_HECATE_BUILD="/x/build", PT_DACAPO_ROOT="/x/dacapo")
        out = subprocess.run([sys.executable, "-c", code], cwd=scripts, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "/x/results|/x/build|/x/dacapo")

    def test_nixrun_forwards_overrides(self):
        with patch.dict(os.environ, {"PT_HECATE_BUILD": "/x/b y", "PT_AGENT_RESULTS_ROOT": ""}, clear=False):
            text = nixrun.forwarded_env()
        self.assertIn("PT_HECATE_BUILD='/x/b y' ", text)
        self.assertNotIn("PT_AGENT_RESULTS_ROOT", text)


if __name__ == "__main__":
    unittest.main()
