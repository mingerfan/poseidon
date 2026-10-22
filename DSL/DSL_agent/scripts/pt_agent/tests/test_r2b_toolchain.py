import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pt_agent import toolchain


toolchain.setup_env()


class ToolchainTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_eva(self):
        argv = toolchain.compile_argv("eva", None, self.path)
        self.assertIn("--eva", argv)
        self.assertNotIn("--pars", argv)
        self.assertNotIn("--dacapo", argv)

    def test_pars(self):
        argv = toolchain.compile_argv("pars", None, self.path)
        self.assertEqual(argv.count("--pars"), 1)
        self.assertNotIn("--eva", argv)

    def test_depth_dp(self):
        argv = toolchain.compile_argv("dacapo-depth-dp", None, self.path)
        self.assertIn("--dacapo", argv)
        self.assertIn("--boot-placement=depth-dp", argv)
        self.assertNotIn("--boot-placement=optimized", argv)

    def test_qwen48(self):
        argv = toolchain.compile_argv("pars", "qwen48", self.path)
        for flag in ("--ckks-config=/out/boot-compiler-config.json", "--waterline=43",
                     "--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5", "--runtime-plan-context-id=qwen-bootstrap48"):
            self.assertIn(flag, argv)
        self.assertFalse(any("/profile.json" in arg for arg in argv))
        self.assertNotIn("--waterline=45", argv)
        self.assertEqual(argv[argv.index("--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5")+1], "-o")

    def test_invalid_modes(self):
        with self.assertRaises(ValueError):
            toolchain.compile_argv("eva", "qwen48", self.path)
        with self.assertRaises(ValueError):
            toolchain.compile_argv("unknown", None, self.path)

    def test_argument_lists_are_independent(self):
        first = toolchain.compile_argv("eva", None, self.path)
        second = toolchain.compile_argv("eva", None, self.path)
        self.assertIsNot(first, second)
        first.append("changed")
        self.assertNotIn("changed", second)

    def test_write_inputs(self):
        samples = [[np.arange(8, dtype=np.float64), np.ones(8)] for _ in range(2)]
        toolchain.write_inputs(self.path, samples, 8)
        self.assertEqual(np.asarray(json.loads((self.path / "inputs.json").read_text())).shape, (2, 2, 8))
        for values, period in ((samples, 12), ((samples*9)[:17], 8), ([[np.full(8, np.nan)]], 8)):
            with self.subTest(period=period, samples=len(values)), self.assertRaises(ValueError):
                toolchain.write_inputs(self.path, values, period)

    def test_new_run_dir(self):
        directory = toolchain.new_run_dir("t-x")
        self.addCleanup(shutil.rmtree, directory)
        self.assertTrue(directory.is_relative_to(toolchain.RESULTS_ROOT))
        self.assertTrue((directory / "payload.json").is_file())
        with self.assertRaises(ValueError):
            toolchain.new_run_dir("Bad_Tag")
