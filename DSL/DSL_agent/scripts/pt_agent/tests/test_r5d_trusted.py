import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import toolchain
from pt_agent.backends import EmitBackend, run_program
from pt_agent.tests.test_r2_backends import SLOTS, p1
from pt_agent.tests._artifacts import COMPILER, require


toolchain.setup_env()


@require(COMPILER)
class TrustedRunnerTests(unittest.TestCase):
    def emit_p1(self, directory):
        backend = EmitBackend(SLOTS)
        run_program(backend, p1, 2)
        backend.write(directory)

    def test_p1_trusted_trace_compile(self):
        directory = toolchain.new_run_dir("r5d-p1")
        self.emit_p1(directory)
        traced = toolchain.trace(directory, trusted=True)
        print("R5d P1 trace=" + json.dumps(dict(run_dir=str(directory), **traced)), flush=True)
        self.assertEqual(traced["exit"], 0, traced["log_tail"])
        self.assertTrue(traced["mlir"])
        self.assertGreater(traced["mlir_bytes"], 0)
        self.assertGreater(traced["peak_rss_mib"], 0)
        compiled = toolchain.compile(directory, "eva", trusted=True)
        print("R5d P1 compile=" + json.dumps(compiled), flush=True)
        self.assertEqual(compiled["exit"], 0, compiled["log_tail"])
        self.assertTrue(compiled["plan"])
        self.assertTrue((directory / "lowered._hecate_golden.runtime-plan.json").is_file())
        self.assertGreater(compiled["peak_rss_mib"], 0)

    def test_retarget_rejects_second_call(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.emit_p1(directory)
            toolchain.retarget(directory)
            text = (directory / "candidate_trace.py").read_text()
            self.assertNotIn("/frontend", text)
            self.assertNotIn("/out/", text)
            self.assertTrue((directory / "candidate_trace.sandbox.py").is_file())
            with self.assertRaises(ValueError):
                toolchain.retarget(directory)

    def test_to_host_argv(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            argv = toolchain.to_host_argv(toolchain.compile_argv("dacapo-depth-dp", "qwen48", directory), directory)
            for path in ("/out/", "/frontend", "/profile.json", "/operator-spec.json"):
                self.assertFalse(any(path in arg for arg in argv))
            self.assertTrue(argv[argv.index("-o")+1].startswith(str(directory)))
            self.assertEqual(argv[0], str(toolchain.host_paths()["hecate_opt"]))

    def test_trusted_run_requires_free_disk(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("pt_agent.toolchain.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
                with self.assertRaises(RuntimeError):
                    toolchain.trusted_run(directory, ["/usr/bin/true"], directory / "small-disk.log")
