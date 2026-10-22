from pathlib import Path
import subprocess
import sys
import unittest


EXAMPLES = Path(__file__).parents[1] / "examples"
SCRIPTS = Path(__file__).parents[2]


class CliTests(unittest.TestCase):
    def run_check(self, file, *args):
        return subprocess.run([sys.executable, "-m", "pt_agent.dsl.check", str(EXAMPLES / file), *args],
                              cwd=SCRIPTS, capture_output=True, text=True, timeout=15)

    def test_mlp_success(self):
        result = self.run_check("mlp.ptdsl", "--weights", "mlp")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_parse_failure(self):
        result = self.run_check("invalid/if_expr.ptdsl")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("E_EXPR", result.stdout)

    def test_shape_failure(self):
        result = self.run_check("invalid/tc_shape.ptdsl", "--weights", "mlp")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("E_SHAPE", result.stdout)

    def test_resnet20_print(self):
        result = self.run_check("resnet20.ptdsl", "--weights", "resnet20", "--print")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("= linear(", result.stdout)
