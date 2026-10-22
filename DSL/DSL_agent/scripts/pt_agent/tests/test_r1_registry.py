import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import config, registry


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(os.environ, {"PT_RESULTS": self.directory.name})
        env.start()
        self.addCleanup(env.stop)

    def row(self, number):
        return dict(run_id=f"test-{number}", round="R1", model="registry", mode="unit",
                    git_rev="test-rev", metrics={"calls": 0, "note": "本地"},
                    artifacts_dir=f"test-{number}", ts="2026-10-20T00:00:00Z")

    def test_two_rows_round_trip(self):
        self.assertEqual(registry.load(), [])
        rows = [self.row(1), self.row(2)]
        for row in rows:
            registry.append(row)
        self.assertEqual(registry.load(), rows)
        self.assertEqual(len((Path(self.directory.name) / "registry.jsonl").read_text().splitlines()), 2)

    def test_invalid_row_preserves_existing_records(self):
        row = self.row(1)
        registry.append(row)
        with self.assertRaises(ValueError):
            registry.append({"run_id": "incomplete"})
        self.assertEqual(registry.load(), [row])

    def test_configuration_and_explicit_results(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PT_RESULTS"):
                config.load()
        with patch.dict(os.environ, {"PT_HECATE_OPT": "/opt/hecate-opt",
                                     "PT_OPERATOR_SPEC": "/opt/operators.json"}):
            loaded = config.load()
        self.assertEqual(loaded.results, Path(self.directory.name))
        self.assertEqual(loaded.hecate_opt, Path("/opt/hecate-opt"))
        self.assertEqual(loaded.operator_spec, Path("/opt/operators.json"))


    def test_run_id_format(self):
        self.assertRegex(registry.new_run_id("R1", "mlp", "unit"),
                         r"^R1-mlp-unit-\d{8}T\d{6}-[0-9a-f]{4}$")

    def test_git_revision_nonempty(self):
        rev = registry.git_rev()
        self.assertIsInstance(rev, str)
        self.assertTrue(rev)


if __name__ == "__main__":
    unittest.main()
