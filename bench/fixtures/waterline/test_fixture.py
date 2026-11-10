#!/usr/bin/env python3
"""Standard-library integrity and no-test-results MLP preparation regression."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

FIXTURES = Path(__file__).resolve().parent
SCRIPT = FIXTURES.parents[2] / "scripts/benchmark_waterline.py"
PLAN = "mlp.reuse._hecate_MLP.runtime-plan.json"
BUNDLE = PLAN.removesuffix(".runtime-plan.json") + ".bundle"
# Byte-for-byte snapshots from 5ef32ba8da97742e4942b99dd6d3a0b735e29f8e.
HASHES = {
    "operator-spec.json": "b37c827c139a227d95cc76b7e7c43677281a0e18752f540db8b714e37b79806b",
    "input.json": "4d310b4e23d7ca1483c56c1642cac10808dfb72f725c85a05ec326e66926738f",
    "mock-result.json": "87ec394c7523609fba87607955c534078a9b91fcaa135ff5aa7503e328c984dd",
    "mlp.optimized.ckks.mlir": "2d779252d6ad88205368abe0755f97bb09e7d1bd4654987ed1dfe41dee7f955e",
    "compiler-profile.json": "993f6933e31f9673433e3b2bee2840b87b8e7787ea08af378dd6342ad0e69247",
    PLAN: "ba10d836878f713b1c353cd50d8a7d8d8fd44f4c5417103372978f7d528bf2e6",
    f"{BUNDLE}/manifest.json": "ee04ed6290ac46254d00cf9963731259d542d0378aa59c2bce6259d3b70776d7",
}


class WaterlineFixtureTest(unittest.TestCase):
    def test_frozen_hashes(self):
        for name, expected in HASHES.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), expected)
        bundle = FIXTURES / BUNDLE
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(len(manifest["blobs"]), 100)
        self.assertEqual(sum(p.is_file() for p in bundle.rglob("*")), 101)
        for blob in manifest["blobs"]:
            content = blob["content"].removeprefix("sha256:")
            data = (bundle / "data" / f"{content}.bin").read_bytes()
            with self.subTest(content=content):
                self.assertEqual(len(data), blob["byte_length"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), content)
        plan = json.loads((FIXTURES / PLAN).read_text())
        self.assertEqual(plan["plaintext_bundle"]["manifest_sha256"],
                         "sha256:" + HASHES[f"{BUNDLE}/manifest.json"])

    def test_no_historical_input_paths(self):
        # Also covers the two-device input paths without invoking its compiler.
        for node in ast.walk(ast.parse(SCRIPT.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn("test-results", node.value)

    def test_single_device_prepare_without_results_directory(self):
        generate = runpy.run_path(str(SCRIPT))["generate"]
        with tempfile.TemporaryDirectory(prefix="poseidon-waterline-test-") as temporary:
            root = Path(temporary)
            fixture_link = root / "bench/fixtures/waterline"
            fixture_link.parent.mkdir(parents=True)
            fixture_link.symlink_to(FIXTURES, target_is_directory=True)
            generate.__globals__["ROOT"] = root
            args = types.SimpleNamespace(kind="mlp", devices=1, output=root / "suite",
                                         budgets=[32], backends=["cpu"], capacity_gib=7,
                                         reserve_gib=1, window=512, count=1024)
            process = generate.__globals__["subprocess"]
            # MLP does not use NumPy; only the unrelated pressure branch needs it.
            with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}), \
                 mock.patch.object(process, "check_output", return_value="fixture-regression\n") as revision, \
                 mock.patch.object(process, "run", side_effect=AssertionError("external execution forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()):
                generate(args)
            revision.assert_called_once_with(["git", "rev-parse", "HEAD"], cwd=root, text=True)
            self.assertFalse((root / "test-results").exists())
            suite = json.loads((args.output / "suite.json").read_text())
            self.assertEqual([case["name"] for case in suite["cases"]], ["eager", "wl-32MiB-cpu-w512"])
            for original, generated in (("input.json", "fixture.json"), ("mock-result.json", "mock.json")):
                self.assertEqual((FIXTURES / original).read_bytes(), (args.output / generated).read_bytes())
            for original in (FIXTURES / BUNDLE).rglob("*"):
                if original.is_file():
                    self.assertEqual(original.read_bytes(),
                                     (args.output / "bundle" / original.relative_to(FIXTURES / BUNDLE)).read_bytes())


if __name__ == "__main__":
    unittest.main()
