"""Offline build guards and opt-in audit of actual GPU primitive evidence."""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest

from build_poseidon_gpu import CUDA, GMP, native_environment


class NativeGpuEnvironmentTests(unittest.TestCase):
    def test_nix_wrappers_cannot_leak_into_native_linking(self):
        original = dict(PATH="/nix/store/wrapped-bin", NIX_LDFLAGS="-rpath /nix/store/glibc/lib",
                        NIX_CFLAGS_COMPILE="-isystem /nix/store/include", COMPILER_PATH="/nix/store/ld",
                        CPATH="/nix/store/include", CMAKE_PREFIX_PATH="/nix/store/zlib",
                        LD_PRELOAD="/nix/store/lib.so", LD_LIBRARY_PATH="/nix/store/lib")
        result = native_environment(original)
        self.assertEqual(result["PATH"], str(CUDA / "bin")+":/usr/bin:/bin")
        self.assertEqual(result["LD_LIBRARY_PATH"], str(GMP / "lib")+":"+str(CUDA / "lib"))
        for name in original:
            if name not in ("PATH", "LD_LIBRARY_PATH"):
                self.assertNotIn(name, result)
        self.assertIn("NIX_LDFLAGS", original)


@unittest.skipUnless(os.environ.get("POSEIDON_GPU_MODSWITCH_RESULTS"), "Requires real Poseidon GPU primitive evidence")
class GpuModswitchEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(os.environ["POSEIDON_GPU_MODSWITCH_RESULTS"])
        cls.report = json.loads((cls.folder / "report.json").read_text())

    def test_native_execution_and_frozen_sources_logs(self):
        data = self.report
        self.assertEqual(data["status"], "passed")
        self.assertTrue(data["gpu_executed"])
        self.assertTrue(data["fhe_executed"])
        self.assertFalse(data["hevm_executed"])
        self.assertEqual(data["agent_calls"], 0)
        for rel, value in data["source_hashes"].items():
            path = (self.folder / rel).resolve()
            self.assertTrue(path.is_relative_to(self.folder.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), value)
        self.assertEqual(len(data["checks"]), 4)
        for name, check in data["checks"].items():
            self.assertEqual(check["exit_code"], 0)
            for suffix in ("stdout", "stderr"):
                raw = (self.folder / (name+"."+suffix)).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), check[suffix+"_sha256"])
            if name != "primitive":
                raw = (self.folder / (name+".stdout")).read_bytes()
                self.assertNotIn(b"/nix/store/", raw)
                self.assertNotIn(b"not found", raw)
        self.assertEqual(json.loads((self.folder / "primitive.stdout").read_text()), data["result"])

    def test_tc128_all_valid_q_prefixes_aliases_negatives_and_decode(self):
        data = self.report["result"]
        self.assertEqual(data["backend"], "poseidon_gpu_primitives")
        self.assertEqual(data["security_request"], "tc128")
        self.assertTrue(data["security_over_budget_rejected"])
        self.assertEqual(data["gpu_device_ordinal"], 0)
        self.assertEqual((data["gpu_compute_major"], data["gpu_compute_minor"]), (8, 9))
        self.assertEqual(data["degree"], 16384)
        self.assertEqual(data["slots"], 8192)
        self.assertEqual(data["scale"], 2**40)
        self.assertEqual([q.bit_length() for q in data["q_moduli"]], [30]*12)
        self.assertEqual([p.bit_length() for p in data["p_moduli"]], [30]*2)
        self.assertEqual([row["q_count"] for row in data["cases"]], list(range(12, 1, -1)))
        self.assertEqual(data["atol"], 1e-5)
        self.assertEqual(data["rtol"], 1e-4)
        for row in data["cases"]:
            for field in ("cpu_gpu_exact", "alias_exact", "sequential_exact"):
                self.assertTrue(row[field])
            self.assertTrue(math.isfinite(row["max_absolute_error"]))
            self.assertLessEqual(row["max_absolute_error"], 1e-5+1e-4*0.5)
            self.assertGreaterEqual(row["mae"], 0)
            self.assertLessEqual(row["mae"], row["max_absolute_error"])
        self.assertTrue(data["source_unchanged"])
        self.assertEqual(data["negative_checks"], 6)


if __name__ == "__main__":
    unittest.main()
