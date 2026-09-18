"""Offline checks for evidence comparison; fake arrays only, no provider calls."""
from pathlib import Path
import tempfile
import unittest
import zipfile
import copy

from audit_agent_batch import npz_equal, validate_cohort_labels
from run_agent_batch import batch_plan


class BatchAuditTests(unittest.TestCase):
    def test_expanded_labels_hashes_and_legacy_metadata_are_checked(self):
        rows, manifest = batch_plan(True)
        report = dict(cases=rows, benchmark={k:v for k, v in manifest.items() if k != "entries"})
        baseline = copy.deepcopy(report)
        validate_cohort_labels(report, baseline)
        # A retry subset is still compared to the exact same complete cohort.
        subset = copy.deepcopy(report)
        subset["cases"] = subset["cases"][-2:]
        validate_cohort_labels(subset, baseline)
        for field in ("version", "descriptor_set_sha256", "families", "new_family_definitions"):
            changed = copy.deepcopy(report)
            changed["benchmark"][field] = "wrong"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_cohort_labels(changed, baseline)
        changed = copy.deepcopy(report)
        changed["cases"][-1]["benchmark_family"] = "linear"
        with self.assertRaisesRegex(ValueError, "model-family mismatch"):
            validate_cohort_labels(changed, baseline)
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_cohort_labels(dict(cases=rows), baseline)
        legacy, _ = batch_plan()
        validate_cohort_labels(dict(cases=legacy), dict(cases=legacy))

    def test_compares_array_bytes_not_zip_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"a.npz", Path(directory)/"b.npz"
            for path, year in ((a, 2000), (b, 2020)):
                with zipfile.ZipFile(path, "w") as archive:
                    info = zipfile.ZipInfo("reference.npy", (year, 1, 1, 0, 0, 0))
                    archive.writestr(info, b"fake-array-data")
            self.assertTrue(npz_equal(a, b))

    def test_changed_array_or_extra_member_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"a.npz", Path(directory)/"b.npz"
            with zipfile.ZipFile(a, "w") as archive:
                archive.writestr("reference.npy", b"original")
            with zipfile.ZipFile(b, "w") as archive:
                archive.writestr("reference.npy", b"changed")
            self.assertFalse(npz_equal(a, b))
            with zipfile.ZipFile(b, "w") as archive:
                archive.writestr("reference.npy", b"original")
                archive.writestr("extra.npy", b"unexpected")
            self.assertFalse(npz_equal(a, b))


if __name__ == "__main__":
    unittest.main()
