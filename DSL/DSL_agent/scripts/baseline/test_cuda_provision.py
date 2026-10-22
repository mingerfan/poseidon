"""Offline provisioning guards; never download/install from unit tests."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from provision_cuda import LOCK, safe_member


class CudaProvisionTests(unittest.TestCase):
    def test_exact_minimal_official_plan(self):
        plan = json.loads(LOCK.read_text())
        self.assertEqual(plan["release"], "12.5.1")
        self.assertEqual({c["name"] for c in plan["components"]},
                         {"cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_cuobjdump"})
        self.assertEqual(sum(c["bytes"] for c in plan["components"]), 54212408)
        self.assertLess(sum(c["bytes"] for c in plan["components"]), plan["download_limit_bytes"])
        for component in plan["components"]:
            self.assertTrue(component["url"].startswith("https://developer.download.nvidia.com/compute/cuda/redist/"))
            self.assertEqual(len(component["sha256"]), 64)

    def test_archive_paths_links_devices_and_redirected_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe_member(tarfile.TarInfo("component/include/header.h"), root)
            for path in ("/etc/file", "../file", "component/../../file"):
                with self.assertRaises(ValueError):
                    safe_member(tarfile.TarInfo(path), root)
            for target in ("/etc/file", "../../../file"):
                link = tarfile.TarInfo("component/lib/link")
                link.type, link.linkname = tarfile.SYMTYPE, target
                with self.assertRaises(ValueError):
                    safe_member(link, root)
            for kind in (tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.LNKTYPE):
                entry = tarfile.TarInfo("component/special")
                entry.type = kind
                with self.assertRaises(ValueError):
                    safe_member(entry, root)
            (root / "redirect").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_member(tarfile.TarInfo("redirect/outside"), root)


if __name__ == "__main__":
    unittest.main()
