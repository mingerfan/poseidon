"""Developer boundary tests; no backend execution or provider calls."""
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT/"scripts/baseline"))
spec = importlib.util.spec_from_file_location("source_guard_tool", Path(__file__).with_name("check_pytorch_sources.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

from application_execution.backend import ContextBackend
from application_execution.context import WorkerContext, FORMAT, canonical, ContextError
from application_backend import PoseidonBackend

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {"format": FORMAT, "paths": {"task_store": "/tmp/developer-source-guard"}}
        self.worker = WorkerContext(canonical(self.snapshot))
        self.frozen_identity = {"backend": "poseidon_cpu", "digest": "original"}
        self.sources = {"model_capture.py": "original"}
        self.snap = patch("application_execution.context._snapshot", return_value=self.snapshot).start()
        self.read = patch("semantic_benchmark_execution.runtime_sources", return_value=self.sources).start()
        self.identity = patch.object(PoseidonBackend, "identity", return_value=self.frozen_identity).start()
        self.addCleanup(patch.stopall)
        self.backend = ContextBackend(self.worker)
        self.snap.reset_mock()
        self.identity.reset_mock()

    def check(self, backend=None, worker=None):
        tool.check_source_boundary(self.worker if worker is None else worker,
                                   self.backend if backend is None else backend,
                                   dict(self.sources), dict(self.frozen_identity))

    def test_current_worker_checked_once(self):
        self.check()
        self.assertEqual(self.snap.call_count, 1)
        self.assertEqual(self.identity.call_count, 1)
        self.assertEqual(self.read.call_count, 1)

    def test_worker_environment_drift_stops_before_identity(self):
        self.snap.return_value = dict(format=FORMAT, environment="changed")
        with self.assertRaisesRegex(ContextError, "context_worker_changed"):
            self.check()
        self.identity.assert_not_called()

    def test_current_backend_identity_must_match(self):
        self.identity.return_value = {"backend": "poseidon_cpu", "digest": "changed"}
        with self.assertRaisesRegex(ValueError, "source/dependency drift"):
            self.check()

    def test_changed_source_bytes_rejected_without_cached_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"capture.py"
            path.write_bytes(b"original")
            hashes = lambda: {"capture.py": hashlib.sha256(path.read_bytes()).hexdigest()}
            frozen = hashes()
            with patch("semantic_benchmark_execution.runtime_sources", side_effect=hashes):
                tool.check_source_boundary(self.worker, self.backend, frozen, self.frozen_identity)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source/dependency drift"):
                    tool.check_source_boundary(self.worker, self.backend, frozen, self.frozen_identity)
        self.assertEqual(self.snap.call_count, 2)

    def test_unchecked_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "context backend mismatch"):
            self.check(backend=PoseidonBackend())
        self.identity.assert_not_called()

    def test_other_worker_rejected(self):
        other = WorkerContext(canonical(self.snapshot))
        with self.assertRaisesRegex(ValueError, "context backend mismatch"):
            self.check(worker=other)
        self.identity.assert_not_called()

if __name__=="__main__": unittest.main(verbosity=2)
