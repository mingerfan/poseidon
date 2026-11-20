"""Default worker admission and real shared leases; mock backend is not FHE evidence."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_worker as worker
from shared_execution import Batch, Coordinator, LIMITS, KINDS
from component_control import QualificationCancelled


def hold_api(storage, results, limits, output, ready, release):
    coordinator = Coordinator(storage, [results], limits=limits)
    plan = dict(branch="lhy-resnet20", source_binding="b"*64, reserve_bytes=1024,
                wall_seconds=30, stages={kind: int(kind == "api") for kind in KINDS})
    batch, _ = coordinator.register("b-provider", plan, output)
    with batch.stage("api"):
        ready.send(True)
        if not release.poll(8):
            raise RuntimeError("Bounded API-lock fixture wait")
        release.recv()
    batch.finish("succeeded")


class SharedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared-worker-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir(mode=0o700)
        self.coordinator = Coordinator(self.root / "coordination", [self.results],
            limits=dict(LIMITS, evidence_bytes=8*1024**2, batch_bytes=1024**2))
        self.host = dict(backend_identity={"fixture": "identity-only"}, source_closure={"test.py": "a"*64})
        self.raw_environment = worker.environment
        self.environment = patch.object(worker, "environment", return_value=(
            self.coordinator, self.results, "lhy-generalization"))
        self.environment_mock = self.environment.start()
        self.addCleanup(self.environment.stop)
        self.snapshot = patch.object(worker, "snapshot", side_effect=lambda: copy.deepcopy(self.host))
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        self.request = {"request_id": "fixture-only"}
        self.candidate = {"hecate_source": "identity-only unit fixture"}

    def rows(self):
        return self.coordinator.snapshot()[0]["batches"]

    def sole(self):
        rows = list(self.rows().values())
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_default_component_dispatch_uses_shared_adapter(self):
        import component_worker
        job = dict(action="qualify", request=self.request, candidate=self.candidate)
        expected = {"status": "fixture-only"}
        with patch.object(component_worker, "validate_program", return_value={"status": "static_accepted_not_executed"}), \
                patch.object(worker, "qualify", return_value=expected) as shared:
            self.assertIs(component_worker.dispatch(job), expected)
        shared.assert_called_once_with(self.request, self.candidate)

    def test_static_rejection_and_validate_never_enter_shared_native_scope(self):
        import component_worker
        rejected = {"status": "rejected"}
        with patch.object(component_worker, "validate_program", return_value=rejected), \
                patch.object(worker, "qualify") as shared:
            for action in ("validate", "qualify"):
                self.assertIs(component_worker.dispatch(dict(action=action, request={}, candidate={})), rejected)
        shared.assert_not_called()

    def test_actual_shared_lease_and_fresh_artifact_registration_precede_backend_write(self):
        expected = dict(status="passed", encrypted_execution=False, fixture=True)
        def backend(request, candidate, *, artifact_observer):
            row = self.sole()
            self.assertEqual(row["active"]["kind"], "qualify")
            self.assertEqual(row["used"]["qualify"], 1)
            self.assertEqual(request, self.request)
            self.assertEqual(candidate, self.candidate)
            output = self.results / "native-fixture"
            output.mkdir(mode=0o700)
            artifact_observer(output)
            self.assertIn(str(output), [x["path"] for x in self.sole()["artifacts"]])
            (output / "fixture").write_bytes(b"not an encrypted execution")
            return expected
        with patch("component_backend.qualify", side_effect=backend):
            self.assertIs(worker.qualify(self.request, self.candidate), expected)
        row = self.sole()
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["used"]["api"], 0)
        self.assertEqual(row["used"]["qualify"], 1)
        self.assertEqual(row["used"]["encrypted"], 0)
        self.assertEqual(row["plan"]["stages"]["qualify"], 1)
        self.assertTrue(all(v == 0 for k,v in row["plan"]["stages"].items() if k != "qualify"))

    def test_admission_failure_never_calls_backend(self):
        with patch("shared_execution.shutil.disk_usage") as disk, patch("component_backend.qualify") as backend:
            disk.return_value.free = 4*1024**3
            with self.assertRaisesRegex(ValueError, "disk reserve"):
                worker.qualify(self.request, self.candidate)
        backend.assert_not_called()
        self.assertFalse(self.rows())

    def test_backend_failure_keeps_spent_stage_and_uncertainty(self):
        with patch("component_backend.qualify", side_effect=RuntimeError("fixture unknown outcome")):
            with self.assertRaises(RuntimeError):
                worker.qualify(self.request, self.candidate)
        row = self.sole()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["used"]["qualify"], 1)
        self.assertTrue(row["last_stage"]["execution_uncertain"])
        self.assertIsNone(row["active"])

    def test_cancelled_worker_is_terminal_without_refunding_qualification(self):
        with patch("component_backend.qualify", side_effect=QualificationCancelled("fixture cancellation")):
            with self.assertRaises(QualificationCancelled):
                worker.qualify(self.request, self.candidate)
        row = self.sole()
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["used"]["qualify"], 1)
        self.assertTrue(row["last_stage"]["execution_uncertain"])

    def test_source_or_backend_drift_after_return_cannot_finish_successfully(self):
        def backend(*args, **kwargs):
            self.host["backend_identity"] = {"fixture": "changed"}
            return {"status": "passed"}
        with patch("component_backend.qualify", side_effect=backend):
            with self.assertRaisesRegex(ValueError, "source/SDK changed"):
                worker.qualify(self.request, self.candidate)
        row = self.sole()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(json.loads((Path(row["artifacts"][0]["path"]) / "report.json").read_text())["status"], "failed")

    def test_candidate_input_mutation_is_rejected_and_caller_data_preserved(self):
        before = copy.deepcopy(self.candidate)
        def backend(request, candidate, **kwargs):
            candidate["hecate_source"] = "mutated fixture"
            return {"status": "passed"}
        with patch("component_backend.qualify", side_effect=backend):
            with self.assertRaisesRegex(ValueError, "input changed"):
                worker.qualify(self.request, self.candidate)
        self.assertEqual(self.candidate, before)
        self.assertEqual(self.sole()["status"], "failed")

    def test_numerical_rejection_is_retained_as_failed_batch(self):
        result = dict(status="failed", failure=dict(layer="numerical_comparison"))
        with patch("component_backend.qualify", return_value=result):
            self.assertIs(worker.qualify(self.request, self.candidate), result)
        self.assertEqual(self.sole()["status"], "failed")
        self.assertFalse(self.sole()["last_stage"]["execution_uncertain"])

    def test_integration_checkout_uses_the_same_limits_and_shared_ledger(self):
        self.environment_mock.return_value = (self.coordinator, self.results, "lhy-agent-dsl")
        with patch("component_backend.qualify", return_value={"status": "passed"}):
            worker.qualify(self.request, self.candidate)
        self.assertEqual(self.sole()["plan"]["branch"], "lhy-agent-dsl")
        self.assertEqual(self.coordinator.limits["compile_jobs"], 2)
        self.assertEqual(self.coordinator.limits["link_jobs"], 1)

    def test_non_native_worker_never_admits_execution(self):
        with patch.dict(os.environ, IN_NIX_SHELL="impure"), patch("component_backend.qualify") as backend:
            with self.assertRaisesRegex(ValueError, "Pinned shared worker"):
                self.raw_environment()
        backend.assert_not_called()
        self.assertFalse(self.rows())

    def test_component_admission_waits_for_the_other_branch_api_lease(self):
        ctx = mp.get_context("spawn")
        ready, child_ready = ctx.Pipe(duplex=False)
        child_release, release = ctx.Pipe(duplex=False)
        output = self.results / "b-api-fixture"
        output.mkdir(mode=0o700)
        child = ctx.Process(target=hold_api, args=(str(self.coordinator.storage), str(self.results),
            self.coordinator.limits, str(output), child_ready, child_release))
        child.start()
        child_ready.close()
        child_release.close()
        try:
            self.assertTrue(ready.poll(10))
            self.assertTrue(ready.recv())
            original = Batch.stage
            def bounded(batch, kind):
                return original(batch, kind, timeout=.08)
            with patch.object(Batch, "stage", bounded), patch("component_backend.qualify") as backend:
                with self.assertRaises(TimeoutError):
                    worker.qualify(self.request, self.candidate)
            backend.assert_not_called()
            row = next(r for name,r in self.rows().items() if name != "b-provider")
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["used"]["qualify"], 0)
        finally:
            release.send(True)
            child.join(3)
            if child.is_alive():
                child.kill()
                child.join(3)
            ready.close()
            release.close()
        self.assertEqual(child.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
