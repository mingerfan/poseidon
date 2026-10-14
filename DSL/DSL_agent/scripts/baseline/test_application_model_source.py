"""Source intake lifecycle tests. Injected backend evidence is not FHE evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from application_component import Application, Stopped
from application_policy import DEFAULT_BUDGET
from benchmark_graph import digest
from benchmark_suite import Builder
from application_component import Application as SourceApplication
from application_model_source import FORMAT
from test_application_component import Backend as BaseBackend, Provider

class Backend(BaseBackend):
    def export(self, result, destination, level, *, model_source=None):
        super().export(result,destination,level)
        if model_source is not None:
            (destination/"model-source-test.json").write_text(json.dumps(model_source))



class SourceSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = Backend()
        self.app = SourceApplication(Path(self.tmp.name)/"tasks", self.backend)
        b = Builder([(2,)])
        self.model = b.finish(b.node("square", ["input0"]))
        self.source = dict(format=FORMAT, kind="graph", model=self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, **kw):
        return self.app.submit_source(copy.deepcopy(self.source), **kw)

    def test_submit_has_one_queued_id_and_never_generates(self):
        task = self.submit()
        self.assertEqual(list(self.app.store.iterdir()), [self.app._dir(task)])
        result = self.app.status(task)
        self.assertEqual((result["status"], result["phase"]), ("queued", "intake_queued"))
        self.assertEqual((result["generations"], result["http_attempts"]), (0, 0))
        self.assertEqual(self.backend.calls, 0)
        self.assertFalse((self.app._dir(task)/"task.json").exists())

    def test_original_json_request_hash_preserved(self):
        original = Application(Path(self.tmp.name)/"legacy", Backend())
        legacy = original.submit(self.model)
        task = self.submit()
        self.assertEqual(self.app.prepare_source(task)["phase"], "queued")
        expected = original._checked_task_metadata(legacy)
        actual = self.app._checked_task_metadata(task)
        self.assertEqual(expected["request"], actual["request"])
        self.assertEqual(expected["validation_level"], actual["validation_level"])
        self.assertLess(actual["budget"]["wall_seconds"], DEFAULT_BUDGET["wall_seconds"])
        self.assertEqual(actual["budget"]["generations"], DEFAULT_BUDGET["generations"])

    def test_status_hides_paths_and_model_data(self):
        task = self.submit()
        public = json.dumps(self.app.status(task))
        self.assertNotIn(self.tmp.name, public)
        self.assertNotIn("source.json", public)
        self.assertNotIn(self.model["id"], public)
        self.assertIsNone(self.app.status(task)["acceptance"])

    def test_graph_snapshotted_at_submit(self):
        source = copy.deepcopy(self.source)
        task = self.app.submit_source(source)
        source["model"]["nodes"][0]["op"] = "negate"
        self.app.prepare_source(task)
        actual = self.app._checked_task_metadata(task)["request"]["model"]
        self.assertEqual(actual, self.model)

    def test_bad_source_and_requirement_fields_are_terminal(self):
        cases = [None, [], {}, dict(format=FORMAT,kind="onnx",model={}),
                 dict(self.source,command="echo secret")]
        for source in cases:
            with self.subTest(source=type(source).__name__):
                task = self.app.submit_source(source)
                self.assertEqual(self.app.status(task)["status"], "unsupported")
        for requirements in [dict(backend="seal_cpu"), dict(validation_level="maybe"),
                             dict(limits={}), dict(options=[]), dict(unknown=True)]:
            with self.subTest(requirements=requirements):
                task = self.submit(**requirements)
                self.assertEqual(self.app.status(task)["status"], "unsupported")

    def test_trusted_source_requires_explicit_true(self):
        for trust in [False, 1, None]:
            task = self.app.submit_source(dict(format=FORMAT, kind="trusted_pytorch",
                                              manifest="/not-read.json", trust_local_code=trust))
            self.assertEqual(self.app.status(task)["status"], "unsupported")
            self.assertFalse((self.app._dir(task)/"source").exists())

    def test_snapshot_payload_tamper_before_worker_rejected(self):
        task = self.submit()
        (self.app._dir(task)/"source/model.json").write_text("{}")
        result = self.app.prepare_source(task)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(result["failure"]["layer"], "integrity")
        self.assertEqual(self.backend.calls, 0)

    def test_snapshot_extra_file_rejected(self):
        task = self.submit()
        (self.app._dir(task)/"source/extra.py").write_text("print('must remain inert')")
        self.assertEqual(self.app.prepare_source(task)["status"], "service_error")

    def test_snapshot_symlink_rejected(self):
        task = self.submit()
        p = self.app._dir(task)/"source/model.json"
        raw = p.read_bytes()
        target = Path(self.tmp.name)/"target"
        target.write_bytes(raw)
        p.unlink()
        p.symlink_to(target)
        self.assertEqual(self.app.prepare_source(task)["status"], "service_error")

    def test_source_metadata_tamper_rejected(self):
        task = self.submit()
        p = self.app._dir(task)/"source.json"
        data = json.loads(p.read_text())
        data["budget"]["generations"] = 100
        p.write_text(json.dumps(data))
        self.assertEqual(self.app.prepare_source(task)["status"], "service_error")

    def test_backend_changed_between_submit_and_intake_rejected(self):
        task = self.submit()
        self.backend.version = 2
        result = self.app.prepare_source(task)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(result["failure"]["layer"], "integrity")

    def test_cancel_before_import_prevents_provider_and_capture(self):
        task = self.submit()
        self.app.cancel(task)
        provider = Provider()
        self.assertEqual(self.app.run(task, provider)["status"], "cancelled")
        self.assertEqual(provider.calls, 0)
        self.assertFalse((self.app._dir(task)/"source-import.log").exists())

    def test_model_import_budget_is_not_reset_for_generation(self):
        # Delay only preparation, then block generation. Both phases consume
        # one requested two-second active execution limit.
        task = self.submit(limits=dict(DEFAULT_BUDGET, wall_seconds=2))
        original = self.app._admit
        def delayed(*args, **kwargs):
            time.sleep(.65)
            return original(*args, **kwargs)
        self.app._admit = delayed
        started = time.monotonic()
        result = self.app.run(task, Provider(sleep=3))
        self.assertEqual(result["status"], "timed_out")
        self.assertLess(time.monotonic()-started, 2.2)
        meta = self.app._checked_task_metadata(task)
        self.assertEqual(meta["budget"]["wall_seconds"], 1)

    def test_slow_intake_timeout_does_not_start_generation(self):
        task = self.submit(limits=dict(DEFAULT_BUDGET, wall_seconds=1))
        def blocked(*args, **kwargs):
            time.sleep(3)
        self.app._admit = blocked
        provider = Provider()
        started = time.monotonic()
        result = self.app.run(task, provider)
        self.assertEqual(result["status"], "timed_out")
        self.assertLess(time.monotonic()-started, 1.6)
        self.assertEqual(provider.calls, 0)

    def test_cancel_during_preparation_from_another_process(self):
        import multiprocessing
        task = self.submit()
        original = self.app._admit
        def delayed(*args, **kwargs):
            time.sleep(5)
            return original(*args, **kwargs)
        self.app._admit = delayed
        proc = multiprocessing.get_context("fork").Process(
            target=self.app.run, args=(task,Provider()))
        proc.start()
        try:
            end = time.monotonic()+3
            while self.app.status(task)["phase"] != "model_import" and time.monotonic()<end:
                time.sleep(.02)
            self.app.cancel(task)
            proc.join(2)
            self.assertFalse(proc.is_alive())
            self.assertEqual(self.app.status(task)["status"], "cancelled")
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join()

    def test_recover_interrupted_import_does_not_capture_again(self):
        task = self.submit()
        p = self.app._dir(task)
        state = self.app._load(p/"state.json")
        state.update(status="running", phase="model_import")
        self.app._write(p/"state.json", state)
        self.assertEqual(self.app.recover(task)["status"], "service_error")
        self.assertEqual(self.app.run(task, Provider())["status"], "service_error")

    def test_rule_provider_and_fake_backend_use_same_task(self):
        task = self.submit()
        result = self.app.run(task, Provider())
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["task_id"], task)
        self.assertEqual(result["package"]["task_id"], task)
        self.assertEqual(result["generation"]["provider"], "scripted_replay")
        self.assertFalse(result["generation"]["agent_generated"])
        self.assertTrue((self.app._dir(task)/"source.json").exists())
        self.assertTrue((self.app._dir(task)/"task.json").exists())
        self.assertEqual(len(list(self.app.store.iterdir())), 1)

    def test_repeat_same_source_reuses_without_provider(self):
        task = self.submit()
        self.app.run(task, Provider())
        again = self.submit()
        provider = Provider()
        result = self.app.run(again, provider)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["package"]["reused"])
        self.assertEqual(provider.calls, 0)

    def test_changed_task_budget_reuses_identical_source(self):
        first=self.submit(limits=dict(DEFAULT_BUDGET,wall_seconds=120))
        self.assertEqual(self.app.run(first,Provider())["status"],"succeeded")
        second=self.submit(limits=dict(DEFAULT_BUDGET,wall_seconds=240))
        provider=Provider()
        result=self.app.run(second,provider)
        self.assertEqual(result["status"],"succeeded")
        self.assertTrue(result["package"]["reused"])
        self.assertEqual(provider.calls,0)
        left=self.app._checked_task_metadata(first)["model_source"]
        right=self.app._checked_task_metadata(second)["model_source"]
        self.assertNotEqual(left["snapshot_binding"],right["snapshot_binding"])
        self.assertEqual(left["files"],right["files"])

    def test_changed_source_binding_does_not_reuse(self):
        task = self.submit()
        self.app.run(task, Provider())
        source = copy.deepcopy(self.source)
        source["model"]["id"] += "_different"
        again = self.app.submit_source(source)
        provider = Provider()
        result = self.app.run(again, provider)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(provider.calls, 1)
        self.assertFalse(result["package"]["reused"])

    def test_worker_environment_error_is_service_failure(self):
        from application_model_source import SourceEnvironmentError
        source = copy.deepcopy(self.source)
        source["model"]["format"] = "poseidon-operator-graph-v1"
        task = self.app.submit_source(source)
        def unavailable(*args):
            raise SourceEnvironmentError("PRIVATE /host/environment")
        self.app._capture = unavailable
        result = self.app.prepare_source(task)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(result["failure"]["layer"], "environment")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("/host/environment", json.dumps(result))

    def test_controller_drift_after_preparation_stops_generation(self):
        import application_model_source as module
        task = self.submit()
        self.app.prepare_source(task)
        provider = Provider()
        with patch.object(module, "own_identity", return_value={"format":"changed"}):
            result = self.app.run(task, provider)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(provider.calls, 0)

    def test_resource_profile_is_intake_requirement_not_generation_option(self):
        task = self.submit(resource_profile="legacy-v1")
        self.assertEqual(self.app.prepare_source(task)["phase"], "queued")
        meta = self.app._checked_task_metadata(task)
        self.assertNotIn("resource_profile", meta["context"])
        rejected = self.submit(options={"resource_profile":"legacy-v1"})
        self.assertEqual(self.app.status(rejected)["status"], "unsupported")

    def test_invalid_generation_option_rejected_before_capture(self):
        task = self.submit(options={"user_shell_command":"PRIVATE"})
        self.assertEqual(self.app.status(task)["status"], "unsupported")
        self.assertFalse((self.app._dir(task)/"source").exists())

    def test_core_graph_resource_profile_mismatch_rejected(self):
        task = self.submit(resource_profile="general-tensors-v1")
        self.assertEqual(self.app.status(task)["status"], "unsupported")

    def test_optional_none_budget_keeps_old_default(self):
        task = self.submit(limits=None,options=None)
        self.assertEqual(self.app.prepare_source(task)["phase"], "queued")
        meta = self.app._checked_task_metadata(task)
        self.assertEqual(meta["budget"]["generations"], DEFAULT_BUDGET["generations"])

    def test_delivery_binds_source_record_and_rejects_source_tamper(self):
        task = self.submit()
        result = self.app.run(task,Provider())
        self.assertEqual(result["status"],"succeeded")
        meta = self.app._checked_task_metadata(task)
        delivered = json.loads((self.app._dir(task)/"program/model-source-test.json").read_text())
        self.assertEqual(delivered,meta["model_source"])
        from model_source_provenance import verify
        verify(delivered,meta["request"])
        meta["model_source"]["snapshot_binding"]="0"*64
        meta["model_source"]["binding"]=digest({k:v for k,v in meta["model_source"].items() if k!="binding"})
        meta["binding"]=digest({k:v for k,v in meta.items() if k!="binding"})
        self.app._write(self.app._dir(task)/"task.json",meta)
        self.assertEqual(self.app.result(task)["status"],"service_error")

    def test_source_receipt_is_not_sent_to_generation_model(self):
        task = self.submit()
        self.app.prepare_source(task)
        meta = self.app._checked_task_metadata(task)
        self.assertIn("model_source",meta)
        self.assertNotIn("model_source",meta["request"])
        provider = Provider()
        result = self.app.run(task,provider)
        self.assertEqual(result["status"],"succeeded")
        self.assertFalse(result["generation"]["agent_generated"])


if __name__ == "__main__":
    unittest.main()
