"""Existing context adapter source forwarding; no compiler or FHE."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application_execution.context import WorkerContext, canonical
from application_execution.backend import ContextBackend, application
from application_model_source import FORMAT
from application_policy import prepare
from hecate_python_env import WORK
from application_execution.tests.test_context import model


class SourceContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results",prefix="source-context-unit-")
        cls.root=Path(cls.temp.name)
        cls.context=WorkerContext.capture(cls.root/"tasks")
        cls.context.persist()
        cls.request=prepare(model(),"poseidon_cpu","numerical",dict(chunk_period=4))[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evidence(self,name):
        p=self.root/name;p.mkdir()
        (p/"request.json").write_bytes(canonical(self.request))
        return dict(evidence=str(p),run_context=self.context.request(self.request,"numerical").record())

    def test_context_adapter_forwards_source_origin(self):
        backend=ContextBackend(self.context)
        result=self.evidence("source-export")
        source={"fixture":"data-only record"}
        destination=self.context.task_store/"source-delivery"/"program"
        with patch("application_backend.PoseidonBackend.export") as call:
            backend.export(result,destination,"numerical",model_source=source)
        call.assert_called_once_with(result,destination,"numerical",
            model_provenance=None,generation_provenance=None,model_source=source)

    def test_legacy_context_export_signature_is_preserved(self):
        backend=ContextBackend(self.context)
        result=self.evidence("legacy-export")
        destination=self.context.task_store/"legacy-delivery"/"program"
        with patch("application_backend.PoseidonBackend.export") as call:
            backend.export(result,destination,"numerical")
        call.assert_called_once_with(result,destination,"numerical",
            model_provenance=None,generation_provenance=None)

    def test_actual_context_application_prepares_source_under_same_task(self):
        app=application(self.context)
        task=app.submit_source(dict(format=FORMAT,kind="graph",model=model()),
            options=dict(chunk_period=4))
        initial=app.status(task)
        self.assertEqual(initial["phase"],"intake_queued")
        prepared=app.prepare_source(task)
        self.assertEqual(prepared["status"],"queued")
        self.assertEqual(prepared["generations"],0)
        self.assertEqual(prepared["http_attempts"],0)
        meta=app._checked_task_metadata(task)
        self.assertEqual(meta["request"],self.request)
        self.assertEqual(meta["model_source"]["kind"],"graph")
        self.assertEqual(meta["model_source"]["resource_profile"],"general-tensors-v1")
        self.assertEqual(app.cancel(task)["status"],"cancelled")


if __name__=="__main__":
    unittest.main()
