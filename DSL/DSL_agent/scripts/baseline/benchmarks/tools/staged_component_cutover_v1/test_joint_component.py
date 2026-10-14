"""Application intake and public-context composition; no generator/backend work."""
import base64,copy,json,unittest
from pathlib import Path
from benchmark_graph import canonical,digest
from candidate_bundle import check_provenance
from application_component import Application
from deepseek_provider import public_request
from test_facade_package import package
from test_intake import captured,PROFILE

MEASUREMENTS=[]
class InertBackend:
    def identity(self):return dict(test_only=True,version=1)
    def qualify(self,*args):raise AssertionError("Backend must remain idle during submit")
    def export(self,*args):raise AssertionError("No artificial program export")

def folder(name,suffix):
    provenance,model,_=package(name)
    result=Path("/out")/("joint-source-"+name+"-"+suffix)
    result.mkdir();(result/"trusted-python").mkdir()
    for key,filename in (("original","original.json"),("decomposition","decomposition.json"),
                         ("reference_report","reference-report.json")):
        (result/filename).write_text(json.dumps(provenance[key],allow_nan=False))
    (result/"model.json").write_text(json.dumps(model,allow_nan=False))
    source=provenance["trusted_pytorch"]
    (result/"trusted-python/manifest.json").write_text(json.dumps(source["manifest"]))
    for filename,text in source["files"].items():(result/"trusted-python"/filename).write_text(text)
    for filename,encoded in source["weight_files"].items():
        (result/"trusted-python"/filename).write_bytes(base64.b64decode(encoded,validate=True))
    return result,provenance,model

class JointApplicationTests(unittest.TestCase):
    def test_v12_import_queues_existing_application_and_keeps_sources_private(self):
        for name in ("linear","norm","split_weight"):
            with self.subTest(case=name):
                source,provenance,model=folder(name,"submit")
                app=Application(Path("/out")/("joint-task-store-"+name),InertBackend())
                task=app.submit_import(source,options=dict(generation_guidance="explicit-v11",chunk_period=4))
                state=app.status(task)
                self.assertEqual(state["status"],"queued")
                self.assertEqual(state["generations"],0)
                self.assertEqual(state["http_attempts"],0)
                self.assertIsNone(state["acceptance"])
                metadata=app._load(app._dir(task)/"task.json")
                self.assertEqual(metadata["model_provenance"],provenance)
                self.assertEqual(metadata["request"]["model"],model)
                self.assertEqual(metadata["context"]["guidance"],"explicit-v11")
                self.assertEqual(metadata["request"]["generation_guidance"]["version"],"explicit-v11")
                self.assertEqual(check_provenance(provenance,model),provenance["decomposition"])
                outbound=public_request(metadata["request"])
                text=json.dumps(outbound)
                for forbidden in ("weight_files","trusted_pytorch","reference_report",
                                  "capture_metadata","capture_emissions","fx-source.txt"):
                    self.assertNotIn(forbidden,text)
                for source_text in provenance["trusted_pytorch"]["files"].values():
                    self.assertNotIn(source_text,text)
                for encoded in provenance["trusted_pytorch"]["weight_files"].values():
                    self.assertNotIn(encoded,text)
                MEASUREMENTS.append(dict(case=name,task_status=state["status"],
                    request_id=metadata["request"]["request_id"],source_format=provenance["trusted_pytorch"]["manifest"]["format"],
                    generation_guidance="explicit-v11",public_context_verified=True,
                    generations=0,http_attempts=0,encrypted_execution=False))

    def test_changed_npy_is_terminal_rejection_before_generation(self):
        source,provenance,_=folder("linear","tamper")
        filename=next(iter(provenance["trusted_pytorch"]["weight_files"]))
        path=source/"trusted-python"/filename
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        app=Application(Path("/out")/"joint-tamper-store",InertBackend())
        task=app.submit_import(source,options=dict(generation_guidance="explicit-v11",chunk_period=4))
        state=app.status(task)
        self.assertEqual(state["status"],"unsupported")
        self.assertEqual(state["phase"],"complete")
        self.assertEqual(state["generations"],0);self.assertEqual(state["http_attempts"],0)
        self.assertNotIn("reference",json.dumps(state["failure"]))

    def test_v12_queued_task_can_be_cancelled_without_generation(self):
        source,_,_=folder("split_weight","cancel")
        app=Application(Path("/out")/"joint-cancel-store",InertBackend())
        task=app.submit_import(source,options=dict(generation_guidance="explicit-v11",chunk_period=4))
        self.assertEqual(app.status(task)["status"],"queued")
        app.cancel(task)
        state=app.status(task)
        self.assertEqual(state["status"],"cancelled")
        self.assertEqual(state["generations"],0);self.assertEqual(state["http_attempts"],0)

    def test_unified_source_api_snapshots_and_admits_checked_v12_capture(self):
        import time
        from application_model_source import FORMAT as SOURCE_FORMAT
        for name in ("linear","norm","split_weight"):
            with self.subTest(case=name):
                app=Application(Path("/out")/("unified-source-store-"+name),InertBackend())
                task=app.submit_source(dict(format=SOURCE_FORMAT,kind="trusted_pytorch",
                    manifest="/fixtures/"+name+"/manifest.json",trust_local_code=True),
                    resource_profile=PROFILE,options=dict(generation_guidance="explicit-v11",chunk_period=4))
                state=app.status(task)
                self.assertEqual(state["status"],"queued");self.assertEqual(state["phase"],"intake_queued")
                path=app._dir(task);snapshot=app._source(path)
                provenance,model,_=package(name)
                app._admit(path,snapshot,app._load(path/"state.json"),model,provenance,time.monotonic()+60)
                metadata=app._checked_task_metadata(task)
                self.assertEqual(app.status(task)["phase"],"queued")
                self.assertEqual(metadata["model_source"]["kind"],"trusted_pytorch")
                self.assertEqual(metadata["model_source"]["files"],snapshot["files"])
                self.assertEqual(metadata["model_provenance"],provenance)
                self.assertEqual(metadata["budget"]["generations"],snapshot["budget"]["generations"])
                self.assertLessEqual(metadata["budget"]["wall_seconds"],60)
                self.assertEqual(app.status(task)["generations"],0)
                self.assertEqual(app.status(task)["http_attempts"],0)
                MEASUREMENTS.append(dict(case=name,task_status="queued",source_api="snapshot_then_admit_checked_capture",
                    capture_supervisor_executed=False,public_context_verified=True,generations=0,http_attempts=0,
                    encrypted_execution=False))

    def test_unified_source_snapshot_tamper_rejected_before_capture_launch(self):
        from application_model_source import FORMAT as SOURCE_FORMAT
        app=Application(Path("/out")/"unified-source-tamper-store",InertBackend())
        task=app.submit_source(dict(format=SOURCE_FORMAT,kind="trusted_pytorch",
            manifest="/fixtures/linear/manifest.json",trust_local_code=True),resource_profile=PROFILE,
            options=dict(generation_guidance="explicit-v11",chunk_period=4))
        self.assertEqual(app.status(task)["phase"],"intake_queued")
        snapshot=app._source(app._dir(task));name=next(n for n in snapshot["files"] if n.endswith(".npy"))
        path=app._dir(task)/"source"/name;raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        state=app.prepare_source(task)
        self.assertEqual(state["status"],"service_error")
        self.assertEqual(state["failure"]["code"],"model_source_integrity_rejected")
        self.assertEqual(state["generations"],0);self.assertEqual(state["http_attempts"],0)
