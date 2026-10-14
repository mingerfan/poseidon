"""Explicit worker/request boundaries; mocked calls here are not FHE evidence."""
import copy,json,os,tempfile,unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch
from application_execution.context import (WorkerContext,RequestContext,ContextError,
    canonical,sha,check_request_record,request_receipt)
from application_execution.backend import ContextBackend,application
from application_policy import prepare
from hecate_python_env import WORK,ROOT

def model(n=4):
    return dict(format="poseidon-model-graph-v2",id="context_square",resource_profile="general-tensors-v1",
        inputs=[dict(name="x",shape=[n])],constants={},
        nodes=[dict(id="s",op="square",inputs=["x"],attrs={},outputs=["v"])],
        outputs=[dict(name="out",value="v")])

class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(dir=WORK/"results",prefix="context-unit-")
        cls.root=Path(cls.directory.name);cls.store=cls.root/"tasks"
        cls.context=WorkerContext.capture(cls.store)
        cls.request=prepare(model(),"poseidon_cpu","numerical",dict(chunk_period=4))[0]
    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_actual_pinned_sdk_and_private_paths(self):
        record=self.context.record()
        self.assertEqual(record["platform"]["configured_platform"],"aarch64-linux" if os.uname().machine=="aarch64" else "x86_64-linux")
        self.assertEqual(record["paths"]["work_root"],str(WORK))
        self.assertEqual(record["paths"]["source_root"],str(ROOT))
        self.assertEqual(record["backend_identity"]["tools"]["format"],"poseidon-backend-identity-v2")
        self.assertEqual(self.context.check_current(),self.context)

    def test_record_copies_cannot_mutate_frozen_context(self):
        value=self.context.record();value["paths"]["work_root"]="/different"
        self.assertEqual(self.context.record()["paths"]["work_root"],str(WORK))
        with self.assertRaises(FrozenInstanceError):self.context.payload=b"{}"
        for raw in (b"{}",b'{"format":"poseidon-worker-context-v1","format":"duplicate"}',b"x"*524289):
            with self.assertRaises((ContextError,ValueError)):WorkerContext(raw)

    def test_per_request_shape_and_acceptance_bindings(self):
        first=self.context.request(self.request,"numerical")
        compiled=self.context.request(self.request,"compiled")
        changed=prepare(model(8),"poseidon_cpu","numerical",dict(chunk_period=4))[0]
        alternate=self.context.request(changed)
        self.assertNotEqual(first.binding,compiled.binding)
        self.assertNotEqual(first.binding,alternate.binding)
        self.assertEqual(first.record()["layout_sha256"],sha(canonical(self.request["layout"])))
        self.assertEqual(first.record()["compiler_configuration_sha256"],sha(canonical(self.request["compiler_configuration"])))
        self.assertEqual(first.check(self.context,self.request,"numerical"),first)

    def test_wrong_level_and_mutated_request_rejected(self):
        context=self.context.request(self.request)
        changed=copy.deepcopy(self.request);changed["layout"]["input_slot_period"]=16
        for request,level in ((changed,"numerical"),(self.request,"compiled")):
            with self.assertRaises((ContextError,ValueError)):context.check(self.context,request,level)
        for level in ("auto",None,True):
            with self.assertRaises(ContextError):self.context.request(self.request,level)

    def test_resigned_request_receipt_cannot_change_binding(self):
        receipt=self.context.request(self.request).record()
        for key,value in (("worker_binding","0"*64),("validation_level","compiled"),("request_sha256","0"*64)):
            changed=copy.deepcopy(receipt);changed[key]=value
            changed["binding"]=sha(canonical({k:v for k,v in changed.items() if k!="binding"}))
            with self.assertRaises(ContextError):check_request_record(changed,self.context,self.request,"numerical")
        receipt["binding"]="0"*64
        with self.assertRaises(ContextError):check_request_record(receipt,self.context,self.request,"numerical")

    def test_work_root_change_is_rejected_without_rebinding_globals(self):
        with patch.dict(os.environ,POSEIDON_WORK_ROOT=str(self.root/"new-work")):
            with self.assertRaises(ContextError):self.context.check_current()
        self.context.check_current()

    def test_wrong_platform_and_impure_worker_rejected(self):
        wrong="x86_64-linux" if os.uname().machine=="aarch64" else "aarch64-linux"
        with patch.dict(os.environ,POSEIDON_PLATFORM=wrong):
            with self.assertRaises((ContextError,RuntimeError)):self.context.check_current()
        with patch.dict(os.environ,IN_NIX_SHELL="impure"):
            with self.assertRaises(ContextError):self.context.check_current()

    def test_tool_manifest_change_is_not_hidden_by_cached_paths(self):
        with patch("poseidon_runtime_paths.read_paths",return_value={}):
            with self.assertRaises(ContextError):self.context.check_current()

    def test_library_environment_change_rejected(self):
        with patch.dict(os.environ,LD_LIBRARY_PATH="/different"):
            with self.assertRaises(ContextError):self.context.check_current()

    def test_store_paths_reject_indirection_and_source_checkout(self):
        for store in ("relative-tasks",Path("/"),Path.home(),ROOT/"tasks",self.root/".." /"other"):
            with self.assertRaises(ContextError):WorkerContext.capture(store)
        indirect=self.root/"indirect";indirect.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ContextError):WorkerContext.capture(indirect/"tasks")

    def test_private_configuration_does_not_read_credential_environment(self):
        allowed={"POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","IN_NIX_SHELL","HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH"}
        original=os.environ.get;observed=[]
        def checked(key,*args):
            self.assertIn(key,allowed);observed.append(key);return original(key,*args)
        with patch("application_execution.context.os.environ.get",side_effect=checked):
            self.context.check_current()
        self.assertTrue(observed)
        self.assertEqual(set(self.context.record()["library_environment"]),{"LD_LIBRARY_PATH","HECATE_PYTHON_LIBRARY_PATH"})

    def test_receipts_are_private_idempotent_and_conflict_checked(self):
        self.context.persist();self.context.persist()
        self.assertEqual((self.store/"worker-context.json").stat().st_mode&0o777,0o600)
        record=self.context.request(self.request).record()
        path=request_receipt(self.context,record);self.assertEqual(request_receipt(self.context,record),path)
        self.assertEqual(path.stat().st_mode&0o777,0o600)
        conflict=WorkerContext.capture(self.root/"conflict");conflict.task_store.mkdir()
        (conflict.task_store/"worker-context.json").write_text("{}")
        with self.assertRaises(ContextError):conflict.persist()

    def test_qualification_delegates_exactly_and_records_context(self):
        backend=ContextBackend(self.context);candidate=dict(schema=1,request_id=self.request["request_id"],hecate_source="unit fixture")
        fake=dict(status="failed",failure=dict(code="backend_rejected",layer="compiler"))
        with patch("application_backend.PoseidonBackend.qualify",return_value=copy.deepcopy(fake)) as call:
            result=backend.qualify(self.request,candidate,"numerical")
        call.assert_called_once_with(self.request,candidate,"numerical")
        receipt=result.pop("run_context");self.assertEqual(result,fake)
        check_request_record(receipt,self.context,self.request,"numerical")
        self.assertEqual(receipt["worker_binding"],self.context.binding)

    def test_drift_prevents_any_backend_call(self):
        backend=ContextBackend(self.context)
        with patch.dict(os.environ,IN_NIX_SHELL="impure"),patch("application_backend.PoseidonBackend.qualify") as call:
            with self.assertRaises(ContextError):backend.qualify(self.request,{},"numerical")
            call.assert_not_called()

    def test_export_outside_own_store_rejected_before_delivery(self):
        backend=ContextBackend(self.context)
        with patch("application_backend.PoseidonBackend.export") as call:
            with self.assertRaises(ContextError):backend.export({},self.root/"outside"/"program","numerical")
            call.assert_not_called()

    def test_original_application_keeps_budget_state_and_cancel(self):
        context=WorkerContext.capture(self.root/"application")
        app=application(context)
        task=app.submit(model(),options=dict(chunk_period=4),limits=dict(
            generations=2,http_attempts=4,wall_seconds=120,api_timeout_seconds=60,max_tokens=2048))
        payload=json.loads((context.task_store/task/"task.json").read_text())
        self.assertEqual(payload["request"],self.request)
        self.assertEqual(payload["budget"]["generations"],2)
        self.assertEqual(app.status(task)["status"],"queued")
        self.assertEqual(app.cancel(task)["status"],"cancelled")

    def test_environment_failure_precedes_generation(self):
        context=WorkerContext.capture(self.root/"preflight");app=application(context)
        task=app.submit(model(),options=dict(chunk_period=4))
        class Provider:
            kind="scripted_replay";agent_calls=0
            def generate(self,*_):raise AssertionError("Generation must not run")
        with patch.dict(os.environ,IN_NIX_SHELL="impure"):
            result=app.run(task,Provider())
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["generations"],0)

if __name__=="__main__":unittest.main()
