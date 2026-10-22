"""Stage reuse controls; injected results are not Agent/FHE evidence."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from application_identity import check_manifest,_digest,_backend_projection
from application_component import Application
from application_execution.context import WorkerContext,ContextError,canonical
from application_reuse import package_key,legacy_key,matches,ReuseKeyRejected
from test_application_component import Backend,Provider
from test_application_identity import IdentityTests
from benchmark_suite import Builder
from hecate_python_env import WORK

def changed_manifest(value,*,generation=False,backend_field=None,stage=None):
    value=copy.deepcopy(value)
    if generation:
        value["generation"]["sources"]["deepseek_provider.py"]=hashlib.sha256(b"changed provider").hexdigest()
    if backend_field:
        value["backend"][backend_field]=hashlib.sha256(b"changed binary").hexdigest()
        for s in ("generation","compilation","execution","validation"):
            value[s]["backend"]=_backend_projection(value["backend"],s)
    if stage:
        name=next(iter(value[stage]["sources"]))
        value[stage]["sources"][name]=hashlib.sha256(b"changed stage").hexdigest()
    for s in ("generation","compilation","execution","validation"):
        value[s]["sha256"]=_digest({k:v for k,v in value[s].items()if k!="sha256"})
    value["manifest_sha256"]=_digest({k:v for k,v in value.items()if k!="manifest_sha256"})
    return check_manifest(value)

class StagedBackend(Backend):
    def __init__(self,fixture):
        super().__init__();self.fixture=fixture
    def identity_manifest(self):return self.fixture.manifest()

class CacheControls(unittest.TestCase):
    def setUp(self):
        self.fixture=IdentityTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.backend=StagedBackend(self.fixture);self.app=Application(self.tmp.name,self.backend)
        b=Builder([(2,)]);self.model=b.finish(b.node("square",["input0"]))
    def finish(self):
        task=self.app.submit(self.model);provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"succeeded")
        return task,result
    def change_provider(self):
        (self.fixture.base/"deepseek_provider.py").write_text("VALUE=17\n")
    def test_provider_change_reuses_verified_package_without_generation(self):
        first,result=self.finish();self.change_provider()
        second=self.app.submit(self.model);unused=Provider();answer=self.app.run(second,unused)
        self.assertEqual(answer["status"],"succeeded");self.assertTrue(answer["package"]["reused"])
        self.assertEqual(answer["package"]["task_id"],first)
        self.assertEqual(unused.calls,0);self.assertEqual(self.backend.calls,1)
        self.assertEqual(answer["http_attempts"],0)
    def test_provider_change_keeps_original_completed_task_deliverable(self):
        first,_=self.finish();self.change_provider()
        self.assertEqual(self.app.result(first)["status"],"succeeded")
    def test_queued_generation_does_not_silently_adopt_new_provider(self):
        task=self.app.submit(self.model);self.change_provider();provider=Provider()
        result=self.app.run(task,provider)
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["generations"],0);self.assertEqual(provider.calls,0)
    def test_compiler_change_requires_a_new_qualification(self):
        self.finish();self.fixture.tools["compiler_sha256"]=hashlib.sha256(b"new compiler").hexdigest()
        task=self.app.submit(self.model);provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"succeeded");self.assertFalse(result["package"]["reused"])
        self.assertEqual(self.backend.calls,2);self.assertEqual(provider.calls,1)
    def test_weights_and_model_change_prevents_cache_reuse(self):
        self.finish()
        changed=copy.deepcopy(self.model);changed["nodes"][0]["op"]="negate"
        task=self.app.submit(changed);provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"succeeded");self.assertFalse(result["package"]["reused"])
        self.assertEqual(self.backend.calls,2)
    def test_compiled_and_numerical_levels_remain_separate(self):
        self.finish()
        task=self.app.submit(self.model,validation_level="compiled")
        result=self.app.run(task,Provider())
        self.assertEqual(result["status"],"succeeded");self.assertFalse(result["package"]["reused"])
        self.assertFalse(result["acceptance"]["numerically_validated"])
    def test_matching_corrupt_cache_key_rejects_before_generation(self):
        first,_=self.finish()
        path=self.app._dir(first)/"state.json";state=self.app._load(path);state["reuse_key"]="0"*64
        self.app._write(path,state);self.change_provider()
        task=self.app.submit(self.model);provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["code"],"cached_task_binding_rejected")
        self.assertEqual(provider.calls,0)
    def test_stored_task_and_cache_bytes_stay_immutable_during_reuse(self):
        first,_=self.finish()
        path=self.app._dir(first)
        before={p.name:p.read_bytes()for p in path.iterdir()if p.is_file()}
        self.change_provider()
        task=self.app.submit(self.model);self.app.run(task,Provider())
        self.assertEqual(before,{p.name:p.read_bytes()for p in path.iterdir()if p.is_file()})
    def test_invalid_stage_manifest_is_not_a_reuse_key(self):
        self.finish();meta=self.app._checked_task_metadata(next(self.app.store.glob("*/task.json")).parent.name)
        meta["identity"]["execution"]["sha256"]="0"*64
        with self.assertRaises(ValueError):package_key(meta)
    def test_legacy_key_representation_is_unchanged(self):
        backend=Backend();app=Application(Path(self.tmp.name)/"legacy",backend)
        task=app.submit(self.model);meta=app._checked_task_metadata(task)
        self.assertEqual(legacy_key(meta),app.status(task)["reuse_key"]if "reuse_key"in app.status(task)else app._load(app._dir(task)/"state.json")["reuse_key"])
        self.assertEqual(package_key(meta),legacy_key(meta))
    def test_generation_and_operational_budgets_do_not_change_package_key(self):
        task=self.app.submit(self.model);meta=self.app._checked_task_metadata(task)
        other=copy.deepcopy(meta);other["identity"]=changed_manifest(other["identity"],generation=True)
        other["budget"]["wall_seconds"]=other["budget"]["wall_seconds"]+1
        self.assertEqual(package_key(meta),package_key(other))
        self.assertNotEqual(legacy_key(meta),legacy_key(other))
    def test_request_content_and_validation_are_bound(self):
        task=self.app.submit(self.model);meta=self.app._checked_task_metadata(task)
        for k in ("request","validation_level"):
            other=copy.deepcopy(meta)
            if k=="request":other["request"]["layout"]["output_ciphertexts"]=2
            else:other[k]="compiled"
            self.assertNotEqual(package_key(meta),package_key(other))

class StagedContextControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=WORK/"results",prefix="staged-context-unit-")
        cls.root=Path(cls.tmp.name)
        cls.context=WorkerContext.capture(cls.root/"tasks")
        from application_backend import PoseidonBackend
        cls.manifest=PoseidonBackend().identity_manifest()
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_new_stores_use_versioned_sdk_policy(self):
        value=self.context.record()
        self.assertEqual(value["format"],"poseidon-worker-context-v2")
        self.assertEqual(value["identity_policy"],"staged-sdk-v1")
        self.assertEqual(set(value["stage_identity_sha256"]),{"compilation","execution","validation"})
    def test_sdk_source_authority_excludes_generation(self):
        value=self.context.record()
        for name in ("deepseek_provider.py","agent_credentials.py","application_component.py"):
            self.assertNotIn(name,value["source_authority"]["sources"])
        for name in ("application_execution/context.py","application_execution/backend.py","application_execution/context_identity.py"):
            self.assertIn(name,value["source_authority"]["sources"])
    def test_generation_only_change_keeps_worker_current(self):
        after=changed_manifest(self.manifest,generation=True)
        with patch("application_backend.PoseidonBackend.identity_manifest",return_value=after):
            self.assertEqual(self.context.check_current(),self.context)
    def test_compilation_execution_and_validation_changes_reject(self):
        for s in ("compilation","execution","validation"):
            after=changed_manifest(self.manifest,stage=s)
            with patch("application_backend.PoseidonBackend.identity_manifest",return_value=after):
                with self.assertRaises(ContextError):self.context.check_current()
    def test_binary_and_parameter_identity_changes_reject(self):
        for k in ("compiler_sha256","executor_sha256","poseidon_library_sha256","operator_spec_sha256"):
            after=changed_manifest(self.manifest,backend_field=k)
            with patch("application_backend.PoseidonBackend.identity_manifest",return_value=after):
                with self.assertRaises(ContextError):self.context.check_current()
    def test_legacy_context_keeps_full_generation_binding(self):
        context=WorkerContext.capture(self.root/"legacy",identity_policy="legacy-v1")
        self.assertEqual(context.record()["format"],"poseidon-worker-context-v1")
        after=changed_manifest(self.manifest,generation=True)
        with patch("application_backend.PoseidonBackend.identity_manifest",return_value=after):
            with self.assertRaises(ContextError):context.check_current()
    def test_existing_legacy_receipt_is_not_promoted_or_overwritten(self):
        store=self.root/"old-receipt"
        old=WorkerContext.capture(store,identity_policy="legacy-v1");old.persist()
        path=store/"worker-context.json";raw=path.read_bytes()
        automatic=WorkerContext.capture(store)
        self.assertEqual(automatic,old);automatic.persist()
        explicit=WorkerContext.capture(store,identity_policy="staged-sdk-v1")
        with self.assertRaises(ContextError):explicit.persist()
        self.assertEqual(path.read_bytes(),raw)
    def test_staged_receipt_roundtrip_and_conflict_check(self):
        context=WorkerContext.capture(self.root/"roundtrip");context.persist()
        self.assertEqual(WorkerContext.capture(context.task_store),context)
        path=context.task_store/"worker-context.json"
        self.assertEqual(path.stat().st_mode&0o777,0o600)
    def test_unknown_policy_and_generation_hash_are_rejected(self):
        for name in ("unrestricted","all",True,{}):
            with self.assertRaises((ContextError,TypeError)):WorkerContext.capture(self.root/"invalid",identity_policy=name)
        value=json.loads(self.context.payload);value["stage_identity_sha256"]["generation"]="0"*64
        with self.assertRaises(ContextError):WorkerContext(canonical(value))
    def test_resigned_payload_cannot_hide_sdk_drift(self):
        value=json.loads(self.context.payload);value["paths"]["work_root"]="/wrong"
        context=WorkerContext(canonical(value))
        with self.assertRaises(ContextError):context.check_current()
