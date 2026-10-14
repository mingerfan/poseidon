"""Request/provenance cache binding regressions with a test backend; no API/FHE."""
import copy,json,tempfile,unittest,time,multiprocessing
from pathlib import Path
from benchmark_graph import canonical,digest
from benchmark_suite import Builder
from application_component import Application
from test_application_component import Backend,Provider

class BoundBackend(Backend):
    package_delay=0
    entered_path=None
    def check_package(self,*args):
        if self.package_delay:
            if self.entered_path is not None:Path(self.entered_path).touch()
            time.sleep(self.package_delay)
        return super().check_package(*args)
    def qualify(self,request,candidate,level):
        result=super().qualify(request,candidate,level)
        result["mock_request_id"]=request["request_id"]
        return result
    def export(self,result,destination,level):
        destination.mkdir()
        (destination/"manifest.json").write_bytes(canonical(dict(test_only=True,level=level,
                                                   request_id=result["mock_request_id"])))

class CacheBindings(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.backend=BoundBackend()
        self.app=Application(self.temp.name,self.backend)
        b=Builder([(2,)]);self.square=b.finish(b.node("square",["input0"]))
        b=Builder([(2,)]);self.negate=b.finish(b.node("negate",["input0"]))
    def tearDown(self):self.temp.cleanup()
    def succeeded(self,model=None):
        task=self.app.submit(model or self.square,validation_level="compiled")
        outcome=self.app.run(task,Provider(["x*x" if model is None else "-x"]))
        self.assertEqual(outcome["status"],"succeeded")
        return task,outcome
    def test_matching_request_reuses_without_generation(self):
        task,_=self.succeeded()
        target=self.app.submit(self.square,validation_level="compiled")
        provider=Provider();outcome=self.app.run(target,provider)
        self.assertEqual(outcome["status"],"succeeded")
        self.assertEqual(outcome["package"]["task_id"],task)
        self.assertTrue(outcome["package"]["reused"])
        self.assertEqual(provider.calls,0)
    def test_different_model_does_not_reuse(self):
        self.succeeded()
        task=self.app.submit(self.negate,validation_level="compiled")
        provider=Provider(["-x"]);outcome=self.app.run(task,provider)
        self.assertEqual(outcome["status"],"succeeded")
        self.assertFalse(outcome["package"]["reused"])
        self.assertEqual(provider.calls,1)
    def test_copied_cache_key_cannot_deliver_different_request(self):
        origin,_=self.succeeded()
        target=self.app.submit(self.negate,validation_level="compiled")
        old=self.app._load(self.app._dir(origin)/"state.json")
        old["reuse_key"]=self.app._load(self.app._dir(target)/"state.json")["reuse_key"]
        self.app._write(self.app._dir(origin)/"state.json",old)
        provider=Provider(["-x"]);outcome=self.app.run(target,provider)
        self.assertEqual(outcome["status"],"service_error")
        self.assertIsNone(outcome["package"])
        self.assertEqual(provider.calls,0)
        self.assertEqual(self.backend.calls,1)
        self.assertEqual(outcome["failure"]["code"],"cached_request_mismatch")
    def test_cached_metadata_binding_is_rechecked(self):
        origin,_=self.succeeded()
        target=self.app.submit(self.square,validation_level="compiled")
        path=self.app._dir(origin)/"task.json"
        meta=self.app._load(path);meta["context"]["owned_fault"]="changed"
        self.app._write(path,meta)
        provider=Provider();outcome=self.app.run(target,provider)
        self.assertEqual(outcome["status"],"service_error")
        self.assertIsNone(outcome["package"])
        self.assertEqual(provider.calls,0)
        self.assertEqual(outcome["failure"]["code"],"cached_task_binding_rejected")
    def test_model_provenance_must_match_even_with_same_request(self):
        origin,_=self.succeeded()
        target=self.app.submit(self.square,validation_level="compiled")
        path=self.app._dir(origin)/"task.json"
        meta=self.app._load(path);meta["model_provenance"]={"owned_fault":"different_source"}
        meta["binding"]=digest({k:v for k,v in meta.items() if k!="binding"})
        self.app._write(path,meta)
        provider=Provider();outcome=self.app.run(target,provider)
        self.assertEqual(outcome["status"],"service_error")
        self.assertIsNone(outcome["package"])
        self.assertEqual(provider.calls,0)
        self.assertEqual(outcome["failure"]["code"],"cached_provenance_mismatch")
    def test_returned_origin_cannot_be_replaced_by_other_model(self):
        source,_=self.succeeded()
        target,result=self.succeeded(self.negate)
        state=self.app._load(self.app._dir(target)/"state.json")
        state["package"]=self.app._load(self.app._dir(source)/"state.json")["package"]
        self.app._write(self.app._dir(target)/"state.json",state)
        outcome=self.app.result(target)
        self.assertEqual(outcome["status"],"service_error")
        self.assertEqual(outcome["failure"]["code"],"cached_request_mismatch")
        self.assertIsNone(outcome["package"])
    def test_delivery_rechecks_current_task_binding(self):
        task,_=self.succeeded()
        path=self.app._dir(task)/"task.json";meta=self.app._load(path)
        meta["context"]["owned_fault"]="different";self.app._write(path,meta)
        outcome=self.app.result(task)
        self.assertEqual(outcome["status"],"service_error")
        self.assertIsNone(outcome["package"])
    def test_validation_level_is_still_bound(self):
        origin,_=self.succeeded()
        requested=self.app._checked_task_metadata(origin)
        with self.assertRaisesRegex(ValueError,"cached_level_mismatch"):
            self.app._package_task_metadata(origin,requested,"numerical")

    def test_cached_package_check_respects_deadline_before_generation(self):
        from application_policy import DEFAULT_BUDGET
        self.succeeded()
        task=self.app.submit(self.square,validation_level="compiled",
                             limits=dict(DEFAULT_BUDGET,wall_seconds=1))
        self.backend.package_delay=4
        provider=Provider();started=time.monotonic()
        outcome=self.app.run(task,provider)
        self.assertEqual(outcome["status"],"timed_out")
        self.assertLess(time.monotonic()-started,2.5)
        self.assertEqual(provider.calls,0)
    def test_cached_package_check_can_be_cancelled_from_another_process(self):
        self.succeeded()
        task=self.app.submit(self.square,validation_level="compiled")
        marker=Path(self.temp.name)/"inside-package-check"
        self.backend.package_delay=5;self.backend.entered_path=str(marker)
        child=multiprocessing.get_context("fork").Process(target=self.app.run,args=(task,Provider()))
        child.start()
        try:
            deadline=time.monotonic()+3
            while not marker.exists() and child.is_alive() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(marker.exists())
            self.app.cancel(task);child.join(2)
            self.assertFalse(child.is_alive())
            self.assertEqual(self.app.status(task)["status"],"cancelled")
            self.assertEqual(self.app.status(task)["generations"],0)
        finally:
            if child.is_alive():child.terminate();child.join()

if __name__=="__main__":unittest.main()
