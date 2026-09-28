"""Application lifecycle tests with injected fakes. Not real-Agent or FHE evidence."""
import copy,json,tempfile,time,unittest
from pathlib import Path
from benchmark_suite import Builder
from benchmark_graph import digest
from application_component import Application,Controller,Stopped
from application_policy import DEFAULT_BUDGET,structured_failure
from component_contract import failure
class Provider:
    kind="scripted_replay";agent_calls=0
    def __init__(self,variants=None,sleep=0):self.calls=0;self.variants=variants or ["x*x"];self.sleep=sleep
    def generate(self,q,h):
        time.sleep(self.sleep)
        v=self.variants[min(self.calls,len(self.variants)-1)];self.calls+=1
        return json.dumps(dict(schema=1,request_id=q["request_id"],
            hecate_source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return ['+v+']\n'))
class Backend:
    def __init__(self,failures=None):self.calls=0;self.version=1;self.failures=failures or []
    def identity(self):return dict(test_only=True,version=self.version)
    def qualify(self,q,c,level):
        self.calls+=1
        if self.calls<=len(self.failures):
            layer=self.failures[self.calls-1]
            return dict(status="failed",failure=failure(layer,"PRIVATE /host/path"),
                        public_feedback=dict(status="failed",layer=layer,category="candidate",diagnostic="Controlled"))
        return dict(status="passed",validation_level=level,compiled_validated=True,
            encrypted_execution=level=="numerical",numerically_validated=level=="numerical",
            compared_values=8,max_absolute_error=0.)
    def export(self,r,d,level):
        d.mkdir();(d/"manifest.json").write_text(json.dumps(dict(test_only=True,level=level)))
    def check_package(self,d,level):
        m=json.loads((d/"manifest.json").read_text())
        if m["level"]!=level:raise ValueError("level")
        return digest(m)
class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.backend=Backend();self.app=Application(self.tmp.name,self.backend)
        b=Builder([(2,)]);self.model=b.finish(b.node("square",["input0"]))
    def tearDown(self):self.tmp.cleanup()
    def submit(self,**kw):return self.app.submit(self.model,**kw)
    def test_success_and_reuse_without_provider(self):
        t=self.submit();p=Provider();s=self.app.run(t,p)
        self.assertEqual(s["status"],"succeeded");self.assertTrue(s["acceptance"]["numerically_validated"])
        t2=self.submit();unused=Provider();s2=self.app.run(t2,unused)
        self.assertTrue(s2["package"]["reused"]);self.assertEqual(unused.calls,0)
        self.assertEqual(self.backend.calls,1)
    def test_compiled_is_explicit_and_separate_cache(self):
        t=self.submit(validation_level="compiled");s=self.app.run(t,Provider())
        self.assertEqual(s["acceptance"]["level"],"compiled");self.assertFalse(s["acceptance"]["numerically_validated"])
        t2=self.submit();p=Provider();self.app.run(t2,p);self.assertEqual(p.calls,1)
    def test_unsupported_input_never_generates(self):
        t=self.submit(backend="GPU");p=Provider();s=self.app.run(t,p)
        self.assertEqual(s["status"],"unsupported");self.assertEqual(p.calls,0)
    def test_bounded_repair(self):
        self.backend.failures=["compiler"]
        s=self.app.run(self.submit(),Provider(["x*x","x*x+zero_ct"]))
        self.assertEqual(s["status"],"succeeded");self.assertEqual(s["generations"],2)
    def test_repeated_candidate_not_recompiled(self):
        self.backend.failures=["compiler"]
        s=self.app.run(self.submit(),Provider())
        self.assertEqual(s["status"],"validation_failed");self.assertEqual(self.backend.calls,1)
        self.assertEqual(s["failure"]["code"],"repeated_candidate")
    def test_repeated_failure_stops_distinct_candidates(self):
        self.backend.failures=["compiler"]*4
        s=self.app.run(self.submit(),Provider(["x*x","x*x+zero_ct","x*x-zero_ct"]))
        self.assertEqual(s["status"],"validation_failed");self.assertEqual(self.backend.calls,2)
        self.assertEqual(s["failure"]["stop_reason"],"repeated_failure")
    def test_generation_budget(self):
        self.backend.failures=["compiler"]
        s=self.app.run(self.submit(limits=dict(DEFAULT_BUDGET,generations=1)),Provider())
        self.assertEqual(s["status"],"budget_exhausted");self.assertEqual(s["generations"],1)
    def test_integrity_never_repairs(self):
        self.backend.failures=["integrity"]
        p=Provider(["x*x","x*x+zero_ct"]);s=self.app.run(self.submit(),p)
        self.assertEqual(s["status"],"service_error");self.assertEqual(p.calls,1)
        self.assertNotIn("PRIVATE",json.dumps(s));self.assertNotIn("/host",json.dumps(s))
    def test_cancel_queued(self):
        t=self.submit();self.app.cancel(t);p=Provider();s=self.app.run(t,p)
        self.assertEqual(s["status"],"cancelled");self.assertEqual(p.calls,0)
    def test_deadline_interrupts_blocked_provider(self):
        t=self.submit(limits=dict(DEFAULT_BUDGET,wall_seconds=1));start=time.monotonic()
        s=self.app.run(t,Provider(sleep=3))
        self.assertEqual(s["status"],"timed_out");self.assertLess(time.monotonic()-start,2)
    def test_interrupted_worker_no_reissue(self):
        t=self.submit();p=self.app._dir(t);s=self.app._load(p/"state.json")
        s.update(status="running",http_attempts=1);self.app._write(p/"state.json",s)
        r=self.app.recover(t);self.assertEqual(r["status"],"service_error")
        self.assertEqual(self.app.run(t,Provider())["http_attempts"],1)
    def test_dependency_drift_prevents_reuse(self):
        t=self.submit();self.app.run(t,Provider());self.backend.version=2
        self.assertEqual(self.app.result(t)["status"],"service_error")
    def test_package_tampering_not_success(self):
        t=self.submit();self.app.run(t,Provider())
        (self.app._dir(t)/"program/manifest.json").write_text("{}")
        self.assertEqual(self.app.result(t)["status"],"service_error")
    def test_http_reserved_before_attempt(self):
        t=self.submit();p=self.app._dir(t);s=self.app._load(p/"state.json")
        c=Controller(self.app,p,s,dict(DEFAULT_BUDGET,http_attempts=1))
        c.reserve_http()
        with self.assertRaises(Stopped):c.reserve_http()
        self.assertEqual(self.app.status(t)["http_attempts"],1)
    def test_backend_exception_is_not_candidate_error(self):
        def broken(*a):raise ValueError("PRIVATE environment")
        self.backend.qualify=broken
        s=self.app.run(self.submit(),Provider())
        self.assertEqual(s["status"],"service_error");self.assertNotIn("PRIVATE",str(s))
    def test_response_failure_is_repairable(self):
        f=structured_failure(dict(failure=dict(layer="response_parse",diagnostic="SECRET")))
        self.assertEqual(f["retry_policy"],"repair_within_budget");self.assertNotIn("SECRET",str(f))
    def test_cancel_during_cached_package_check(self):
        t=self.submit();self.app.run(t,Provider());t2=self.submit()
        original=self.backend.check_package
        def cancel_before_delivery(*args):
            binding=original(*args);self.app.cancel(t2);return binding
        self.backend.check_package=cancel_before_delivery
        p=Provider();result=self.app.run(t2,p)
        self.assertEqual(result["status"],"cancelled");self.assertEqual(p.calls,0)
    def test_active_cancel_from_separate_process(self):
        import multiprocessing
        t=self.submit()
        proc=multiprocessing.get_context("fork").Process(target=self.app.run,args=(t,Provider(sleep=8)))
        proc.start()
        try:
            end=time.monotonic()+3
            while self.app.status(t)["phase"]!="generating" and time.monotonic()<end:time.sleep(.02)
            self.app.cancel(t);proc.join(3)
            self.assertFalse(proc.is_alive())
            self.assertEqual(self.app.status(t)["status"],"cancelled")
        finally:
            if proc.is_alive():proc.terminate();proc.join()
    def test_malformed_budget_is_terminal_unsupported(self):
        t=self.submit(limits={})
        self.assertEqual(self.app.status(t)["status"],"unsupported")
    def test_tampered_task_no_generation(self):
        t=self.submit();p=self.app._dir(t)/"task.json";m=json.loads(p.read_text())
        m["budget"]["generations"]=100;p.write_text(json.dumps(m))
        provider=Provider();result=self.app.run(t,provider)
        self.assertEqual(result["status"],"service_error");self.assertEqual(provider.calls,0)
    def test_context_selection_not_model_name(self):
        from application_policy import prepare,BACKEND
        a,c=prepare(self.model,BACKEND,"numerical",{})
        renamed=copy.deepcopy(self.model);renamed["id"]="arbitrary_unseen_name"
        b,d=prepare(renamed,BACKEND,"numerical",{})
        self.assertEqual(c,d);self.assertIsNone(c["guidance"])
    def test_http_guard_participates_in_provider_retries(self):
        from deepseek_provider import DeepSeekProvider,Config
        class Transport:
            is_live=False
            def __init__(self):self.calls=0
            def post(self,*a):self.calls+=1;raise TimeoutError()
        transport=Transport();provider=DeepSeekProvider(Config(provider_retries=3),transport=transport)
        t=self.submit(limits=dict(DEFAULT_BUDGET,http_attempts=1))
        from unittest.mock import patch
        with patch("deepseek_provider.time.sleep"):
            result=self.app.run(t,provider)
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertEqual(result["http_attempts"],1);self.assertEqual(transport.calls,1)
    def test_source_policy_does_not_import_experiment_modules(self):
        for name in ("application_component.py","application_policy.py","application_backend.py"):
            src=(Path(__file__).parent/name).read_text()
            for forbidden in ("benchmarks.tools","stage2_expression","r211","r205"):
                self.assertNotIn(forbidden,src)
    def test_path_in_task_id_refused(self):
        with self.assertRaises(ValueError):self.app.status("../outside")
if __name__=="__main__":unittest.main()
