"""Application controls with synthetic adapters; no Agent API or FHE claim."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
from benchmark_graph import digest
from benchmark_suite import Builder
from application_component import Application
from application_policy import DEFAULT_BUDGET
from application_artifact_budget import ArtifactMonitor, OperationalStopped, FORMAT, JOURNAL, MONITOR
from test_application_component import Backend, Provider

def bounds(size=1024*1024, reserve=0):
    return dict(format=FORMAT,max_new_artifact_bytes=size,min_free_disk_bytes=reserve)

class ObserverBackend(Backend):
    artifact_monitor_version=MONITOR
    def __init__(self,root,*,file_bytes=0,failures=None,export_bytes=0):
        super().__init__(failures)
        self.root=root
        self.file_bytes=file_bytes
        self.export_bytes=export_bytes
        self.created=[]
    def operational_artifact_roots(self):
        return (self.root,)
    def _evidence(self,prefix,observer):
        folder=Path(tempfile.mkdtemp(prefix=prefix,dir=self.root))
        self.created.append(folder)
        if observer is not None:observer(folder)
        (folder/"synthetic.bin").write_bytes(b"x"*self.file_bytes)
        return folder
    def qualify(self,q,c,level,*,artifact_observer=None):
        result=super().qualify(q,c,level)
        self._evidence("candidate-replay-",artifact_observer)
        return result
    def export(self,r,d,level,**kwargs):
        super().export(r,d,level)
        if self.export_bytes:(d/"synthetic.bin").write_bytes(b"x"*self.export_bytes)
    def execute(self,folder,samples,*,artifact_observer=None):
        evidence=self._evidence("poseidon-program-execution-",artifact_observer)
        return dict(status="executed",test_only=True,encrypted_execution=False,
                    evidence=str(evidence),agent_calls=0,compiler_calls=0)

class AdmissionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.evidence=self.root/"evidence";self.evidence.mkdir()
        self.backend=ObserverBackend(self.evidence)
        self.app=Application(self.root/"tasks",self.backend)
        b=Builder([(2,)])
        self.model=b.finish(b.node("square",["input0"]))
    def tearDown(self):
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.tmp.cleanup()
    def submit(self,**kwargs):
        return self.app.submit(self.model,**kwargs)
    def test_legacy_payload_and_request_hash_unchanged(self):
        old=self.submit()
        new=self.submit(admission_policy="structured-v1",operational_limits=bounds())
        a=self.app._checked_task_metadata(old);b=self.app._checked_task_metadata(new)
        self.assertNotIn("admission_policy",a);self.assertNotIn("operational_limits",a)
        self.assertEqual(a["request"],b["request"])
        self.assertEqual(a["context"],b["context"])
        self.assertEqual(self.app.status(old).keys(),self.app.status(new).keys()-{"operational_usage"})
        self.assertEqual(self.app._load(self.app._dir(old)/"state.json")["reuse_key"],
                         self.app._load(self.app._dir(new)/"state.json")["reuse_key"])
    def test_precise_backend_failure_is_not_sent_to_provider(self):
        task=self.submit(backend="PRIVATE /environment/path",admission_policy="structured-v1")
        provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"unsupported")
        self.assertEqual(result["failure"]["code"],"backend_not_supported")
        self.assertEqual(provider.calls,0)
        self.assertNotIn("PRIVATE",json.dumps(result))
    def test_admission_witnesses(self):
        cases=[("inputs",self.model["inputs"]*5,"input_count_exceeded"),
               ("inputs",[dict(name="input0",shape=[1]*5)],"input_rank_exceeded"),
               ("inputs",[dict(name="input0",shape=[256,2])],"input_elements_exceeded"),
               ("nodes",self.model["nodes"]*65,"graph_node_count_exceeded")]
        for field,value,code in cases:
            with self.subTest(code=code):
                model=copy.deepcopy(self.model);model[field]=value
                task=self.app.submit(model,admission_policy="structured-v1")
                failure=self.app.status(task)["failure"]
                self.assertEqual(failure["code"],code)
                self.assertEqual(set(failure["context"]),{"actual","limit"})
    def test_invalid_budget_and_policy_fail_closed(self):
        for kwargs,reason in [
            (dict(limits={}),"generation_budget"),
            (dict(validation_level=[]),"validation_level"),
            (dict(admission_policy=[]),"admission_policy"),
            (dict(operational_limits=bounds(True)),"operational_limits")]:
            with self.subTest(reason=reason):
                options=dict(admission_policy="structured-v1");options.update(kwargs)
                result=self.app.status(self.submit(**options))
                self.assertEqual(result["status"],"unsupported")
                self.assertEqual(result["failure"]["reason_code"],reason)
    def test_unknown_validator_error_is_generic_and_redacted(self):
        model=copy.deepcopy(self.model);model["nodes"][0]["op"]="PRIVATE_UNSUPPORTED"
        result=self.app.status(self.app.submit(model,admission_policy="structured-v1"))
        self.assertEqual(result["failure"]["reason_code"],"unknown")
        self.assertNotIn("PRIVATE",json.dumps(result))
    def test_observer_contract_required_before_generation(self):
        self.app.backend=Backend()
        task=self.submit(operational_limits=bounds())
        result=self.app.run(task,Provider())
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["code"],"artifact_monitor_unavailable")
        self.assertEqual(result["generations"],0)
    def test_adapter_environment_exception_is_redacted_before_generation(self):
        def broken():
            raise RuntimeError("PRIVATE /environment/path")
        self.backend.operational_artifact_roots=broken
        result=self.app.status(self.submit(operational_limits=bounds()))
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["generations"],0)
        self.assertNotIn("PRIVATE",json.dumps(result))
    def test_missing_external_root_is_integrity_failure(self):
        task=self.submit(operational_limits=bounds())
        monitor=ArtifactMonitor(self.app,self.app._dir(task),bounds())
        folder=Path(tempfile.mkdtemp(prefix="candidate-replay-",dir=self.evidence))
        monitor.register(folder)
        folder.rename(folder.with_name(folder.name+"-moved"))
        with self.assertRaises(OperationalStopped) as caught:monitor.check(force=True)
        self.assertEqual(caught.exception.public_failure["reason_code"],"integrity")
    def test_disk_reserve_is_environment_not_candidate(self):
        with patch("application_artifact_budget.shutil.disk_usage",
                   return_value=type("Disk",(),dict(free=4))()):
            result=self.app.status(self.submit(operational_limits=bounds(reserve=8)))
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["reason_code"],"disk_reserve")
    def test_input_metadata_exhaustion_before_generation(self):
        task=self.submit(operational_limits=bounds(1))
        provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertEqual(provider.calls,0)
    def test_complete_lifecycle_records_local_and_external_artifacts(self):
        self.backend.file_bytes=4096
        result=self.app.run(self.submit(operational_limits=bounds()),Provider())
        self.assertEqual(result["status"],"succeeded")
        usage=result["operational_usage"]
        self.assertEqual(usage["evidence_directories"],1)
        self.assertGreater(usage["observed_peak_bytes"],4096)
        self.assertFalse(usage["hard_filesystem_quota"])
        self.assertNotIn(self.tmp.name,json.dumps(usage))
    def test_repairs_share_one_nonrefundable_budget(self):
        self.backend.file_bytes=60000;self.backend.failures=["compiler"]
        task=self.submit(operational_limits=bounds(100000))
        result=self.app.run(task,Provider(["x*x","x*x+zero_ct"]))
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertEqual(self.backend.calls,2)
        self.assertEqual(result["generations"],2)
        self.assertEqual(result["failure"]["code"],"artifact_budget_exhausted")
        self.assertIsNone(result["package"])
        self.assertEqual(len(self.backend.created),2)
        self.assertTrue(all((p/"synthetic.bin").is_file() for p in self.backend.created))
    def test_packaging_is_inside_budget(self):
        self.backend.export_bytes=100000
        task=self.submit(operational_limits=bounds(50000))
        result=self.app.run(task,Provider())
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertIsNone(result["package"])
        self.assertTrue((self.app._dir(task)/"program/synthetic.bin").is_file())
    def test_cache_reuse_does_not_charge_old_artifacts_or_generate(self):
        old=self.submit();self.app.run(old,Provider())
        self.backend.file_bytes=1000000
        task=self.submit(operational_limits=bounds(50000));provider=Provider()
        result=self.app.run(task,provider)
        self.assertEqual(result["status"],"succeeded")
        self.assertTrue(result["package"]["reused"])
        self.assertEqual(provider.calls,0)
        self.assertEqual(result["operational_usage"]["evidence_directories"],0)
    def test_journal_tampering_stops_before_provider(self):
        task=self.submit(operational_limits=bounds())
        p=self.app._dir(task)/JOURNAL
        value=json.loads(p.read_text());value["observed_peak_bytes"]=0;p.write_text(json.dumps(value))
        provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["reason_code"],"integrity")
        self.assertEqual(provider.calls,0)
    def test_missing_journal_never_resets_continuation_budget(self):
        task=self.submit(operational_limits=bounds())
        (self.app._dir(task)/JOURNAL).unlink()
        provider=Provider();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["reason_code"],"integrity")
        self.assertEqual(provider.calls,0)
    def test_blocked_generation_is_interrupted_by_artifact_limit(self):
        import time
        task=self.submit(operational_limits=bounds(50000))
        folder=self.app._dir(task)
        class WritingProvider(Provider):
            def generate(self,q,h):
                self.calls+=1
                (folder/"transport.tmp").write_bytes(b"x"*100000)
                time.sleep(3)
                raise AssertionError("Must stop before provider returns")
        provider=WritingProvider()
        start=time.monotonic();result=self.app.run(task,provider)
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertLess(time.monotonic()-start,2)
        self.assertEqual(self.backend.calls,0)
        self.assertEqual(provider.calls,1)
        self.assertTrue((folder/"transport.tmp").exists())
    def test_cancel_precedes_resource_check(self):
        task=self.submit(operational_limits=bounds())
        self.app.cancel(task)
        result=self.app.run(task,Provider())
        self.assertEqual(result["status"],"cancelled")
    def test_invalid_local_limits_do_not_change_task(self):
        task=self.submit();self.app.run(task,Provider())
        before=self.app.result(task)
        result=self.app.execute(task,[{}],operational_limits=bounds(False))
        self.assertEqual(result["status"],"unsupported")
        self.assertEqual(self.app.result(task),before)
    def test_accounting_survives_controller_recreation(self):
        task=self.submit(operational_limits=bounds())
        m=ArtifactMonitor(self.app,self.app._dir(task),bounds())
        folder=Path(tempfile.mkdtemp(prefix="candidate-replay-",dir=self.evidence))
        m.register(folder);(folder/"data").write_bytes(b"a"*10000);m.check(force=True)
        peak=m.usage()["observed_peak_bytes"]
        # A host deletion cannot refund already observed storage.
        (folder/"data").unlink()
        again=ArtifactMonitor(self.app,self.app._dir(task),bounds())
        again.check(force=True)
        self.assertGreaterEqual(again.usage()["observed_peak_bytes"],peak)
        self.assertEqual(again.usage()["evidence_directories"],1)
    def test_never_adopts_nonempty_history_or_arbitrary_path(self):
        task=self.submit(operational_limits=bounds())
        m=ArtifactMonitor(self.app,self.app._dir(task),bounds())
        history=Path(tempfile.mkdtemp(prefix="candidate-replay-",dir=self.evidence))
        (history/"history").write_text("preserve")
        with self.assertRaises(OperationalStopped):m.register(history)
        outside=self.root/"candidate-replay-outside";outside.mkdir()
        with self.assertRaises(OperationalStopped):m.register(outside)
        self.assertEqual((history/"history").read_text(),"preserve")
    def test_symlink_and_directory_replacement_are_integrity_errors(self):
        task=self.submit(operational_limits=bounds())
        m=ArtifactMonitor(self.app,self.app._dir(task),bounds())
        link=self.app._dir(task)/"indirect";link.symlink_to(self.evidence)
        with self.assertRaises(OperationalStopped) as caught:m.check(force=True)
        self.assertEqual(caught.exception.public_failure["reason_code"],"integrity")
    def test_source_budget_continues_through_admission_and_generation(self):
        source=dict(format="poseidon-model-source-request-v1",kind="graph",model=self.model)
        task=self.app.submit_source(source,admission_policy="structured-v1",operational_limits=bounds())
        snapshot=self.app._source(self.app._dir(task))
        self.assertEqual(snapshot["operational_limits"],bounds())
        before=self.app.status(task)["operational_usage"]["observed_peak_bytes"]
        result=self.app.run(task,Provider())
        self.assertEqual(result["status"],"succeeded")
        self.assertGreater(result["operational_usage"]["observed_peak_bytes"],before)
        meta=self.app._checked_task_metadata(task)
        self.assertEqual(meta["admission_policy"],"structured-v1")
        self.assertEqual(meta["operational_limits"],bounds())
    def test_source_snapshot_size_exhaustion_is_terminal(self):
        source=dict(format="poseidon-model-source-request-v1",kind="graph",model=self.model)
        task=self.app.submit_source(source,operational_limits=bounds(1))
        result=self.app.status(task)
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertEqual(result["generations"],0)
    def test_untrusted_local_source_rejected_with_public_reason(self):
        source=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
                    manifest="/PRIVATE/file",trust_local_code=False)
        result=self.app.status(self.app.submit_source(source,admission_policy="structured-v1"))
        self.assertEqual(result["status"],"unsupported")
        self.assertEqual(result["failure"]["reason_code"],"trusted_source")
        self.assertNotIn("PRIVATE",json.dumps(result))
    def test_local_execution_has_separate_budget_without_changing_package(self):
        task=self.submit();self.app.run(task,Provider())
        before=self.app.result(task)["package"]
        self.backend.file_bytes=120000
        result=self.app.execute(task,[{}],operational_limits=bounds(50000))
        self.assertEqual(result["status"],"budget_exhausted")
        self.assertEqual(result["agent_calls"],0)
        self.assertEqual(result["compiler_calls"],0)
        self.assertEqual(self.app.result(task)["package"],before)
        self.assertTrue((self.backend.created[-1]/"synthetic.bin").is_file())
    def test_local_execution_success_uses_existing_package_and_observer(self):
        task=self.submit();self.app.run(task,Provider())
        result=self.app.execute(task,[{}],operational_limits=bounds())
        self.assertEqual(result["status"],"executed")
        self.assertEqual(result["operational_usage"]["evidence_directories"],1)
        self.assertEqual(result["agent_calls"],0)
    def test_old_local_execute_signature_remains_valid(self):
        task=self.submit();self.app.run(task,Provider())
        result=self.app.execute(task,[{}])
        self.assertEqual(result["status"],"executed")
        self.assertNotIn("operational_usage",result)

if __name__=="__main__":unittest.main()
