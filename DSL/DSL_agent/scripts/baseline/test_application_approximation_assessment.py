"""Durable reference-only control tests with injected bound source receipts."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import test_task_numerical_assessment as fixture
from benchmark_graph import canonical,digest
from numerical_results import compare
from assessment_array_dataset import export_dataset,load_dataset
from application_assessment import ApplicationAssessment,AssessmentControllerError,read,seal
from application_approximation_assessment import ApplicationApproximationAssessment
from test_approximation_dataset_assessment import receipt
from program_execution import normalize_samples

class ReferenceControllerTests(unittest.TestCase):
    setUp=fixture.AssessmentTests.setUp
    ready=fixture.AssessmentTests.ready
    def prepared(self,n=17,*,quota=2,criterion=None,part_samples=16,**requirements):
        task,provider,outcome=self.ready()
        metadata=copy.deepcopy(self.app._checked_task_metadata(task))
        metadata["model_source"]=dict(kind="trusted_pytorch",snapshot_binding="e"*64)
        metadata["binding"]=digest({k:v for k,v in metadata.items()if k!="binding"})
        patcher=patch("task_output_adapter.TaskOutputAdapter._metadata",return_value=metadata)
        patcher.start();self.addCleanup(patcher.stop)
        folder=Path(self.tmp.name)/"dataset"
        x=np.arange(n*2,dtype=np.float64).reshape(n,2)/max(100.,n*2)
        manifest=export_dataset(metadata["request"],{"x":x},folder)
        service=ApplicationApproximationAssessment(self.app,Path(self.tmp.name)/"reference-store")
        h=service.submit(task,folder/"dataset.json",expected_dataset_binding=manifest["binding"],
            input_dtypes={"x":"float64"},trust_local_code=True,max_new_assessments=quota,
            application_criterion=criterion,part_samples=part_samples,**requirements)
        self.task=task;self.metadata=metadata;self.service=service;self.handle=h
        self.delta=0.;self.calls=0;self.after_reference=None
        self.folder=folder;self.manifest=manifest
        return service,h,task,provider,outcome
    def worker(self,task,samples,**kwargs):
        self.calls+=1
        path,h=self.service._path(self.handle)
        plan=read(path/"plan.json")
        data=load_dataset(self.metadata["request"],path/"dataset/dataset.json",
            expected_binding=plan["dataset_file_binding"])
        collector=self.service._collector(plan,self.metadata,data,{})
        wire=[{n:a.tolist()for n,a in item.items()}for item in normalize_samples(self.metadata["request"],samples)]
        key=next(name for name,p in collector.parts.items()if p["supplied_input_sha256"]==digest(wire))
        actual=np.square(np.asarray([s["x"]for s in samples],dtype=np.float64))
        r=receipt(collector,key,actual,actual+self.delta)
        if self.after_reference:self.after_reference()
        return r
    def patched(self,service):
        return patch.object(service.numeric,"assess",side_effect=self.worker)
    def reject(self,call):
        with self.assertRaises(AssessmentControllerError):call()

    def test_submit_and_query_do_not_execute(self):
        service,h,task,p,outcome=self.prepared()
        first=service.status(h)
        self.assertEqual(first["status"],"queued");self.assertEqual(first["budget"]["used"],0)
        self.assertFalse(first["encrypted_execution"]);self.assertFalse(first["acceptance"]["numerical_on_declared_dataset"])
        self.assertEqual(service.result(h)["binding"],first["binding"])
        self.assertEqual(p.calls,1);self.assertEqual(self.backend.execute_calls,0)
        self.assertEqual(self.app.result(task),outcome)

    def test_resume_reuses_first_receipt_and_terminal_is_stable(self):
        service,h,task,p,outcome=self.prepared(criterion=dict(kind="max_absolute_error",limit=.1))
        with self.patched(service):first=service.advance(h)
        self.assertEqual(first["status"],"ready");self.assertEqual(first["completed_parts"],1)
        other=ApplicationApproximationAssessment(self.app,service.store)
        with self.patched(other):final=other.advance(h)
        self.assertEqual(final["status"],"succeeded");self.assertEqual(final["budget"]["used"],2)
        self.assertEqual(final["summary"]["evaluated_samples"],17)
        self.assertTrue(final["acceptance"]["application_criterion_validated"])
        self.assertFalse(final["ckks_validation_performed"]);self.assertEqual(self.calls,2)
        self.assertEqual(other.advance(h)["binding"],final["binding"])
        self.assertEqual(other.cancel(h)["binding"],final["binding"])
        self.assertEqual(self.backend.execute_calls,0);self.assertEqual(self.app.result(task),outcome)

    def test_no_quota_and_exhaustion_do_not_repeat(self):
        for quota in (0,1):
            if quota==0:service,h,*_=self.prepared(quota=0)
            else:
                h=service.submit(self.task,self.folder/"dataset.json",
                    expected_dataset_binding=self.manifest["binding"],input_dtypes={"x":"float64"},
                    trust_local_code=True,max_new_assessments=1);self.handle=h
            with self.patched(service):r=service.advance(h,max_new_parts=2)
            self.assertEqual(r["status"],"budget_exhausted");self.assertEqual(r["budget"]["used"],quota)
            calls=self.calls;service.advance(h);self.assertEqual(self.calls,calls)

    def test_cancel_and_running_lock_keep_committed_quota(self):
        service,h,*_=self.prepared()
        seen=[];self.after_reference=lambda:seen.append(service.status(h))
        with self.patched(service):r=service.advance(h)
        self.assertTrue(seen[0]["worker_active"]);self.assertEqual(seen[0]["budget"]["used"],1)
        self.assertEqual(r["status"],"ready")
        self.assertEqual(service.cancel(h)["status"],"cancelled")
        self.assertEqual(service.advance(h)["completed_parts"],1);self.assertEqual(self.calls,1)

    def test_metric_failures_are_recorded_and_independent_parts_continue(self):
        service,h,task,_,outcome=self.prepared(criterion=dict(kind="max_absolute_error",limit=.1))
        self.delta=.25
        with self.patched(service):r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"validation_failed");self.assertEqual(r["completed_parts"],2)
        self.assertEqual(r["summary"]["part_criterion_failures"],2)
        self.assertEqual(self.calls,2);self.assertEqual(self.app.result(task),outcome)

    def test_interruption_keeps_uncertain_attempt_and_never_retries(self):
        service,h,*_=self.prepared()
        class Interrupted(BaseException):pass
        with patch.object(service.numeric,"assess",side_effect=Interrupted):
            with self.assertRaises(Interrupted):service.advance(h)
        again=ApplicationApproximationAssessment(self.app,service.store).status(h)
        self.assertEqual(again["status"],"service_error")
        self.assertEqual(again["budget"]["used"],1)
        self.assertTrue(again["ledger"][0]["execution_uncertain"])
        self.assertEqual(service.advance(h)["status"],"service_error");self.assertEqual(self.calls,0)

    def test_owned_data_receipt_and_identity_tampering_are_rejected(self):
        service,h,*_=self.prepared()
        with self.patched(service):service.advance(h)
        p=service.store/h["evaluation_id"]/"receipts/part_0.json";original=p.read_bytes()
        p.write_bytes(original+b" ")
        self.reject(lambda:service.status(h));p.write_bytes(original)
        with patch("application_approximation_assessment.identity",return_value="0"*64):
            self.reject(lambda:service.advance(h))
        p=service.store/h["evaluation_id"]/"dataset/input_0.npy";raw=p.read_bytes()
        p.write_bytes(raw[:-1]+bytes([raw[-1]^1]));self.reject(lambda:service.advance(h))
        self.assertEqual(self.calls,1);self.assertEqual(self.backend.execute_calls,0)

    def test_source_data_change_does_not_replace_owned_snapshot(self):
        service,h,*_=self.prepared(3,quota=1)
        (self.folder/"input_0.npy").write_bytes(b"untrusted changed data")
        with self.patched(service):r=service.advance(h)
        self.assertEqual(r["status"],"succeeded");self.assertEqual(r["summary"]["evaluated_samples"],3)

    def test_receipt_storage_exhaustion_preserves_call(self):
        service,h,*_=self.prepared(3,quota=1,max_receipt_bytes=1)
        with self.patched(service):r=service.advance(h)
        self.assertEqual(r["status"],"budget_exhausted");self.assertEqual(r["budget"]["used"],1)
        self.assertEqual(r["completed_parts"],0);self.assertEqual(self.calls,1)
        self.assertEqual(service.advance(h)["status"],"budget_exhausted")

    def test_service_failure_and_private_text_are_sanitized(self):
        service,h,*_=self.prepared()
        with patch.object(service.numeric,"assess",side_effect=RuntimeError("PRIVATE 123.456")):
            r=service.advance(h)
        self.assertEqual(r["status"],"service_error");self.assertNotIn("PRIVATE",str(r))
        self.assertEqual(r["budget"]["used"],1);self.assertFalse(r["encrypted_execution"])

    def test_numerical_controller_cannot_execute_reference_only_handle(self):
        service,h,*_=self.prepared()
        numerical=ApplicationAssessment(self.app,service.store)
        self.reject(lambda:numerical.advance(h))
        self.assertEqual(self.backend.execute_calls,0)

    def test_deadline_is_persistent_and_expired_work_is_not_started(self):
        service,h,*_=self.prepared(total_wall_seconds=1)
        with patch("application_assessment.time_budget.stop_reason",
                   return_value=("timed_out","assessment_controller_deadline_reached")):
            r=service.advance(h)
        self.assertEqual(r["status"],"timed_out");self.assertEqual(r["budget"]["used"],0)
        self.assertEqual(self.calls,0)

    def test_all_cached_receipts_complete_under_zero_new_quota(self):
        service,h,*_=self.prepared(3,quota=2,part_samples=2)
        with self.patched(service):done=service.advance(h,max_new_parts=2)
        path=service.store/h["evaluation_id"]
        cache={p.stem:dict(receipt=read(p),binding=read(p)["binding"])for p in (path/"receipts").glob("*.json")}
        other=service.submit(self.task,self.folder/"dataset.json",
            expected_dataset_binding=self.manifest["binding"],input_dtypes={"x":"float64"},
            trust_local_code=True,max_new_assessments=0,part_samples=2,cached_receipts=cache)
        calls=self.calls;r=service.advance(other)
        self.assertEqual(r["status"],"succeeded");self.assertEqual(r["budget"]["used"],0)
        self.assertEqual(r["completed_parts"],2);self.assertEqual(self.calls,calls)

    def test_explicit_unbounded_deadline_is_rejected(self):
        service,h,*_=self.prepared(3,quota=1)
        self.reject(lambda:service.submit(self.task,self.folder/"dataset.json",
            expected_dataset_binding=self.manifest["binding"],input_dtypes={"x":"float64"},
            trust_local_code=True,max_new_assessments=1,total_wall_seconds=None))
        self.assertEqual(self.calls,0)

    def test_result_queries_program_once_and_keeps_return_binding(self):
        service,h,*_=self.prepared(3,quota=1)
        with self.patched(service):done=service.advance(h)
        with patch.object(self.app,"result",wraps=self.app.result)as getter:
            result=service.result(h)
        self.assertEqual(getter.call_count,1)
        self.assertEqual(result["binding"],done["binding"])

    def test_declared_grade_disagreement_rejects_before_reference(self):
        service,h,*_=self.prepared(3,quota=1)
        root=service.store/h["evaluation_id"];plan=read(root/"plan.json")
        plan["package_validation"]=dict(level="compiled",encrypted_execution=False,numerically_validated=False)
        plan=seal(plan);h=dict(h,binding=plan["binding"])
        state=read(root/"state.json");state["submission_binding"]=plan["binding"]
        (root/"plan.json").write_bytes(canonical(plan));(root/"state.json").write_bytes(canonical(seal(state)))
        self.reject(lambda:service.advance(h))
        self.assertEqual(self.calls,0);self.assertEqual(self.backend.execute_calls,0)

    def test_frame_metadata_drift_rejects_instead_of_using_an_old_copy(self):
        service,h,*_=self.prepared(3,quota=1)
        changed=copy.deepcopy(self.metadata);changed["binding"]="0"*64
        with patch.object(service.numeric.outputs,"_metadata",side_effect=[self.metadata,changed]):
            self.reject(lambda:service.status(h))
        self.assertEqual(self.calls,0)

    def test_cached_first_part_runs_only_original_remaining_part(self):
        service,h,*_=self.prepared()
        with self.patched(service):service.advance(h)
        p=service.store/h["evaluation_id"]/"receipts/part_0.json";r=read(p)
        h=service.submit(self.task,self.folder/"dataset.json",expected_dataset_binding=self.manifest["binding"],
            input_dtypes={"x":"float64"},trust_local_code=True,max_new_assessments=1,
            cached_receipts={"part_0":dict(receipt=r,binding=r["binding"])})
        self.handle=h
        with self.patched(service):done=service.advance(h)
        self.assertEqual(done["status"],"succeeded");self.assertEqual(done["budget"]["used"],1)
        self.assertEqual(done["completed_parts"],2);self.assertEqual(self.calls,2)

if __name__=="__main__":unittest.main()
