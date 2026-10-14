"""Application assessment control tests with injected backend executions."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import canonical,digest
from application_assessment import ApplicationAssessment,AssessmentControllerError,seal,read,write_new,identity
from task_numerical_assessment import POLICY,NATIVE_POLICY
from assessment_array_dataset import export_dataset,load_dataset
from test_task_numerical_assessment import AssessmentTests,relu
class ApplicationAssessmentTests(unittest.TestCase):
    setUp=AssessmentTests.setUp
    ready=AssessmentTests.ready
    def prepared(self,n=17,*,graph=None,provenance=None):
        task,provider,outcome=self.ready(graph,provenance)
        request=self.app._checked_task_metadata(task)["request"]
        x=np.arange(n*2,dtype=np.float64).reshape(n,2)/max(100.,n*2)
        folder=Path(self.tmp.name)/"dataset"
        manifest=export_dataset(request,{"x":x},folder)
        service=ApplicationAssessment(self.app,Path(self.tmp.name)/"assessment-store")
        return service,task,folder,manifest,provider,outcome
    def submitted(self,n=17,quota=2,**kwargs):
        service,task,folder,manifest,p,outcome=self.prepared(n)
        handle=service.submit(task,folder/"dataset.json",expected_dataset_binding=manifest["binding"],
            max_new_assessments=quota,**kwargs)
        return service,handle,task,p,outcome,folder,manifest
    def rejection(self,call,code=None):
        with self.assertRaises(AssessmentControllerError)as error:call()
        if code:self.assertEqual(error.exception.code,code)
        self.assertNotIn("PRIVATE",str(error.exception.feedback()))
    def test_submit_status_result_do_not_execute_or_call_provider(self):
        service,h,task,p,outcome,*_=self.submitted()
        self.assertEqual(self.backend.execute_calls,0)
        first=service.status(h);self.assertEqual(first["status"],"queued")
        self.assertEqual(first["budget"]["used"],0);self.assertIsNone(first["summary"])
        self.assertEqual(service.result(h),first);self.assertEqual(p.calls,1)
        self.assertEqual(self.app.result(task),outcome)
    def test_first_receipt_is_obtained_inside_the_same_budget(self):
        service,h,task,p,outcome,*_=self.submitted()
        first=service.advance(h)
        self.assertEqual(first["status"],"ready");self.assertEqual(first["budget"]["used"],1)
        self.assertEqual(first["completed_parts"],1);self.assertEqual(first["summary"]["evaluated_samples"],16)
        self.assertEqual(self.backend.execute_calls,1)
        final=service.advance(h)
        self.assertEqual(final["status"],"succeeded");self.assertEqual(final["budget"]["used"],2)
        self.assertEqual(final["summary"]["evaluated_samples"],17)
        self.assertEqual(final["summary"]["ckks_comparison"]["compared_values"],34)
        self.assertTrue(final["acceptance"]["numerical_on_declared_dataset"])
        self.assertEqual(self.backend.execute_calls,2);self.assertEqual(p.calls,1)
        self.assertEqual(self.app.result(task),outcome)
    def test_zero_budget_is_terminal_without_bootstrap_execution(self):
        service,h,*_=self.submitted(quota=0)
        r=service.advance(h);self.assertEqual(r["status"],"budget_exhausted")
        self.assertEqual(r["budget"]["used"],0);self.assertEqual(self.backend.execute_calls,0)
    def test_exhaustion_does_not_retry_completed_or_pending_part(self):
        service,h,*_=self.submitted(quota=1)
        r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"budget_exhausted");self.assertEqual(r["completed_parts"],1)
        self.assertEqual(r["summary"]["parts"][1]["status"],"not_run")
        service.advance(h,max_new_parts=4096);self.assertEqual(self.backend.execute_calls,1)
    def test_reload_continues_from_receipts_without_reexecuting_first_part(self):
        service,h,task,p,outcome,*_=self.submitted()
        service.advance(h);self.assertEqual(self.backend.execute_calls,1)
        other=ApplicationAssessment(self.app,service.store)
        r=other.advance(h);self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["budget"]["used"],2);self.assertEqual(self.backend.execute_calls,2)
        again=ApplicationAssessment(self.app,service.store).result(h)
        self.assertEqual(again["binding"],r["binding"]);self.assertEqual(p.calls,1)
    def test_cancel_before_execution_is_durable_and_preserves_program_task(self):
        service,h,task,_,outcome,*_=self.submitted()
        r=service.cancel(h);self.assertEqual(r["status"],"cancelled")
        again=ApplicationAssessment(self.app,service.store).advance(h)
        self.assertEqual(again["status"],"cancelled");self.assertEqual(again["budget"]["used"],0)
        self.assertEqual(self.backend.execute_calls,0);self.assertEqual(self.app.result(task),outcome)
        self.assertFalse((self.app.store/task/"cancel.json").exists())
    def test_cancel_during_call_is_seen_before_next_part(self):
        service,h,*_=self.submitted()
        self.backend.after_execute=lambda:service.cancel(h)
        r=service.advance(h,max_new_parts=2);self.backend.after_execute=None
        self.assertEqual(r["status"],"cancelled");self.assertEqual(r["budget"]["used"],1)
        self.assertEqual(r["completed_parts"],1);self.assertEqual(self.backend.execute_calls,1)
    def test_running_status_observes_lock_and_committed_admission(self):
        service,h,*_=self.submitted(3,quota=1);seen=[]
        self.backend.after_execute=lambda:seen.append(service.status(h))
        r=service.advance(h);self.backend.after_execute=None
        self.assertEqual(r["status"],"succeeded")
        self.assertEqual(seen[0]["status"],"running");self.assertTrue(seen[0]["worker_active"])
        self.assertEqual(seen[0]["budget"]["used"],1)
        self.assertEqual(seen[0]["ledger"][0]["phase"],"started")
    def test_interruption_before_receipt_never_retries_unknown_execution(self):
        service,h,*_=self.submitted()
        class HostStopped(BaseException):pass
        with patch.object(service.numeric,"assess",side_effect=HostStopped):
            with self.assertRaises(HostStopped):service.advance(h)
        state=read(service.store/h["evaluation_id"]/"state.json")
        self.assertEqual(state["status"],"running");self.assertEqual(state["used"],1)
        resumed=ApplicationAssessment(self.app,service.store).status(h)
        self.assertEqual(resumed["status"],"service_error")
        self.assertEqual(resumed["failure"]["code"],"assessment_controller_execution_uncertain")
        self.assertTrue(resumed["ledger"][0]["execution_uncertain"])
        service.advance(h);self.assertEqual(self.backend.execute_calls,0)
    def test_orphan_uncommitted_receipt_is_preserved_without_adoption(self):
        service,h,*_=self.submitted(3,quota=1)
        path=service.store/h["evaluation_id"]
        class HostStopped(BaseException):pass
        import application_assessment
        real_write=application_assessment.write_new
        def interrupted(p,v):
            real_write(p,v)
            if p.parent.name=="receipts":raise HostStopped()
        with patch("application_assessment.write_new",side_effect=interrupted):
            with self.assertRaises(HostStopped):service.advance(h)
        orphan=path/"receipts/part_0.json";before=hashlib.sha256(orphan.read_bytes()).hexdigest()
        r=ApplicationAssessment(self.app,service.store).result(h)
        self.assertEqual(r["status"],"service_error");self.assertEqual(r["completed_parts"],0)
        self.assertEqual(r["budget"]["used"],1);self.assertEqual(self.backend.execute_calls,1)
        self.assertEqual(hashlib.sha256(orphan.read_bytes()).hexdigest(),before)
    def test_numerical_failures_remain_visible_and_independent_parts_continue(self):
        service,h,*_=self.submitted();self.backend.delta=.01
        r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"validation_failed");self.assertEqual(r["budget"]["used"],2)
        self.assertEqual(r["summary"]["part_numerical_failures"],2)
        self.assertFalse(r["acceptance"]["numerical_on_declared_dataset"])
    def test_application_accuracy_failure_does_not_downgrade_original_task(self):
        graph,provenance=relu()
        service,task,folder,m,p,outcome=self.prepared(graph=graph,provenance=provenance)
        h=service.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],max_new_assessments=2,
            application_criterion=dict(kind="max_absolute_error",limit=.001))
        r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"validation_failed")
        self.assertTrue(r["acceptance"]["numerical_on_declared_dataset"])
        self.assertFalse(r["acceptance"]["application_criterion_validated"])
        self.assertEqual(self.app.result(task),outcome);self.assertEqual(p.calls,1)
    def test_service_exception_preserves_used_quota_and_redacts_private_message(self):
        service,h,*_=self.submitted()
        with patch.object(service.numeric,"assess",side_effect=RuntimeError("PRIVATE 123.456")):
            r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"service_error");self.assertEqual(r["budget"]["used"],1)
        self.assertNotIn("PRIVATE",str(r));self.assertEqual(r["ledger"][0]["phase"],"failed")
        service.advance(h);self.assertEqual(self.backend.execute_calls,0)
    def test_timeout_and_unsupported_do_not_become_numeric_success(self):
        for status in ("timed_out","unsupported"):
            with self.subTest(status=status):
                # Independent state directories; one already submitted task is reused.
                if status=="timed_out":
                    service,h,task,_,_,folder,m=self.submitted()
                else:
                    h=service.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],max_new_assessments=2)
                with patch.object(service.numeric,"assess",return_value=dict(status=status,
                    failure=dict(code="source_reference_timeout",layer="source_reference"))):
                    r=service.advance(h,max_new_parts=2)
                self.assertEqual(r["status"],status);self.assertEqual(r["budget"]["used"],1)
                self.assertFalse(r["acceptance"]["numerical_on_declared_dataset"])
    def test_dataset_source_changes_after_submit_do_not_change_owned_snapshot(self):
        service,h,_,_,_,folder,m=self.submitted(3,quota=1)
        (folder/m["inputs"]["x"]["file"]).write_bytes(b"PRIVATE")
        r=service.advance(h);self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["summary"]["evaluated_samples"],3)
    def test_owned_snapshot_mutation_is_rejected_before_execution(self):
        service,h,*_=self.submitted(3,quota=1)
        p=service.store/h["evaluation_id"]/"dataset/input_0.npy";raw=p.read_bytes();p.write_bytes(raw[:-1]+b"x")
        self.rejection(lambda:service.advance(h));self.assertEqual(self.backend.execute_calls,0)
    def test_wrong_handle_and_path_traversal_are_rejected(self):
        service,h,*_=self.submitted()
        for bad in (dict(h,binding="0"*64),dict(h,evaluation_id="../PRIVATE"),dict(h,extra="PRIVATE")):
            self.rejection(lambda v=bad:service.status(v))
        self.assertEqual(self.backend.execute_calls,0)
    def test_budget_tampering_even_with_new_state_hash_is_rejected(self):
        service,h,*_=self.submitted()
        p=service.store/h["evaluation_id"]/"state.json";s=read(p);s["used"]=0
        s["ledger"]=[dict(number=1,part_id="part_0",phase="started",outcome=None)]
        p.write_bytes(canonical(seal(s)))
        self.rejection(lambda:service.advance(h),"assessment_controller_state_rejected")
        self.assertEqual(self.backend.execute_calls,0)
    def test_controller_or_reference_identity_drift_stops_before_execution(self):
        service,h,*_=self.submitted()
        with patch("application_assessment.identity",return_value="0"*64):
            self.rejection(lambda:service.advance(h),"assessment_controller_binding_changed")
        self.assertEqual(self.backend.execute_calls,0)
    def test_receipt_tampering_is_rejected_on_reload(self):
        service,h,*_=self.submitted();service.advance(h)
        p=service.store/h["evaluation_id"]/"receipts/part_0.json";r=json.loads(p.read_text());r["samples"]=1
        p.write_bytes(canonical(r))
        self.rejection(lambda:ApplicationAssessment(self.app,service.store).advance(h))
        self.assertEqual(self.backend.execute_calls,1)
    def test_receipt_storage_budget_exhaustion_preserves_attempt_and_is_terminal(self):
        service,h,*_=self.submitted(3,quota=1,max_receipt_bytes=1)
        r=service.advance(h);self.assertEqual(r["status"],"budget_exhausted")
        self.assertEqual(r["failure"]["code"],"assessment_controller_receipt_budget_exhausted")
        self.assertEqual(r["budget"]["used"],1);self.assertEqual(r["completed_parts"],0)
        self.assertEqual(self.backend.execute_calls,1)
        service.advance(h);self.assertEqual(self.backend.execute_calls,1)
    def test_cached_receipts_complete_without_new_execution(self):
        service,task,folder,m,_,outcome=self.prepared(3)
        request=self.app._checked_task_metadata(task)["request"]
        data=load_dataset(request,folder/"dataset.json",expected_binding=m["binding"])
        receipt=self.adapter.assess(task,data.batches()[0]["samples"],expected_package_binding=outcome["package"]["binding"])
        before=self.backend.execute_calls
        h=service.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],
            cached_receipts={"part_0":dict(receipt=receipt,binding=receipt["binding"])})
        r=service.advance(h);self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["budget"]["used"],0);self.assertEqual(self.backend.execute_calls,before)
        self.assertEqual(r["summary"]["evaluated_samples"],3)
    def test_result_and_input_options_are_detached_from_external_mutation(self):
        service,h,*_=self.submitted(3,quota=1)
        r=service.advance(h);binding=r["binding"];r["ledger"][0]["number"]=999
        self.assertEqual(service.result(h)["binding"],binding)
    def test_native_policy_requires_explicit_trust_source_and_exact_dtype(self):
        service,task,folder,m,*_=self.prepared(3)
        self.rejection(lambda:service.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],
            reference_policy=NATIVE_POLICY,max_new_assessments=1,input_dtypes={"x":"float64"},trust_local_code=True))
        self.assertEqual(self.backend.execute_calls,0)
    def test_invalid_limits_and_partition_are_rejected_before_execution(self):
        service,task,folder,m,*_=self.prepared(3)
        for options in (dict(max_new_assessments=True),dict(max_new_assessments=4097),
                        dict(part_samples=17),dict(max_receipt_bytes=0),dict(input_dtypes={"x":"float32"})):
            self.rejection(lambda v=options:service.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],**v))
        self.assertEqual(self.backend.execute_calls,0)
    def test_nonempty_unowned_store_and_symlink_store_are_rejected(self):
        path=Path(self.tmp.name)/"unowned";path.mkdir();(path/"keep").write_text("original")
        self.rejection(lambda:ApplicationAssessment(self.app,path))
        alias=Path(self.tmp.name)/"alias";alias.symlink_to(path,target_is_directory=True)
        self.rejection(lambda:ApplicationAssessment(self.app,alias))
        self.assertEqual((path/"keep").read_text(),"original")
    def test_explicit_partition_size_keeps_complete_numeric_scope(self):
        service,h,*_=self.submitted(3,quota=2,part_samples=2)
        r=service.advance(h,max_new_parts=2)
        self.assertEqual(r["status"],"succeeded");self.assertEqual(r["planned_parts"],2)
        self.assertEqual(r["summary"]["ckks_comparison"]["compared_values"],6)
        self.assertEqual(self.backend.execute_calls,2)
if __name__=="__main__":unittest.main()
