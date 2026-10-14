"""Count/cancellation/cache control tests with injected backend outputs."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import digest
from assessment_array_dataset import export_dataset,load_dataset
from task_dataset_assessment import reference_anchor,DatasetAssessmentError
from task_dataset_evaluation import TaskDatasetEvaluation
from test_task_numerical_assessment import AssessmentTests,relu
class DatasetEvaluationTests(unittest.TestCase):
    setUp=AssessmentTests.setUp
    ready=AssessmentTests.ready
    def prepared(self,n=17,limit=0,cache_first=False,metric=None,graph=None,provenance=None,dtype=np.float64):
        task,provider,outcome=self.ready(graph,provenance)
        request=self.app._checked_task_metadata(task)["request"]
        x=np.arange(n*2,dtype=dtype).reshape(n,2)/max(100.,n*2)
        folder=Path(self.tmp.name)/"dataset"
        labels=None
        if metric and metric["kind"]=="top1_accuracy_drop":
            labels=dict(output="out",class_axis=0,values=np.zeros(n,np.int64))
        manifest=export_dataset(request,{"x":x},folder,labels=labels)
        data=load_dataset(request,folder/"dataset.json",expected_binding=manifest["binding"])
        first=data.batches()[0]
        profile=None
        if labels is not None:
            profile=dict(kind="top1_accuracy_drop",output="out",class_axis=0,
                         labels=[0]*len(first["samples"]),max_drop=metric["max_drop"])
        elif metric:profile=metric
        receipt=self.adapter.assess(task,first["samples"],expected_package_binding=outcome["package"]["binding"],
            application_criterion=profile)
        anchor=reference_anchor(receipt,expected_binding=receipt["binding"])
        cache={"part_0":dict(receipt=receipt,binding=receipt["binding"])} if cache_first else None
        evaluation=TaskDatasetEvaluation(self.app,task,data,anchor=anchor,
            application_criterion=metric,max_new_assessments=limit,cached_receipts=cache)
        return evaluation,data,task,provider,outcome,receipt
    def test_default_zero_quota_has_no_execution_or_generation(self):
        e,_,task,p,outcome,_=self.prepared()
        before=self.backend.execute_calls;calls=p.calls
        r=e.advance();self.assertEqual(r["status"],"budget_exhausted")
        self.assertEqual(r["new_assessments"],0);self.assertEqual(self.backend.execute_calls,before)
        self.assertEqual(r["summary"]["parts"][0]["status"],"not_run")
        self.assertEqual(self.app.result(task),outcome);self.assertEqual(p.calls,calls)
    def test_cache_only_default_advance_reuses_receipt_without_execution(self):
        e,_,task,p,outcome,_=self.prepared(3,cache_first=True)
        before=self.backend.execute_calls
        r=e.advance();self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["new_assessments"],0);self.assertEqual(r["summary"]["evaluated_samples"],3)
        self.assertEqual(self.backend.execute_calls,before);self.assertEqual(p.calls,1)
        self.assertEqual(self.app.result(task),outcome)
        self.assertEqual(e.advance()["binding"],r["binding"])
    def test_explicit_quota_admits_one_of_two_parts_and_never_expands(self):
        e,*_=self.prepared(limit=1);before=self.backend.execute_calls
        r=e.advance(max_new_parts=4096)
        self.assertEqual(r["status"],"budget_exhausted");self.assertEqual(r["new_assessments"],1)
        self.assertEqual(r["summary"]["evaluated_samples"],16)
        self.assertEqual(r["summary"]["parts"][1]["status"],"not_run")
        self.assertEqual(self.backend.execute_calls,before+1)
        e.advance();self.assertEqual(self.backend.execute_calls,before+1)
    def test_small_advances_finish_three_parts_without_repeated_execution(self):
        e,_,task,p,outcome,_=self.prepared(33,limit=3);before=self.backend.execute_calls
        self.assertEqual(e.advance()["status"],"ready")
        self.assertEqual(e.advance()["status"],"ready")
        r=e.advance();self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["summary"]["evaluated_samples"],33)
        self.assertEqual(r["summary"]["ckks_comparison"]["compared_values"],66)
        self.assertEqual(self.backend.execute_calls,before+3)
        self.assertEqual(p.calls,1);self.assertEqual(self.app.result(task),outcome)
    def test_zero_increment_does_not_admit_new_execution(self):
        e,*_=self.prepared(limit=2);before=self.backend.execute_calls
        self.assertEqual(e.advance(max_new_parts=0)["status"],"ready")
        self.assertEqual(self.backend.execute_calls,before)
    def test_cache_plus_new_part_uses_only_one_admission(self):
        e,*_=self.prepared(limit=1,cache_first=True);before=self.backend.execute_calls
        r=e.advance();self.assertEqual(r["status"],"succeeded")
        self.assertEqual(r["new_assessments"],1)
        self.assertEqual([x["new_assessments"] for x in r["ledger"]],[0,1])
        self.assertEqual(self.backend.execute_calls,before+1)
    def test_cancel_before_execution_preserves_task_and_budget(self):
        e,_,task,_,outcome,_=self.prepared(limit=2);before=self.backend.execute_calls
        self.assertEqual(e.cancel()["status"],"cancelled")
        self.assertEqual(e.advance()["new_assessments"],0)
        self.assertEqual(self.backend.execute_calls,before)
        self.assertEqual(self.app.result(task),outcome)
        self.assertFalse((self.app.store/task/"cancel.json").exists())
    def test_cancel_during_call_is_observed_before_next_part(self):
        e,*_=self.prepared(limit=2);before=self.backend.execute_calls
        self.backend.after_execute=e.cancel
        r=e.advance(max_new_parts=2);self.backend.after_execute=None
        self.assertEqual(r["status"],"cancelled");self.assertEqual(r["new_assessments"],1)
        self.assertEqual(r["summary"]["evaluated_samples"],16)
        self.assertEqual(self.backend.execute_calls,before+1)
    def test_ckks_failure_is_retained_and_independent_parts_continue(self):
        e,*_=self.prepared(limit=2);self.backend.delta=.01
        r=e.advance(max_new_parts=2)
        self.assertEqual(r["status"],"validation_failed");self.assertEqual(r["new_assessments"],2)
        self.assertEqual(r["summary"]["part_numerical_failures"],2)
        self.assertEqual(r["summary"]["numerically_validated_samples"],0)
    def test_service_failure_stops_without_automatic_retry_or_private_error(self):
        e,*_=self.prepared(limit=2)
        with patch.object(e.numeric,"assess",side_effect=RuntimeError("PRIVATE 9.12345")) as call:
            r=e.advance(max_new_parts=2);self.assertEqual(call.call_count,1)
            e.advance();self.assertEqual(call.call_count,1)
        self.assertEqual(r["status"],"service_error");self.assertEqual(r["new_assessments"],1)
        self.assertEqual(r["ledger"][0]["encrypted_execution"],None)
        self.assertNotIn("PRIVATE",str(r["failure"]))
    def test_timeout_is_terminal_and_counts_attempt_before_call(self):
        e,*_=self.prepared(limit=2)
        def timeout(*args,**kwargs):
            self.assertEqual(e.used,1);self.assertEqual(e.ledger[0]["status"],"started")
            return dict(status="timed_out",failure=dict(code="fixture_timeout",layer="execution"))
        with patch.object(e.numeric,"assess",side_effect=timeout) as call:
            r=e.advance(max_new_parts=2);e.advance();self.assertEqual(call.call_count,1)
        self.assertEqual(r["status"],"timed_out")
        self.assertEqual(r["summary"]["parts"][0]["status"],"timed_out")
    def test_wrong_cache_pin_or_input_is_rejected_before_execution(self):
        e,data,task,_,_,receipt=self.prepared();before=self.backend.execute_calls
        anchor=reference_anchor(receipt,expected_binding=receipt["binding"])
        for cache in (dict(part_0=dict(receipt=receipt,binding="0"*64)),
                      dict(part_1=dict(receipt=receipt,binding=receipt["binding"]))):
            with self.assertRaises(DatasetAssessmentError):
                TaskDatasetEvaluation(self.app,task,data,anchor=anchor,max_new_assessments=2,cached_receipts=cache)
        self.assertEqual(self.backend.execute_calls,before)
    def test_receipt_cache_is_detached_from_caller_mutation(self):
        e,data,task,_,_,receipt=self.prepared(3);cache={"part_0":dict(receipt=receipt,binding=receipt["binding"])}
        fresh=TaskDatasetEvaluation(self.app,task,data,anchor=reference_anchor(receipt,expected_binding=receipt["binding"]),
            cached_receipts=cache)
        receipt["bundle_binding"]="0"*64;cache.clear()
        self.assertEqual(fresh.advance()["status"],"succeeded")
    def test_identity_drift_returns_failed_state_with_historical_summary_only(self):
        e,*_=self.prepared(limit=2);before=self.backend.execute_calls
        with patch("task_dataset_evaluation.evaluation_identity",return_value="0"*64):
            r=e.advance();state=e.state()
        self.assertEqual(r["status"],"service_error");self.assertFalse(state["summary_identity_current"])
        self.assertEqual(r["failure"]["code"],"dataset_reference_changed")
        self.assertEqual(self.backend.execute_calls,before)
    def test_collector_identity_drift_cannot_escape_failure_handler(self):
        e,*_=self.prepared(limit=2)
        with patch("task_dataset_assessment.adapter_identity",return_value="0"*64):
            r=e.advance()
        self.assertEqual(r["status"],"service_error");self.assertFalse(r["summary_identity_current"])
    def test_returned_state_is_detached_from_live_control_and_reports(self):
        e,*_=self.prepared(3,cache_first=True);r=e.advance();binding=r["binding"]
        r["ledger"][0]["new_assessments"]=999;r["summary"]["parts"][0]["planned_samples"]=999
        self.assertEqual(e.state()["binding"],binding)
        with self.assertRaises(AttributeError):e.limit=99
    def test_invalid_quota_and_increment_are_rejected(self):
        e,data,task,_,_,receipt=self.prepared()
        anchor=reference_anchor(receipt,expected_binding=receipt["binding"])
        for n in (-1,4097,True):
            with self.assertRaises(DatasetAssessmentError):
                TaskDatasetEvaluation(self.app,task,data,anchor=anchor,max_new_assessments=n)
            with self.assertRaises(DatasetAssessmentError):e.advance(max_new_parts=n)
    def test_explicit_application_criterion_preserves_original_task_grade(self):
        graph,provenance=relu()
        e,_,task,_,outcome,_=self.prepared(17,2,metric=dict(kind="max_absolute_error",limit=.001),
            graph=graph,provenance=provenance)
        r=e.advance(max_new_parts=2)
        self.assertEqual(r["status"],"validation_failed")
        self.assertTrue(r["summary"]["supplied_inputs_numerically_validated"])
        self.assertEqual(r["summary"]["application_criterion"]["status"],"failed")
        self.assertEqual(self.app.result(task),outcome)
    def test_native_dataset_dtype_must_match_pinned_reference(self):
        from task_numerical_assessment import NATIVE_POLICY,NATIVE_FORMAT,implementation
        e,data,task,_,_,r=self.prepared(3,dtype=np.float32)
        r.update(format=NATIVE_FORMAT,reference_policy=NATIVE_POLICY,
            reference_identity_sha256=implementation(policy=NATIVE_POLICY),
            source_binding="a"*64,source_snapshot_binding="b"*64,
            source_reference_evaluated=True,source_reference_binding="c"*64,source_forward_binding="d"*64,
            source_input_dtypes={"x":"float64"},source_output_dtypes={"out":"float64"},
            source_total_comparison=copy.deepcopy(r["total_comparison"]))
        r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        anchor=reference_anchor(r,expected_binding=r["binding"])
        with self.assertRaises(DatasetAssessmentError) as error:
            TaskDatasetEvaluation(self.app,task,data,anchor=anchor)
        self.assertEqual(error.exception.code,"dataset_binding_rejected")
    def test_explicit_smaller_parts_keep_samples_and_bound_execution_calls(self):
        e,data,task,_,_,receipt=self.prepared(3)
        fresh=TaskDatasetEvaluation(self.app,task,data,
            anchor=reference_anchor(receipt,expected_binding=receipt["binding"]),
            max_new_assessments=2,part_samples=2)
        before=self.backend.execute_calls
        first=fresh.advance();self.assertEqual(first["status"],"ready")
        self.assertEqual(first["summary"]["evaluated_samples"],2)
        final=fresh.advance();self.assertEqual(final["status"],"succeeded")
        self.assertEqual(final["summary"]["evaluated_samples"],3)
        self.assertEqual(self.backend.execute_calls,before+2)
    def test_invalid_partition_size_rejects_before_admitting_calls(self):
        e,data,task,_,_,receipt=self.prepared(3)
        for size in (0,17,True):
            with self.assertRaises(ValueError):
                TaskDatasetEvaluation(self.app,task,data,
                    anchor=reference_anchor(receipt,expected_binding=receipt["binding"]),part_samples=size)
if __name__=="__main__":unittest.main()
