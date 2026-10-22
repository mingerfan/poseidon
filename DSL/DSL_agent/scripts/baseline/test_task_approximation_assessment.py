"""Reference-only controls with injected source frames; no FHE/API."""
import copy,json,unittest
from unittest.mock import patch
import numpy as np
import test_task_native_assessment as native_fixture
from test_model_output_contract import model
from task_numerical_assessment import SOURCE_SELECTION
from benchmark_graph import digest
from task_approximation_assessment import TaskApproximationAssessment,SUBJECT,FORMAT

class ApproximationAssessmentTests(unittest.TestCase):
    setUp=native_fixture.NativeAssessmentTests.setUp
    ready=native_fixture.NativeAssessmentTests.ready
    selection=native_fixture.NativeAssessmentTests.selection
    frame=native_fixture.NativeAssessmentTests.frame

    def call(self,task,outcome,frame,*,selection=None,**kwargs):
        selector=self.selection() if selection is None else selection
        with patch("task_approximation_assessment.TaskSourceReference.reference",return_value=frame) as worker,\
             patch.object(self.app,"_source",return_value=getattr(self,"snapshot",{})),\
             patch.object(self.app,"execute",side_effect=AssertionError("FHE must not execute")):
            result=TaskApproximationAssessment(self.app).assess(task,self.samples,
                expected_package_binding=kwargs.pop("binding",outcome["package"]["binding"]),
                native_source=selector,**kwargs)
        return result,worker

    def test_same_reference_and_package_grade_retained(self):
        for level in ("compiled","numerical"):
            with self.subTest(level=level):
                task,provider,outcome=self.ready(level=level);frame=self.frame(task,outcome)
                result,worker=self.call(task,outcome,frame,
                    application_criterion=dict(kind="max_absolute_error",limit=0.))
                self.assertEqual(result["status"],"passed");self.assertEqual(result["format"],FORMAT)
                self.assertTrue(result["application_criterion_validated"])
                self.assertEqual(result["package_validation"]["level"],level)
                self.assertFalse(result["encrypted_execution"])
                self.assertFalse(result["ckks_validation_performed"])
                self.assertFalse(result["supplied_inputs_numerically_validated"])
                self.assertFalse(result["original_source_accuracy_certified"])
                self.assertEqual(self.backend.execute_calls,0)
                self.assertEqual(self.app.result(task),outcome);self.assertEqual(provider.calls,1)
                self.assertEqual(worker.call_count,1)
                self.assertEqual(result["binding"],digest({k:v for k,v in result.items()if k!="binding"}))

    def test_explicit_metric_failure_preserves_mathematical_difference(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome,delta=.25)
        result,_=self.call(task,outcome,frame,application_criterion=dict(kind="max_absolute_error",limit=.01))
        self.assertEqual(result["status"],"validation_failed")
        self.assertEqual(result["failure"]["code"],"assessment_application_criterion_failed")
        self.assertEqual(result["original_source_vs_approximate_comparison"]["max_absolute_error"],.25)
        self.assertFalse(result["application_criterion_validated"])

    def test_float32_source_precision_not_replaced(self):
        task,_,outcome=self.ready();self.samples=[dict(x=[.1,.2])]
        frame=self.frame(task,outcome,dtype="float32")
        result,_=self.call(task,outcome,frame,selection=self.selection(input_dtypes={"x":"float32"}),
            application_criterion=dict(kind="max_absolute_error",limit=0.))
        self.assertEqual(result["source_input_dtypes"],{"x":"float32"})
        self.assertEqual(result["status"],"validation_failed")
        self.assertGreater(result["application_criterion"]["max_absolute_error"],0.)

    def test_metric_absent_does_not_certify_approximation(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome,delta=.5)
        result,_=self.call(task,outcome,frame)
        self.assertEqual(result["status"],"passed")
        self.assertFalse(result["application_criterion_validated"])
        self.assertEqual(result["application_criterion"]["status"],"not_requested")
        self.assertFalse(result["original_source_vs_approximate_comparison"]["passed"])

    def test_top1_labels_describe_approximation_not_encrypted_result(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome,
            transform=lambda values:{"out":values["out"][::-1].copy()})
        result,_=self.call(task,outcome,frame,application_criterion=dict(kind="top1_accuracy_drop",
            output="out",class_axis=0,labels=[1],max_drop=0.))
        metric=result["application_criterion"]
        self.assertEqual(metric["subject"],SUBJECT)
        self.assertEqual(metric["original_source_top1"],1.)
        self.assertEqual(metric["approximate_top1"],0.)
        self.assertNotIn("encrypted_top1",metric)
        self.assertEqual(result["status"],"validation_failed")

    def test_wrong_binding_and_invalid_inputs_stop_before_source(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome)
        result,worker=self.call(task,outcome,frame,binding="0"*64)
        self.assertEqual(result["failure"]["code"],"assessment_binding_rejected")
        worker.assert_not_called()
        for invalid in ([],[dict(x=[True,0.])],[dict(x=[0.])],[dict(x=[float("nan"),0.])]):
            self.samples=invalid
            result,worker=self.call(task,outcome,frame)
            self.assertEqual(result["status"],"validation_failed");worker.assert_not_called()

    def test_none_and_unsafe_source_selection_stop_before_source(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome)
        adapter=TaskApproximationAssessment(self.app)
        with patch("task_approximation_assessment.TaskSourceReference.reference") as worker:
            result=adapter.assess(task,self.samples,expected_package_binding=outcome["package"]["binding"],
                native_source=None)
            self.assertEqual(result["failure"]["code"],"assessment_input_rejected");worker.assert_not_called()
        for selection in (True,{},self.selection(input_dtypes={"x":"int64"}),
                          self.selection(trust_local_code=1),self.selection(evidence_directory="relative")):
            result,worker=self.call(task,outcome,frame,selection=selection)
            self.assertEqual(result["status"],"validation_failed");worker.assert_not_called()

    def test_tampered_source_frame_and_shape_are_rejected(self):
        task,_,outcome=self.ready();original=self.frame(task,outcome)
        for changes in (dict(bundle_binding="0"*64),dict(supplied_input_sha256="0"*64),
                        dict(source_snapshot_binding="0"*64),dict(encrypted_execution=True)):
            frame=copy.deepcopy(original);frame.update(changes)
            frame["binding"]=digest({k:v for k,v in frame.items()if k!="binding"})
            result,_=self.call(task,outcome,frame)
            self.assertEqual(result["failure"]["code"],"assessment_source_reference_rejected")
        frame=copy.deepcopy(original);frame["source_reference"]["outputs"]=[{"out":[0.]}]
        frame["source_reference"]["binding"]=digest({k:v for k,v in frame["source_reference"].items()if k!="binding"})
        frame["binding"]=digest({k:v for k,v in frame.items()if k!="binding"})
        result,_=self.call(task,outcome,frame)
        self.assertEqual(result["status"],"service_error")

    def test_domain_guard_precedes_native_worker(self):
        graph,provenance=native_fixture.relu();task,_,outcome=self.ready(graph,provenance)
        self.samples=[dict(x=[9.123456789,0.])]
        result,worker=self.call(task,outcome,dict(status="unused"))
        self.assertEqual(result["failure"]["code"],"approximation_domain_rejected")
        self.assertNotIn("9.123456789",json.dumps(result));worker.assert_not_called()

    def test_invalid_metric_stops_before_source(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome)
        for value in (True,dict(kind="max_absolute_error",limit=True),
                      dict(kind="max_absolute_error",limit=float("inf")),
                      dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[True],max_drop=.01)):
            result,worker=self.call(task,outcome,frame,application_criterion=value)
            self.assertEqual(result["failure"]["code"],"assessment_input_rejected");worker.assert_not_called()

    def test_original_task_and_caller_selector_unchanged(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome);selector=self.selection()
        before=copy.deepcopy(selector)
        result,_=self.call(task,outcome,frame,selection=selector)
        self.assertEqual(selector,before);self.assertEqual(self.app.result(task),outcome)
        self.assertEqual(result["agent_calls"],0);self.assertEqual(result["compiler_calls"],0)

    def test_source_terminal_failure_kept_without_execution(self):
        task,_,outcome=self.ready()
        for status,code in (("timed_out","source_reference_timeout"),
                            ("validation_failed","source_reference_nondeterminism"),
                            ("service_error","source_reference_binding_changed")):
            result,_=self.call(task,outcome,dict(status=status,failure=dict(code=code,layer="source_reference")))
            self.assertEqual(result["status"],status);self.assertEqual(result["failure"]["code"],code)
            self.assertFalse(result["encrypted_execution"])

    def test_nonterminal_task_does_not_run_reference(self):
        task=self.app.submit(model())
        with patch("task_approximation_assessment.TaskSourceReference.reference") as worker:
            result=TaskApproximationAssessment(self.app).assess(task,self.samples,
                expected_package_binding="a"*64,native_source=self.selection())
            self.assertEqual(result["status"],"queued");worker.assert_not_called()

    def test_metadata_change_during_reference_is_rejected(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome)
        with patch("task_approximation_assessment.implementation",side_effect=["a"*64,"b"*64]),\
             patch("task_approximation_assessment.TaskSourceReference.reference",return_value=frame),\
             patch.object(self.app,"_source",return_value=self.snapshot):
            result=TaskApproximationAssessment(self.app).assess(task,self.samples,
                expected_package_binding=outcome["package"]["binding"],native_source=self.selection())
        self.assertEqual(result["failure"]["code"],"assessment_identity_changed")
        self.assertFalse(result["encrypted_execution"])

if __name__=="__main__":unittest.main()
