"""Numerical/source boundary controls; injected outputs, no FHE or paid calls."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import digest
from task_numerical_assessment import (TaskNumericalAssessment,SOURCE_SELECTION,NATIVE_FORMAT,
    NATIVE_POLICY,POLICY,mathematical,native_identity,SOURCE_FRAME_FORMAT,NATIVE_REPORT_FORMAT,
    source_binding,implementation)
from test_task_numerical_assessment import AssessmentTests,AssessmentBackend,relu
from test_model_output_contract import model

class NativeAssessmentTests(unittest.TestCase):
    setUp=AssessmentTests.setUp
    ready=AssessmentTests.ready
    def selection(self,**change):
        value=dict(format=SOURCE_SELECTION,input_dtypes={"x":"float64"},trust_local_code=True,
                   evidence_directory=str(Path(self.tmp.name)/"native"))
        value.update(change);return value
    def call(self,task,outcome,selection=None,**kwargs):
        return self.adapter.assess(task,self.samples,
            expected_package_binding=kwargs.pop("binding",outcome["package"]["binding"]),
            native_source=self.selection() if selection is None else selection,**kwargs)
    def frame(self,task,outcome,*,delta=0.,transform=None,dtype="float64"):
        metadata=self.app._checked_task_metadata(task);request=metadata["request"]
        outputs=[]
        for sample in self.samples:
            values=mathematical(request["model"],sample)
            values={n:np.asarray(v,dtype=dtype).astype(np.float64) for n,v in values.items()}
            if transform:values=transform(values)
            elif delta:values={n:v+delta for n,v in values.items()}
            outputs.append({n:v.tolist() for n,v in values.items()})
        native=dict(format=NATIVE_REPORT_FORMAT,status="evaluated",
            native_dtype_preserved=True,model_imported_in_host=False,encrypted_execution=False,
            application_accuracy_certified=False,agent_generated=False,
            source_identity=native_identity(),samples=len(outputs),outputs=outputs,
            output_dtypes={o["name"]:dtype for o in request["model"]["outputs"]},
            input_dtypes={"x":dtype})
        native["binding"]=digest(native)
        self.snapshot=dict(binding=digest({"controlled_source_fixture":task}))
        grade=dict(level=outcome["acceptance"]["level"],
            encrypted_execution=outcome["acceptance"]["level"]=="numerical",
            numerically_validated=outcome["acceptance"]["level"]=="numerical")
        frame=dict(format=SOURCE_FRAME_FORMAT,status="evaluated",source_reference_evaluated=True,
            task_id=task,task_status="succeeded",request_id=request["request_id"],
            task_metadata_binding=metadata["binding"],bundle_binding=outcome["package"]["binding"],
            package_validation=grade,source_binding=source_binding(metadata),
            source_snapshot_binding=self.snapshot["binding"],source_reference=native,
            supplied_input_sha256=digest(self.samples),reference_policy=NATIVE_POLICY,
            encrypted_execution=False,application_accuracy_certified=False,api_calls=0,compiler_calls=0,
            evidence=str(Path(self.tmp.name)/"native/native-reference.json"))
        frame["binding"]=digest(frame)
        return frame
    def controlled(self,task,outcome,frame,**kwargs):
        with patch("task_numerical_assessment.TaskSourceReference.reference",return_value=frame) as run,\
             patch.object(self.app,"_source",return_value=self.snapshot):
            result=self.call(task,outcome,**kwargs)
        return result,run
    def test_native_exact_outputs_share_existing_application_and_keep_grade(self):
        task,provider,outcome=self.ready()
        frame=self.frame(task,outcome)
        result,run=self.controlled(task,outcome,frame,application_criterion=dict(kind="max_absolute_error",limit=0.))
        self.assertEqual(result["format"],NATIVE_FORMAT);self.assertEqual(result["status"],"passed")
        self.assertEqual(result["reference_policy"],NATIVE_POLICY)
        self.assertTrue(result["ckks_comparison"]["passed"])
        self.assertTrue(result["source_application_criterion_validated"])
        self.assertFalse(result["source_application_validation_scope"]["dataset_accuracy_certified"])
        self.assertFalse(result["original_source_accuracy_certified"])
        self.assertEqual(result["source_forward_binding"],frame["source_reference"]["binding"])
        self.assertEqual(result["binding"],digest({k:v for k,v in result.items() if k!="binding"}))
        self.assertEqual(provider.calls,1);self.assertEqual(self.backend.execute_calls,1)
        self.assertEqual(self.app.result(task),outcome);self.assertEqual(run.call_count,1)
    def test_native_float32_rounding_is_preserved_in_source_metric(self):
        task,_,outcome=self.ready();self.samples=[dict(x=[.1,.2])]
        frame=self.frame(task,outcome,dtype="float32")
        result,_=self.controlled(task,outcome,frame,selection=self.selection(input_dtypes={"x":"float32"}),
            application_criterion=dict(kind="max_absolute_error",limit=0.))
        self.assertTrue(result["ckks_comparison"]["passed"])
        self.assertGreater(result["source_total_comparison"]["max_absolute_error"],0.)
        self.assertEqual(result["source_input_dtypes"],{"x":"float32"})
        self.assertEqual(result["source_output_dtypes"],{"out":"float32"})
        self.assertFalse(result["source_vs_registered_mathematics"]["passed"])
        self.assertEqual(result["failure"]["code"],"assessment_application_criterion_failed")
    def test_registered_original_is_not_substituted_for_native_source(self):
        for shape in ((2,),(1,2),(1,1,2)):
            with self.subTest(shape=shape):
                graph,provenance=relu(shape);task,_,outcome=self.ready(graph,provenance)
                self.samples=[dict(x=np.array([-.5,.5]).reshape(shape).tolist())]
                frame=self.frame(task,outcome,delta=.25)
                result,_=self.controlled(task,outcome,frame,
                    application_criterion=dict(kind="max_absolute_error",limit=.01))
                self.assertEqual(result["reference_policy"],NATIVE_POLICY)
                self.assertEqual(result["application_criterion"]["scope"],NATIVE_POLICY)
                self.assertEqual(result["source_total_comparison"]["max_absolute_error"],.25)
                self.assertEqual(result["approximation_comparison"]["max_absolute_error"],.125)
                self.assertFalse(result["source_vs_registered_mathematics"]["passed"])
                self.assertTrue(result["ckks_comparison"]["passed"])
                self.assertFalse(result["source_application_criterion_validated"])
    def test_native_top1_uses_actual_source_scores_and_never_hides_ckks(self):
        task,_,outcome=self.ready()
        frame=self.frame(task,outcome,transform=lambda values:{"out":values["out"][::-1].copy()})
        metric=dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[1],max_drop=0.)
        result,_=self.controlled(task,outcome,frame,application_criterion=metric)
        self.assertTrue(result["ckks_comparison"]["passed"])
        self.assertEqual(result["application_criterion"]["original_source_top1"],1.)
        self.assertEqual(result["application_criterion"]["encrypted_top1"],0.)
        self.assertEqual(result["application_criterion"]["accuracy_drop"],1.)
        self.assertNotIn("original_mathematical_top1",result["application_criterion"])
        self.assertEqual(result["failure"]["code"],"assessment_application_criterion_failed")
        self.backend.delta=.001
        result,_=self.controlled(task,outcome,frame,application_criterion=dict(
            kind="max_absolute_error",limit=10.))
        self.assertTrue(result["application_criterion"]["passed"])
        self.assertEqual(result["failure"]["code"],"assessment_numerical_mismatch")
    def test_no_metric_does_not_certify_native_application_accuracy(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome,delta=.5)
        result,_=self.controlled(task,outcome,frame)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["application_criterion"]["status"],"not_requested")
        self.assertFalse(result["source_application_criterion_validated"])
        self.assertFalse(result["source_total_comparison"]["passed"])
    def test_compiled_package_is_not_regraded(self):
        task,_,outcome=self.ready(level="compiled");frame=self.frame(task,outcome)
        result,_=self.controlled(task,outcome,frame,application_criterion=dict(kind="max_absolute_error",limit=0.))
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["package_validation"]["level"],"compiled")
        self.assertFalse(result["package_validation"]["numerically_validated"])
        self.assertEqual(self.app.result(task),outcome)
    def test_explicit_source_trust_and_existing_source_required_before_fhe(self):
        task,_,outcome=self.ready()
        result=self.call(task,outcome,selection=self.selection(trust_local_code=False))
        self.assertEqual(result["status"],"unsupported")
        self.assertEqual(result["failure"]["code"],"source_reference_trust_required")
        result=self.call(task,outcome)
        self.assertEqual(result["failure"]["code"],"source_reference_missing_source")
        self.assertEqual(self.backend.execute_calls,0)
    def test_nonterminal_states_do_not_run_source_or_fhe(self):
        task=self.app.submit(model())
        for expected in ("queued","cancelled"):
            with patch("task_numerical_assessment.TaskSourceReference.reference") as run:
                result=self.adapter.assess(task,self.samples,expected_package_binding="a"*64,
                    native_source=self.selection())
                self.assertEqual(result["status"],expected);run.assert_not_called()
            if expected=="queued":self.app.cancel(task)
        self.assertEqual(self.backend.execute_calls,0)
    def test_invalid_source_selector_is_controlled_and_detached(self):
        task,_,outcome=self.ready()
        for selection in (True,{},self.selection(format="unknown"),
            self.selection(input_dtypes={"x":"float16"}),self.selection(input_dtypes={"y":"float64"}),
            self.selection(trust_local_code=1),self.selection(evidence_directory="relative"),
            self.selection(extra=1)):
            with self.subTest(selection=selection),patch("task_numerical_assessment.TaskSourceReference.reference") as run:
                result=self.call(task,outcome,selection=selection)
                self.assertEqual(result["failure"]["code"],"assessment_input_rejected")
                run.assert_not_called()
        self.assertEqual(self.backend.execute_calls,0)
    def test_wrong_package_invalid_samples_or_metric_do_not_launch_source(self):
        task,_,outcome=self.ready()
        with patch("task_numerical_assessment.TaskSourceReference.reference") as run:
            out=self.call(task,outcome,binding="0"*64)
            self.assertEqual(out["failure"]["code"],"assessment_binding_rejected")
            self.samples=[dict(x=[True,.1])]
            self.assertEqual(self.call(task,outcome)["status"],"validation_failed")
            self.samples=[dict(x=[.1,.2])]
            out=self.call(task,outcome,application_criterion=dict(kind="max_absolute_error",limit=True))
            self.assertEqual(out["failure"]["code"],"assessment_input_rejected")
            run.assert_not_called()
        self.assertEqual(self.backend.execute_calls,0)
    def test_approximation_domain_rejection_precedes_source_and_fhe(self):
        graph,provenance=relu();task,_,outcome=self.ready(graph,provenance)
        self.samples=[dict(x=[9.123456789,0.])]
        with patch("task_numerical_assessment.TaskSourceReference.reference") as run:
            out=self.call(task,outcome);run.assert_not_called()
        self.assertEqual(out["failure"]["code"],"approximation_domain_rejected")
        self.assertNotIn("9.123456789",json.dumps(out));self.assertEqual(self.backend.execute_calls,0)
    def test_source_failure_terminal_is_preserved_without_fhe(self):
        task,_,outcome=self.ready()
        for status,code in (("timed_out","source_reference_timeout"),
                            ("validation_failed","source_reference_nondeterminism"),
                            ("service_error","source_reference_binding_changed")):
            frame=dict(status=status,failure=dict(code=code,layer="source_reference"))
            with patch("task_numerical_assessment.TaskSourceReference.reference",return_value=frame):
                out=self.call(task,outcome)
            self.assertEqual(out["status"],status);self.assertEqual(out["failure"]["code"],code)
            self.assertFalse(out["execution_requested"])
        self.assertEqual(self.backend.execute_calls,0)
    def test_tampered_source_frame_is_rejected_before_fhe_without_payload(self):
        task,_,outcome=self.ready();original=self.frame(task,outcome)
        for change in (dict(request_id="0"*64),dict(supplied_input_sha256="0"*64),
                       dict(source_snapshot_binding="0"*64),dict(bundle_binding="0"*64),
                       dict(api_calls=True),dict(encrypted_execution=True)):
            frame=copy.deepcopy(original);frame.update(change)
            frame["binding"]=digest({k:v for k,v in frame.items() if k!="binding"})
            result,_=self.controlled(task,outcome,frame)
            self.assertEqual(result["failure"]["code"],"assessment_source_reference_rejected")
        frame=copy.deepcopy(original);frame["source_reference"]["outputs"][0]["out"]=[9.123456789,0.]
        result,_=self.controlled(task,outcome,frame)
        self.assertEqual(result["status"],"service_error")
        self.assertNotIn("9.123456789",json.dumps(result))
        self.assertEqual(self.backend.execute_calls,0)
    def test_wrong_native_shapes_dtypes_or_count_are_rejected_before_fhe(self):
        task,_,outcome=self.ready();original=self.frame(task,outcome)
        for change in (dict(output_dtypes={"out":"int64"}),dict(samples=True),
            dict(outputs=[{"out":[0.]}]),dict(outputs=[{"other":[0.,0.]}]),dict(input_dtypes={"x":"float32"})):
            frame=copy.deepcopy(original);frame["source_reference"].update(change)
            frame["source_reference"]["binding"]=digest({k:v for k,v in frame["source_reference"].items() if k!="binding"})
            frame["binding"]=digest({k:v for k,v in frame.items() if k!="binding"})
            result,_=self.controlled(task,outcome,frame)
            self.assertEqual(result["failure"]["code"],"assessment_source_reference_rejected")
        self.assertEqual(self.backend.execute_calls,0)
    def test_source_snapshot_drift_during_execution_is_rejected(self):
        task,_,outcome=self.ready();frame=self.frame(task,outcome)
        self.backend.after_execute=lambda:self.snapshot.update(binding="0"*64)
        result,_=self.controlled(task,outcome,frame)
        self.assertEqual(result["status"],"service_error")
        self.assertEqual(result["failure"]["code"],"assessment_identity_changed")
    def test_reference_identity_distinguishes_native_and_mathematical_modes(self):
        self.assertNotEqual(implementation(),implementation(policy=NATIVE_POLICY))
    def test_repeated_assessment_does_not_regenerate_compile_or_mutate_source_selection(self):
        task,provider,outcome=self.ready();frame=self.frame(task,outcome);selection=self.selection()
        saved=copy.deepcopy(selection)
        first,_=self.controlled(task,outcome,frame,selection=selection)
        second,_=self.controlled(task,outcome,frame,selection=selection)
        self.assertEqual(first["binding"],second["binding"]);self.assertEqual(selection,saved)
        self.assertEqual(provider.calls,1);self.assertEqual(self.backend.execute_calls,2)
        self.assertEqual(self.app.result(task),outcome)
    def test_private_conversion_hooks_in_selector_are_not_executed(self):
        class Unsafe:
            def __eq__(self,value):raise AssertionError("compare hook")
            def __deepcopy__(self,memo):raise AssertionError("copy hook")
            def __array__(self,*args):raise AssertionError("array hook")
        task,_,outcome=self.ready()
        result=self.call(task,outcome,selection=self.selection(input_dtypes={"x":Unsafe()}))
        self.assertEqual(result["failure"]["code"],"assessment_input_rejected")
        self.assertEqual(self.backend.execute_calls,0)
if __name__=="__main__":unittest.main()
