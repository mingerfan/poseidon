"""Application assessment controls with injected executions; no FHE or API."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from application_component import Application
from benchmark_graph import digest
from benchmark_math import evaluate
from model_decomposition import FORMAT,decompose
from program_execution import normalize_samples
from task_numerical_assessment import TaskNumericalAssessment,mathematical
from test_application_component import Provider
from test_task_output_adapter import OutputBackend
from test_model_output_contract import model

class AssessmentBackend(OutputBackend):
    def __init__(self):
        super().__init__();self.after_execute=None;self.received=None;self.delta=0.
    def execute(self,folder,samples):
        self.execute_calls+=1;self.received=copy.deepcopy(samples)
        request,level=self.tasks[str(folder)]
        values=[{n:a.tolist() for n,a in evaluate(request["model"],s).items()}
                for s in normalize_samples(request,samples)]
        if self.delta:
            a=np.asarray(values[0][next(iter(values[0]))]);a.flat[0]+=self.delta
            values[0][next(iter(values[0]))]=a.tolist()
        result=dict(format="poseidon-local-execution-v1",status="executed",backend="poseidon_cpu",
            request_id=request["request_id"],bundle_binding=self.check_package(folder,level),
            package_validation=dict(level=level,encrypted_execution=level=="numerical",
                                    numerically_validated=level=="numerical"),
            encrypted_execution=True,executor_secret_key=False,samples=len(samples),outputs=values,
            agent_calls=0,compiler_calls=0,execution=dict(executor_secret_key=False),
            numerically_validated_on_supplied_inputs=False,application_accuracy_certified=False)
        if self.override:result.update(copy.deepcopy(self.override))
        if self.after_execute:self.after_execute()
        return result
def relu(shape=(2,)):
    original=dict(format=FORMAT,id="assessment_relu",inputs=[dict(name="x",shape=list(shape))],
        constants={},nodes=[dict(id="activation",op="relu",inputs=["x"],attrs=dict(
            approximation=dict(method="polynomial",basis="power",domain=[-1.,1.],
                               coefficients=[0.,.5,.5])),outputs=["y"])],
        outputs=[dict(name="out",value="y")])
    lowered,binding=decompose(original)
    provenance=dict(original=original,decomposition=binding,reference_report=dict(
        original_sha256=digest(original),model_sha256=digest(lowered),
        reference_status="passed",approximation_accuracy_certified=False))
    return lowered,provenance

class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.backend=AssessmentBackend();self.app=Application(Path(self.tmp.name)/"tasks",self.backend)
        self.adapter=TaskNumericalAssessment(self.app);self.samples=[dict(x=[-.5,.25])]
    def ready(self,graph=None,provenance=None,level="numerical"):
        task=self.app.submit(graph or model(),validation_level=level,model_provenance=provenance)
        provider=Provider();outcome=self.app.run(task,provider)
        self.assertEqual(outcome["status"],"succeeded")
        return task,provider,outcome
    def assess(self,task,outcome,**kwargs):
        return self.adapter.assess(task,self.samples,
            expected_package_binding=kwargs.pop("binding",outcome["package"]["binding"]),**kwargs)
    def test_three_shapes_compare_all_values_without_source_accuracy_claim(self):
        for shape in ((2,),(1,2),(1,1,2)):
            with self.subTest(shape=shape):
                task,provider,outcome=self.ready(model(shape))
                self.samples=[dict(x=np.array([-.5,.25]).reshape(shape))]
                result=self.assess(task,outcome)
                self.assertEqual(result["status"],"passed")
                self.assertTrue(result["supplied_inputs_numerically_validated"])
                self.assertEqual(result["ckks_comparison"]["compared_values"],2)
                self.assertEqual(result["ckks_comparison"]["atol"],1e-5)
                self.assertEqual(result["ckks_comparison"]["rtol"],1e-4)
                self.assertFalse(result["original_source_accuracy_certified"])
                self.assertFalse(result["formal_equivalence"])
                self.assertEqual(result["application_criterion"]["status"],"not_requested")
                self.assertEqual(provider.calls,1)
                self.assertEqual(self.app.result(task),outcome)
                self.assertEqual(result["binding"],digest({k:v for k,v in result.items() if k!="binding"}))
    def test_explicit_max_error_criterion_cannot_hide_ckks_failure(self):
        task,_,outcome=self.ready();self.backend.delta=.01
        result=self.assess(task,outcome,application_criterion=dict(kind="max_absolute_error",limit=1.))
        self.assertEqual(result["failure"]["code"],"assessment_numerical_mismatch")
        self.assertFalse(result["ckks_comparison"]["passed"])
        self.assertTrue(result["application_criterion"]["passed"])
        self.assertEqual(self.app.result(task),outcome)
    def test_frozen_threshold_boundary_is_inclusive(self):
        task,_,outcome=self.ready();self.samples=[dict(x=[0.,0.])]
        self.backend.delta=1e-5
        self.assertEqual(self.assess(task,outcome)["status"],"passed")
        self.backend.delta=float(np.nextafter(1e-5,np.inf))
        self.assertEqual(self.assess(task,outcome)["failure"]["code"],"assessment_numerical_mismatch")
    def test_approximation_and_ckks_errors_are_separate_in_three_contexts(self):
        for shape in ((2,),(1,2),(1,1,2)):
            with self.subTest(shape=shape):
                graph,provenance=relu(shape);task,_,outcome=self.ready(graph,provenance)
                self.samples=[dict(x=np.array([-.5,.5]).reshape(shape))]
                result=self.assess(task,outcome,application_criterion=dict(kind="max_absolute_error",limit=.01))
                self.assertEqual(result["status"],"validation_failed")
                self.assertEqual(result["failure"]["code"],"assessment_application_criterion_failed")
                self.assertTrue(result["ckks_comparison"]["passed"])
                self.assertTrue(result["decomposition_comparison"]["passed"])
                self.assertEqual(result["approximation_comparison"]["max_absolute_error"],.125)
                self.assertEqual(result["ckks_comparison"]["max_absolute_error"],0.)
                self.assertEqual(self.app.result(task),outcome)
    def test_no_application_criterion_does_not_impose_original_threshold(self):
        graph,provenance=relu();task,_,outcome=self.ready(graph,provenance)
        self.samples=[dict(x=[-.5,.5])]
        result=self.assess(task,outcome)
        self.assertEqual(result["status"],"passed")
        self.assertFalse(result["total_comparison"]["passed"])
        self.assertFalse(result["original_source_accuracy_certified"])
    def test_top1_drop_and_label_shapes_are_explicit(self):
        task,_,outcome=self.ready();self.samples=[dict(x=[-.5,.25]),dict(x=[.25,-.5])]
        criterion=dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[0,1],max_drop=.01)
        result=self.assess(task,outcome,application_criterion=criterion)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["application_criterion"]["encrypted_top1"],1.)
        self.assertEqual(result["application_criterion"]["labels_compared"],2)
        self.assertFalse(result["original_source_accuracy_certified"])
    def test_top1_success_never_replaces_elementwise_comparison(self):
        task,_,outcome=self.ready();self.backend.delta=.1
        result=self.assess(task,outcome,application_criterion=dict(
            kind="top1_accuracy_drop",output="out",class_axis=-1,labels=[0],max_drop=0.))
        self.assertTrue(result["application_criterion"]["passed"])
        self.assertFalse(result["ckks_comparison"]["passed"])
        self.assertEqual(result["status"],"validation_failed")
    def test_compiled_package_grade_is_not_rewritten(self):
        task,_,outcome=self.ready(level="compiled")
        result=self.assess(task,outcome)
        self.assertEqual(result["status"],"passed")
        self.assertTrue(result["supplied_inputs_numerically_validated"])
        self.assertEqual(result["package_validation"]["level"],"compiled")
        self.assertFalse(result["package_validation"]["numerically_validated"])
        self.assertEqual(self.app.result(task),outcome)
    def test_repeat_and_cached_task_do_not_regenerate_or_compile(self):
        first,provider,original=self.ready()
        task=self.app.submit(model());unused=Provider();outcome=self.app.run(task,unused)
        self.assertTrue(outcome["package"]["reused"])
        a=self.assess(task,outcome);b=self.assess(task,outcome)
        self.assertEqual(a["binding"],b["binding"])
        self.assertEqual(provider.calls,1);self.assertEqual(unused.calls,0)
        self.assertEqual(self.backend.calls,1);self.assertEqual(self.backend.execute_calls,2)
        self.assertEqual(self.app.result(first),original)
    def test_cancelled_and_pending_tasks_do_not_execute(self):
        pending=self.app.submit(model())
        self.assertEqual(self.adapter.assess(pending,self.samples,expected_package_binding="a"*64)["status"],"queued")
        self.app.cancel(pending)
        self.assertEqual(self.adapter.assess(pending,self.samples,expected_package_binding="a"*64)["status"],"cancelled")
        self.assertEqual(self.backend.execute_calls,0)
    def test_wrong_package_binding_stops_before_execution(self):
        task,_,outcome=self.ready()
        result=self.assess(task,outcome,binding="0"*64)
        self.assertEqual(result["failure"]["code"],"assessment_binding_rejected")
        self.assertEqual(self.backend.execute_calls,0)
    def test_domain_rejection_stops_before_execution_and_redacts_values(self):
        graph,provenance=relu();task,_,outcome=self.ready(graph,provenance)
        self.samples=[dict(x=[9.123456789,0.])]
        result=self.assess(task,outcome)
        self.assertEqual(result["failure"]["code"],"approximation_domain_rejected")
        self.assertEqual(self.backend.execute_calls,0)
        self.assertNotIn("9.123456789",json.dumps(result))
    def test_invalid_samples_stop_before_execution(self):
        task,_,outcome=self.ready()
        for sample in (dict(y=[0.,0.]),dict(x=[True,0.]),dict(x=[np.nan,0.]),dict(x=[0.])):
            self.samples=[sample];self.assertEqual(self.assess(task,outcome)["status"],"validation_failed")
        self.assertEqual(self.backend.execute_calls,0)
    def test_bad_metrics_do_not_execute(self):
        task,_,outcome=self.ready()
        for value in (
            dict(kind="max_absolute_error",limit=True),dict(kind="max_absolute_error",limit=-1),
            dict(kind="max_absolute_error",limit=np.nan),dict(kind="max_absolute_error",limit=1,extra=0),
            dict(kind="unknown"),dict(kind="top1_accuracy_drop",output="missing",class_axis=0,labels=[0],max_drop=.01),
            dict(kind="top1_accuracy_drop",output="out",class_axis=True,labels=[0],max_drop=.01),
            dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[True],max_drop=.01),
            dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[2],max_drop=.01),
            dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[],max_drop=.01)):
            result=self.assess(task,outcome,application_criterion=value)
            self.assertEqual(result["failure"]["code"],"assessment_input_rejected")
        self.assertEqual(self.backend.execute_calls,0)
    def test_conversion_hooks_in_criterion_are_not_executed(self):
        class Unsafe:
            def __deepcopy__(self,memo):raise AssertionError("copy hook")
            def __eq__(self,value):raise AssertionError("comparison hook")
            def __array__(self,*a,**kw):raise AssertionError("array hook")
        task,_,outcome=self.ready()
        result=self.assess(task,outcome,application_criterion=dict(kind=Unsafe()))
        self.assertEqual(result["failure"]["code"],"assessment_input_rejected")
        self.assertEqual(self.backend.execute_calls,0)
    def test_sample_snapshot_is_detached_from_caller_buffers(self):
        task,_,outcome=self.ready();array=np.array([-.5,.25]);self.samples=[dict(x=array)]
        self.backend.after_execute=lambda:array.fill(9.)
        result=self.assess(task,outcome)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(self.backend.received,[dict(x=[-.5,.25])])
    def test_sixteen_samples_and_seventeen_rejection(self):
        task,_,outcome=self.ready();self.samples=[dict(x=[.1,.2])]*16
        self.assertEqual(self.assess(task,outcome)["samples"],16)
        self.samples*=2
        self.assertEqual(self.assess(task,outcome)["status"],"validation_failed")
        self.assertEqual(self.backend.execute_calls,1)
    def test_wrong_execution_identity_and_private_key_flag_are_rejected(self):
        task,_,outcome=self.ready()
        for override in (dict(request_id="0"*64),dict(bundle_binding="0"*64),dict(backend="seal_cpu"),
                         dict(executor_secret_key=True),dict(execution=dict(executor_secret_key=True)),
                         dict(samples=True),dict(samples=2),dict(agent_calls=1),dict(compiler_calls=1)):
            self.backend.override=override
            result=self.assess(task,outcome)
            self.assertEqual(result["failure"]["code"],"assessment_execution_rejected")
            self.assertTrue(result["execution_requested"])
    def test_bad_outputs_are_rejected_without_disclosing_them(self):
        task,_,outcome=self.ready()
        for override in (dict(outputs=[dict(secret=[9.123456789,0.])]),
                         dict(outputs=[dict(out=[float("inf"),0.])]),
                         dict(outputs=[dict(out=[0.])])):
            self.backend.override=override
            result=self.assess(task,outcome)
            self.assertEqual(result["status"],"validation_failed")
            self.assertNotIn("9.123456789",json.dumps(result))
    def test_original_execution_failure_stays_a_failure(self):
        task,_,outcome=self.ready()
        self.backend.override=dict(status="validation_failed",failure=dict(code="approximation_domain_rejected",layer="input"),
                                   encrypted_execution=False)
        result=self.assess(task,outcome)
        self.assertEqual(result["status"],"validation_failed")
        self.assertFalse(result["supplied_inputs_numerically_validated"])
        self.assertEqual(result["execution"]["failure"]["code"],"approximation_domain_rejected")
    def test_dependency_drift_after_execution_is_rejected(self):
        task,_,outcome=self.ready();self.backend.after_execute=lambda:setattr(self.backend,"version",2)
        result=self.assess(task,outcome)
        self.assertEqual(result["failure"]["code"],"assessment_identity_changed")
        self.assertEqual(result["status"],"service_error")
    def test_reference_mismatch_stops_before_execution(self):
        graph,provenance=relu();task,_,outcome=self.ready(graph,provenance)
        original=mathematical
        def wrong(g,inputs,**kwargs):
            value=original(g,inputs,**kwargs)
            if g is provenance["original"]:raise AssertionError("identity shortcut")
            if g.get("format")==FORMAT and not kwargs.get("ideal",False):
                return {n:a+1. for n,a in value.items()}
            return value
        with patch("task_numerical_assessment.mathematical",side_effect=wrong):
            result=self.assess(task,outcome)
        self.assertEqual(result["failure"]["code"],"assessment_decomposition_mismatch")
        self.assertEqual(self.backend.execute_calls,0)
