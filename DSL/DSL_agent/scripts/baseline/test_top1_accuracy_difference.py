"""Metric and receipt/controller regressions; injected backends, no native/API."""
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from top1_accuracy_metrics import threshold
from component_contract import prepare_task
from task_numerical_assessment import criterion,AssessmentError
from task_dataset_assessment import part_criterion
from test_model_output_contract import model
from assessment_array_dataset import export_dataset,load_dataset

class DifferenceMetricTests(unittest.TestCase):
    def test_absolute_change_checks_improvement_and_loss(self):
        for original,compared in ((0,1),(1,0)):
            result=threshold("top1_accuracy_difference",original,compared,1,.01)
            self.assertFalse(result["passed"]);self.assertEqual(result["accuracy_difference"],1.)
        self.assertEqual(threshold("top1_accuracy_drop",0,1,1,.01),
                         dict(accuracy_drop=-1.,max_drop=.01,passed=True))

    def test_one_percentage_point_boundary_uses_exact_counts(self):
        for original,compared in ((99,100),(100,99)):
            self.assertTrue(threshold("top1_accuracy_difference",original,compared,100,.01)["passed"])
            self.assertFalse(threshold("top1_accuracy_difference",original,compared,100,
                                       float(np.nextafter(.01,0.)))["passed"])
        self.assertFalse(threshold("top1_accuracy_difference",98,100,100,.01)["passed"])

    def test_invalid_counts_and_unknown_metric_are_rejected(self):
        for args in ((True,0,1,.1),(0,2,1,.1),(0,0,0,.1),(0,0,1,True)):
            with self.assertRaises(ValueError):threshold("top1_accuracy_difference",*args)
        with self.assertRaises(ValueError):threshold("top1_unknown",0,0,1,.1)

    def test_typed_criterion_and_part_labels_preserve_new_kind(self):
        value=dict(kind="top1_accuracy_difference",output="out",class_axis=-1,
                   labels=[0,1,0],max_difference=.01)
        checked=criterion(value,model(),3);self.assertEqual(checked["class_axis"],0)
        self.assertEqual(part_criterion(checked,1,2,model())["labels"],[1,0])
        self.assertEqual(value["class_axis"],-1)
        for edit in (lambda x:x.update(max_drop=.01),lambda x:x.update(max_difference=True)):
            bad=copy.deepcopy(value);edit(bad)
            with self.assertRaises(AssessmentError):criterion(bad,model(),3)

    def test_bound_array_labels_expand_absolute_difference(self):
        q=prepare_task(model())
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp)/"dataset"
            manifest=export_dataset(q,{"x":np.array([[.1,.2],[.3,.4]])},folder,
                labels=dict(output="out",class_axis=0,values=np.array([0,1],dtype=np.int64)))
            dataset=load_dataset(q,folder/"dataset.json",expected_binding=manifest["binding"])
            profile=dataset.criterion(dict(kind="top1_accuracy_difference",max_difference=.01))
            self.assertEqual(profile,dict(kind="top1_accuracy_difference",output="out",
                class_axis=0,labels=[0,1],max_difference=.01))

import test_approximation_dataset_assessment as approximation_fixtures
class DifferenceApproximationDatasetTests(unittest.TestCase):
    setUp=approximation_fixtures.ApproximationDatasetTests.setUp
    collector=approximation_fixtures.ApproximationDatasetTests.collector

    def record_arrays(self,collector,name,approximate,original):
        value=approximation_fixtures.receipt(collector,name,np.asarray(approximate,dtype=np.float64),
                      np.asarray(original,dtype=np.float64))
        collector.record(name,value,expected_binding=value["binding"])

    def test_difference_is_taken_after_aggregating_signed_correct_counts(self):
        metric=dict(kind="top1_accuracy_difference",output="out",class_axis=0,
                    labels=[1]*16,max_difference=.01)
        c=self.collector(sizes=(8,8),metric=metric)
        self.record_arrays(c,"part_0",[[0.,1.]]*8,[[1.,0.]]*8)
        self.assertIsNone(c.summary()["application_criterion"]["passed"])
        self.record_arrays(c,"part_1",[[1.,0.]]*8,[[0.,1.]]*8)
        result=c.summary();self.assertTrue(result["application_criterion_validated"])
        self.assertEqual(result["application_criterion"]["accuracy_difference"],0.)
        self.assertEqual(result["part_criterion_failures"],2)

    def test_unequal_part_improvement_does_not_pass_absolute_limit(self):
        metric=dict(kind="top1_accuracy_difference",output="out",class_axis=0,
                    labels=[1]+[0]*16,max_difference=.05)
        c=self.collector(sizes=(1,16),metric=metric)
        self.record_arrays(c,"part_0",[[0.,1.]],[[1.,0.]])
        self.record_arrays(c,"part_1",[[1.,0.]]*16,[[1.,0.]]*16)
        result=c.summary();self.assertEqual(result["status"],"failed")
        self.assertAlmostEqual(result["application_criterion"]["accuracy_difference"],1/17)
        self.assertFalse(result["encrypted_execution"])

import test_task_dataset_assessment as dataset_fixtures
from test_task_numerical_assessment import relu
class DifferenceNumericalDatasetTests(unittest.TestCase):
    setUp=dataset_fixtures.DatasetTests.setUp
    ready=dataset_fixtures.DatasetTests.ready
    batches=dataset_fixtures.DatasetTests.batches
    prepared=dataset_fixtures.DatasetTests.prepared
    record_all=dataset_fixtures.DatasetTests.record_all

    def test_application_failure_keeps_independent_numerical_pass(self):
        graph,provenance=relu();batches=self.batches((1,2))
        for batch in batches:batch["samples"]=[dict(x=[-.5,-.8])]*len(batch["samples"])
        c,rs,bs,_,_,outcome=self.prepared(batches,graph=graph,provenance=provenance,
            metric=dict(kind="top1_accuracy_difference",output="out",class_axis=0,
                        labels=[1]*3,max_difference=.01))
        result=self.record_all(c,rs,bs)
        self.assertEqual(result["status"],"validation_failed")
        self.assertTrue(result["supplied_inputs_numerically_validated"])
        self.assertEqual(result["application_criterion"]["accuracy_difference"],1.)
        self.assertEqual(outcome["status"],"succeeded")

import test_task_numerical_assessment as assessment_fixtures
from application_assessment import ApplicationAssessment
class DifferencePersistentControllerTests(unittest.TestCase):
    setUp=assessment_fixtures.AssessmentTests.setUp
    ready=assessment_fixtures.AssessmentTests.ready

    def test_unmet_absolute_criterion_is_terminal_and_never_repairs_program(self):
        graph,provenance=relu();task,provider,outcome=self.ready(graph,provenance)
        q=self.app._checked_task_metadata(task)["request"];folder=Path(self.tmp.name)/"dataset"
        manifest=export_dataset(q,{"x":np.array([[-.5,-.8]]*3)},folder,
            labels=dict(output="out",class_axis=0,values=np.array([1]*3,dtype=np.int64)))
        service=ApplicationAssessment(self.app,Path(self.tmp.name)/"assessment")
        handle=service.submit(task,folder/"dataset.json",expected_dataset_binding=manifest["binding"],
            application_criterion=dict(kind="top1_accuracy_difference",max_difference=.01),
            part_samples=2,max_new_assessments=2)
        result=service.advance(handle,max_new_parts=2)
        self.assertEqual(result["status"],"validation_failed")
        self.assertEqual(result["failure"]["code"],"assessment_application_criterion_failed")
        calls=self.backend.execute_calls
        self.assertEqual(service.advance(handle)["binding"],result["binding"])
        self.assertEqual(service.result(handle)["binding"],result["binding"])
        self.assertEqual(self.backend.execute_calls,calls);self.assertEqual(provider.calls,1)
        self.assertEqual(self.app.result(task),outcome)

if __name__=="__main__":unittest.main()
