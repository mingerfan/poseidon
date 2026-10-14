"""Host-pinned receipt controls; synthetic references, no native/FHE/API."""
import copy,unittest
from unittest.mock import patch
import numpy as np
from benchmark_graph import digest
from test_model_output_contract import model
from program_execution import normalize_samples
from numerical_results import compare
from task_numerical_assessment import ATOL,RTOL,REFERENCE_ATOL,REFERENCE_RTOL,NATIVE_POLICY
from task_approximation_assessment import FORMAT as RECEIPT,SUBJECT,approximation_metric
from task_dataset_assessment import DatasetAssessmentError,part_criterion,shaped
from approximation_dataset_assessment import ApproximationDatasetAssessment

def anchor(request):
    return dict(request_id=request["request_id"],bundle_binding="b"*64,task_id="a"*32,
        task_metadata_binding="c"*64,source_binding="d"*64,source_snapshot_binding="e"*64,
        package_validation=dict(level="numerical",encrypted_execution=True,numerically_validated=True),
        reference_policy=NATIVE_POLICY,reference_identity_sha256="9"*64,source_input_dtypes={"x":"float64"})
def receipt(collector,name,actual,original):
    part=collector.parts[name];profile=part_criterion(collector.declaration["application_criterion"],
        part["start"],part["samples"],collector.request["model"])
    approximate=shaped(actual,collector.request["model"]);native=shaped(original,collector.request["model"])
    metric=approximation_metric(profile,native,approximate,collector.request["model"])
    record=dict(collector.anchor,format=RECEIPT,status="validation_failed"if metric["passed"]is False else "passed",
        task_status="succeeded",application_metric_subject=SUBJECT,source_reference_evaluated=True,
        source_reference_requested=True,encrypted_execution=False,execution_requested=False,
        ckks_validation_performed=False,supplied_inputs_numerically_validated=False,
        original_source_accuracy_certified=False,formal_equivalence=False,
        samples=part["samples"],supplied_input_sha256=part["supplied_input_sha256"],
        criterion_sha256=part["criterion_sha256"],agent_calls=0,compiler_calls=0,
        source_reference_binding="1"*64,source_forward_binding=digest({"part":name}),
        source_selection_sha256="2"*64,source_output_dtypes={"out":"float64"},
        original_source_vs_approximate_comparison=compare(actual,original,ATOL,RTOL),
        decomposition_comparison=compare(actual,actual,REFERENCE_ATOL,REFERENCE_RTOL),
        application_criterion=metric,application_criterion_validated=metric["passed"]is True,
        failure=None if metric["passed"]is not False else dict(code="assessment_application_criterion_failed"))
    record["binding"]=digest(record);return record

class ApproximationDatasetTests(unittest.TestCase):
    def setUp(self):
        for target,value in (("approximation_dataset_assessment.reference_identity","9"*64),
                             ("approximation_dataset_assessment.implementation","8"*64)):
            p=patch(target,return_value=value);p.start();self.addCleanup(p.stop)
        self.request=dict(request_id="f"*64,model=model())
    def collector(self,sizes=(16,1),metric=None):
        batches=[];start=0
        for index,n in enumerate(sizes):
            batches.append(dict(id="part_"+str(index),sample_ids=["s"+str(i)for i in range(start,start+n)],
                samples=[dict(x=[.2,.3])for _ in range(n)]));start+=n
        return ApproximationDatasetAssessment(self.request,batches,anchor=anchor(self.request),application_criterion=metric)
    def record(self,collector,name,error=0.):
        n=collector.parts[name]["samples"];actual=np.tile([.04,.09],(n,1))
        value=receipt(collector,name,actual,actual+error)
        collector.record(name,value,expected_binding=value["binding"]);return value
    def reject(self,call,code=None):
        with self.assertRaises(DatasetAssessmentError)as caught:call()
        if code:self.assertEqual(caught.exception.code,code)

    def test_partial_and_failed_parts_never_become_complete(self):
        c=self.collector(metric=dict(kind="max_absolute_error",limit=.01))
        self.record(c,"part_0",.2);state=c.summary()
        self.assertFalse(state["complete"]);self.assertEqual(state["status"],"incomplete")
        self.assertIsNone(state["application_criterion"]["passed"])
        self.assertFalse(state["application_criterion_validated"])
        self.record(c,"part_1");state=c.summary()
        self.assertEqual(state["status"],"failed");self.assertEqual(state["part_criterion_failures"],1)
        self.assertFalse(state["encrypted_execution"]);self.assertFalse(state["ckks_validation_performed"])

    def test_maximum_and_mae_are_weighted_by_actual_values(self):
        c=self.collector(metric=dict(kind="max_absolute_error",limit=.25))
        self.record(c,"part_0",.125);self.record(c,"part_1",.25)
        state=c.summary()
        self.assertAlmostEqual(state["maximum_absolute_error"],.25)
        self.assertAlmostEqual(state["mae"],(16*.125+.25)/17)
        self.assertTrue(state["application_criterion_validated"])
        self.assertEqual(state["evaluated_samples"],17);self.assertEqual(state["compared_values"],34)

    def test_identical_receipt_is_idempotent_and_conflict_is_rejected(self):
        c=self.collector();r=self.record(c,"part_0")
        c.record("part_0",r,expected_binding=r["binding"])
        self.assertEqual(c.summary()["evaluated_samples"],16)
        bad=copy.deepcopy(r);bad["source_forward_binding"]="3"*64
        bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
        self.reject(lambda:c.record("part_0",bad,expected_binding=bad["binding"]),"dataset_part_conflict")

    def test_identity_input_and_criterion_cannot_mix(self):
        c=self.collector(metric=dict(kind="max_absolute_error",limit=.1))
        original=receipt(c,"part_0",np.zeros((16,2)),np.zeros((16,2)))
        for changes in (dict(reference_identity_sha256="0"*64),dict(supplied_input_sha256="0"*64),
                        dict(criterion_sha256="0"*64),dict(bundle_binding="0"*64),dict(samples=True)):
            r=copy.deepcopy(original);r.update(changes);r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
            self.reject(lambda:c.record("part_0",r,expected_binding=r["binding"]))

    def test_wrong_errors_metrics_and_encrypted_flags_are_rejected(self):
        c=self.collector(metric=dict(kind="max_absolute_error",limit=.1))
        original=receipt(c,"part_0",np.zeros((16,2)),np.zeros((16,2)))
        for path,value in (("encrypted_execution",True),("ckks_validation_performed",True),
                           ("application_criterion_validated",False),("application_metric_subject","encrypted")):
            r=copy.deepcopy(original);r[path]=value;r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
            self.reject(lambda:c.record("part_0",r,expected_binding=r["binding"]))
        for field in ("mae","max_absolute_error"):
            r=copy.deepcopy(original);r["original_source_vs_approximate_comparison"][field]=.2
            r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
            self.reject(lambda:c.record("part_0",r,expected_binding=r["binding"]))

    def test_output_dtype_is_fixed_across_parts(self):
        c=self.collector();self.record(c,"part_0")
        r=receipt(c,"part_1",np.zeros((1,2)),np.zeros((1,2)))
        r["source_output_dtypes"]={"out":"float32"};r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
        self.reject(lambda:c.record("part_1",r,expected_binding=r["binding"]),"dataset_binding_rejected")

    def test_source_or_collector_identity_drift_stops_summary(self):
        c=self.collector()
        with patch("approximation_dataset_assessment.reference_identity",return_value="0"*64):
            self.reject(c.summary,"dataset_reference_changed")
        with patch("approximation_dataset_assessment.implementation",return_value="0"*64):
            self.reject(c.summary,"dataset_reference_changed")

    def test_no_metric_means_no_application_accuracy_claim(self):
        c=self.collector(sizes=(1,));self.record(c,"part_0",.25)
        state=c.summary();self.assertEqual(state["status"],"passed")
        self.assertFalse(state["application_criterion_validated"])
        self.assertFalse(state["original_source_accuracy_certified"])

    def test_top1_uses_counts_and_boundary_for_ten_thousand_synthetic_rows(self):
        c=self.collector(sizes=(16,)*625,metric=dict(kind="top1_accuracy_drop",output="out",
            class_axis=0,labels=[1]*10000,max_drop=.01))
        remaining_wrong=100
        for name,part in c.parts.items():
            original=np.tile([0.,1.],(part["samples"],1));actual=original.copy()
            wrong=min(part["samples"],remaining_wrong)
            actual[:wrong]=[1.,0.];remaining_wrong-=wrong
            r=receipt(c,name,actual,original);c.record(name,r,expected_binding=r["binding"])
        state=c.summary();metric=state["application_criterion"]
        self.assertTrue(state["complete"]);self.assertTrue(state["application_criterion_validated"])
        self.assertEqual(metric["accuracy_drop"],.01);self.assertEqual(metric["original_correct"],10000)
        self.assertEqual(metric["approximate_correct"],9900)
        self.assertGreater(state["part_criterion_failures"],0)

    def test_top1_tail_is_not_equal_weight_to_full_part(self):
        c=self.collector(sizes=(16,1),metric=dict(kind="top1_accuracy_drop",output="out",
            class_axis=0,labels=[1]*17,max_drop=.06))
        original=np.tile([0.,1.],(16,1));r=receipt(c,"part_0",original,original)
        c.record("part_0",r,expected_binding=r["binding"])
        r=receipt(c,"part_1",np.array([[1.,0.]]),np.array([[0.,1.]]))
        c.record("part_1",r,expected_binding=r["binding"])
        metric=c.summary()["application_criterion"]
        self.assertAlmostEqual(metric["accuracy_drop"],1/17);self.assertTrue(metric["passed"])

    def test_owned_request_and_declaration_mutation_are_rejected(self):
        c=self.collector();c.request["model"]["inputs"][0]["shape"]=[1]
        self.reject(c.summary,"dataset_reference_changed")
        c=self.collector();c.declaration["parts"][0]["samples"]=1
        self.reject(c.summary,"dataset_reference_changed")

if __name__=="__main__":unittest.main()
