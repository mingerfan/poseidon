"""Dataset controls over injected Application outputs; no real FHE or API."""
import copy,json,unittest
from unittest.mock import patch
from benchmark_graph import digest
from task_dataset_assessment import (DatasetNumericalAssessment,DatasetAssessmentError,
    declare_dataset,reference_anchor,part_criterion)
from task_numerical_assessment import NATIVE_POLICY,NATIVE_FORMAT
from test_task_numerical_assessment import AssessmentTests,relu
from test_task_native_assessment import NativeAssessmentTests
from test_model_output_contract import model
class DatasetTests(unittest.TestCase):
    setUp=AssessmentTests.setUp
    ready=AssessmentTests.ready
    def batches(self,counts=(1,2)):
        start=0;out=[]
        for j,n in enumerate(counts):
            out.append(dict(id="part_"+str(j),sample_ids=["sample_"+str(i) for i in range(start,start+n)],
                            samples=[dict(x=[.25,-.5]) for _ in range(n)]));start+=n
        return out
    def prepared(self,batches=None,metric=None,graph=None,provenance=None,delta=0.):
        batches=batches or self.batches()
        if graph is not None and len(graph["outputs"])>1:
            from test_application_component import Provider
            task=self.app.submit(graph,model_provenance=provenance)
            # The scripted source must return one expression for each output.
            provider=Provider(variants=[",".join(["x*x"]*len(graph["outputs"]))])
            outcome=self.app.run(task,provider)
            self.assertEqual(outcome["status"],"succeeded",outcome.get("failure"))
        else:
            task,_,outcome=self.ready(graph,provenance)
        request=self.app._checked_task_metadata(task)["request"]
        from task_numerical_assessment import criterion
        global_profile=criterion(metric,request["model"],sum(len(b["samples"]) for b in batches))
        records=[];start=0
        for b in batches:
            self.backend.delta=delta
            result=self.adapter.assess(task,b["samples"],expected_package_binding=outcome["package"]["binding"],
                application_criterion=part_criterion(global_profile,start,len(b["samples"]),request["model"]))
            records.append(result);start+=len(b["samples"])
        self.backend.delta=0.
        anchor=reference_anchor(records[0],expected_binding=records[0]["binding"])
        declaration=declare_dataset(request,batches,anchor=anchor,application_criterion=metric)
        collector=DatasetNumericalAssessment(request,declaration,expected_declaration_binding=declaration["binding"])
        return collector,records,batches,request,declaration,outcome
    def record_all(self,collector,records,batches):
        for r,b in zip(records,batches):collector.record(b["id"],r,expected_binding=r["binding"])
        return collector.summary()
    def error(self,code,call):
        with self.assertRaises(DatasetAssessmentError) as ctx:call()
        self.assertEqual(ctx.exception.code,code)
    def test_complete_disjoint_dataset_compares_all_values_without_new_execution(self):
        c,rs,bs,_,d,_=self.prepared(self.batches((1,8,16)))
        calls=self.backend.execute_calls;result=self.record_all(c,rs,bs)
        self.assertEqual(result["status"],"passed");self.assertEqual(result["planned_samples"],25)
        self.assertEqual(result["evaluated_samples"],25);self.assertEqual(result["numerically_validated_samples"],25)
        self.assertEqual(result["part_status_counts"],{"passed":3})
        self.assertEqual(result["ckks_comparison"]["compared_values"],50)
        self.assertEqual(result["new_encrypted_calls"],0);self.assertEqual(self.backend.execute_calls,calls)
        self.assertFalse(result["dataset_criterion_validated"]);self.assertFalse(result["original_source_accuracy_certified"])
        self.assertEqual(result["binding"],digest({k:v for k,v in result.items() if k!="binding"}))
        self.assertNotIn('"x"',json.dumps(d))
    def test_missing_part_never_becomes_dataset_success(self):
        c,rs,bs,*_=self.prepared(metric=dict(kind="max_absolute_error",limit=0.))
        c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"])
        r=c.summary();self.assertEqual(r["status"],"incomplete");self.assertFalse(r["complete"])
        self.assertEqual(r["application_criterion"]["status"],"incomplete")
        self.assertFalse(r["supplied_inputs_numerically_validated"])
        self.assertEqual(r["parts"][1]["status"],"not_run")
    def test_identical_receipt_is_idempotent_and_counts_once(self):
        c,rs,bs,*_=self.prepared();self.record_all(c,rs,bs)
        r=c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"])
        self.assertEqual(r["status"],"already_recorded")
        self.assertEqual(c.summary()["evaluated_samples"],3)
    def test_same_receipt_cannot_inflate_two_different_partitions(self):
        c,rs,bs,*_=self.prepared(self.batches((1,1)))
        self.assertEqual(rs[0]["binding"],rs[1]["binding"])
        c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"])
        self.error("dataset_part_conflict",lambda:c.record(bs[1]["id"],rs[1],expected_binding=rs[1]["binding"]))
        self.assertEqual(c.summary()["evaluated_samples"],1)
        self.assertEqual(c.summary()["numerical_receipts"],1)
    def test_conflicting_receipt_does_not_replace_historical_result(self):
        c,rs,bs,*_=self.prepared();self.record_all(c,rs,bs)
        r=copy.deepcopy(rs[0]);r["task_id"]="e"*32;r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        self.error("dataset_part_conflict",lambda:c.record(bs[0]["id"],r,expected_binding=r["binding"]))
        self.assertEqual(c.summary()["parts"][0]["receipt_binding"],rs[0]["binding"])
    def test_unequal_top1_parts_are_weighted_and_failed_part_is_preserved(self):
        graph,provenance=relu();bs=self.batches((1,8,16))
        bs[0]["samples"]=[dict(x=[-.5,-.25])]
        bs[1]["samples"]=[dict(x=[.5,.25])]*8;bs[2]["samples"]=[dict(x=[.5,.25])]*16
        profile=dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[0]*25,max_drop=.1)
        c,rs,bs,*_=self.prepared(bs,profile,graph,provenance)
        self.assertEqual(rs[0]["status"],"validation_failed")
        result=self.record_all(c,rs,bs)
        self.assertEqual(result["status"],"passed")
        metric=result["application_criterion"]
        self.assertEqual(metric["accuracy_drop"],.04);self.assertEqual(metric["encrypted_top1"],.96)
        self.assertEqual(metric["counts"]["labels_compared"],25)
        self.assertEqual(result["part_application_failures"],1)
        self.assertEqual(result["parts"][0]["status"],"validation_failed")
        self.assertTrue(result["dataset_criterion_validated"])
    def test_numerical_failure_cannot_be_hidden_by_global_top1(self):
        metric=dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[1]*3,max_drop=1.)
        c,rs,bs,*_=self.prepared(metric=metric,delta=.1)
        result=self.record_all(c,rs,bs)
        self.assertTrue(result["application_criterion"]["passed"])
        self.assertEqual(result["status"],"validation_failed")
        self.assertEqual(result["part_numerical_failures"],2)
        self.assertEqual(result["numerically_validated_samples"],0)
        self.assertEqual(result["part_status_counts"],{"validation_failed":2})
        self.assertFalse(result["dataset_criterion_validated"])
    def test_max_error_is_dataset_maximum_not_average(self):
        graph,provenance=relu();bs=self.batches((1,2))
        bs[0]["samples"]=[dict(x=[-.5,.5])]
        bs[1]["samples"]=[dict(x=[0.,1.])]*2
        c,rs,bs,*_=self.prepared(bs,dict(kind="max_absolute_error",limit=.1),graph,provenance)
        result=self.record_all(c,rs,bs)
        self.assertEqual(result["application_criterion"]["max_absolute_error"],.125)
        self.assertEqual(result["total_vs_original_reference"]["mae"],1/24)
        self.assertEqual(result["status"],"validation_failed")
    def test_zero_reference_cosine_is_undefined_not_fabricated(self):
        bs=self.batches();[b.update(samples=[dict(x=[0.,0.])]*len(b["samples"])) for b in bs]
        c,rs,bs,*_=self.prepared(bs)
        r=self.record_all(c,rs,bs)
        self.assertIsNone(r["ckks_comparison"]["cosine_similarity"])
        self.assertEqual(r["ckks_comparison"]["nonzero_reference_count"],0)
        self.assertIsNone(r["ckks_comparison"]["max_nonzero_reference_relative_error"])
    def test_multiple_outputs_preserve_declared_flattening_and_class_axis(self):
        metric=dict(kind="top1_accuracy_drop",output="right",class_axis=0,labels=[1]*3,max_drop=0.)
        c,rs,bs,*_=self.prepared(metric=metric,graph=model(names=("left","right")))
        result=self.record_all(c,rs,bs)
        self.assertEqual(result["ckks_comparison"]["compared_values"],12)
        self.assertEqual(result["application_criterion"]["encrypted_top1"],1.)
    def test_missing_terminal_failure_is_visible_and_never_passes(self):
        c,rs,bs,*_=self.prepared()
        c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"])
        for status in ("timed_out","cancelled","unsupported","service_error","budget_exhausted"):
            # Separate collectors preserve the immutable outcome observation.
            fresh=DatasetNumericalAssessment(c.request,c.declaration,expected_declaration_binding=c.declaration["binding"])
            outcome=dict(status=status,failure=dict(code="fixture",layer="source_reference"))
            fresh.record_unavailable(bs[1]["id"],outcome,expected_outcome_sha256=digest(outcome))
            result=fresh.summary();self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["parts"][1]["status"],status);self.assertEqual(result["evaluated_samples"],0)
    def test_queued_or_executed_receipts_cannot_be_marked_unavailable(self):
        c,rs,bs,*_=self.prepared()
        for value in (dict(status="queued"),dict(status="validation_failed",encrypted_execution=True),rs[0]):
            self.error("dataset_receipt_rejected",lambda:c.record_unavailable(
                bs[0]["id"],value,expected_outcome_sha256=digest(value)))
    def test_part_input_and_criterion_bindings_must_match(self):
        c,rs,bs,*_=self.prepared()
        for key in ("supplied_input_sha256","criterion_sha256"):
            r=copy.deepcopy(rs[0]);r[key]="0"*64;r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
            self.error("dataset_receipt_rejected",lambda:c.record(bs[0]["id"],r,expected_binding=r["binding"]))
        self.assertEqual(c.summary()["evaluated_samples"],0)
    def test_report_hash_and_trusted_pin_are_both_required(self):
        c,rs,bs,*_=self.prepared()
        r=copy.deepcopy(rs[0]);r["samples"]=2
        self.error("dataset_binding_rejected",lambda:c.record(bs[0]["id"],r,expected_binding=rs[0]["binding"]))
        self.error("dataset_binding_rejected",lambda:c.record(bs[0]["id"],rs[0],expected_binding="0"*64))
    def test_summary_statistics_are_recalculated_not_trusted(self):
        c,rs,bs,*_=self.prepared()
        for key,value in (("passed",False),("mae",.01),("max_absolute_error",.01),("elementwise_pass",[[False,True]])):
            r=copy.deepcopy(rs[0]);r["ckks_comparison"][key]=value
            r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
            self.error("dataset_receipt_rejected",lambda:c.record(bs[0]["id"],r,expected_binding=r["binding"]))
    def test_different_package_reference_policy_and_dtype_cannot_mix(self):
        c,rs,bs,*_=self.prepared()
        for change in (dict(bundle_binding="a"*64),dict(reference_identity_sha256="a"*64),
                       dict(reference_policy=NATIVE_POLICY,format=NATIVE_FORMAT)):
            r=copy.deepcopy(rs[0]);r.update(change);r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
            with self.assertRaises(DatasetAssessmentError):c.record(bs[0]["id"],r,expected_binding=r["binding"])
        self.assertEqual(c.summary()["evaluated_samples"],0)
    def test_dataset_declaration_rejects_duplicate_ids_and_changed_model(self):
        c,rs,bs,request,d,_=self.prepared()
        bad=copy.deepcopy(bs);bad[1]["sample_ids"][0]=bad[0]["sample_ids"][0]
        self.error("dataset_declaration_rejected",lambda:declare_dataset(request,bad,anchor=d["anchor"]))
        bad=copy.deepcopy(bs);bad[1]["id"]=bad[0]["id"]
        self.error("dataset_declaration_rejected",lambda:declare_dataset(request,bad,anchor=d["anchor"]))
        changed=copy.deepcopy(request);changed["model"]["id"]+="changed"
        self.error("dataset_declaration_rejected",lambda:DatasetNumericalAssessment(changed,d,
            expected_declaration_binding=d["binding"]))
    def test_partition_size_and_declaration_limits_are_bounded(self):
        c,rs,bs,request,d,_=self.prepared()
        bad=self.batches((17,))
        self.error("dataset_declaration_rejected",lambda:declare_dataset(request,bad,anchor=d["anchor"]))
        self.error("dataset_declaration_rejected",lambda:declare_dataset(request,[bs[0]]*4097,anchor=d["anchor"]))
        bad=copy.deepcopy(d);bad["samples"]=True;bad["binding"]=digest({k:v for k,v in bad.items() if k!="binding"})
        self.error("dataset_declaration_rejected",lambda:DatasetNumericalAssessment(request,bad,
            expected_declaration_binding=bad["binding"]))
    def test_current_dependency_drift_stops_collection(self):
        c,rs,bs,*_=self.prepared()
        with patch("task_dataset_assessment.adapter_identity",return_value="0"*64):
            self.error("dataset_reference_changed",lambda:c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"]))
            self.error("dataset_reference_changed",c.summary)
    def test_summary_and_declaration_do_not_alias_caller_state(self):
        c,rs,bs,request,d,_=self.prepared();result=self.record_all(c,rs,bs)
        before=c.summary()["binding"];d["parts"][0]["id"]="changed"
        request["model"]["id"]="changed";rs[0]["samples"]=3
        result["parts"][0]["ckks_statistics"]["max_absolute_error"]=999.
        result["anchor"]["bundle_binding"]="0"*64
        self.assertEqual(c.summary()["binding"],before)
    def native_prepared(self,metric=None):
        from task_numerical_assessment import implementation,application_metric,criterion,flatten
        from task_dataset_assessment import shaped
        from numerical_results import compare
        c,rs,bs,request,d,outcome=self.prepared(metric=metric)
        # Explicit injected native receipt fixtures; no source or FHE execution.
        for r,b in zip(rs,bs):
            original=r["total_comparison"]["reference"]
            actual=r["ckks_comparison"]["actual"]
            r.update(format=NATIVE_FORMAT,reference_policy=NATIVE_POLICY,
                reference_identity_sha256=implementation(policy=NATIVE_POLICY),
                source_binding=digest({"native_fixture":"source"}),
                source_snapshot_binding=digest({"native_fixture":"snapshot"}),
                source_reference_evaluated=True,
                source_reference_binding=digest({"fixture":"native-frame","part":b["id"]}),
                source_forward_binding=digest({"fixture":"native-forward","part":b["id"]}),
                source_input_dtypes={"x":"float64"},
                source_output_dtypes={"out":"float64"},
                source_total_comparison=copy.deepcopy(r["total_comparison"]))
            profile=part_criterion(d["application_criterion"],d["parts"][bs.index(b)]["start"],len(b["samples"]),request["model"])
            import numpy as np
            r["application_criterion"]=application_metric(profile,
                shaped(np.asarray(original),request["model"]),
                shaped(np.asarray(r["decomposition_comparison"]["reference"]),request["model"]),
                shaped(np.asarray(actual),request["model"]),request["model"],scope=NATIVE_POLICY)
            r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        anchor=reference_anchor(rs[0],expected_binding=rs[0]["binding"])
        declaration=declare_dataset(request,bs,anchor=anchor,application_criterion=metric)
        collector=DatasetNumericalAssessment(request,declaration,expected_declaration_binding=declaration["binding"])
        return collector,rs,bs
    def test_native_dataset_metric_uses_bound_native_policy(self):
        metric=dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[1]*3,max_drop=0.)
        c,rs,bs=self.native_prepared(metric)
        r=self.record_all(c,rs,bs)
        self.assertEqual(r["status"],"passed");self.assertEqual(r["application_criterion"]["scope"],NATIVE_POLICY)
        self.assertEqual(r["application_criterion"]["encrypted_top1"],1.)
        self.assertTrue(r["dataset_criterion_validated"]);self.assertFalse(r["original_source_accuracy_certified"])
    def test_native_dtypes_cannot_change_between_parts(self):
        c,rs,bs=self.native_prepared()
        c.record(bs[0]["id"],rs[0],expected_binding=rs[0]["binding"])
        r=copy.deepcopy(rs[1]);r["source_input_dtypes"]={"x":"float32"}
        r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        self.error("dataset_receipt_rejected",lambda:c.record(bs[1]["id"],r,expected_binding=r["binding"]))
        self.assertEqual(c.summary()["status"],"incomplete")
    def test_native_source_comparison_and_forward_receipt_are_required(self):
        c,rs,bs=self.native_prepared()
        for change in (dict(source_total_comparison=None),dict(source_forward_binding=None),
                       dict(source_reference_binding=None)):
            r=copy.deepcopy(rs[0]);r.update(change)
            r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
            self.error("dataset_receipt_rejected",lambda:c.record(bs[0]["id"],r,expected_binding=r["binding"]))
        self.assertEqual(c.summary()["evaluated_samples"],0)
    def test_native_reference_values_cannot_be_replaced_by_math_values(self):
        c,rs,bs=self.native_prepared(dict(kind="max_absolute_error",limit=.01))
        r=copy.deepcopy(rs[0]);r["source_total_comparison"]["reference"]=[[.5,.5]]
        r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        self.error("dataset_receipt_rejected",lambda:c.record(bs[0]["id"],r,expected_binding=r["binding"]))
    def test_unsafe_hooks_rejected_and_private_values_do_not_enter_errors(self):
        c,rs,bs,*_=self.prepared()
        class Unsafe:
            def __deepcopy__(self,memo):raise AssertionError("copy hook")
            def __eq__(self,value):raise AssertionError("compare hook")
            def __array__(self,*args):raise AssertionError("array hook")
        self.error("dataset_binding_rejected",lambda:c.record(bs[0]["id"],dict(value=Unsafe()),expected_binding="0"*64))
        r=copy.deepcopy(rs[0]);r["ckks_comparison"]["actual"]=[[9.123456789]]
        r["binding"]=digest({k:v for k,v in r.items() if k!="binding"})
        with self.assertRaises(DatasetAssessmentError) as error:c.record(bs[0]["id"],r,expected_binding=r["binding"])
        self.assertNotIn("9.123456789",str(error.exception.feedback()))
if __name__=="__main__":unittest.main()
