"""Receipt-only original-source/approximation dataset assessment.

Inputs and labels are frozen before collection. Returned source receipts are
host-pinned, rechecked against their partitions, and never counted as FHE.
"""
import copy,hashlib,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest
from numerical_results import compare
from model_output_contract import is_hash
from task_numerical_assessment import NATIVE_POLICY,ATOL,RTOL,REFERENCE_ATOL,REFERENCE_RTOL
from task_approximation_assessment import (
    FORMAT as RECEIPT_FORMAT,SUBJECT,implementation as reference_identity,approximation_metric)
from task_dataset_assessment import (
    DatasetAssessmentError,require,public,pinned,copied,declare_dataset,part_criterion,
    shaped,signature,MAX_PARTS,MAX_SAMPLES,RECEIPT_BYTES)

FORMAT="poseidon-approximation-dataset-assessment-v1"
ANCHOR_FIELDS=("request_id","bundle_binding","task_id","task_metadata_binding",
    "source_binding","source_snapshot_binding","package_validation","reference_policy",
    "reference_identity_sha256","source_input_dtypes")
def implementation():
    return digest(dict(subject=SUBJECT,collector=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       reference=reference_identity()))

class ApproximationDatasetAssessment:
    """Collect independently pinned receipts; no execution or user callbacks."""
    @public("dataset_declaration_rejected")
    def __init__(self,request,batches,*,anchor,application_criterion=None):
        self.request=copied(request,16*1024**2,"dataset_declaration_rejected")
        anchor=copied(anchor,131072,"dataset_declaration_rejected")
        require(type(anchor)is dict and set(anchor)==set(ANCHOR_FIELDS))
        require(anchor["reference_policy"]==NATIVE_POLICY and
            anchor["reference_identity_sha256"]==reference_identity(),"dataset_reference_changed")
        require(all(is_hash(anchor[k])for k in ("request_id","bundle_binding","task_metadata_binding",
            "source_binding","source_snapshot_binding","reference_identity_sha256")))
        require(type(anchor["task_id"])is str and len(anchor["task_id"])==32 and
            all(x in "0123456789abcdef" for x in anchor["task_id"]))
        grade=anchor["package_validation"]
        require(type(grade)is dict and set(grade)=={"level","encrypted_execution","numerically_validated"}
            and grade["level"]in ("compiled","numerical") and
            grade["encrypted_execution"]is (grade["level"]=="numerical") and
            grade["numerically_validated"]is (grade["level"]=="numerical"))
        require(type(anchor["source_input_dtypes"])is dict and
            set(anchor["source_input_dtypes"])=={i["name"]for i in self.request["model"]["inputs"]} and
            all(type(v)is str and v in ("float32","float64")for v in anchor["source_input_dtypes"].values()))
        self.declaration=declare_dataset(self.request,batches,anchor=anchor,
            application_criterion=application_criterion)
        self.anchor=anchor;self.identity=implementation();self.records={}
        self.request_binding=digest(self.request);self.declaration_binding=self.declaration["binding"]
        self.parts={p["id"]:p for p in self.declaration["parts"]};self.output_dtypes=None

    @public("dataset_reference_changed")
    def _current(self):
        require(implementation()==self.identity and
            self.anchor["reference_identity_sha256"]==reference_identity() and
            digest(self.request)==self.request_binding and
            self.declaration["binding"]==self.declaration_binding and
            digest({k:v for k,v in self.declaration.items()if k!="binding"})==self.declaration_binding and
            self.anchor==self.declaration["anchor"],"dataset_reference_changed")

    @public("dataset_receipt_rejected")
    def record(self,part_id,receipt,*,expected_binding):
        self._current()
        require(type(part_id)is str and part_id in self.parts,"dataset_receipt_rejected")
        report=pinned(receipt,expected_binding,RECEIPT_BYTES);part=self.parts[part_id]
        if part_id in self.records:
            require(self.records[part_id]["binding"]==expected_binding,"dataset_part_conflict")
            return
        require(report.get("format")==RECEIPT_FORMAT and
            report.get("status")in ("passed","validation_failed") and
            report.get("task_status")=="succeeded" and
            report.get("application_metric_subject")==SUBJECT and
            report.get("source_reference_evaluated")is True and
            report.get("source_reference_requested")is True and
            report.get("encrypted_execution")is False and report.get("execution_requested")is False and
            report.get("ckks_validation_performed")is False and
            report.get("supplied_inputs_numerically_validated")is False and
            report.get("original_source_accuracy_certified")is False and report.get("formal_equivalence")is False,
            "dataset_receipt_rejected")
        require(all(report.get(k)==v for k,v in self.anchor.items()),"dataset_binding_rejected")
        require(type(report.get("samples"))is int and report["samples"]==part["samples"] and
            report.get("supplied_input_sha256")==part["supplied_input_sha256"] and
            report.get("criterion_sha256")==part["criterion_sha256"],"dataset_binding_rejected")
        require(type(report.get("agent_calls"))is int and report["agent_calls"]==0 and
            type(report.get("compiler_calls"))is int and report["compiler_calls"]==0,
            "dataset_receipt_rejected")
        for name in ("source_reference_binding","source_forward_binding","source_selection_sha256"):
            require(is_hash(report.get(name)),"dataset_receipt_rejected")
        dtypes=report.get("source_output_dtypes")
        require(type(dtypes)is dict and set(dtypes)=={n for n,_ in signature(self.request["model"])} and
            all(type(v)is str and v in ("float32","float64")for v in dtypes.values()),"dataset_receipt_rejected")
        if self.output_dtypes is not None:require(dtypes==self.output_dtypes,"dataset_binding_rejected")
        comparison=report.get("original_source_vs_approximate_comparison")
        require(type(comparison)is dict,"dataset_receipt_rejected")
        actual=np.asarray(comparison.get("actual"),dtype=np.float64)
        reference=np.asarray(comparison.get("reference"),dtype=np.float64)
        width=sum(math.prod(s)for _,s in signature(self.request["model"]))
        require(actual.shape==reference.shape==(part["samples"],width) and
            np.isfinite(actual).all()and np.isfinite(reference).all(),"dataset_receipt_rejected")
        require(compare(actual,reference,ATOL,RTOL)==comparison,"dataset_receipt_rejected")
        profile=part_criterion(self.declaration["application_criterion"],part["start"],
            part["samples"],self.request["model"])
        approximate=shaped(actual,self.request["model"]);native=shaped(reference,self.request["model"])
        metric=approximation_metric(profile,native,approximate,self.request["model"])
        require(metric==report.get("application_criterion") and
            report.get("application_criterion_validated")is (metric["passed"]is True) and
            report["status"]==("validation_failed"if metric["passed"]is False else "passed"),
            "dataset_receipt_rejected")
        failure=report.get("failure")
        require(failure is None if metric["passed"]is not False else
            type(failure)is dict and failure.get("code")=="assessment_application_criterion_failed",
            "dataset_receipt_rejected")
        decomposition=report.get("decomposition_comparison")
        require(type(decomposition)is dict and decomposition.get("passed")is True,
            "dataset_receipt_rejected")
        da=np.asarray(decomposition.get("actual"),dtype=np.float64)
        dr=np.asarray(decomposition.get("reference"),dtype=np.float64)
        require(da.shape==dr.shape==actual.shape and
            compare(da,dr,REFERENCE_ATOL,REFERENCE_RTOL)==decomposition,"dataset_receipt_rejected")
        counts=None
        if profile is not None and profile["kind"]=="top1_accuracy_drop":
            counts=dict(original_correct=0,approximate_correct=0,total=0)
            for src,approx,label in zip(native,approximate,profile["labels"]):
                target=np.asarray(label,dtype=np.int64)
                counts["original_correct"]+=int(np.sum(np.argmax(src[profile["output"]],
                    axis=profile["class_axis"])==target))
                counts["approximate_correct"]+=int(np.sum(np.argmax(approx[profile["output"]],
                    axis=profile["class_axis"])==target))
                counts["total"]+=int(target.size)
        self.output_dtypes=copy.deepcopy(dtypes)
        self.records[part_id]=dict(binding=expected_binding,compared_values=int(actual.size),
            maximum_absolute_error=float(np.abs(actual-reference).max()),
            absolute_error_sum=float(np.abs(actual-reference).sum()),counts=counts,
            criterion_passed=metric["passed"],source_forward_binding=report["source_forward_binding"])

    @public("dataset_reference_changed")
    def summary(self):
        self._current()
        complete=len(self.records)==len(self.parts)
        count=sum(r["compared_values"]for r in self.records.values())
        evaluated=sum(self.parts[n]["samples"]for n in self.records)
        maximum=max((r["maximum_absolute_error"]for r in self.records.values()),default=None)
        mae=sum(r["absolute_error_sum"]for r in self.records.values())/count if count else None
        profile=self.declaration["application_criterion"]
        metric=dict(status="not_requested"if profile is None else "pending",
            passed=None,subject=SUBJECT)
        if profile is not None and self.records:
            if profile["kind"]=="max_absolute_error":
                passed=maximum<=profile["limit"]
                metric=dict(kind=profile["kind"],limit=profile["limit"],
                    max_absolute_error=maximum,passed=passed if complete else None,
                    status=("passed"if passed else "failed")if complete else "pending",subject=SUBJECT)
            else:
                original=sum(r["counts"]["original_correct"]for r in self.records.values())
                approximate=sum(r["counts"]["approximate_correct"]for r in self.records.values())
                total=sum(r["counts"]["total"]for r in self.records.values())
                drop=Fraction(original-approximate,total)
                passed=drop<=Fraction(str(profile["max_drop"]))
                metric=dict(kind=profile["kind"],output=profile["output"],class_axis=profile["class_axis"],
                    original_source_top1=original/total,approximate_top1=approximate/total,
                    original_correct=original,approximate_correct=approximate,
                    labels_compared=total,accuracy_drop=float(drop),max_drop=profile["max_drop"],
                    passed=passed if complete else None,
                    status=("passed"if passed else "failed")if complete else "pending",subject=SUBJECT)
        result=dict(format=FORMAT,status=("passed"if metric["passed"]is not False else "failed")if complete else "incomplete",
            dataset_binding=self.declaration["binding"],reference_identity_sha256=self.anchor["reference_identity_sha256"],
            complete=complete,planned_samples=self.declaration["samples"],evaluated_samples=evaluated,
            planned_parts=len(self.parts),completed_parts=len(self.records),
            parts=[dict(id=n,status="recorded"if n in self.records else "not_run",
                receipt_binding=self.records[n]["binding"]if n in self.records else None)for n in self.parts],
            compared_values=count,maximum_absolute_error=maximum,mae=mae,
            application_criterion=metric,application_criterion_validated=complete and metric["passed"]is True,
            part_criterion_failures=sum(r["criterion_passed"]is False for r in self.records.values()),
            encrypted_execution=False,ckks_validation_performed=False,
            supplied_inputs_numerically_validated=False,original_source_accuracy_certified=False,
            formal_equivalence=False,agent_calls=0,compiler_calls=0)
        result["binding"]=digest(result);return result
