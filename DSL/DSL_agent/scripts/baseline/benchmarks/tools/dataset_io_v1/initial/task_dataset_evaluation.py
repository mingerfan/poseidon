"""Serial, count-bounded dataset assessment over one existing Application.

This adapter admits at most the explicitly allowed number of new assessments.
The existing source/runtime workers keep their single-call limits. Cancellation
is observed between parts; this is not a hard-deadline or background service.
"""
import copy
from pathlib import Path
from benchmark_graph import digest
from model_output_contract import snapshot,is_hash
from task_numerical_assessment import TaskNumericalAssessment,SOURCE_SELECTION,source_selection
from task_dataset_assessment import (DatasetNumericalAssessment,DatasetAssessmentError,
    declare_dataset,part_criterion,adapter_identity,MAX_PARTS)
from assessment_array_dataset import ArrayDataset,DatasetFileError
FORMAT="poseidon-task-dataset-evaluation-v1"
class TaskDatasetEvaluation:
    """Host-owned optional evaluation over a delivered package; no new Agent."""
    def __init__(self,application,task,dataset,*,anchor,application_criterion=None,
                 max_new_assessments=0,native_source=None,cached_receipts=None):
        if not isinstance(dataset,ArrayDataset):raise DatasetFileError("dataset_file_rejected")
        if type(max_new_assessments)is not int or not 0<=max_new_assessments<=MAX_PARTS:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        self.numeric=TaskNumericalAssessment(application);self.application=application;self.task=task
        self.numeric.outputs._task(task);self.outcome=application.result(task)
        if self.outcome["status"]!="succeeded":raise DatasetAssessmentError("dataset_receipt_rejected")
        self.metadata=self.numeric.outputs._metadata(task);package,grade=self.numeric.outputs._package(self.outcome,self.metadata)
        self.request=self.metadata["request"];self.package_binding=package["binding"]
        if anchor.get("bundle_binding")!=self.package_binding:
            raise DatasetAssessmentError("dataset_binding_rejected")
        self.batches=dataset.batches()
        profile=dataset.criterion(application_criterion)
        self.declaration=declare_dataset(self.request,self.batches,anchor=anchor,application_criterion=profile)
        self.collector=DatasetNumericalAssessment(self.request,self.declaration,
            expected_declaration_binding=self.declaration["binding"])
        self.limit=max_new_assessments;self.used=0;self.position=0;self.cancel_requested=False
        self.status="queued";self.receipts={};self.ledger=[];self.failure=None
        self.native_source=None
        if native_source is not None:
            native_source=snapshot(native_source,limit=131072)
            if type(native_source)is not dict or set(native_source)!={"input_dtypes","trust_local_code","evidence_root"}:
                raise DatasetAssessmentError("dataset_declaration_rejected")
            selection=dict(format=SOURCE_SELECTION,input_dtypes=native_source["input_dtypes"],
                trust_local_code=native_source["trust_local_code"],
                evidence_directory=str(Path(native_source["evidence_root"])/"part_0"))
            source_selection(selection,self.request)
            if anchor["reference_policy"]!="bound-source-native-dtype-finite-v1" or selection["trust_local_code"]is not True or                 selection["input_dtypes"]!=anchor["source_input_dtypes"]:
                raise DatasetAssessmentError("dataset_binding_rejected")
            self.native_source=native_source
        elif anchor["reference_policy"]=="bound-source-native-dtype-finite-v1" and max_new_assessments>0:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        self.cache=cached_receipts or {}
        if type(self.cache)is not dict or not set(self.cache)<=set(self.collector.parts):
            raise DatasetAssessmentError("dataset_declaration_rejected")
        self.identity=adapter_identity(anchor["reference_policy"])
    def cancel(self):
        self.cancel_requested=True
        return self.state()
    def state(self):
        summary=self.collector.summary()
        result=dict(format=FORMAT,status=self.status,task_id=self.task,dataset_binding=self.declaration["binding"],
            package_binding=self.package_binding,max_new_assessments=self.limit,new_assessments=self.used,
            remaining_assessments=self.limit-self.used,cancel_requested=self.cancel_requested,
            completed_parts=len(self.collector.records),planned_parts=len(self.batches),
            summary=summary,ledger=copy.deepcopy(self.ledger),failure=copy.deepcopy(self.failure),
            new_agent_calls=0,new_compiler_calls=0)
        result["binding"]=digest(result);return result
    def _current(self):
        if (self.application.result(self.task)!=self.outcome or
            self.numeric.outputs._metadata(self.task)!=self.metadata or
            adapter_identity(self.collector.policy)!=self.identity):
            raise DatasetAssessmentError("dataset_reference_changed")
    def advance(self,*,max_new_parts=1):
        """At most this many new calls; cached receipts do not consume admission quota."""
        if type(max_new_parts)is not int or not 0<=max_new_parts<=self.limit:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        if self.status in ("succeeded","validation_failed","cancelled","service_error"):return self.state()
        calls=0
        try:
            self._current()
            while self.position<len(self.batches):
                if self.cancel_requested:
                    self.status="cancelled";return self.state()
                batch=self.batches[self.position];name=batch["id"]
                if name in self.cache:
                    cache=snapshot(self.cache[name],limit=16*1024**2)
                    if type(cache)is not dict or set(cache)!={"receipt","binding"}:
                        raise DatasetAssessmentError("dataset_binding_rejected")
                    self.collector.record(name,cache["receipt"],expected_binding=cache["binding"])
                    self.ledger.append(dict(part_id=name,kind="receipt_reuse",receipt_binding=cache["binding"],
                                            new_assessments=0))
                    self.receipts[name]=cache["receipt"]
                else:
                    if self.used>=self.limit:
                        self.status="budget_exhausted";return self.state()
                    if calls>=max_new_parts:
                        self.status="ready";return self.state()
                    # Account the attempt before entering the existing bounded validator.
                    self.used+=1;calls+=1;self.status="running"
                    part=self.declaration["parts"][self.position]
                    profile=part_criterion(self.declaration["application_criterion"],part["start"],part["samples"],self.request["model"])
                    selection=None
                    if self.native_source is not None:
                        selection=dict(format=SOURCE_SELECTION,input_dtypes=self.native_source["input_dtypes"],
                            trust_local_code=True,evidence_directory=str(Path(self.native_source["evidence_root"])/name))
                    report=self.numeric.assess(self.task,batch["samples"],expected_package_binding=self.package_binding,
                        application_criterion=profile,native_source=selection)
                    self.receipts[name]=report
                    if "ckks_comparison"in report and report.get("encrypted_execution")is True and is_hash(report.get("binding")):
                        self.collector.record(name,report,expected_binding=report["binding"])
                    else:
                        self.collector.record_unavailable(name,report,expected_outcome_sha256=digest(report))
                    self.ledger.append(dict(part_id=name,kind="assessment",status=report["status"],
                        receipt_binding=report.get("binding"),new_assessments=1,
                        encrypted_execution=report.get("encrypted_execution")is True))
                    if report["status"]in ("service_error","timed_out"):
                        self.failure=copy.deepcopy(report.get("failure"));self.status=report["status"]
                        self.position+=1;return self.state()
                self.position+=1;self._current()
            summary=self.collector.summary()
            self.status="succeeded" if summary["status"]=="passed" else "validation_failed"
            return self.state()
        except DatasetAssessmentError as error:
            self.failure=error.feedback();self.status="service_error";return self.state()
