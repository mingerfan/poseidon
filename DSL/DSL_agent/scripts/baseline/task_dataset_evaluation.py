"""Serial dataset assessment over one delivered Application package.

Only explicitly admitted calls execute. Source/runtime worker limits remain in
force. Cancellation is observed between parts; no global wall deadline or
background inference service is introduced.
"""
import copy,hashlib
from pathlib import Path
from benchmark_graph import digest,canonical
from model_output_contract import is_hash
from task_numerical_assessment import TaskNumericalAssessment,SOURCE_SELECTION,source_selection,NATIVE_POLICY
from task_dataset_assessment import (DatasetNumericalAssessment,DatasetAssessmentError,
    declare_dataset,part_criterion,adapter_identity,MAX_PARTS,copied)
from assessment_array_dataset import ArrayDataset,DatasetFileError,MAX_ARRAY_BYTES
FORMAT="poseidon-task-dataset-evaluation-v1"
TERMINAL={"succeeded","validation_failed","cancelled","service_error","timed_out","budget_exhausted"}
def evaluation_identity(policy):
    root=Path(__file__).resolve().parent
    return digest(dict(assessment=adapter_identity(policy),sources={
        n:hashlib.sha256((root/n).read_bytes()).hexdigest()
        for n in ("task_dataset_evaluation.py","assessment_array_dataset.py")}))
def service_failure():
    return dict(code="dataset_assessment_service_failed",layer="assessment",
        message="Dataset assessment stopped after a service or environment failure.",
        retry_policy="never_automatic")
class TaskDatasetEvaluation:
    """Optional host adapter; reuses Application and its checked program."""
    def __init__(self,application,task,dataset,*,anchor,application_criterion=None,
                 max_new_assessments=0,native_source=None,cached_receipts=None,part_samples=16):
        if type(dataset)is not ArrayDataset:raise DatasetFileError("dataset_file_rejected")
        if type(max_new_assessments)is not int or not 0<=max_new_assessments<=MAX_PARTS:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        anchor=copied(anchor,131072,"dataset_declaration_rejected")
        if type(anchor)is not dict:raise DatasetAssessmentError("dataset_declaration_rejected")
        self.numeric=TaskNumericalAssessment(application);self.application=application;self.task=task
        self.numeric.outputs._task(task)
        self.outcome=copied(application.result(task),16*1024**2,"dataset_receipt_rejected")
        if self.outcome["status"]!="succeeded":raise DatasetAssessmentError("dataset_receipt_rejected")
        self.metadata=copied(self.numeric.outputs._metadata(task),16*1024**2,"dataset_receipt_rejected")
        package,_=self.numeric.outputs._package(self.outcome,self.metadata)
        self.request=self.metadata["request"];self.package_binding=package["binding"]
        if anchor.get("bundle_binding")!=self.package_binding:
            raise DatasetAssessmentError("dataset_binding_rejected")
        self._batches=dataset.batches(part_samples)
        profile=dataset.criterion(application_criterion)
        self.declaration=declare_dataset(self.request,self._batches,anchor=anchor,application_criterion=profile)
        self.collector=DatasetNumericalAssessment(self.request,self.declaration,
            expected_declaration_binding=self.declaration["binding"])
        if anchor["reference_policy"]==NATIVE_POLICY:
            dtype_names={"<f4":"float32","<f8":"float64"}
            actual={n:dtype_names[v["dtype"]] for n,v in dataset.manifest["inputs"].items()}
            if actual!=anchor["source_input_dtypes"]:
                raise DatasetAssessmentError("dataset_binding_rejected")
        self._limit=max_new_assessments;self.used=0;self.position=0;self.cancel_requested=False
        self.status="queued";self.receipts={};self.ledger=[];self.failure=None
        self._native_source=None;self._evidence_inode=None
        if native_source is not None:
            native_source=copied(native_source,131072,"dataset_declaration_rejected")
            if type(native_source)is not dict or set(native_source)!={"input_dtypes","trust_local_code","evidence_root"}:
                raise DatasetAssessmentError("dataset_declaration_rejected")
            try:
                selection=dict(format=SOURCE_SELECTION,input_dtypes=native_source["input_dtypes"],
                    trust_local_code=native_source["trust_local_code"],
                    evidence_directory=str(Path(native_source["evidence_root"])/"part_0"))
                source_selection(selection,self.request)
            except (ValueError,TypeError,OSError):
                raise DatasetAssessmentError("dataset_declaration_rejected") from None
            if (anchor["reference_policy"]!=NATIVE_POLICY or
                selection["trust_local_code"]is not True or
                selection["input_dtypes"]!=anchor["source_input_dtypes"]):
                raise DatasetAssessmentError("dataset_binding_rejected")
            evidence=Path(native_source["evidence_root"])
            if (not evidence.is_absolute() or evidence!=evidence.resolve() or
                evidence.exists() or evidence.is_symlink() or not evidence.parent.is_dir()):
                raise DatasetAssessmentError("dataset_declaration_rejected")
            self._native_source=native_source
        elif anchor["reference_policy"]==NATIVE_POLICY and max_new_assessments>0:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        cache={} if cached_receipts is None else cached_receipts
        if type(cache)is not dict or not set(cache)<=set(self.collector.parts):
            raise DatasetAssessmentError("dataset_declaration_rejected")
        self._cache={};total=0
        # Check all cached receipts before admitting any new execution.
        checked=DatasetNumericalAssessment(self.request,self.declaration,
            expected_declaration_binding=self.declaration["binding"])
        for name,value in cache.items():
            value=copied(value,16*1024**2,"dataset_binding_rejected")
            if type(value)is not dict or set(value)!={"receipt","binding"}:
                raise DatasetAssessmentError("dataset_binding_rejected")
            total+=len(canonical(value))
            if total>MAX_ARRAY_BYTES:raise DatasetAssessmentError("dataset_declaration_rejected")
            checked.record(name,value["receipt"],expected_binding=value["binding"])
            self._cache[name]=value
        self.identity=evaluation_identity(anchor["reference_policy"])
        self._plan=dict(dataset_file_binding=dataset.manifest["binding"],
            declaration_binding=self.declaration["binding"],package_binding=self.package_binding,
            max_new_assessments=self._limit,part_samples=part_samples,native_source=self._native_source,
            cached_receipts={n:v["binding"] for n,v in self._cache.items()},implementation=self.identity)
        self._plan_binding=digest(self._plan)
        self._last_summary=self.collector.summary()
    @property
    def limit(self):return self._limit
    def cancel(self):
        if self.status not in TERMINAL:
            self.cancel_requested=True
            # An admitted call finishes within its existing worker limit.
            if self.status!="running":self.status="cancelled"
        return self.state()
    def _current(self):
        if (self.application.result(self.task)!=self.outcome or
            self.numeric.outputs._metadata(self.task)!=self.metadata or
            evaluation_identity(self.collector.policy)!=self.identity or
            digest(self._plan)!=self._plan_binding or self._limit!=self._plan["max_new_assessments"]):
            raise DatasetAssessmentError("dataset_reference_changed")
    def state(self):
        summary_current=True
        try:
            self._current();summary=self.collector.summary()
            self._last_summary=copy.deepcopy(summary)
        except DatasetAssessmentError as error:
            self.failure=error.feedback();self.status="service_error"
            summary=copy.deepcopy(self._last_summary);summary_current=False
        except Exception:
            self.failure=service_failure();self.status="service_error"
            summary=copy.deepcopy(self._last_summary);summary_current=False
        result=dict(format=FORMAT,status=self.status,task_id=self.task,
            dataset_file_binding=self._plan["dataset_file_binding"],
            dataset_binding=self.declaration["binding"],evaluation_binding=self._plan_binding,
            implementation_identity=self.identity,package_binding=self.package_binding,
            max_new_assessments=self.limit,new_assessments=self.used,
            remaining_assessments=max(0,self.limit-self.used),cancel_requested=self.cancel_requested,
            completed_parts=len(self.collector.records),planned_parts=len(self._batches),
            summary=summary,summary_identity_current=summary_current,
            ledger=copy.deepcopy(self.ledger),failure=copy.deepcopy(self.failure),
            new_agent_calls=0,new_compiler_calls=0)
        result["binding"]=digest(result);return result
    def _source_directory(self,name):
        root=Path(self._native_source["evidence_root"])
        if self._evidence_inode is None:
            # The source worker creates each part itself, under this fresh parent.
            if root.exists() or root.is_symlink() or root!=root.resolve():
                raise DatasetAssessmentError("dataset_binding_rejected")
            root.mkdir(mode=0o700)
            stat=root.stat();self._evidence_inode=(stat.st_dev,stat.st_ino)
        else:
            stat=root.lstat()
            if (not root.is_dir() or root.is_symlink() or root!=root.resolve() or
                (stat.st_dev,stat.st_ino)!=self._evidence_inode):
                raise DatasetAssessmentError("dataset_binding_rejected")
        return str(root/name)
    def advance(self,*,max_new_parts=1):
        """Admit this many new calls; cached receipts consume no call quota."""
        if type(max_new_parts)is not int or not 0<=max_new_parts<=MAX_PARTS:
            raise DatasetAssessmentError("dataset_declaration_rejected")
        if self.status in TERMINAL:return self.state()
        calls=0
        try:
            self._current()
            while self.position<len(self._batches):
                if self.cancel_requested:
                    self.status="cancelled";return self.state()
                batch=self._batches[self.position];name=batch["id"]
                if name in self._cache:
                    cache=self._cache[name]
                    self.collector.record(name,cache["receipt"],expected_binding=cache["binding"])
                    self.ledger.append(dict(part_id=name,kind="receipt_reuse",
                        receipt_binding=cache["binding"],new_assessments=0))
                    self.receipts[name]=copy.deepcopy(cache["receipt"])
                else:
                    if self.used>=self.limit:
                        self.status="budget_exhausted";return self.state()
                    if calls>=max_new_parts:
                        self.status="ready";return self.state()
                    part=self.declaration["parts"][self.position]
                    profile=part_criterion(self.declaration["application_criterion"],
                        part["start"],part["samples"],self.request["model"])
                    selection=None
                    if self._native_source is not None:
                        selection=dict(format=SOURCE_SELECTION,input_dtypes=self._native_source["input_dtypes"],
                            trust_local_code=True,evidence_directory=str(Path(self._native_source["evidence_root"])/name))
                    # Reserve and journal before entering the existing validator.
                    self.used+=1;calls+=1;self.status="running"
                    entry=dict(part_id=name,kind="assessment",status="started",
                        new_assessments=1,receipt_binding=None,encrypted_execution=None)
                    self.ledger.append(entry)
                    try:
                        if selection is not None:
                            selection["evidence_directory"]=self._source_directory(name)
                        report=self.numeric.assess(self.task,batch["samples"],
                            expected_package_binding=self.package_binding,
                            application_criterion=profile,native_source=selection)
                        report=copied(report,16*1024**2,"dataset_receipt_rejected")
                        self.receipts[name]=report
                        if ("ckks_comparison"in report and report.get("encrypted_execution")is True and
                            is_hash(report.get("binding"))):
                            self.collector.record(name,report,expected_binding=report["binding"])
                        else:
                            self.collector.record_unavailable(name,report,expected_outcome_sha256=digest(report))
                        entry.update(status=report["status"],receipt_binding=report.get("binding"),
                            encrypted_execution=report.get("encrypted_execution")is True)
                    except DatasetAssessmentError:
                        entry["status"]="evidence_rejected";raise
                    except Exception:
                        entry["status"]="service_error";self.failure=service_failure()
                        self.status="service_error";return self.state()
                    if report["status"]in ("service_error","timed_out","cancelled"):
                        self.failure=copy.deepcopy(report.get("failure"));self.status=report["status"]
                        self.position+=1;return self.state()
                self.position+=1;self._last_summary=self.collector.summary();self._current()
            summary=self.collector.summary()
            self.status="succeeded" if summary["status"]=="passed" else "validation_failed"
            return self.state()
        except DatasetAssessmentError as error:
            self.failure=error.feedback();self.status="service_error";return self.state()
        except Exception:
            self.failure=service_failure();self.status="service_error";return self.state()
