"""Persistent original-versus-approximation dataset assessment.

Reuses the existing assessment journal, locking, cancellation, quota and clock
handling. Reference-only interpretation is explicit; the numerical controller
and its historical plans are not modified.
"""
import hashlib,os,copy
from pathlib import Path
from benchmark_graph import canonical,digest
from model_output_contract import snapshot,is_hash
from assessment_array_dataset import load_dataset,MAX_ARRAY_BYTES
from task_numerical_assessment import NATIVE_POLICY,SOURCE_SELECTION,CODES as METRIC_CODES
from task_approximation_assessment import TaskApproximationAssessment,SUBJECT,implementation as reference_identity
from task_output_adapter import source_binding
from approximation_dataset_assessment import ApproximationDatasetAssessment,ANCHOR_FIELDS,implementation as collector_identity
from application_assessment import (
    ApplicationAssessment,AssessmentControllerError,require,public,read,bound,seal,
    update,write_new,STATE,RECEIPT_BYTES,STATE_BYTES,TERMINAL)
from model_weight_package import regular_bytes
from source_precision_reference import CODES as SOURCE_CODES
from task_dataset_assessment import MAX_PARTS,DatasetAssessmentError,part_criterion
import assessment_time_budget as time_budget

FORMAT="poseidon-application-approximation-assessment-v1"

def identity():
    return digest(dict(format=FORMAT,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_controller_sha256=hashlib.sha256(Path(__file__).with_name("application_assessment.py").read_bytes()).hexdigest(),
        collector=collector_identity(),clock=time_budget.implementation()))

class ApplicationApproximationAssessment(ApplicationAssessment):
    @public("assessment_controller_input_rejected")
    def __init__(self,application,store):
        super().__init__(application,store)
        self.numeric=TaskApproximationAssessment(application)

    def _anchor(self,plan,metadata):
        # The plan grade comes from a checked program result at submission and
        # is compared to a fresh result by _scene. Collection has no side effects.
        return dict(request_id=metadata["request"]["request_id"],bundle_binding=plan["package_binding"],
            task_id=plan["program_task"],task_metadata_binding=metadata["binding"],
            source_binding=source_binding(metadata),
            source_snapshot_binding=metadata["model_source"]["snapshot_binding"],
            package_validation=plan["package_validation"],reference_policy=NATIVE_POLICY,
            reference_identity_sha256=plan["reference_identity"],
            source_input_dtypes=plan["input_dtypes"])

    def _collector(self,plan,metadata,dataset,cache):
        collector=ApproximationDatasetAssessment(metadata["request"],dataset.batches(plan["part_samples"]),
            anchor=self._anchor(plan,metadata),
            application_criterion=dataset.criterion(plan["application_criterion"]))
        for name,item in cache.items():
            collector.record(name,item["receipt"],expected_binding=item["binding"])
        return collector

    @public("assessment_controller_input_rejected")
    def submit(self,program_task,dataset_path,*,expected_dataset_binding,
               application_criterion=None,max_new_assessments=0,part_samples=16,
               input_dtypes=None,trust_local_code=False,max_receipt_bytes=64*1024**2,
               cached_receipts=None,total_wall_seconds=3600):
        require(type(total_wall_seconds)is int and 1<=total_wall_seconds<=time_budget.MAX_SECONDS)
        # Base submission freezes validated data and creates a fresh journal;
        # no source/FHE execution occurs. The public handle is returned only
        # after this explicit reference-only plan and state are bound.
        handle=super().submit(program_task,dataset_path,expected_dataset_binding=expected_dataset_binding,
            reference_policy=NATIVE_POLICY,application_criterion=application_criterion,
            max_new_assessments=max_new_assessments,part_samples=part_samples,
            input_dtypes=input_dtypes,trust_local_code=trust_local_code,
            max_receipt_bytes=max_receipt_bytes,cached_receipts=None,
            total_wall_seconds=total_wall_seconds)
        path,_=self._path(handle)
        plan=bound(read(path/"plan.json"),handle["binding"])
        plan.update(assessment_kind=SUBJECT,reference_identity=reference_identity(),
            implementation_identity=identity(),cached_receipt_bindings={})
        cache={} if cached_receipts is None else snapshot(cached_receipts,limit=max_receipt_bytes)
        require(type(cache)is dict,"assessment_controller_input_rejected")
        metadata=self.numeric.outputs._metadata(program_task)
        outcome=self.application.result(program_task)
        package,grade=self.numeric.outputs._package(outcome,metadata)
        require(outcome["status"]=="succeeded" and digest(outcome)==plan["task_result_sha256"] and
            metadata["binding"]==plan["task_metadata_binding"] and
            package["binding"]==plan["package_binding"],"assessment_controller_binding_changed")
        plan["package_validation"]=grade
        dataset=load_dataset(metadata["request"],path/"dataset/dataset.json",
                             expected_binding=expected_dataset_binding)
        self._collector(plan,metadata,dataset,cache).summary()
        state=bound(read(path/"state.json"))
        for name,item in cache.items():
            require(type(item)is dict and set(item)=={"receipt","binding"})
            raw=canonical(item["receipt"]);state["receipt_bytes"]+=len(raw)
            require(state["receipt_bytes"]<=max_receipt_bytes)
            write_new(path/"receipts"/(name+".json"),item["receipt"])
            state["receipts"][name]=dict(binding=item["binding"],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            plan["cached_receipt_bindings"][name]=item["binding"]
        plan=seal(plan);handle["binding"]=plan["binding"]
        state["submission_binding"]=plan["binding"]
        update(path/"plan.json",plan);self._save(path,state)
        return handle

    def _scene(self,path,handle):
        plan=bound(read(path/"plan.json"),handle["binding"])
        require(plan.get("assessment_kind")==SUBJECT and plan.get("reference_policy")==NATIVE_POLICY and
            plan["evaluation_id"]==handle["evaluation_id"] and
            plan["implementation_identity"]==identity() and
            plan["reference_identity"]==reference_identity(),"assessment_controller_binding_changed")
        time_budget.plan_budget(plan)
        metadata=self.numeric.outputs._metadata(plan["program_task"])
        outcome=self.application.result(plan["program_task"])
        package,grade=self.numeric.outputs._package(outcome,metadata)
        require(outcome["status"]=="succeeded" and digest(outcome)==plan["task_result_sha256"] and
            metadata["binding"]==plan["task_metadata_binding"] and package["binding"]==plan["package_binding"] and
            grade==plan["package_validation"],
            "assessment_controller_binding_changed")
        dataset=load_dataset(metadata["request"],path/"dataset/dataset.json",
                             expected_binding=plan["dataset_file_binding"])
        state=bound(read(path/"state.json"))
        require(state["format"]==STATE and state["submission_binding"]==handle["binding"] and
            state["status"]in ({"queued","ready","running"}|TERMINAL) and
            type(state["used"])is int and 0<=state["used"]<=plan["max_new_assessments"] and
            type(state["ledger"])is list and len(state["ledger"])==state["used"] and
            type(state["receipts"])is dict and type(state["receipt_bytes"])is int and
            0<=state["receipt_bytes"]<=plan["max_receipt_bytes"],"assessment_controller_state_rejected")
        batches=dataset.batches(plan["part_samples"]);names={b["id"]for b in batches}
        require(set(state["receipts"])<=names and type(plan["cached_receipt_bindings"])is dict and
            set(plan["cached_receipt_bindings"])<=names,"assessment_controller_state_rejected")
        attempted=set();recorded=set()
        for index,entry in enumerate(state["ledger"]):
            require(type(entry)is dict and type(entry["number"])is int and entry["number"]==index+1 and
                entry["part_id"]in names and entry["part_id"]not in attempted and
                entry["phase"]in ("started","recorded","failed"),"assessment_controller_state_rejected")
            attempted.add(entry["part_id"])
            if entry["phase"]=="recorded":recorded.add(entry["part_id"])
        require(set(state["receipts"])==recorded|set(plan["cached_receipt_bindings"]),
                "assessment_controller_state_rejected")
        cache={}
        for name,item in state["receipts"].items():
            require(type(item)is dict and set(item)=={"binding","sha256","bytes"} and
                is_hash(item["binding"])and is_hash(item["sha256"])and type(item["bytes"])is int and
                0<item["bytes"]<=RECEIPT_BYTES,"assessment_controller_state_rejected")
            raw=regular_bytes(path/"receipts"/(name+".json"),RECEIPT_BYTES)
            require(len(raw)==item["bytes"]and hashlib.sha256(raw).hexdigest()==item["sha256"],
                    "assessment_controller_binding_changed")
            receipt=bound(read(path/"receipts"/(name+".json")),item["binding"])
            if name in plan["cached_receipt_bindings"]:
                require(item["binding"]==plan["cached_receipt_bindings"][name],
                        "assessment_controller_binding_changed")
            cache[name]=dict(receipt=receipt,binding=item["binding"])
        require(sum(item["bytes"]for item in state["receipts"].values())==state["receipt_bytes"],
                "assessment_controller_state_rejected")
        return plan,state,metadata,dataset,batches,cache

    def _frame(self,path,handle,plan,state,dataset,cache,*,busy=False):
        metadata=self.numeric.outputs._metadata(plan["program_task"])
        require(metadata["binding"]==plan["task_metadata_binding"],"assessment_controller_binding_changed")
        summary=self._collector(plan,metadata,dataset,cache).summary()
        result=dict(format=FORMAT,evaluation_id=handle["evaluation_id"],submission_binding=handle["binding"],
            status=state["status"],program_task=plan["program_task"],package_binding=plan["package_binding"],
            assessment_kind=SUBJECT,reference_policy=NATIVE_POLICY,dataset_binding=plan["dataset_file_binding"],
            worker_active=busy,cancel_requested=self._cancelled(path,handle),
            budget=dict(max_new_assessments=plan["max_new_assessments"],used=state["used"],
                remaining=plan["max_new_assessments"]-state["used"],max_receipt_bytes=plan["max_receipt_bytes"],
                receipt_bytes=state["receipt_bytes"]),
            planned_parts=plan["planned_parts"],completed_parts=len(cache),planned_samples=plan["planned_samples"],
            ledger=state["ledger"],summary=summary,failure=state["failure"],
            acceptance=dict(original_versus_approximate_on_declared_dataset=summary["complete"],
                application_criterion_validated=summary["application_criterion_validated"],
                numerical_on_declared_dataset=False,original_source_accuracy_certified=False,formal_equivalence=False),
            encrypted_execution=False,ckks_validation_performed=False,
            evidence=dict(receipts={n:v["binding"]for n,v in cache.items()}),
            new_agent_calls=0,new_compiler_calls=0,new_encrypted_calls=0)
        budget=time_budget.plan_budget(plan)
        if budget is not None:
            result["budget"].update(total_wall_seconds=budget["seconds"],deadline_wall=budget["deadline_wall"])
        return seal(snapshot(result,limit=STATE_BYTES))

    @public("assessment_controller_environment_unavailable")
    def advance(self,handle,*,max_new_parts=1):
        require(type(max_new_parts)is int and 0<=max_new_parts<=MAX_PARTS)
        path,handle=self._path(handle)
        with self._lock(path):
            plan,state,metadata,dataset,batches,cache=self._scene(path,handle)
            state=self._recover(path,state,plan,handle)
            collector=self._collector(plan,metadata,dataset,cache);calls=0
            while state["status"]not in TERMINAL:
                state=self._apply_stop(path,handle,plan,state)
                if state["status"]in TERMINAL:break
                if self._cancelled(path,handle):
                    state["status"]="cancelled";state=self._save(path,state);break
                pending=[batch for batch in batches if batch["id"]not in cache]
                if not pending:
                    summary=collector.summary()
                    state["status"]="succeeded"if summary["status"]=="passed"else "validation_failed"
                    if state["status"]=="validation_failed":
                        state["failure"]=dict(code="assessment_application_criterion_failed",
                            layer="approximation_assessment",retry_policy="never_automatic")
                    state=self._save(path,state);break
                if state["used"]>=plan["max_new_assessments"]:
                    state["status"]="budget_exhausted"
                    state["failure"]=AssessmentControllerError("assessment_controller_budget_exhausted").feedback()
                    state=self._save(path,state);break
                if calls>=max_new_parts:
                    state["status"]="ready";state=self._save(path,state);break
                batch=pending[0];name=batch["id"]
                state["used"]+=1;calls+=1;state["status"]="running"
                state["ledger"].append(dict(number=state["used"],part_id=name,phase="started",outcome=None))
                state=self._save(path,state);entry=state["ledger"][-1]
                try:
                    stop=self._stop(path,handle,plan,state)
                    if stop is not None:
                        state["status"],code=stop
                        entry.update(phase="failed",outcome=state["status"],execution_uncertain=False)
                        state["failure"]=AssessmentControllerError(code).feedback()
                        state=self._save(path,state);break
                    parent=path/"native";parent.mkdir(mode=0o700,exist_ok=True)
                    require(parent==parent.resolve() and not parent.is_symlink() and
                        parent.stat().st_uid==os.getuid(),"assessment_controller_binding_changed")
                    selection=dict(format=SOURCE_SELECTION,input_dtypes=plan["input_dtypes"],
                        trust_local_code=True,evidence_directory=str(parent/name))
                    part=collector.parts[name]
                    profile=part_criterion(collector.declaration["application_criterion"],
                        part["start"],part["samples"],metadata["request"]["model"])
                    report=self.numeric.assess(plan["program_task"],batch["samples"],
                        expected_package_binding=plan["package_binding"],
                        native_source=selection,application_criterion=profile)
                    report=snapshot(report,limit=RECEIPT_BYTES)
                    if not is_hash(report.get("binding")) or report.get("source_reference_evaluated")is not True:
                        entry.update(phase="failed",outcome=report.get("status"))
                        state["status"]=report["status"]if report.get("status")in TERMINAL else "service_error"
                        given=report.get("failure")
                        code=given.get("code")if type(given)is dict else None
                        state["failure"]=dict(code=code if code in {*METRIC_CODES,*SOURCE_CODES,"approximation_domain_rejected"} else
                            "assessment_controller_environment_unavailable",
                            layer="approximation_assessment",retry_policy="never_automatic")
                        state=self._save(path,state);break
                    collector.record(name,report,expected_binding=report["binding"])
                    raw=canonical(report)
                    if state["receipt_bytes"]+len(raw)>plan["max_receipt_bytes"]:
                        entry.update(phase="failed",outcome="budget_exhausted")
                        state["status"]="budget_exhausted"
                        state["failure"]=AssessmentControllerError("assessment_controller_receipt_budget_exhausted").feedback()
                        state=self._save(path,state);break
                    write_new(path/"receipts"/(name+".json"),report)
                    state["receipts"][name]=dict(binding=report["binding"],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
                    state["receipt_bytes"]+=len(raw)
                    entry.update(phase="recorded",outcome=report["status"])
                    state["status"]="ready";state=self._save(path,state)
                    cache[name]=dict(receipt=report,binding=report["binding"])
                except Exception:
                    entry.update(phase="failed",outcome="service_error")
                    state["status"]="service_error"
                    state["failure"]=AssessmentControllerError("assessment_controller_environment_unavailable").feedback()
                    state=self._save(path,state);break
            return self._frame(path,handle,plan,state,dataset,cache)
