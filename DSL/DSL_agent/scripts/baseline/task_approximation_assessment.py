"""Bound original-source versus approximate-graph assessment, without FHE.

This explicit host adapter reuses Application and its native source reference.
Numerical package validation, generation, and execution are unchanged.
"""
import copy
from pathlib import Path
from benchmark_graph import digest
from model_output_contract import OutputError,is_hash,snapshot
from task_output_adapter import TaskOutputAdapter,source_binding
from task_source_reference import TaskSourceReference
from source_precision_reference import source_identity as native_identity
from task_numerical_assessment import (
    AssessmentError,CODES,NATIVE_POLICY,POLICY,REFERENCE_ATOL,REFERENCE_RTOL,
    ATOL,RTOL,require,criterion,source_selection,native_values,mathematical,
    flatten,application_metric,
)
from numerical_results import compare
from program_execution import normalize_samples
from execution_input_domain import check_approximation_domains,ApproximationDomainRejected

FORMAT="poseidon-task-approximation-assessment-v1"
SUBJECT="declared-approximate-computation-v1"
ROOT=(Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[2].name == "DSL_agent" and Path(__file__).resolve().parents[3].name == "DSL" else Path(__file__).resolve().parents[2])

def implementation():
    from application_identity import source_closure,DYNAMIC
    authorities=dict(DYNAMIC)
    authorities[("source_precision_reference.py","importlib.import_module(module_name)")]=(
        "checked trusted source manifest; readonly /model; bound task and forward request")
    closure=source_closure(ROOT/"scripts/baseline",("task_approximation_assessment.py",),
                          dynamic_authorities=authorities)
    import numpy as np
    return digest(dict(format=FORMAT,subject=SUBJECT,sources=closure["sources"],
        import_edges=closure["import_edges"],dynamic_dependencies=closure["dynamic_dependencies"],
        external_imports=closure["external_imports"],numpy=np.__version__))

def approximation_metric(profile,original,approximate,model):
    # Reuse the exact-count threshold and typed criterion. The compared result
    # here is mathematical approximation, so it must not be labeled encrypted.
    value=application_metric(profile,original,approximate,approximate,model,scope=NATIVE_POLICY)
    if value.get("kind")=="top1_accuracy_drop":
        value["approximate_top1"]=value.pop("encrypted_top1")
        value.pop("approximate_computation_top1")
    value["subject"]=SUBJECT
    return value

class TaskApproximationAssessment:
    """Assess one explicit 1..16-sample group; the host owns dataset scheduling."""
    def __init__(self,application):
        self.application=application
        self.outputs=TaskOutputAdapter(application)

    @staticmethod
    def _base(task,outcome):
        return dict(format=FORMAT,task_id=task,task_status=outcome["status"],
            execution_requested=False,encrypted_execution=False,
            ckks_validation_performed=False,supplied_inputs_numerically_validated=False,
            agent_calls=0,compiler_calls=0,original_source_accuracy_certified=False,
            formal_equivalence=False,reference_policy=NATIVE_POLICY,
            application_metric_subject=SUBJECT)

    def _failed(self,task,outcome,code,*,source_requested=False):
        result=self._base(task,outcome)
        result.update(status="service_error" if code in (
            "assessment_identity_changed","assessment_environment_unavailable",
            "assessment_source_reference_rejected") else "validation_failed",
            source_reference_requested=source_requested,
            application_criterion_validated=False,
            failure=dict(code=code,layer="approximation_assessment",message=CODES[code]))
        return result

    def assess(self,task,samples,*,expected_package_binding,native_source,
               application_criterion=None):
        """Reference execution is explicit; it never invokes Application.execute."""
        self.outputs._task(task)
        outcome=self.application.result(task)
        result=self._base(task,outcome)
        if outcome["status"]!="succeeded":
            result.update(status=outcome["status"],source_reference_requested=False,
                application_criterion_validated=False,failure=copy.deepcopy(outcome.get("failure")))
            return result
        requested=False
        try:
            metadata=self.outputs._metadata(task)
            package,grade=self.outputs._package(outcome,metadata)
            require(is_hash(expected_package_binding) and package["binding"]==expected_package_binding,
                    "assessment_binding_rejected")
            request=metadata["request"];provenance=metadata.get("model_provenance")
            if provenance is not None:
                from candidate_bundle import check_provenance
                check_provenance(provenance,request["model"])
            original=provenance["original"] if provenance is not None else request["model"]
            normalized=normalize_samples(request,samples)
            profile=criterion(application_criterion,request["model"],len(normalized))
            selection=source_selection(native_source,request)
            require(selection is not None,"assessment_input_rejected")
            frozen=implementation()
            domain=check_approximation_domains(original,normalized)
            approximate=[mathematical(original,item,ideal=False) for item in normalized]
            lowered=[mathematical(request["model"],item) for item in normalized]
            decomposition=compare(flatten(lowered,request["model"]),
                flatten(approximate,request["model"]),REFERENCE_ATOL,REFERENCE_RTOL)
            require(decomposition["passed"],"assessment_decomposition_mismatch")
            wire=[{name:value.tolist() for name,value in item.items()} for item in normalized]
            requested=True
            frame=TaskSourceReference(self.application).reference(task,wire,
                selection["evidence_directory"],expected_package_binding=package["binding"],
                input_dtypes=selection["input_dtypes"],trust_local_code=selection["trust_local_code"])
            if frame.get("status")!="evaluated":
                require(implementation()==frozen,"assessment_identity_changed")
                result.update(status=frame.get("status","service_error"),
                    source_reference_requested=True,application_criterion_validated=False,
                    failure=copy.deepcopy(frame.get("failure")))
                return result
            native,frame=native_values(frame,task,metadata,package,grade,wire,selection)
            submission=snapshot(self.application._source(self.application._dir(task)),
                                limit=16*1024**2)
            require(submission["binding"]==frame["source_snapshot_binding"],
                    "assessment_source_reference_rejected")
            comparison=compare(flatten(approximate,request["model"]),
                               flatten(native,request["model"]),ATOL,RTOL)
            metric=approximation_metric(profile,native,approximate,request["model"])
            require(self.application.result(task)==outcome and
                self.application._checked_task_metadata(task)==metadata and
                self.application._source(self.application._dir(task))==submission and
                native_identity()==frame["source_reference"]["source_identity"] and
                implementation()==frozen,"assessment_identity_changed")
            passed=metric["passed"] is not False
            result.update(status="passed" if passed else "validation_failed",
                source_reference_requested=True,source_reference_evaluated=True,
                samples=len(normalized),request_id=request["request_id"],
                bundle_binding=package["binding"],task_metadata_binding=metadata["binding"],
                source_binding=source_binding(metadata),source_snapshot_binding=submission["binding"],
                source_reference_binding=frame["binding"],
                source_forward_binding=frame["source_reference"]["binding"],
                source_reference_evidence=frame["evidence"],
                source_input_dtypes=frame["source_reference"]["input_dtypes"],
                source_output_dtypes=frame["source_reference"]["output_dtypes"],
                source_selection_sha256=digest(selection),supplied_input_sha256=digest(wire),
                reference_identity_sha256=frozen,criterion_sha256=digest(profile),
                package_validation=copy.deepcopy(grade),input_domain_check=domain,
                mathematical_reference_policy=POLICY,decomposition_comparison=decomposition,
                original_source_vs_approximate_comparison=comparison,
                application_criterion=metric,application_criterion_validated=metric["passed"] is True,
                validation_scope=dict(subject=SUBJECT,declared_samples_only=True,
                    samples=len(normalized),input_sha256=digest(wire),
                    criterion_sha256=digest(profile),dataset_accuracy_certified=False),
                failure=None if passed else dict(code="assessment_application_criterion_failed",
                    layer="approximation_assessment",message=CODES["assessment_application_criterion_failed"]))
            result["binding"]=digest(result)
            return result
        except ApproximationDomainRejected as exc:
            result.update(status="validation_failed",source_reference_requested=requested,
                application_criterion_validated=False,failure=exc.public_failure)
            return result
        except AssessmentError as exc:
            return self._failed(task,outcome,exc.code,source_requested=requested)
        except OutputError:
            return self._failed(task,outcome,"assessment_binding_rejected",source_requested=requested)
        except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError):
            return self._failed(task,outcome,"assessment_reference_unavailable",source_requested=requested)
        except (OSError,RuntimeError,ImportError):
            return self._failed(task,outcome,"assessment_environment_unavailable",source_requested=requested)
