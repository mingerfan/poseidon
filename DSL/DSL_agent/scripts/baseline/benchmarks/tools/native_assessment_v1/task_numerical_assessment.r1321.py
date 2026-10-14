"""Explicit, bounded numerical assessment of an existing Application package.

Reference calculations use the registered mathematical graphs in float64. The
existing Application owns package validation and execution. No generation,
compilation, Provider calls, package rewriting, or automatic retry is added.
"""
import copy
import math
from pathlib import Path
import numpy as np

from benchmark_graph import canonical, digest, validate
from model_output_contract import decoded_tensor, is_hash, snapshot, OutputError
from task_output_adapter import TaskOutputAdapter, source_binding
from program_execution import normalize_samples
from numerical_results import compare
from execution_input_domain import check_approximation_domains, ApproximationDomainRejected

FORMAT = "poseidon-task-numerical-assessment-v1"
POLICY = "registered-mathematical-float64-v1"
ATOL, RTOL = 1e-5, 1e-4
REFERENCE_ATOL, REFERENCE_RTOL = 1e-12, 1e-12
ROOT = Path(__file__).resolve().parents[2]
CODES = {
    "assessment_binding_rejected": "Program or task binding does not match.",
    "assessment_input_rejected": "Input samples or application criterion are invalid.",
    "assessment_reference_unavailable": "The registered mathematical reference could not assess these inputs.",
    "assessment_decomposition_mismatch": "The declared approximate computation and decomposed graph differ.",
    "assessment_execution_rejected": "Execution output or identity does not match the checked program.",
    "assessment_numerical_mismatch": "Decrypted output exceeds the fixed CKKS comparison threshold.",
    "assessment_application_criterion_failed": "The explicit application criterion was not met.",
    "assessment_identity_changed": "Assessment dependencies changed during evaluation.",
    "assessment_environment_unavailable": "Assessment dependencies are unavailable.",
}
class AssessmentError(ValueError):
    def __init__(self, code):
        if code not in CODES:raise ValueError("Unknown assessment diagnostic")
        self.code=code
        super().__init__(CODES[code])
def require(ok, code):
    if not ok:raise AssessmentError(code)
def implementation():
    from application_identity import source_closure
    closure=source_closure(ROOT/"scripts/baseline",("task_numerical_assessment.py",))
    return digest(dict(policy=POLICY,sources=closure["sources"],
                       import_edges=closure["import_edges"],
                       dynamic_dependencies=closure["dynamic_dependencies"],
                       external_imports=closure["external_imports"],numpy=np.__version__))
def mathematical(model, inputs, *, ideal=False):
    from operator_arithmetic_graph import FORMATS,evaluate_reference
    if model.get("format") in FORMATS:
        return evaluate_reference(model,inputs,ideal=ideal)
    from benchmark_math import evaluate
    return evaluate(model,inputs)
def criterion(value, model, count):
    """Only declared metrics; no user callbacks, implicit budgets or fitting."""
    if value is None:return None
    try:
        value=snapshot(value,limit=4*1024**2)
    except OutputError:
        raise AssessmentError("assessment_input_rejected") from None
    require(type(value)is dict,"assessment_input_rejected")
    kind=value.get("kind")
    require(type(kind)is str,"assessment_input_rejected")
    if kind=="max_absolute_error":
        require(set(value)=={"kind","limit"} and type(value["limit"])in(int,float)
                and math.isfinite(value["limit"]) and 0<=value["limit"]<=1024,
                "assessment_input_rejected")
        return dict(kind=kind,limit=float(value["limit"]))
    require(kind=="top1_accuracy_drop" and
            set(value)=={"kind","output","class_axis","labels","max_drop"} and
            type(value["class_axis"])is int and
            type(value["output"])is str and
            type(value["labels"])is list and len(value["labels"])==count and
            type(value["max_drop"])in(int,float) and math.isfinite(value["max_drop"]) and
            0<=value["max_drop"]<=1,"assessment_input_rejected")
    checked=validate(model);refs={o["name"]:o["value"] for o in model["outputs"]}
    require(value["output"] in refs,"assessment_input_rejected")
    shape=checked["shapes"][refs[value["output"]]]
    axis=value["class_axis"]
    require(-len(shape)<=axis<len(shape),"assessment_input_rejected")
    axis%=len(shape);label_shape=list(shape[:axis]+shape[axis+1:])
    def labels(item,dims):
        if dims:
            require(type(item)is list and len(item)==dims[0],"assessment_input_rejected")
            return [labels(x,dims[1:]) for x in item]
        require(type(item)is int and 0<=item<shape[axis],"assessment_input_rejected")
        return item
    return dict(kind=kind,output=value["output"],class_axis=axis,
                labels=[labels(item,label_shape) for item in value["labels"]],
                max_drop=float(value["max_drop"]))
def flatten(values,model):
    return np.stack([np.concatenate([sample[o["name"]].reshape(-1)
                    for o in model["outputs"]]) for sample in values])
def decoded(execution,request,binding,grade,count):
    require(type(execution)is dict and execution.get("status")=="executed" and
            execution.get("format")=="poseidon-local-execution-v1" and
            execution.get("backend")=="poseidon_cpu" and
            execution.get("request_id")==request["request_id"] and
            execution.get("bundle_binding")==binding and
            execution.get("encrypted_execution")is True and execution.get("executor_secret_key")is False and
            type(execution.get("samples"))is int and execution["samples"]==count and
            type(execution.get("agent_calls"))is int and execution["agent_calls"]==0 and
            type(execution.get("compiler_calls"))is int and execution["compiler_calls"]==0 and
            execution.get("package_validation")==grade,
            "assessment_execution_rejected")
    metadata=execution.get("execution")
    require(type(metadata)is dict and metadata.get("executor_secret_key")is False,
            "assessment_execution_rejected")
    values=execution.get("outputs");checked=validate(request["model"])
    names={o["name"] for o in request["model"]["outputs"]}
    require(type(values)is list and len(values)==count,"assessment_execution_rejected")
    outputs=[]
    for sample in values:
        require(type(sample)is dict and set(sample)==names,"assessment_execution_rejected")
        outputs.append({o["name"]:np.asarray(decoded_tensor(sample[o["name"]],
                       list(checked["shapes"][o["value"]])),dtype=np.float64)
                       for o in request["model"]["outputs"]})
    return outputs
def application_metric(profile,original,approximate,decrypted,model):
    if profile is None:
        return dict(status="not_requested",passed=None,scope=POLICY)
    if profile["kind"]=="max_absolute_error":
        error=np.abs(flatten(decrypted,model)-flatten(original,model))
        value=float(error.max())
        return dict(kind=profile["kind"],status="passed" if value<=profile["limit"] else "failed",
                    passed=value<=profile["limit"],limit=profile["limit"],
                    max_absolute_error=value,scope=POLICY)
    name=profile["output"];axis=profile["class_axis"]
    labels=[np.asarray(x,dtype=np.int64) for x in profile["labels"]]
    refs=(original,approximate,decrypted);scores=[]
    for group in refs:
        correct=total=0
        for sample,target in zip(group,labels):
            prediction=np.argmax(sample[name],axis=axis)
            correct+=int(np.sum(prediction==target));total+=int(target.size)
        scores.append(correct/total)
    drop=scores[0]-scores[2]
    return dict(kind=profile["kind"],status="passed" if drop<=profile["max_drop"] else "failed",
        passed=drop<=profile["max_drop"],output=name,class_axis=axis,
        original_mathematical_top1=scores[0],approximate_mathematical_top1=scores[1],
        encrypted_top1=scores[2],accuracy_drop=drop,max_drop=profile["max_drop"],
        labels_compared=sum(t.size for t in labels),tie_policy="first_index",scope=POLICY)
class TaskNumericalAssessment:
    """Optional host adapter over the same Application; no new Agent instance."""
    def __init__(self, application):
        self.outputs=TaskOutputAdapter(application)
        self.application=application
    def _failure(self, task, outcome, code, requested=False):
        return dict(format=FORMAT,status=("service_error" if code in
                    ("assessment_identity_changed","assessment_environment_unavailable")
                    else "validation_failed"),task_id=task,execution_requested=requested,
                    task_status=outcome.get("status"),failure=dict(code=code,layer="assessment",
                    message=CODES[code]),agent_calls=0,compiler_calls=0,
                    supplied_inputs_numerically_validated=False,
                    original_source_accuracy_certified=False)
    def assess(self,task,samples,*,expected_package_binding,application_criterion=None):
        """Assess 1..16 new inputs. Retain the returned binding in trusted host code."""
        self.outputs._task(task)
        outcome=self.application.result(task)
        if outcome["status"]!="succeeded":
            return dict(format=FORMAT,status=outcome["status"],task_id=task,
                        task_status=outcome["status"],failure=copy.deepcopy(outcome.get("failure")),
                        agent_calls=0,compiler_calls=0,
                        supplied_inputs_numerically_validated=False,
                        original_source_accuracy_certified=False)
        requested=False
        try:
            metadata=self.outputs._metadata(task)
            package,grade=self.outputs._package(outcome,metadata)
            require(is_hash(expected_package_binding) and package["binding"]==expected_package_binding,
                    "assessment_binding_rejected")
            request=metadata["request"]
            provenance=metadata.get("model_provenance")
            if provenance is not None:
                from candidate_bundle import check_provenance
                check_provenance(provenance,request["model"])
            original=provenance["original"] if provenance is not None else request["model"]
            normalized=normalize_samples(request,samples)
            profile=criterion(application_criterion,request["model"],len(normalized))
            identity=implementation();domain=check_approximation_domains(original,normalized)
            approximate=[mathematical(original,s,ideal=False) for s in normalized]
            lowered=[mathematical(request["model"],s) for s in normalized]
            lower_flat=flatten(lowered,request["model"])
            approximate_flat=flatten(approximate,request["model"])
            decomposition=compare(lower_flat,approximate_flat,REFERENCE_ATOL,REFERENCE_RTOL)
            require(decomposition["passed"],"assessment_decomposition_mismatch")
            try:
                ideal=[mathematical(original,s,ideal=True) for s in normalized]
                ideal_flat=flatten(ideal,request["model"])
                require(ideal_flat.shape==lower_flat.shape and np.isfinite(ideal_flat).all(),
                        "assessment_reference_unavailable")
            except (ValueError,KeyError,TypeError,OverflowError,FloatingPointError):
                if profile is not None:raise AssessmentError("assessment_reference_unavailable") from None
                ideal=None;ideal_flat=None
            # Execution receives detached values, never the caller's mutable buffers.
            wire=[{n:a.tolist() for n,a in s.items()} for s in normalized]
            requested=True
            execution=self.application.execute(task,wire)
            if execution.get("status")!="executed":
                result=dict(format=FORMAT,status=execution.get("status","service_error"),task_id=task,
                            task_status=outcome["status"],execution=snapshot(execution,limit=4*1024**2),
                            agent_calls=0,compiler_calls=0,supplied_inputs_numerically_validated=False,
                            original_source_accuracy_certified=False)
                require(implementation()==identity,"assessment_identity_changed")
                return result
            values=decoded(execution,request,package["binding"],grade,len(normalized))
            numerical=compare(flatten(values,request["model"]),lower_flat,ATOL,RTOL)
            if ideal is not None:
                approximation=compare(approximate_flat,ideal_flat,ATOL,RTOL)
                total=compare(flatten(values,request["model"]),ideal_flat,ATOL,RTOL)
                metric=application_metric(profile,ideal,approximate,values,request["model"])
            else:
                approximation=total=None
                metric=dict(status="not_requested",passed=None,scope=POLICY)
            require(self.application.result(task)==outcome and
                    self.application._checked_task_metadata(task)==metadata and
                    implementation()==identity,"assessment_identity_changed")
            passed=numerical["passed"] and metric["passed"]is not False
            code=None if passed else ("assessment_numerical_mismatch" if not numerical["passed"]
                                      else "assessment_application_criterion_failed")
            result=dict(format=FORMAT,status="passed" if passed else "validation_failed",
                task_id=task,task_status=outcome["status"],execution_requested=True,encrypted_execution=True,request_id=request["request_id"],
                bundle_binding=package["binding"],task_metadata_binding=metadata["binding"],
                source_binding=source_binding(metadata),reference_policy=POLICY,
                reference_identity_sha256=identity,samples=len(normalized),
                supplied_input_sha256=digest(wire),criterion_sha256=digest(profile),
                package_validation=copy.deepcopy(grade),input_domain_check=domain,
                decomposition_comparison=decomposition,ckks_comparison=numerical,
                approximation_comparison=approximation,total_comparison=total,
                original_mathematical_reference="available" if ideal is not None else "unavailable",
                application_criterion=metric,supplied_inputs_numerically_validated=numerical["passed"],
                original_source_accuracy_certified=False,formal_equivalence=False,
                execution_evidence=execution.get("evidence"),agent_calls=0,compiler_calls=0,
                failure=None if code is None else dict(code=code,layer="assessment",message=CODES[code]))
            result["binding"]=digest(result)
            return result
        except ApproximationDomainRejected as exc:
            return dict(format=FORMAT,status="validation_failed",task_id=task,
                        task_status=outcome["status"],failure=exc.public_failure,
                        agent_calls=0,compiler_calls=0,supplied_inputs_numerically_validated=False,
                        original_source_accuracy_certified=False)
        except AssessmentError as exc:
            return self._failure(task,outcome,exc.code,requested)
        except OutputError:
            return self._failure(task,outcome,"assessment_execution_rejected",requested)
        except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError):
            return self._failure(task,outcome,"assessment_reference_unavailable",requested)
        except (OSError,RuntimeError,ImportError):
            return self._failure(task,outcome,"assessment_environment_unavailable",requested)
