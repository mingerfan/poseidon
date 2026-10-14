"""Receipt-only dataset assessment over the existing numerical adapter.

No model loading, generation, compilation, execution, filesystem evidence
loading or automatic retries. Trusted host code supplies pinned receipts and
declares input partitions before collecting results.
"""
import functools,hashlib,math,re
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest,validate
from model_output_contract import snapshot,is_hash,decoded_tensor,OutputError
from program_execution import normalize_samples
from numerical_results import compare
from task_numerical_assessment import (FORMAT as MATH_FORMAT,NATIVE_FORMAT,POLICY,NATIVE_POLICY,
    ATOL,RTOL,REFERENCE_ATOL,REFERENCE_RTOL,criterion,application_metric,implementation as reference_identity)

FORMAT="poseidon-dataset-numerical-assessment-v1"
DECLARATION="poseidon-assessment-dataset-declaration-v1"
MAX_PARTS=4096
MAX_SAMPLES=65536
DECLARATION_BYTES=4*1024**2
RECEIPT_BYTES=16*1024**2
IDENTIFIER=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
CODES={
 "dataset_declaration_rejected":"Dataset partitions, signature or criterion are invalid.",
 "dataset_binding_rejected":"The declared dataset or supplied receipt does not match its trusted binding.",
 "dataset_reference_changed":"Assessment implementation or reference identity changed.",
 "dataset_receipt_rejected":"The receipt is incomplete, inconsistent or uses a different assessment scope.",
 "dataset_part_conflict":"This partition already has different evidence.",
}
class DatasetAssessmentError(ValueError):
    def __init__(self,code):
        if code not in CODES:raise ValueError("Unknown dataset diagnostic")
        self.code=code;super().__init__(CODES[code])
    def feedback(self):
        return dict(code=self.code,layer="assessment",message=CODES[self.code],retry_policy="never_automatic")
def require(ok,code="dataset_declaration_rejected"):
    if not ok:raise DatasetAssessmentError(code)
def public(code):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            try:return function(*args,**kwargs)
            except DatasetAssessmentError:raise
            except (ValueError,TypeError,KeyError,IndexError,OverflowError,ArithmeticError,OSError,RuntimeError):
                raise DatasetAssessmentError(code) from None
        return wrapped
    return decorate
def copied(value,limit,code):
    try:return snapshot(value,limit=limit)
    except OutputError:raise DatasetAssessmentError(code) from None
def pinned(value,expected,limit):
    value=copied(value,limit,"dataset_binding_rejected")
    require(type(value)is dict and is_hash(expected) and value.get("binding")==expected and
        digest({k:v for k,v in value.items() if k!="binding"})==expected,"dataset_binding_rejected")
    return value
def adapter_identity(policy):
    return digest(dict(adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       reference_identity=reference_identity(policy=policy),policy=policy))
@public("dataset_receipt_rejected")
def reference_anchor(receipt,*,expected_binding):
    """Only host-pinned numerical receipts establish the collection identity."""
    report=pinned(receipt,expected_binding,RECEIPT_BYTES)
    policy=report.get("reference_policy")
    require(policy in (POLICY,NATIVE_POLICY) and
        report.get("format")== (NATIVE_FORMAT if policy==NATIVE_POLICY else MATH_FORMAT) and
        report.get("encrypted_execution")is True and report.get("execution_requested")is True and
        report.get("task_status")=="succeeded" and report.get("formal_equivalence")is False and
        report.get("original_source_accuracy_certified")is False,
        "dataset_receipt_rejected")
    names=("request_id","bundle_binding","reference_policy","reference_identity_sha256",
           "source_binding","package_validation")
    require(all(n in report for n in names),"dataset_receipt_rejected")
    require(all(is_hash(report[n]) for n in ("request_id","bundle_binding","reference_identity_sha256")) and
        (report["source_binding"]is None or is_hash(report["source_binding"])),"dataset_receipt_rejected")
    require(report["reference_identity_sha256"]==reference_identity(policy=policy),"dataset_reference_changed")
    grade=report["package_validation"]
    require(type(grade)is dict and set(grade)=={"level","encrypted_execution","numerically_validated"} and
        grade["level"]in ("compiled","numerical") and
        grade["encrypted_execution"]is (grade["level"]=="numerical") and
        grade["numerically_validated"]is (grade["level"]=="numerical"),"dataset_receipt_rejected")
    if policy==NATIVE_POLICY:
        names+=("source_snapshot_binding","source_input_dtypes","source_output_dtypes")
        require(all(n in report for n in names) and report.get("source_reference_evaluated")is True and
            is_hash(report["source_snapshot_binding"]) and is_hash(report["source_binding"]),
            "dataset_receipt_rejected")
    return {n:report[n] for n in names}
def part_criterion(profile,start,count,model):
    if profile is None:return None
    value=dict(profile)
    if value["kind"]=="top1_accuracy_drop":value["labels"]=profile["labels"][start:start+count]
    return criterion(value,model,count)
@public("dataset_declaration_rejected")
def declare_dataset(request,batches,*,anchor,application_criterion=None):
    """Hash bounded input partitions; returned declaration contains no input arrays."""
    request=copied(request,DECLARATION_BYTES,"dataset_declaration_rejected")
    anchor=copied(anchor,131072,"dataset_declaration_rejected")
    require(type(request)is dict and type(request.get("model"))is dict and is_hash(request.get("request_id")))
    validate(request["model"])
    require(type(anchor)is dict and anchor.get("request_id")==request["request_id"] and
        anchor.get("reference_policy")in (POLICY,NATIVE_POLICY))
    require(type(batches)is list and 1<=len(batches)<=MAX_PARTS)
    parts=[];sample_names=set();part_names=set();start=0
    for batch in batches:
        require(type(batch)is dict and set(batch)=={"id","sample_ids","samples"})
        name=batch["id"];ids=batch["sample_ids"]
        require(type(name)is str and IDENTIFIER.fullmatch(name) and name not in part_names)
        require(type(ids)is list and 1<=len(ids)<=16 and all(type(i)is str and IDENTIFIER.fullmatch(i) for i in ids)
                and len(set(ids))==len(ids) and not sample_names.intersection(ids))
        normalized=normalize_samples(request,batch["samples"])
        require(len(normalized)==len(ids) and start+len(ids)<=MAX_SAMPLES)
        wire=[{n:a.tolist() for n,a in sample.items()} for sample in normalized]
        require(len(canonical(wire))<=4*1024**2)
        parts.append(dict(id=name,sample_ids=list(ids),start=start,samples=len(ids),supplied_input_sha256=digest(wire)))
        part_names.add(name);sample_names.update(ids);start+=len(ids)
    profile=criterion(application_criterion,request["model"],start)
    for part in parts:
        part["criterion_sha256"]=digest(part_criterion(profile,part["start"],part["samples"],request["model"]))
    declaration=dict(format=DECLARATION,request_id=request["request_id"],model_sha256=digest(request["model"]),
        anchor=anchor,application_criterion=profile,samples=start,parts=parts)
    require(len(canonical(declaration))<=DECLARATION_BYTES)
    declaration["binding"]=digest(declaration)
    return declaration
def signature(model):
    checked=validate(model)
    return [(o["name"],list(checked["shapes"][o["value"]])) for o in model["outputs"]]
def shaped(flat,model):
    result=[];spec=signature(model)
    for row in flat:
        item={};offset=0
        for name,shape in spec:
            n=math.prod(shape);item[name]=row[offset:offset+n].reshape(shape);offset+=n
        result.append(item)
    return result
def comparison(report,name,count,elements,atol,rtol):
    value=report.get(name)
    require(type(value)is dict and value.get("atol")==atol and value.get("rtol")==rtol and
        type(value.get("compared_values"))is int and value["compared_values"]==count*elements,
        "dataset_receipt_rejected")
    try:
        actual=np.asarray(decoded_tensor(value["actual"],[count,elements]),dtype=np.float64)
        reference=np.asarray(decoded_tensor(value["reference"],[count,elements]),dtype=np.float64)
        checked=compare(actual,reference,atol,rtol)
    except (KeyError,ValueError,TypeError,OverflowError):
        raise DatasetAssessmentError("dataset_receipt_rejected") from None
    # Recalculate elementwise gates and scalar errors; cosine remains report-only.
    require(all(value.get(k)==checked[k] for k in (
        "passed","mae","max_absolute_error","max_nonzero_reference_relative_error",
        "nonzero_reference_count","absolute_error","relative_error","elementwise_pass")),
        "dataset_receipt_rejected")
    return actual,reference,checked
def statistics(actual,reference):
    error=np.abs(actual-reference);scale=float(max(np.abs(actual).max(),np.abs(reference).max()))
    if scale:
        a=actual/scale;r=reference/scale
        squared_a=float(np.sum(a*a));squared_r=float(np.sum(r*r));dot=float(np.sum(a*r))
    else:squared_a=squared_r=dot=0.
    nz=reference!=0
    maximum_relative=float(np.max(error[nz]/np.abs(reference[nz]))) if nz.any() else None
    result=dict(values=int(actual.size),absolute_error_sum=float(np.sum(error)),
        max_absolute_error=float(np.max(error)),nonzero_reference_count=int(nz.sum()),
        max_nonzero_reference_relative_error=maximum_relative,
        scale=scale,squared_actual=squared_a,squared_reference=squared_r,scaled_dot=dot)
    require(all(math.isfinite(v) for v in result.values() if type(v)is float),"dataset_receipt_rejected")
    return result
def combined_statistics(items):
    if not items:return None
    values=sum(x["values"] for x in items);scale=max(x["scale"] for x in items)
    squared_a=squared_r=dot=0.
    for x in items:
        ratio=(x["scale"]/scale)**2 if scale else 0.
        squared_a+=x["squared_actual"]*ratio;squared_r+=x["squared_reference"]*ratio;dot+=x["scaled_dot"]*ratio
    cosine=dot/math.sqrt(squared_a*squared_r) if squared_a>0 and squared_r>0 else None
    return dict(compared_values=values,mae=math.fsum(x["absolute_error_sum"] for x in items)/values,
        max_absolute_error=max(x["max_absolute_error"] for x in items),
        nonzero_reference_count=sum(x["nonzero_reference_count"] for x in items),
        max_nonzero_reference_relative_error=max((x["max_nonzero_reference_relative_error"] for x in items
            if x["max_nonzero_reference_relative_error"]is not None),default=None),
        cosine_similarity=cosine)
class DatasetNumericalAssessment:
    """Collect receipts once; missing, failed and stale partitions stay visible."""
    @public("dataset_declaration_rejected")
    def __init__(self,request,declaration,*,expected_declaration_binding):
        self.request=copied(request,DECLARATION_BYTES,"dataset_declaration_rejected")
        self.declaration=pinned(declaration,expected_declaration_binding,DECLARATION_BYTES)
        data=self.declaration
        require(data.get("format")==DECLARATION and data.get("request_id")==self.request.get("request_id") and
            data.get("model_sha256")==digest(self.request["model"]))
        self.anchor=data["anchor"];self.policy=self.anchor["reference_policy"]
        require(self.policy in (POLICY,NATIVE_POLICY) and
            self.anchor["reference_identity_sha256"]==reference_identity(policy=self.policy),"dataset_reference_changed")
        names={"request_id","bundle_binding","reference_policy","reference_identity_sha256",
               "source_binding","package_validation"}
        if self.policy==NATIVE_POLICY:names.update(
            ("source_snapshot_binding","source_input_dtypes","source_output_dtypes"))
        require(type(self.anchor)is dict and set(self.anchor)==names and
            self.anchor["request_id"]==self.request["request_id"] and
            is_hash(self.anchor["bundle_binding"]) and
            (self.anchor["source_binding"]is None or is_hash(self.anchor["source_binding"])))
        if self.policy==NATIVE_POLICY:
            require(is_hash(self.anchor["source_snapshot_binding"]) and is_hash(self.anchor["source_binding"]))
            input_names={i["name"] for i in self.request["model"]["inputs"]}
            output_names={o["name"] for o in self.request["model"]["outputs"]}
            for field,keys in (("source_input_dtypes",input_names),("source_output_dtypes",output_names)):
                dtype=self.anchor[field]
                require(type(dtype)is dict and set(dtype)==keys and
                    all(type(v)is str and v in ("float32","float64") for v in dtype.values()))
        self.expected_binding=expected_declaration_binding
        self.identity=adapter_identity(self.policy);self.records={}
        # Validate the partition index without reading input values again.
        require(type(data["parts"])is list and 1<=len(data["parts"])<=MAX_PARTS)
        ids=set();names=set();start=0
        for part in data["parts"]:
            require(type(part)is dict and set(part)=={"id","sample_ids","start","samples","supplied_input_sha256","criterion_sha256"} and
                type(part["id"])is str and IDENTIFIER.fullmatch(part["id"]) and part["id"]not in names and
                type(part["samples"])is int and 1<=part["samples"]<=16 and
                type(part["start"])is int and part["start"]==start and
                type(part["sample_ids"])is list and len(part["sample_ids"])==part["samples"] and
                all(type(i)is str and IDENTIFIER.fullmatch(i) for i in part["sample_ids"]) and
                len(set(part["sample_ids"]))==part["samples"] and not ids.intersection(part["sample_ids"]) and
                is_hash(part["supplied_input_sha256"]) and is_hash(part["criterion_sha256"]))
            profile=part_criterion(data["application_criterion"],start,part["samples"],self.request["model"])
            require(digest(profile)==part["criterion_sha256"])
            names.add(part["id"]);ids.update(part["sample_ids"]);start+=part["samples"]
        require(type(data["samples"])is int and data["samples"]==start<=MAX_SAMPLES)
        self.parts={p["id"]:p for p in data["parts"]}
    def _check(self,part_id=None):
        require(self.declaration["binding"]==self.expected_binding and
            digest({k:v for k,v in self.declaration.items() if k!="binding"})==self.expected_binding and
            digest(self.request["model"])==self.declaration["model_sha256"] and
            self.request["request_id"]==self.declaration["request_id"],"dataset_binding_rejected")
        require(adapter_identity(self.policy)==self.identity,"dataset_reference_changed")
        if part_id is not None:require(type(part_id)is str and part_id in self.parts)
    @public("dataset_receipt_rejected")
    def record(self,part_id,receipt,*,expected_binding):
        self._check(part_id);part=self.parts[part_id]
        report=pinned(receipt,expected_binding,RECEIPT_BYTES)
        if part_id in self.records:
            require(self.records[part_id]["receipt_binding"]==expected_binding,"dataset_part_conflict")
            return dict(status="already_recorded",part_id=part_id)
        require(all(r["receipt_binding"]!=expected_binding for r in self.records.values()),
                "dataset_part_conflict")
        require(reference_anchor(report,expected_binding=expected_binding)==self.anchor,"dataset_receipt_rejected")
        require(report.get("supplied_input_sha256")==part["supplied_input_sha256"] and
            report.get("criterion_sha256")==part["criterion_sha256"] and
            type(report.get("samples"))is int and report["samples"]==part["samples"] and
            type(report.get("agent_calls"))is int and report["agent_calls"]==0 and
            type(report.get("compiler_calls"))is int and report["compiler_calls"]==0,
            "dataset_receipt_rejected")
        count=part["samples"];model=self.request["model"];elements=sum(math.prod(s) for _,s in signature(model))
        actual,reference,numerical=comparison(report,"ckks_comparison",count,elements,ATOL,RTOL)
        lower,approximate,decomposition=comparison(report,"decomposition_comparison",count,elements,REFERENCE_ATOL,REFERENCE_RTOL)
        require(decomposition["passed"] and np.array_equal(reference,lower),"dataset_receipt_rejected")
        target=report.get("source_total_comparison") if self.policy==NATIVE_POLICY else report.get("total_comparison")
        source_stats=None;original=None
        if target is not None:
            name="source_total_comparison" if self.policy==NATIVE_POLICY else "total_comparison"
            final,original,_=comparison(report,name,count,elements,ATOL,RTOL)
            require(np.array_equal(actual,final),"dataset_receipt_rejected")
            source_stats=statistics(actual,original)
        profile=part_criterion(self.declaration["application_criterion"],part["start"],count,model)
        require(original is not None or profile is None,"dataset_receipt_rejected")
        metric=application_metric(profile,shaped(original,model) if original is not None else None,
            shaped(approximate,model),shaped(actual,model),model,scope=self.policy)
        require(report.get("application_criterion")==metric,"dataset_receipt_rejected")
        passed=numerical["passed"] and metric["passed"]is not False
        failure=None if passed else ("assessment_numerical_mismatch" if not numerical["passed"] else
                                    "assessment_application_criterion_failed")
        require(report.get("status")==("passed" if passed else "validation_failed") and
            report.get("supplied_inputs_numerically_validated")is numerical["passed"] and
            (report.get("failure")is None if passed else
                type(report.get("failure"))is dict and report["failure"].get("code")==failure),
            "dataset_receipt_rejected")
        counts=None
        if profile is not None and profile["kind"]=="top1_accuracy_drop":
            labels=[np.asarray(x,dtype=np.int64) for x in profile["labels"]]
            scores=[]
            for group in (shaped(original,model),shaped(approximate,model),shaped(actual,model)):
                scores.append(sum(int(np.sum(np.argmax(s[profile["output"]],axis=profile["class_axis"])==t))
                                  for s,t in zip(group,labels)))
            counts=dict(original_correct=scores[0],approximate_correct=scores[1],encrypted_correct=scores[2],
                        labels_compared=sum(int(t.size) for t in labels))
        self._check()
        self.records[part_id]=dict(status=report["status"],receipt_binding=expected_binding,samples=count,
            ckks_passed=numerical["passed"],application_passed=metric["passed"],failure_code=failure,
            ckks_statistics=statistics(actual,reference),original_statistics=source_stats,top1_counts=counts)
        return dict(status="recorded",part_id=part_id,part_status=report["status"])
    @public("dataset_receipt_rejected")
    def record_unavailable(self,part_id,outcome,*,expected_outcome_sha256):
        """A trusted missing/failed execution can never become numerical success."""
        self._check(part_id);outcome=copied(outcome,RECEIPT_BYTES,"dataset_receipt_rejected")
        require(type(outcome)is dict and is_hash(expected_outcome_sha256) and
            digest(outcome)==expected_outcome_sha256,"dataset_binding_rejected")
        status=outcome.get("status")
        require(status in ("cancelled","timed_out","unsupported","service_error","validation_failed","budget_exhausted") and
            outcome.get("encrypted_execution")is not True and "ckks_comparison"not in outcome,
            "dataset_receipt_rejected")
        if part_id in self.records:
            require(self.records[part_id]["receipt_binding"]==expected_outcome_sha256,"dataset_part_conflict")
            return dict(status="already_recorded",part_id=part_id)
        self.records[part_id]=dict(status=status,receipt_binding=expected_outcome_sha256,
            samples=0,ckks_passed=None,application_passed=None,failure_code="dataset_part_unavailable",
            ckks_statistics=None,original_statistics=None,top1_counts=None)
        return dict(status="recorded",part_id=part_id,part_status=status)
    @public("dataset_reference_changed")
    def summary(self):
        self._check();data=self.declaration
        entries=[dict(id=p["id"],sample_ids=list(p["sample_ids"]),planned_samples=p["samples"],
            **self.records.get(p["id"],dict(status="not_run",receipt_binding=None,samples=0,
                ckks_passed=None,application_passed=None,failure_code=None,
                ckks_statistics=None,original_statistics=None,top1_counts=None))) for p in data["parts"]]
        complete=all(p["samples"]==p["planned_samples"] for p in entries)
        numeric=all(p["ckks_passed"]is True for p in entries) if complete else None
        evaluated=[p for p in entries if p["ckks_statistics"]is not None]
        ckks=combined_statistics([p["ckks_statistics"] for p in evaluated])
        targets=[p["original_statistics"] for p in evaluated if p["original_statistics"]is not None]
        total=combined_statistics(targets)
        profile=data["application_criterion"];metric=dict(status="not_requested",passed=None,scope=self.policy)
        if profile is not None:
            if not complete:
                metric=dict(status="incomplete",passed=None,scope=self.policy)
            elif profile["kind"]=="max_absolute_error":
                value=total["max_absolute_error"];passed=value<=profile["limit"]
                metric=dict(kind=profile["kind"],status="passed" if passed else "failed",passed=passed,
                            limit=profile["limit"],max_absolute_error=value,scope=self.policy)
            else:
                scores={key:sum(p["top1_counts"][key] for p in evaluated) for key in (
                    "original_correct","approximate_correct","encrypted_correct","labels_compared")}
                n=scores["labels_compared"];drop=(scores["original_correct"]-scores["encrypted_correct"])/n
                passed=drop<=profile["max_drop"]
                metric=dict(kind=profile["kind"],status="passed" if passed else "failed",passed=passed,
                    original_top1=scores["original_correct"]/n,approximate_top1=scores["approximate_correct"]/n,
                    encrypted_top1=scores["encrypted_correct"]/n,accuracy_drop=drop,max_drop=profile["max_drop"],
                    counts=scores,scope=self.policy)
        passed=complete and numeric is True and metric["passed"]is not False
        status="passed" if passed else "incomplete" if not complete else "validation_failed"
        # All original per-part outcomes remain visible even when the declared
        # global Top-1 criterion passes after weighting by label counts.
        result=dict(format=FORMAT,status=status,dataset_binding=data["binding"],request_id=data["request_id"],
            anchor=data["anchor"],aggregation_identity=self.identity,planned_samples=data["samples"],
            evaluated_samples=sum(p["samples"] for p in evaluated),
            numerically_validated_samples=sum(p["samples"] for p in evaluated if p["ckks_passed"]is True),
            part_status_counts={status:sum(p["status"]==status for p in entries)
                for status in sorted({p["status"] for p in entries})},complete=complete,
            supplied_inputs_numerically_validated=(numeric is True),
            application_criterion=metric,criterion_sha256=digest(profile),ckks_comparison=ckks,
            total_vs_original_reference=total,parts=entries,
            part_application_failures=sum(p["application_passed"]is False for p in evaluated),
            part_numerical_failures=sum(p["ckks_passed"]is False for p in evaluated),
            numerical_receipts=len(evaluated),
            dataset_criterion_validated=(passed and metric["passed"]is True),
            original_source_accuracy_certified=False,formal_equivalence=False,
            declared_dataset_only=True,agent_calls=0,compiler_calls=0,new_encrypted_calls=0)
        result=copied(result,RECEIPT_BYTES,"dataset_receipt_rejected")
        result["binding"]=digest(result)
        return result
