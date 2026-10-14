"""Application metrics over supplied reference/output data; no execution certification."""
import hashlib,json,math
from fractions import Fraction
FORMAT="poseidon-application-metric-calculation-v1"
MAX_SAMPLES=16
MAX_VALUES=4096

def require(condition,message):
    if not condition:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def matrix(value):
    require(type(value)is list and 1<=len(value)<=MAX_SAMPLES,"Metric sample count")
    columns=None;result=[]
    for row in value:
        require(type(row)is list and 1<=len(row)<=MAX_VALUES,"Metric output count")
        require(columns is None or columns==len(row),"Metric rectangular outputs");columns=len(row)
        require(all(type(v)in(int,float)and math.isfinite(v)for v in row),"Metric finite numeric outputs")
        result.append([float(v)for v in row])
    return result
def checked_inputs(original,approximate,decrypted):
    a,b,c=map(matrix,(original,approximate,decrypted))
    require(len(a)==len(b)==len(c)and len(a[0])==len(b[0])==len(c[0]),"Metric shape mismatch")
    return a,b,c
def errors(actual,reference):
    differences=[abs(a-b)for ra,rb in zip(actual,reference)for a,b in zip(ra,rb)]
    require(all(math.isfinite(x)for x in differences),"Metric error outside float64 range")
    return dict(values=len(differences),mae=math.fsum(differences)/len(differences),
        maximum_absolute_error=max(differences),
        elementwise_threshold_passed=all(abs(a-b)<=1e-5+1e-4*abs(b)
            for ra,rb in zip(actual,reference)for a,b in zip(ra,rb)))
def top1(values,labels):
    predictions=[max(range(len(row)),key=lambda i:row[i])for row in values]
    correct=sum(a==b for a,b in zip(predictions,labels))
    return dict(correct=correct,samples=len(labels),accuracy=correct/len(labels),
        percentage=100.*correct/len(labels),tie_policy="first_index")
def calculation(original,approximate,decrypted,metric):
    original,approximate,decrypted=checked_inputs(original,approximate,decrypted)
    require(type(metric)is dict,"Explicit application metric")
    require(metric.get("kind")in("top1_drop","mae","maximum_absolute_error"),"Registered application metric")
    if metric["kind"]=="top1_drop":
        require(set(metric)=={"kind","labels","maximum_drop_percentage_points"},"Classification metric fields")
        labels=metric["labels"];limit=metric["maximum_drop_percentage_points"]
        require(type(labels)is list and len(labels)==len(original)and
            all(type(v)is int and 0<=v<len(original[0])for v in labels),"Classification labels")
        require(type(limit)in(int,float)and math.isfinite(limit)and 0<=limit<=100,"Explicit accuracy drop limit")
        exact,plain,encrypted=(top1(v,labels)for v in(original,approximate,decrypted))
        plain_fraction=Fraction(100*(exact["correct"]-plain["correct"]),len(labels))
        encrypted_fraction=Fraction(100*(exact["correct"]-encrypted["correct"]),len(labels))
        plain_drop=float(plain_fraction);encrypted_drop=float(encrypted_fraction)
        declared_limit=Fraction(str(limit))
        app=dict(original=exact,approximate=plain,decrypted=encrypted,
            approximation_drop_percentage_points=plain_drop,total_drop_percentage_points=encrypted_drop,
            approximation_metric_passed=plain_fraction<=declared_limit,decrypted_metric_passed=encrypted_fraction<=declared_limit,
            threshold_comparison="exact counts against declared JSON decimal limit")
    else:
        require(set(metric)=={"kind","maximum_error"},"Regression metric fields")
        limit=metric["maximum_error"]
        require(type(limit)in(int,float)and math.isfinite(limit)and limit>=0,"Explicit application error limit")
        plain,total=errors(approximate,original),errors(decrypted,original)
        field="mae"if metric["kind"]=="mae"else"maximum_absolute_error"
        app=dict(original_reference="supplied",approximation_value=plain[field],total_value=total[field],
            approximation_metric_passed=plain[field]<=limit,decrypted_metric_passed=total[field]<=limit)
    result=dict(format=FORMAT,samples=len(original),outputs_per_sample=len(original[0]),
        dataset_outputs_sha256=digest(dict(original=original,approximate=approximate,decrypted=decrypted)),
        metric=dict(metric),application_metric=app,approximation_error=errors(approximate,original),
        execution_error=errors(decrypted,approximate),total_error=errors(decrypted,original),
        numerical_tolerance=dict(atol=1e-5,rtol=1e-4),
        outputs_supplied_by_caller=True,execution_verified=False,original_reference_verified=False,
        application_accuracy_certified=False,production_active=False)
    result["binding"]=digest(result);return result
def check_calculation(declared,original,approximate,decrypted,metric):
    require(type(declared)is dict and canonical(declared)==canonical(calculation(original,approximate,decrypted,metric)),
        "Metric calculation binding")
    return declared
