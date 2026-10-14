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


MAX_DATASET_SAMPLES=10000
def aggregate(chunks,*,sample_count,identity):
    """Combine 1..16-row calculations without expanding an execution request."""
    require(type(sample_count)is int and 1<=sample_count<=MAX_DATASET_SAMPLES,"Metric dataset sample count")
    require(type(identity)is dict and set(identity)=={"model_sha256","package_binding","dataset_sha256"}and
        all(type(v)is str and len(v)==64 and all(c in"0123456789abcdef"for c in v)for v in identity.values()),
        "Metric explicit model/package/dataset identity")
    require(type(chunks)is list and 1<=len(chunks)<=sample_count,"Metric chunk count")
    seen=set();outputs=None;kind=None;spec=None;counts={"original":0,"approximate":0,"decrypted":0}
    total_values=0;stats={key:dict(sum_absolute=0.,maximum_absolute_error=0.,elementwise_threshold_passed=True)
        for key in("approximation_error","execution_error","total_error")}
    fingerprints=[]
    for chunk in chunks:
        require(type(chunk)is dict and set(chunk)=={"indices","calculation"},"Metric chunk fields")
        indices,r=chunk["indices"],chunk["calculation"]
        require(type(indices)is list and 1<=len(indices)<=MAX_SAMPLES and
            all(type(i)is int and 0<=i<sample_count for i in indices)and
            len(set(indices))==len(indices)and not seen.intersection(indices),"Metric duplicate/missing dataset positions")
        require(type(r)is dict and r.get("binding")==digest({k:v for k,v in r.items()if k!="binding"})and
            r.get("format")==FORMAT and type(r.get("samples"))is int and r["samples"]==len(indices)and
            r.get("execution_verified")is False and r.get("application_accuracy_certified")is False,
            "Metric unverified calculation receipt")
        require(type(r.get("outputs_per_sample"))is int and 1<=r["outputs_per_sample"]<=MAX_VALUES,"Metric chunk output count")
        if outputs is None:outputs=r["outputs_per_sample"]
        require(outputs==r["outputs_per_sample"],"Metric chunk output signature")
        this_spec={k:v for k,v in r["metric"].items()if k!="labels"}
        if spec is None:spec=this_spec;kind=spec["kind"]
        require(canonical(this_spec)==canonical(spec),"Metric mixed requests")
        if kind=="top1_drop":
            app=r["application_metric"]
            for key in counts:
                correct=app[key]["correct"]
                require(type(correct)is int and 0<=correct<=r["samples"]and app[key]["samples"]==r["samples"],"Metric class counts")
                counts[key]+=correct
        for key in stats:
            item=r[key];values=len(indices)*outputs
            require(type(item["values"])is int and item["values"]==values and
                type(item["mae"])in(int,float)and type(item["maximum_absolute_error"])in(int,float)and
                math.isfinite(item["mae"])and math.isfinite(item["maximum_absolute_error"])and
                0<=item["mae"]<=item["maximum_absolute_error"]and type(item["elementwise_threshold_passed"])is bool,
                "Metric chunk error summary")
            stats[key]["sum_absolute"]+=item["mae"]*values
            stats[key]["maximum_absolute_error"]=max(stats[key]["maximum_absolute_error"],item["maximum_absolute_error"])
            stats[key]["elementwise_threshold_passed"] &= item["elementwise_threshold_passed"]
        total_values+=len(indices)*outputs;seen.update(indices)
        fingerprints.append(dict(indices=list(indices),calculation_binding=r["binding"]))
    require(seen==set(range(sample_count)),"Metric incomplete dataset")
    if kind=="top1_drop":
        limit=Fraction(str(spec["maximum_drop_percentage_points"]))
        plain_drop=Fraction(100*(counts["original"]-counts["approximate"]),sample_count)
        total_drop=Fraction(100*(counts["original"]-counts["decrypted"]),sample_count)
        app=dict(correct_counts=counts,samples=sample_count,
            accuracy={key:value/sample_count for key,value in counts.items()},
            approximation_drop_percentage_points=float(plain_drop),total_drop_percentage_points=float(total_drop),
            approximation_metric_passed=plain_drop<=limit,decrypted_metric_passed=total_drop<=limit,
            threshold_comparison="exact aggregate counts against declared JSON decimal limit")
    else:
        field="mae"if kind=="mae"else"maximum_absolute_error"
        approximate=stats["approximation_error"]["sum_absolute"]/total_values if field=="mae"else stats["approximation_error"][field]
        total=stats["total_error"]["sum_absolute"]/total_values if field=="mae"else stats["total_error"][field]
        app=dict(approximation_value=approximate,total_value=total,
            approximation_metric_passed=approximate<=spec["maximum_error"],decrypted_metric_passed=total<=spec["maximum_error"])
    result=dict(format="poseidon-application-metric-aggregate-v1",identity=dict(identity),metric=spec,samples=sample_count,
        values=total_values,chunks=fingerprints,application_metric=app,
        error_summaries={key:dict(mae=value["sum_absolute"]/total_values,
            maximum_absolute_error=value["maximum_absolute_error"],elementwise_threshold_passed=value["elementwise_threshold_passed"])
            for key,value in stats.items()},dataset_positions_complete=True,
        execution_verified=False,original_reference_verified=False,application_accuracy_certified=False,
        authority="caller-supplied metric calculations; no backend or original-model certification",
        per_execution_sample_limit=MAX_SAMPLES,production_active=False)
    result["binding"]=digest(result);return result
