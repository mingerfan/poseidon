"""Independent source-function comparison for explicit GELU variants.

The existing polynomial/decomposed reference and frozen CKKS tolerance are
unchanged. This module only identifies the original Torch source function.
"""
import hashlib, math, random
from pathlib import Path
from benchmark_graph import digest, require, samples
from model_resources import resolve
from pytorch_gelu_source import VARIANTS, check_record

FORMAT="poseidon-gelu-ordered-source-reference-v1"

def original_scalar(x, variant):
    require(type(variant) is str and variant in VARIANTS,"Registered GELU source variant")
    if variant=="none":
        return .5*x*(1+math.erf(x/math.sqrt(2)))
    return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))

def evaluate_source(model, inputs, record):
    """Compose independent original operators; never call a DSL lowerer."""
    import numpy as np
    from operator_graph import evaluate_reference
    from operator_reference import interval
    from model_decomposition import polynomial_profile
    require(record["graph_sha256"]==digest(model) and record["binding"]==
            digest({k:v for k,v in record.items() if k!="binding"}),"GELU source reference binding")
    rows={r["node_id"]:r for r in record["entries"]}
    require(len(rows)==len(record["entries"]) and set(rows)==
            {n["id"] for n in model["nodes"] if n["op"]=="gelu"},
            "GELU source reference inventory")
    require(set(inputs)=={s["name"] for s in model["inputs"]},"Source reference input names")
    values={n:np.asarray(v,dtype=np.float64) for n,v in model["constants"].items()}
    for spec in model["inputs"]:
        x=np.asarray(inputs[spec["name"]])
        require(x.dtype==np.float64 and list(x.shape)==spec["shape"] and np.isfinite(x).all(),
                "Source reference input signature")
        values[spec["name"]]=x
    for node in model["nodes"]:
        if node["op"]=="gelu":
            row=rows[node["id"]];variant=row["variant"]
            require(type(variant) is str and variant in VARIANTS and
                    row["source_identity"]==VARIANTS[variant] and
                    row["approximation_sha256"]==digest(node["attrs"]["approximation"]) and
                    len(node["inputs"])==1 and len(node["outputs"])==1,
                    "GELU source reference identity")
            x=values[node["inputs"][0]]
            interval(x,polynomial_profile(node["attrs"]["approximation"]))
            result={node["outputs"][0]:np.vectorize(
                lambda v:original_scalar(float(v),variant),otypes=[float])(x)}
        else:
            names=list(dict.fromkeys(node["inputs"]))
            fragment=dict(format=model["format"],id=model["id"],constants={},
                inputs=[dict(name=n,shape=list(values[n].shape)) for n in names],
                nodes=[node],outputs=[dict(name=n,value=n) for n in node["outputs"]])
            result=evaluate_reference(fragment,{n:values[n] for n in names},ideal=True)
        for name in node["outputs"]:
            value=np.asarray(result[name],dtype=np.float64)
            require(np.isfinite(value).all(),"Source reference nonfinite output")
            values[name]=value
    return {o["name"]:values[o["value"]] for o in model["outputs"]}

def validate(prepared,inputs,outputs,*,resource_profile="general-tensors-v1",count=16):
    import numpy as np
    import torch
    from pytorch_source_contract import output_tree
    from operator_graph import evaluate_reference
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as independent_torch
    high,lowered,binding,context=prepared
    require(type(context) is dict and set(context)=={"original_module","normalized_module"},
            "Trusted GELU reference context")
    require(high["inputs"]==inputs and [o["name"] for o in high["outputs"]]==outputs and
            binding["pytorch_capture"]["graph_sha256"]==digest(high),"Source reference graph binding")
    require(context["original_module"] is not context["normalized_module"],"Separate reference modules required")
    metadata=binding["pytorch_capture"];alias=metadata["ordered_alias_capture"]
    source=check_record(metadata["gelu_source_capture"],high,
        emissions=alias["capture_emissions"],
        normalized_origins=alias["normalization"]["mathematical_node_origins"],
        resource_profile=resource_profile)
    structure=metadata["output_structure"]
    modules=[context["original_module"],context["normalized_module"]]
    before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in modules]
    rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    maximum=ideal_error=normalized_error=approximation_error=ratio=normal_ratio=0.;values=0
    def forward(module,batch):
        args=[torch.from_numpy(batch[s["name"]].copy()) for s in inputs]
        raw,tree=output_tree(module(*args),lambda v:isinstance(v,torch.Tensor),outputs,
                            scalar_tensor=lambda v:v.ndim==0)
        require(tree==structure,"Source reference output structure")
        require(all(v.dtype is torch.float64 and v.device.type=="cpu" for v in raw),"Reference output dtype")
        return {n:v.detach().numpy().copy().reshape(1) if v.ndim==0 else v.detach().numpy().copy()
                for n,v in zip(outputs,raw)}
    def compare(a,b):
        nonlocal ratio
        require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                "Reference shape or nonfinite output")
        ratio=max(ratio,float(np.max(np.abs(a-b)/(1e-12+1e-12*np.abs(b)))))
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
        return float(np.max(np.abs(a-b)))
    with torch.no_grad():
        for batch in samples(lowered,count):
            saved={n:v.copy() for n,v in batch.items()}
            original=forward(modules[0],batch);normalized=forward(modules[1],batch)
            approximate=evaluate_reference(high,batch);ideal=evaluate_source(high,batch,source)
            decomposed=evaluate(lowered,batch,resource_profile=resource_profile)
            independent=independent_torch(lowered,batch,resource_profile=resource_profile)
            for n in outputs:
                ideal_error=max(ideal_error,compare(original[n],ideal[n]))
                normalized_error=max(normalized_error,compare(normalized[n],original[n]))
                normal_ratio=max(normal_ratio,float(np.max(np.abs(normalized[n]-original[n])/
                                                            (1e-12+1e-12*np.abs(original[n])))))
                maximum=max(maximum,compare(decomposed[n],approximate[n]),
                            compare(independent[n],approximate[n]))
                approximation_error=max(approximation_error,float(np.max(np.abs(approximate[n]-original[n]))))
                values+=original[n].size
            for n in batch:require(np.array_equal(batch[n],saved[n]),"Reference input was mutated")
            require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py_rng and
                    all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng)),
                    "Random forward rejected")
            for m,state in zip(modules,before):
                require(state.keys()==m.state_dict().keys() and all(torch.equal(v,m.state_dict()[k])
                        for k,v in state.items()),"Public-state mutation rejected")
    result=dict(format=FORMAT,status="passed",groups=count,compared_values=values,graph_sha256=digest(high),
        lowered_sha256=digest(lowered),capture_binding_sha256=binding["binding_sha256"],
        reference_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_original_torch_vs_independent_ideal=ideal_error,max_original_vs_normalized_torch=normalized_error,
        max_decomposition_error=maximum,max_approximation_error=approximation_error,
        approximation_accuracy_certified=False,comparison_tolerance=dict(atol=1e-12,rtol=1e-12),
        max_reference_tolerance_ratio=ratio,max_normalization_tolerance_ratio=normal_ratio,
        input_storage_isolated=True,public_state_immutable=True,reference_scope="finite CPU float64 probe inputs",
        physical_layout_validated=False,compiler_validated=False,encrypted_execution_validated=False,
        gelu_source_binding=source["binding"],paid_calls=0)
    result["binding"]=digest(result)
    return result
