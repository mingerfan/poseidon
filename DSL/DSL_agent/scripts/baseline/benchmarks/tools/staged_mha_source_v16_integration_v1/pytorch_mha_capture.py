"""Temporary v16 native-MHA-only worker boundary; NOT final ordered integration.

Normal production integration must expand each registered MHA node inside the
ordered graph. This prototype uses the already audited fixed MHA-only capture
grammar to make complete source/weight/None/checker boundaries reviewable.
"""
import copy, hashlib, json, math, random, struct
from pathlib import Path
from benchmark_graph import digest, canonical, require
from model_resources import resolve, identity
from pytorch_mha_plan import capture, verify, source_reference
from pytorch_mha_outputs import SOURCE_FORMAT, flatten, validate as validate_tree

FORMAT = "poseidon-trusted-native-mha-capture-overlay-v1"
REFERENCE = "poseidon-native-mha-overlay-finite-reference-v1"
FIELDS = {"format","torch_version","trusted_code","training","graph_sha256","dtype_policy",
          "source_precision_execution_compared","source_state_dtypes","output_structure",
          "mha_source_capture","weight_binding"}

def _scalars(value):
    if type(value) is list:
        for item in value:yield from _scalars(item)
    else:yield value

def check_state(public, records, expected):
    from pytorch_weight_package import check_binding
    check_binding(expected, records)
    require(type(public) is dict and set(public)==set(records),"Complete original MHA state")
    for name, record in public.items():
        require(type(record) is dict and set(record)=={"dtype","shape","bytes_sha256","values"}
                and record["dtype"] in ("<f8","|b1")
                and records[name]["dtype"]==record["dtype"]
                and records[name]["shape"]==record["shape"],"Original MHA state declaration")
        leaves=list(_scalars(record["values"]))
        require(len(leaves)==math.prod(record["shape"]),"Original MHA state element count")
        if record["dtype"]=="|b1":
            require(all(type(v) is bool for v in leaves),"Boolean MHA state values")
            raw=bytes(leaves)
        else:
            require(all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1024
                        for v in leaves),"Finite original MHA state")
            raw=b"".join(struct.pack("<d",v) for v in leaves)
        require(hashlib.sha256(raw).hexdigest()==record["bytes_sha256"],"Original MHA state bytes")
        normalized=b"".join(struct.pack("<d",float(v)) for v in leaves)
        require(hashlib.sha256(normalized).hexdigest()==expected["normalized_float64_sha256"][name],
                "Original MHA state differs from loaded package")
    return public

def prepare_module(module,inputs,outputs,*,approximations,resource_profile,
                   dtype_policy,**unused):
    import numpy as np, torch
    require(not unused and dtype_policy=="strict-float64-v1"
            and resource_profile=="general-tensors-v1","Explicit bounded MHA prototype policy")
    rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
    plan,context=capture(module,inputs,outputs,approximation=approximations,trusted=True)
    require(torch.equal(rng,torch.get_rng_state()) and random.getstate()==py
            and all(np.array_equal(a,b) for a,b in zip(nr,np.random.get_state())),"MHA trace RNG mutation")
    high=plan["mathematical_graph"];lowered=plan["lowered_graph"]
    metadata=dict(format=FORMAT,torch_version=torch.__version__,trusted_code=True,training=False,
        graph_sha256=digest(high),dtype_policy=dtype_policy,source_precision_execution_compared=False,
        source_state_dtypes={name:"bool" if value["dtype"]=="|b1" else "float64"
                             for name,value in plan["source_request"]["public_state"].items()},
        output_structure=plan["output_record"],mha_source_capture=plan)
    bound=copy.deepcopy(plan["decomposition"]);bound["pytorch_capture"]=metadata
    bound.pop("binding_sha256",None);bound["binding_sha256"]=digest(bound)
    return high,lowered,bound,dict(original_module=context["original_module"],
                                  normalized_module=copy.deepcopy(context["source_fx"]))

def check_capture_metadata(metadata,original,spec,*,resource_profile,expected_weight_binding):
    from pytorch_weight_package import declarations
    from pytorch_ordered_contract import shim
    import trusted_pytorch_import as legacy
    legacy.validate_manifest(shim(spec))
    require(spec["format"]==SOURCE_FORMAT and resource_profile=="general-tensors-v1"
            and spec["dtype_policy"]=="strict-float64-v1","Explicit v16 prototype source policy")
    require(type(metadata) is dict and set(metadata)==FIELDS and metadata["format"]==FORMAT
            and metadata["torch_version"]==legacy.runtime_identity()["torch"]
            and metadata["trusted_code"] is True and metadata["training"] is False
            and metadata["dtype_policy"]==spec["dtype_policy"]
            and metadata["source_precision_execution_compared"] is False,"MHA metadata contract")
    plan=metadata["mha_source_capture"];q=plan["source_request"]
    require(q["source_format"]==spec["format"] and q["inputs"]==spec["inputs"]
            and q["outputs"]==spec["outputs"] and q["approximation"]==spec["approximations"],
            "Complete manifest/request binding")
    verify(plan,expected_source_request_sha256=digest(q))
    require(canonical(original)==canonical(plan["mathematical_graph"])
            and metadata["graph_sha256"]==digest(original)
            and metadata["output_structure"]==plan["output_record"],"MHA graph/output binding")
    validate_tree(metadata["output_structure"])
    records=declarations(spec["weights"],resource_profile=resource_profile)
    require(canonical(metadata["weight_binding"])==canonical(expected_weight_binding),
            "MHA loaded weight binding")
    check_state(q["public_state"],records,expected_weight_binding)
    require(metadata["source_state_dtypes"]=={
            n:"bool" if r["dtype"]=="|b1" else "float64" for n,r in records.items()},
            "MHA complete dtype inventory")
    return plan["lowered_graph"]

def validate(prepared,inputs,outputs,*,resource_profile="general-tensors-v1",count=16):
    import numpy as np, torch
    from benchmark_graph import samples
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as independent_torch
    from operator_graph import evaluate_reference
    high,lowered,bound,context=prepared
    require(count==16 and high["inputs"]==inputs and [o["name"] for o in high["outputs"]]==outputs,
            "Fixed MHA finite reference signature")
    metadata=bound["pytorch_capture"];plan=metadata["mha_source_capture"]
    require(metadata["graph_sha256"]==digest(high),"MHA source reference graph binding")
    modules=[context["original_module"],context["normalized_module"]]
    require(modules[0] is not modules[1],"Separate MHA reference modules")
    before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in modules]
    rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
    maxima=dict(ideal=0.,normalized=0.,decomposed=0.,approximation=0.,ratio=0.,normal_ratio=0.)
    values=0
    def forward(module,batch):
        raw,tree=flatten(module(*[torch.from_numpy(batch[s["name"]].copy()) for s in inputs]),
                         lambda v:isinstance(v,torch.Tensor),outputs,source_format=SOURCE_FORMAT)
        require(tree==metadata["output_structure"] and
                all(v.dtype is torch.float64 and v.device.type=="cpu" for v in raw),"MHA native output structure")
        return {name:value.detach().numpy().copy() for name,value in zip(outputs,raw)}
    def compare(a,b):
        require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),"MHA finite shape")
        maxima["ratio"]=max(maxima["ratio"],float(np.max(np.abs(a-b)/(1e-12+1e-12*np.abs(b)))))
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
        return float(np.max(np.abs(a-b)))
    with torch.no_grad():
        for batch in samples(lowered,count):
            saved={n:v.copy() for n,v in batch.items()}
            original=forward(modules[0],batch);normalized=forward(modules[1],batch)
            ideal=source_reference(plan,batch,approximate=False)
            approximate=source_reference(plan,batch,approximate=True)
            high_reference=evaluate_reference(high,batch)
            actual=evaluate(lowered,batch,resource_profile=resource_profile)
            tensor=independent_torch(lowered,batch,resource_profile=resource_profile)
            for name in outputs:
                maxima["ideal"]=max(maxima["ideal"],compare(original[name],ideal[name]))
                maxima["normalized"]=max(maxima["normalized"],compare(original[name],normalized[name]))
                maxima["normal_ratio"]=max(maxima["normal_ratio"],float(np.max(
                    np.abs(original[name]-normalized[name])/(1e-12+1e-12*np.abs(original[name])))))
                maxima["decomposed"]=max(maxima["decomposed"],compare(actual[name],approximate[name]),
                    compare(tensor[name],approximate[name]),compare(high_reference[name],approximate[name]))
                maxima["approximation"]=max(maxima["approximation"],float(np.max(
                    np.abs(approximate[name]-ideal[name]))))
                values+=int(original[name].size)
            require(all(np.array_equal(batch[n],saved[n]) for n in batch),"MHA input mutation")
            require(torch.equal(rng,torch.get_rng_state()) and random.getstate()==py and
                    all(np.array_equal(a,b) for a,b in zip(nr,np.random.get_state())),"MHA random forward")
            for module,state in zip(modules,before):
                require(state.keys()==module.state_dict().keys() and all(
                    torch.equal(v,module.state_dict()[k]) for k,v in state.items()),"MHA public state mutation")
    result=dict(format=REFERENCE,status="passed",groups=count,compared_values=values,
        graph_sha256=digest(high),lowered_sha256=digest(lowered),capture_binding_sha256=bound["binding_sha256"],
        reference_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mha_source_binding=plan["binding"],max_original_torch_vs_independent_ideal=maxima["ideal"],
        max_original_vs_normalized_torch=maxima["normalized"],max_decomposition_error=maxima["decomposed"],
        max_approximation_error=maxima["approximation"],max_reference_tolerance_ratio=maxima["ratio"],
        max_normalization_tolerance_ratio=maxima["normal_ratio"],
        comparison_tolerance=dict(atol=1e-12,rtol=1e-12),input_storage_isolated=True,
        public_state_immutable=True,approximation_accuracy_certified=False,
        physical_layout_validated=False,compiler_validated=False,encrypted_execution_validated=False,paid_calls=0)
    result["binding"]=digest(result);return result

def check_reference(report,original,lowered,spec,*,resource_profile,expected_weight_binding):
    from pytorch_ordered_contract import source_identity, capture_binding, reference_version
    import trusted_pytorch_import as legacy
    require(type(report) is dict and report.get("format")==reference_version(spec)
            and report.get("mode")=="isolated_trusted_pytorch_reference"
            and report.get("original_sha256")==digest(original) and report.get("model_sha256")==digest(lowered)
            and report.get("manifest_sha256")==digest(spec) and report.get("source_hashes")==source_identity()
            and report.get("runtime")==legacy.runtime_identity() and report.get("probes")==16
            and report.get("resource_profile")==identity(resource_profile),"MHA report identity")
    require(report.get("reference_status")=="passed" and report.get("reference_dtype")=="float64"
            and all(report.get(n) is False for n in ("source_precision_execution_compared",
                "approximation_accuracy_certified","runtime_layout_validated","encrypted_execution","agent_generated"))
            and report.get("paid_calls")==0,"MHA reference scope")
    checked,bound=capture_binding(original,spec,resource_profile=resource_profile,
        capture_metadata=report.get("capture_metadata"),expected_weight_binding=expected_weight_binding)
    require(canonical(checked)==canonical(lowered),"MHA reference graph")
    finite=report.get("finite_reference")
    require(type(finite) is dict and finite.get("binding")==digest({k:v for k,v in finite.items() if k!="binding"})
            and finite.get("format")==REFERENCE and finite.get("status")=="passed" and finite.get("groups")==16
            and finite.get("graph_sha256")==digest(original) and finite.get("lowered_sha256")==digest(lowered)
            and finite.get("capture_binding_sha256")==bound["binding_sha256"]
            and finite.get("mha_source_binding")==report["capture_metadata"]["mha_source_capture"]["binding"]
            and finite.get("reference_implementation_sha256")==source_identity()["pytorch_mha_capture.py"]
            and finite.get("comparison_tolerance")==dict(atol=1e-12,rtol=1e-12)
            and finite.get("input_storage_isolated") is True and finite.get("public_state_immutable") is True,
            "MHA finite receipt identity")
    from benchmark_graph import validate as validate_graph
    sizes=validate_graph(lowered,resource_profile=resource_profile)["output_shapes"]
    require(type(finite.get("compared_values")) is int and finite["compared_values"]==
            16*sum(math.prod(s) for s in sizes.values()),"MHA compared value count")
    for name in ("max_original_torch_vs_independent_ideal","max_original_vs_normalized_torch",
                 "max_decomposition_error","max_approximation_error","max_reference_tolerance_ratio",
                 "max_normalization_tolerance_ratio"):
        require(type(finite.get(name)) in (int,float) and math.isfinite(finite[name]) and finite[name]>=0,
                "MHA finite metric")
    require(finite["max_reference_tolerance_ratio"]<=1 and finite["max_normalization_tolerance_ratio"]<=1
            and all(finite.get(n) is False for n in ("approximation_accuracy_certified",
                    "physical_layout_validated","compiler_validated","encrypted_execution_validated"))
            and finite.get("paid_calls")==0,"MHA finite scope")
    return report
