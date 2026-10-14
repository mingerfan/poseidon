"""Explicit calibration and profile freezing before DSL/model graph generation."""
import copy,hashlib
from pathlib import Path
from benchmark_graph import digest,canonical
from model_output_contract import snapshot,is_hash
from model_weight_package import regular_bytes
from strict_json_files import strict_file
from source_calibration import run,check_report,implementation
from source_calibration_contract import FX_FORMAT
from source_precision_reference import prepare_source
from pytorch_approximation import validate as validate_approximations,Resolver

FORMAT="poseidon-source-approximation-freeze-v1"
FUNCTION_TARGETS={
 "torch.exp":"exp","torch.reciprocal":"reciprocal","torch.rsqrt":"rsqrt",
 "torch.relu":"relu","torch.sigmoid":"sigmoid","torch.tanh":"tanh","torch.log":"log","torch.abs":"abs",
 "torch.nn.functional.relu":"relu","torch.nn.functional.gelu":"gelu","torch.nn.functional.silu":"silu",
}
METHOD_TARGETS={n:n for n in("exp","reciprocal","rsqrt","relu","sigmoid","tanh","log","abs")}
CODES={"approximation_preparation_input_rejected","approximation_preparation_binding_changed",
       "approximation_calibration_missing","approximation_calibration_domain_exceeded",
       "approximation_calibration_location_unsupported","approximation_preparation_environment_unavailable"}

class PreparationError(ValueError):
    def __init__(self,code):
        if code not in CODES:raise ValueError("Unknown preparation diagnostic")
        self.code=code;super().__init__(code)
    def feedback(self):
        return dict(code=self.code,layer="approximation_preparation",retry_policy="supply_explicit_configuration")

def require(value,code="approximation_preparation_input_rejected"):
    if not value:raise PreparationError(code)

def inspect_source(manifest_path,samples,output_directory,*,expected_manifest_sha256,
                   source_signature,input_dtypes,output_structure,
                   resource_profile="general-tensors-v1",trust_local_code=False):
    """Execute only an explicitly trusted original source; no model graph is required."""
    require(trust_local_code is True)
    selection=dict(format=FX_FORMAT,purpose="calibration",mode="describe",
                   graph_sha256=None,selectors=[])
    return run(manifest_path,None,samples,output_directory,
        expected_manifest_sha256=expected_manifest_sha256,selection=selection,
        expected_selection_sha256=digest(selection),input_dtypes=input_dtypes,
        output_structure=output_structure,resource_profile=resource_profile,trusted=True,
        source_signature=source_signature)

def observe_source(manifest_path,samples,output_directory,*,expected_manifest_sha256,
                   source_signature,selection,expected_selection_sha256,input_dtypes,
                   output_structure,resource_profile="general-tensors-v1",trust_local_code=False):
    require(trust_local_code is True)
    require(type(selection)is dict and selection.get("format")==FX_FORMAT and selection.get("mode")=="observe")
    return run(manifest_path,None,samples,output_directory,
        expected_manifest_sha256=expected_manifest_sha256,selection=selection,
        expected_selection_sha256=expected_selection_sha256,input_dtypes=input_dtypes,
        output_structure=output_structure,resource_profile=resource_profile,trusted=True,
        source_signature=source_signature)

def evidence(manifest,specification,item):
    from hecate_python_env import WORK
    from trusted_pytorch_import import runtime_identity
    require(type(item)is dict and set(item)=={"directory","report_binding"}and is_hash(item["report_binding"]))
    path=Path(item["directory"])
    require(path.is_absolute()and path==path.resolve()and path.is_relative_to(WORK/"results")and
            path.is_dir()and not path.is_symlink())
    request=strict_file(path/"request.json",16*1024**2)
    report=strict_file(path/"calibration.json",16*1024**2)
    require(report["binding"]==item["report_binding"]and request["manifest"]==manifest and
            request.get("source_signature")==specification and request["implementation"]==implementation()and
            request["runtime"]==runtime_identity(),"approximation_preparation_binding_changed")
    require(request["selection"]["format"]==FX_FORMAT and request["selection"]["mode"]=="observe")
    check_report(report,request)
    return request,report

def freeze(manifest_path,output_directory,*,expected_manifest_sha256,source_signature,
           calibration_evidence,configuration,trust_local_code=False):
    """Freeze caller-supplied profiles; no fitting or automatic domain expansion."""
    require(trust_local_code is True)
    spec,files=prepare_source(Path(manifest_path))
    require(is_hash(expected_manifest_sha256)and digest(spec)==expected_manifest_sha256,
            "approximation_preparation_binding_changed")
    signature=snapshot(source_signature,limit=131072)
    records=snapshot(calibration_evidence,limit=131072)
    require(type(records)is list and 1<=len(records)<=4096)
    config=snapshot(configuration,limit=131072);validate_approximations(config)
    require(config["modules"]==[],"approximation_calibration_location_unsupported")
    reports=[];bindings=set();graph=None;domain_rows={};runtime_profile=None;output_tree=None
    for record in records:
        request,report=evidence(spec,signature,record)
        require(report["binding"]not in bindings,"approximation_preparation_input_rejected")
        bindings.add(report["binding"]);reports.append(report["binding"])
        if graph is None:
            graph=report["fx_graph"];runtime_profile=request["resource_profile"];output_tree=request["output_structure"]
        require(request["resource_profile"]==runtime_profile and request["output_structure"]==output_tree,
                "approximation_preparation_binding_changed")
        require(report["fx_graph"]==graph,"approximation_preparation_binding_changed")
        for selector in request["selection"]["selectors"]:
            require(selector["point"]=="input"and selector["projection"]=={"kind":"tensor"}and
                    selector["argument"]in(0,"input"),"approximation_calibration_location_unsupported")
            actual=report["observations"][selector["id"]]
            domain_rows.setdefault(selector["node"],[]).append(actual)
    resolver=Resolver(config);resolved=[]
    for node in graph["nodes"]:
        function=(FUNCTION_TARGETS.get(node["target"])if node["op"]=="call_function"else
                  METHOD_TARGETS.get(node["target"])if node["op"]=="call_method"else None)
        if function is None:continue
        require(node["name"]in domain_rows,"approximation_calibration_missing")
        rows=domain_rows[node["name"]]
        require(all(r["op"]==node["op"]and r["target"]==node["target"]for r in rows),
                "approximation_preparation_binding_changed")
        profile=resolver.select(node["name"],function,function,[])
        lo=min(r["minimum"]for r in rows);hi=max(r["maximum"]for r in rows)
        require(profile["domain"][0]<=lo and hi<=profile["domain"][1],
                "approximation_calibration_domain_exceeded")
        resolved.append(dict(source_node=node["name"],operator=function,function=function,
            target=node["target"],profile_sha256=digest(profile),domain=list(profile["domain"]),
            observed_minimum=lo,observed_maximum=hi,
            samples=sum(r["samples"]for r in rows),values=sum(r["values"]for r in rows)))
    require(resolved,"approximation_calibration_location_unsupported")
    selection=resolver.finish()
    frozen=copy.deepcopy(spec);frozen["approximations"]=config
    require({k:v for k,v in frozen.items()if k!="approximations"}==
            {k:v for k,v in spec.items()if k!="approximations"})
    from pytorch_weight_package import read
    from model_resources import resolve
    # Source paths and raw weight bytes were already validated by prepare_source/read.
    _,weights=read(Path(manifest_path).parent,spec["weights"],resource_profile=runtime_profile)
    payload={name:regular_bytes(path,65536)for name,path in files.items()}
    payload.update(weights)
    from hecate_python_env import WORK
    out=Path(output_directory)
    require(out.is_absolute()and out==out.resolve()and out.parent.is_relative_to(WORK/"results")and
            not out.exists()and not out.is_symlink())
    out.mkdir(mode=0o700);source=out/"source";source.mkdir(mode=0o700)
    for name,raw in payload.items():
        with (source/name).open("xb")as stream:stream.write(raw)
    with (source/"manifest.json").open("xb")as stream:stream.write(canonical(frozen))
    current,_=prepare_source(Path(manifest_path))
    require(current==spec and all(regular_bytes(files[n],65536)==payload[n]for n in files),
            "approximation_preparation_binding_changed")
    for name,raw in weights.items():
        require(regular_bytes(Path(manifest_path).parent/name,len(raw))==raw,
                "approximation_preparation_binding_changed")
    checked,_=prepare_source(source/"manifest.json")
    require(checked==frozen,"approximation_preparation_binding_changed")
    result=dict(format=FORMAT,status="frozen",original_manifest_sha256=digest(spec),
       configured_manifest_sha256=digest(frozen),graph_sha256=graph["graph_sha256"],
       source_signature=signature,resource_profile=runtime_profile,output_structure=output_tree,
       calibration_report_bindings=reports,configuration=config,
       approximation_selection=selection,source_locations=resolved,
       source_payload_sha256={n:hashlib.sha256(raw).hexdigest()for n,raw in payload.items()},
       observer_implementation=implementation(),
       preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       configured_manifest=str(source/"manifest.json"),model_graph_constructed=False,
       deterministic_dsl_answer_required=False,profile_fitted=False,
       domain_covered_on_declared_calibration_inputs=True,complete_model_capability_certified=False,
       universal_domain_guarantee=False,
       original_application_accuracy_certified=False,ckks_validation_performed=False,
       api_calls=0,compiler_calls=0,encrypted_calls=0)
    result["binding"]=digest(result)
    with (out/"freeze.json").open("xb")as stream:stream.write(canonical(result))
    return result
