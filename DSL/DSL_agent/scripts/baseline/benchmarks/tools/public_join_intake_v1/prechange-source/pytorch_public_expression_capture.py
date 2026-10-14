from pytorch_public_mask_source import SOURCE_FORMAT as MASK_SOURCE_FORMAT, PACKAGE_FORMAT as MASK_PACKAGE_FORMAT, REFERENCE_FORMAT as MASK_REFERENCE_FORMAT
"""V23 public-expression intake, using the retained V22 capture and reference."""
import copy,hashlib,io,json
from pathlib import Path
from benchmark_graph import require,digest
from pytorch_public_value_ops import POLICY,CAST_POLICY,ARITHMETIC_POLICY,TRANSFORM_POLICY,CONSTANT_POLICY,MASK_POLICY
from pytorch_public_constant_source import SOURCE_FORMAT as CONSTANT_SOURCE_FORMAT, PACKAGE_FORMAT as CONSTANT_PACKAGE_FORMAT, REFERENCE_FORMAT as CONSTANT_REFERENCE_FORMAT
from pytorch_public_tensor_transform_source import SOURCE_FORMAT as STATE_TRANSFORM_SOURCE_FORMAT, PACKAGE_FORMAT as STATE_TRANSFORM_PACKAGE_FORMAT, REFERENCE_FORMAT as STATE_TRANSFORM_REFERENCE_FORMAT
from pytorch_public_tensor_transform_source import CAPTURE_FORMAT as STATE_TRANSFORM_CAPTURE_FORMAT
from pytorch_public_expression_flow import normalize,check as check_flow,_arrays,state_identity
from pytorch_public_arithmetic_capture import (
    TRACE_OPTIONS,prepare as prepare_base,state_records,structural,
    check_record as check_base,derived_arrays as base_arrays)
from pytorch_public_state_binding import check_original,check_trace,check_normalization
from pytorch_spatial_policy import SOURCE_FORMAT as SPATIAL_SOURCE_FORMAT, PUBLIC_FORMAT as SPATIAL_BASE_FORMAT, EXPRESSION_FORMAT as SPATIAL_FORMAT, POLICY as SPATIAL_POLICY, supports_input_call, supports_spatial

from pytorch_static_broadcast import SOURCE_FORMAT as BROADCAST_SOURCE_FORMAT, CAPTURE_FORMAT as BROADCAST_FORMAT, POLICY as BROADCAST_POLICY

from pytorch_source_package_policy import SOURCE_FORMAT as SOURCE_PACKAGE_FORMAT, CAPTURE_FORMAT as SOURCE_PACKAGE_CAPTURE_FORMAT
from pytorch_static_broadcast import supports_source as supports_broadcast

from pytorch_static_tensor_transforms import SOURCE_FORMAT as TENSOR_SOURCE_FORMAT, PACKAGE_FORMAT as TENSOR_PACKAGE_FORMAT, REFERENCE_FORMAT as TENSOR_REFERENCE_FORMAT
from pytorch_static_tensor_transforms import POLICY as TENSOR_POLICY, CAPTURE_FORMAT as TENSOR_CAPTURE_FORMAT, selected as supports_transforms

FORMAT="poseidon-public-expression-source-capture-v1"
SOURCE_FORMAT="poseidon-trusted-pytorch-v23"
CAST_SOURCE_FORMAT="poseidon-trusted-pytorch-v24"
ARITHMETIC_SOURCE_FORMAT="poseidon-trusted-pytorch-v25"
INPUT_SOURCE_FORMAT="poseidon-trusted-pytorch-v26"
INPUT_CAPTURE_FORMAT="poseidon-public-expression-source-capture-v4"
SOURCE_FORMATS=(SOURCE_FORMAT,CAST_SOURCE_FORMAT,ARITHMETIC_SOURCE_FORMAT,INPUT_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT)
CAST_FORMAT="poseidon-public-expression-source-capture-v2"
ARITHMETIC_FORMAT="poseidon-public-expression-source-capture-v3"

def source_policy(spec):
    require(type(spec)is dict and spec.get("format")in SOURCE_FORMATS,"Explicit public expression source contract")
    return {SOURCE_FORMAT:POLICY,CAST_SOURCE_FORMAT:CAST_POLICY,ARITHMETIC_SOURCE_FORMAT:ARITHMETIC_POLICY,INPUT_SOURCE_FORMAT:ARITHMETIC_POLICY,SPATIAL_SOURCE_FORMAT:ARITHMETIC_POLICY,BROADCAST_SOURCE_FORMAT:ARITHMETIC_POLICY,SOURCE_PACKAGE_FORMAT:ARITHMETIC_POLICY,TENSOR_SOURCE_FORMAT:ARITHMETIC_POLICY,STATE_TRANSFORM_SOURCE_FORMAT:TRANSFORM_POLICY,CONSTANT_SOURCE_FORMAT:CONSTANT_POLICY,MASK_SOURCE_FORMAT:MASK_POLICY}[spec["format"]]

def capture_format(spec):
    if spec.get("format")==MASK_SOURCE_FORMAT:
        from pytorch_public_mask_source import CAPTURE_FORMAT
        return CAPTURE_FORMAT
    if spec.get("format")==CONSTANT_SOURCE_FORMAT:
        from pytorch_public_constant_source import CAPTURE_FORMAT
        return CAPTURE_FORMAT
    if spec.get("format")==STATE_TRANSFORM_SOURCE_FORMAT:return STATE_TRANSFORM_CAPTURE_FORMAT
    if supports_transforms(spec):return TENSOR_CAPTURE_FORMAT
    if spec["format"]==SOURCE_PACKAGE_FORMAT:return SOURCE_PACKAGE_CAPTURE_FORMAT
    if supports_broadcast(spec):return BROADCAST_FORMAT
    if spec["format"]==SPATIAL_SOURCE_FORMAT:return SPATIAL_FORMAT
    if supports_input_call(spec):return INPUT_CAPTURE_FORMAT
    return {POLICY:FORMAT,CAST_POLICY:CAST_FORMAT,ARITHMETIC_POLICY:ARITHMETIC_FORMAT}[source_policy(spec)]

BASE_FORMAT="poseidon-trusted-pytorch-v22"
PROFILE="general-tensors-v1"

def converted_arrays(arrays):
    import numpy as np
    return {n:np.array(a,dtype="<f8"if a.dtype.str=="<f4"else a.dtype.str,copy=True,order="C")
            for n,a in arrays.items()}

def array_declarations(arrays,*,resource_profile):
    import numpy as np
    from pytorch_weight_package import declarations
    result={}
    for index,name in enumerate(sorted(arrays)):
        value=np.array(arrays[name],copy=True,order="C");stream=io.BytesIO()
        np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
        result[name]=dict(file=f"state_{index:04d}.npy",sha256=hashlib.sha256(raw).hexdigest(),
                          bytes=len(raw),shape=list(value.shape),dtype=value.dtype.str)
    declarations(result,resource_profile=resource_profile)
    return result

def conversion_record(before,converted,float_policy):
    return dict(format="poseidon-public-expression-float-normalization-v1",
                float_policy=float_policy,raw_state_sha256=digest(before),
                normalized_state_sha256=digest(converted),
                converted_names=sorted(n for n,v in before.items()if v["dtype"]=="<f4"),
                compute_dtype="float64",source_precision_execution_compared=False)

def prepare(module,spec,output,*,expected_weight_binding,resource_profile=PROFILE):
    import torch
    from pytorch_weight_package import export_state,read,binding
    from pytorch_static_trace import trace
    from pytorch_mha_ordered_reference import validate
    policy=source_policy(spec)
    source=copy.deepcopy(module).cpu().eval()
    raw=_arrays(source);before=state_records(raw)
    check_original(before,spec["weights"],float_policy=spec["dtype_policy"])
    require(binding(raw,spec["weights"])==expected_weight_binding,"Original expression weights changed")
    reference_source=copy.deepcopy(source).double().eval();converted=_arrays(reference_source)
    check_normalization(before,state_records(converted),float_policy=spec["dtype_policy"])
    call_plan=None
    if supports_input_call(spec):
        from pytorch_input_binding import plan,signature_record
        call_plan=plan(spec,signature_record(reference_source))
    source_graph=trace(reference_source,spec["inputs"],resource_profile=resource_profile,
                       **TRACE_OPTIONS,public_expression_policy=policy,input_plan=call_plan,
                       **({"spatial_source_policy":SPATIAL_POLICY}if supports_spatial(spec) else {}),
                       **({"broadcast_source_policy":BROADCAST_POLICY}if supports_broadcast(spec) else {}),
                       **({"tensor_transform_source_policy":TENSOR_POLICY}if supports_transforms(spec) else {}),
                       **({"public_mask_source_policy":__import__("pytorch_public_mask_source").POLICY}if spec["format"]==MASK_SOURCE_FORMAT else {}))
    input_flow=None;input_reference=reference_source
    if call_plan is not None:
        from pytorch_input_binding import normalize as normalize_inputs,reference_module
        Path(output,"fx-input-source.txt").write_text(str(source_graph.graph))
        source_graph,input_flow=normalize_inputs(source_graph,call_plan)
        input_reference=reference_module(reference_source,call_plan,
            input_effect_policy="registered-working-copy-inputs-v1"if spec["format"]==MASK_SOURCE_FORMAT else"immutable-inputs-v1")
    broadcast_flow=None
    if supports_broadcast(spec):
        from pytorch_static_broadcast import normalize as normalize_broadcast
        source_graph,broadcast_flow=normalize_broadcast(source_graph,spec["inputs"],resource_profile=resource_profile)
    transform_flow=None
    if supports_transforms(spec) and spec["format"] not in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT):
        from pytorch_static_tensor_transforms import normalize as normalize_transforms
        source_graph,transform_flow=normalize_transforms(source_graph,spec["inputs"],resource_profile=resource_profile)
    traced=_arrays(source_graph);traced_records=state_records(traced)
    unreferenced=sorted(set(converted)-set(traced))
    check_trace(state_records(converted),traced_records,unreferenced)
    require(state_records(_arrays(source))==before and
            state_records(_arrays(reference_source))==state_records(converted),"Public expression trace mutated state")
    normalized,flow=normalize(source_graph,traced,
                              expected_state=state_identity(traced,resource_profile=resource_profile),
                              resource_profile=resource_profile,policy=policy)
    if spec["format"] in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT):
        from pytorch_static_tensor_transforms import normalize as normalize_transforms
        normalized,transform_flow=normalize_transforms(normalized,spec["inputs"],resource_profile=resource_profile)
    mask_flow=None
    if spec["format"]==MASK_SOURCE_FORMAT:
        from pytorch_public_mask_selection import normalize as normalize_mask_selection
        normalized,mask_flow=normalize_mask_selection(normalized,spec["inputs"],resource_profile=resource_profile)
    metadata={n.name:copy.deepcopy(n.meta)for n in normalized.graph.nodes}
    effective=torch.fx.GraphModule(normalized,copy.deepcopy(normalized.graph))
    for n in effective.graph.nodes:n.meta=metadata[n.name]
    effective_arrays=_arrays(effective)
    require(state_identity(effective_arrays,resource_profile=resource_profile)==(mask_flow["effective_state"]if mask_flow is not None else flow["effective_state"]),
            "Public expression effective state projection")
    weights_path=Path(output)/"expression-weight-snapshot";weights_path.mkdir()
    weights=export_state(effective,weights_path,resource_profile=resource_profile)
    effective_arrays,_=read(weights_path,weights,resource_profile=resource_profile)
    base_spec=copy.deepcopy(spec);base_spec.pop("call",None);base_spec.update(format=SPATIAL_BASE_FORMAT if supports_spatial(spec) else BASE_FORMAT,weights=weights)
    base_path=Path(output)/"expression-base-capture";base_path.mkdir()
    base,context=prepare_base(effective,base_spec,base_path,
                              expected_weight_binding=binding(effective_arrays,weights),
                              resource_profile=resource_profile,_return_context=True)
    require(structural(base["division_capture"]["source_inventory"])==
            structural(mask_flow["normalized_inventory"]if mask_flow is not None else transform_flow["normalized_inventory"] if spec["format"] in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT) else
                       flow["normalized_inventory"]),"Expression/retained tracing structural bridge")
    high,lowered,bound,base_context=context
    original_context=dict(base_context);original_context["original_module"]=input_reference
    finite=validate((high,lowered,bound,original_context),spec["inputs"],spec["outputs"],
                    resource_profile=resource_profile,count=16)
    require(state_records(_arrays(source))==before and
            state_records(_arrays(reference_source))==state_records(converted),"Expression reference mutated state")
    result=dict(format=capture_format(spec),source_intake_version=spec["format"],
                original_manifest_sha256=digest(spec),original_weight_binding=expected_weight_binding,
                original_public_state=before,normalized_source_state=state_records(converted),
                float_normalization=conversion_record(before,state_records(converted),spec["dtype_policy"]),
                traced_public_state=traced_records,unreferenced_state_names=unreferenced,
                public_expression_flow=flow,internal_manifest=base_spec,base_capture=base,
                high_graph=high,lowered_graph=lowered,original_reference=finite,
                production_intake_integrated=True,provider_calls=0,compiler_calls=0,encrypted_calls=0,
                resource_profile=resource_profile)
    if mask_flow is not None:
        result["public_mask_selection_flow"]=mask_flow
        result["input_effect_policy"]="registered-working-copy-inputs-v1"
    if transform_flow is not None:result["static_tensor_transform_flow"]=transform_flow
    if input_flow is not None:result["input_call_flow"]=input_flow
    if broadcast_flow is not None:result["static_broadcast_flow"]=broadcast_flow
    result["binding"]=digest(result)
    Path(output,"fx-public-source.txt").write_text(str(source_graph.graph))
    Path(output,"fx-public-normalized.txt").write_text(str(effective.graph))
    return result

def replay(record,spec,arrays,*,resource_profile):
    from pytorch_weight_package import binding
    require(type(arrays)is dict and set(arrays)==set(spec["weights"]),"Original expression public arrays required")
    raw=state_records(arrays);check_original(raw,spec["weights"],float_policy=spec["dtype_policy"])
    require(record["original_public_state"]==raw and
            record["original_weight_binding"]==binding(arrays,spec["weights"]),"Original expression state changed")
    converted=converted_arrays(arrays);states=state_records(converted)
    check_normalization(raw,states,float_policy=spec["dtype_policy"])
    require(record["normalized_source_state"]==states and
            record["float_normalization"]==conversion_record(raw,states,spec["dtype_policy"]),
            "Public expression float conversion binding")
    traced=record["traced_public_state"]
    check_trace(states,traced,record["unreferenced_state_names"])
    flow=record["public_expression_flow"];selected={n:converted[n]for n in traced}
    source_inventory=flow["source_inventory"]
    if supports_transforms(spec):
        from pytorch_static_tensor_transforms import check as check_transforms
        transforms=record["static_tensor_transform_flow"]
        if spec["format"] in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT):
            full,effective=check_flow(flow,selected,expected_binding=flow["binding"],
                expected_source_inventory=flow["source_inventory"],resource_profile=resource_profile,policy=source_policy(spec))
            check_transforms(transforms,spec["inputs"],resource_profile=resource_profile,
                             expected_source_inventory=flow["normalized_inventory"])
            require(transforms["public_state"]==state_records(full),"public_transform_state_bridge")
        else:
            require(check_transforms(transforms,spec["inputs"],resource_profile=resource_profile)==source_inventory and
                    transforms["public_state"]==traced,"static_transform_public_flow_bridge")
            source_inventory=transforms["source_inventory"]
    else:require("static_tensor_transform_flow" not in record,"Unexpected static tensor transform flow")
    if supports_broadcast(spec):
        from pytorch_static_broadcast import check as check_broadcast
        broadcast=record["static_broadcast_flow"]
        require(check_broadcast(broadcast,spec["inputs"],resource_profile=resource_profile)==source_inventory and
                broadcast["public_state"]==traced,"static_broadcast_public_flow_bridge")
        source_inventory=broadcast["source_inventory"]
    else:require("static_broadcast_flow"not in record,"Unexpected static broadcast flow")
    if supports_input_call(spec):
        from pytorch_input_binding import check_flow as check_inputs
        input_flow=check_inputs(record["input_call_flow"],spec)
        require(input_flow["normalized_inventory"]==source_inventory,"input_call_public_flow_bridge")
    else:require("input_call_flow"not in record,"Unexpected structured input flow")
    require(flow["original_state"]==state_identity(selected,resource_profile=resource_profile),
            "Public expression original FX state")
    if spec["format"] not in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT):
        _,effective=check_flow(flow,selected,expected_binding=flow["binding"],
                               expected_source_inventory=flow["source_inventory"],resource_profile=resource_profile,policy=source_policy(spec))
    if spec["format"]==MASK_SOURCE_FORMAT:
        from pytorch_public_mask_selection import check as check_mask_selection
        require(record.get("input_effect_policy")=="registered-working-copy-inputs-v1","public_mask_input_effect_policy")
        masks=record["public_mask_selection_flow"]
        _,effective=check_mask_selection(masks,spec["inputs"],full,resource_profile=resource_profile,
            expected_source_inventory=record["static_tensor_transform_flow"]["normalized_inventory"])
    else:require("public_mask_selection_flow"not in record and "input_effect_policy"not in record,"Unexpected public mask selection flow")
    base_spec=record["internal_manifest"]
    require(base_spec["format"]==(SPATIAL_BASE_FORMAT if supports_spatial(spec) else BASE_FORMAT)and
            {k:v for k,v in base_spec.items()if k not in("format","weights")}==
            {k:v for k,v in spec.items()if k not in("format","weights","call")},
            "Public expression source changed beyond format and derived weights")
    require(base_spec["weights"]==array_declarations(effective,resource_profile=resource_profile),
            "Public expression derived NPY declarations")
    final=base_arrays(record["base_capture"],base_spec,effective,resource_profile=resource_profile)
    return effective,final

def check_record(record,spec,arrays,*,expected_record_binding,resource_profile=PROFILE):
    from pytorch_ordered_contract import check_reference
    require(type(record)is dict and record.get("binding")==expected_record_binding==
            digest({k:v for k,v in record.items()if k!="binding"}),"Public expression external capture authority")
    require(record["format"]==capture_format(spec) and record["source_intake_version"]==spec["format"] and
            spec.get("format")in SOURCE_FORMATS and record["original_manifest_sha256"]==digest(spec)and
            record["resource_profile"]==resource_profile and record["production_intake_integrated"]is True,
            "Public expression capture identity")
    effective,final=replay(record,spec,arrays,resource_profile=resource_profile)
    base=record["base_capture"];base_spec=record["internal_manifest"]
    check_base(base,base_spec,effective,final,expected_record_binding=base["binding"],
               resource_profile=resource_profile)
    require(record["high_graph"]==base["high_graph"]and record["lowered_graph"]==base["lowered_graph"]and
            structural(base["division_capture"]["source_inventory"])==
            structural(record["public_mask_selection_flow"]["normalized_inventory"]if spec["format"]==MASK_SOURCE_FORMAT else record["static_tensor_transform_flow"]["normalized_inventory"] if spec["format"] in (STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT) else
                       record["public_expression_flow"]["normalized_inventory"]),
            "Public expression complete graph bridge")
    require(record["provider_calls"]==record["compiler_calls"]==record["encrypted_calls"]==0,
            "Public expression capture-only scope")
    source_reference=copy.deepcopy(base["base_reference"])
    source_reference["finite_reference"]=copy.deepcopy(record["original_reference"])
    check_reference(source_reference,base["high_graph"],base["lowered_graph"],base["internal_manifest"],
                    resource_profile=resource_profile,expected_weight_binding=base["internal_weight_binding"])
    return record

def derived_arrays(record,spec,arrays,*,resource_profile):
    """Authenticate and reconstruct derived NPY bytes without importing the model."""
    check_record(record,spec,arrays,expected_record_binding=record["binding"],resource_profile=resource_profile)
    return replay(record,spec,arrays,resource_profile=resource_profile)[1]

def wrapped_metadata(record):
    from pytorch_public_arithmetic_capture import wrapped_metadata as wrap_base
    metadata=wrap_base(record["base_capture"])
    metadata["weight_binding"]=copy.deepcopy(record["original_weight_binding"])
    types={"<f4":"float32","<f8":"float64","<i4":"int32","<i8":"int64","|b1":"bool"}
    metadata["source_state_dtypes"]={n:types[v["dtype"]]for n,v in record["original_public_state"].items()}
    metadata["public_value_expressions"]=copy.deepcopy(record)
    return metadata

def check_capture_metadata(metadata,original,spec,*,resource_profile,expected_weight_binding,expected_weight_arrays):
    from pytorch_weight_package import binding
    require(type(metadata)is dict and "public_value_expressions"in metadata,"Explicit public expression metadata")
    require(type(expected_weight_arrays)is dict,"Authenticated expression public payloads required")
    record=metadata["public_value_expressions"]
    require(binding(expected_weight_arrays,spec["weights"])==expected_weight_binding==
            record["original_weight_binding"],"Public expression expected weights differ")
    check_record(record,spec,expected_weight_arrays,expected_record_binding=record["binding"],
                 resource_profile=resource_profile)
    require(metadata==wrapped_metadata(record)and original==record["high_graph"],
            "Public expression graph or wrapped metadata changed")
    return record["lowered_graph"]

def make_reference(record,spec,metadata,*,resource_profile,expected_weight_arrays):
    from pytorch_ordered_contract import source_identity,reference_version,capture_binding
    expected=copy.deepcopy(record["base_capture"]["base_reference"])
    expected.update(format=reference_version(spec),manifest_sha256=digest(spec),source_hashes=source_identity(),
                    capture_metadata=copy.deepcopy(metadata))
    _,bound=capture_binding(record["high_graph"],spec,resource_profile=resource_profile,capture_metadata=metadata,
                           expected_weight_binding=record["original_weight_binding"],
                           expected_weight_arrays=expected_weight_arrays)
    finite=copy.deepcopy(record["original_reference"])
    finite["capture_binding_sha256"]=bound["binding_sha256"];finite.pop("binding",None);finite["binding"]=digest(finite)
    expected["finite_reference"]=finite
    return expected

def check_reference(report,original,lowered,spec,*,resource_profile,expected_weight_binding,expected_weight_arrays):
    checked=check_capture_metadata(report.get("capture_metadata"),original,spec,resource_profile=resource_profile,
                                   expected_weight_binding=expected_weight_binding,
                                   expected_weight_arrays=expected_weight_arrays)
    require(checked==lowered,"Public expression decomposition changed")
    record=report["capture_metadata"]["public_value_expressions"]
    require(report==make_reference(record,spec,report["capture_metadata"],resource_profile=resource_profile,
                                   expected_weight_arrays=expected_weight_arrays),
            "Public expression reference identity, metrics or scope changed")
    return report

def capture_loaded_module(request,module,arrays,loaded,output_root):
    import sys,numpy as np,torch
    from pytorch_ordered_contract import source_identity
    from platform_config import require_python_packages
    require_python_packages(torch,np)
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
                 cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
    require(request["source_hashes"]==source_identity()and request["runtime"]==runtime,
            "Pinned public expression worker identity")
    torch.set_num_threads(1);require(isinstance(module,torch.nn.Module),"Trusted factory must return nn.Module")
    spec=request["manifest"];profile=request["resource_profile"]
    record=prepare(module,spec,output_root,expected_weight_binding=loaded,resource_profile=profile)
    metadata=wrapped_metadata(record)
    report=make_reference(record,spec,metadata,resource_profile=profile,expected_weight_arrays=arrays)
    check_reference(report,record["high_graph"],record["lowered_graph"],spec,resource_profile=profile,
                    expected_weight_binding=loaded,expected_weight_arrays=arrays)
    Path(output_root,"original.json").write_text(json.dumps(record["high_graph"],allow_nan=False))
    Path(output_root,"capture-reference.json").write_text(json.dumps(report,allow_nan=False))
