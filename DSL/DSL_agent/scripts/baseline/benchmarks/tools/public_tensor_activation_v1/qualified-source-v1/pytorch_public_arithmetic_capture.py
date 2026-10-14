"""Versioned public tensor normalization before the retained ordered source capture."""
import copy,hashlib,json
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from pytorch_public_tensor_division import POLICY,normalize,check as check_division,state_payload
from pytorch_public_state_binding import check_original,check_trace,direct_divisors,float64_state,check_normalization
from pytorch_static_trace import trace
from pytorch_public_boolean_arithmetic import normalize as normalize_masks,check as check_masks
from pytorch_static_trace import ALIAS_POLICY
FORMAT="poseidon-public-tensor-source-capture-v1"
SOURCE_FORMAT="poseidon-trusted-pytorch-v22"
BASE_FORMAT="poseidon-trusted-pytorch-v21"
PROFILE="general-tensors-v1"
TRACE_OPTIONS=dict(policy=ALIAS_POLICY,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
 mha_source_policy="poseidon-mha-ordered-source-v1",partition_source_policy="poseidon-pytorch-static-partition-v2",
 einsum_source_policy="poseidon-pytorch-static-einsum-v2",exact_arithmetic_policy="poseidon-pytorch-exact-arithmetic-v1",
 public_tensor_division_policy=POLICY)
CAPTURE_OPTIONS=dict(common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
 gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
 partition_source_policy="poseidon-pytorch-static-partition-v2",einsum_source_policy="poseidon-pytorch-static-einsum-v2",
 exact_arithmetic_policy="poseidon-pytorch-exact-arithmetic-v1",approximation_policy="node-module-default-v1",
 matrix_policy="static-tensor-matmul-v1")
def structural(inventory):
 require(type(inventory)is list and len({n["name"]for n in inventory})==len(inventory),"Unique FX inventory")
 names={n["name"]:i for i,n in enumerate(inventory)}
 def convert(v):
  if type(v)is dict:
   if v.get("kind")=="fx_value":
    require(set(v)=={"kind","name"}and v["name"]in names,"Known FX inventory operand")
    return dict(kind="fx_value",index=names[v["name"]])
   return {k:convert(x)for k,x in v.items()}
  if type(v)is list:return [convert(x)for x in v]
  return v
 return [convert({k:v for k,v in n.items()if k!="name"})for n in inventory]
def state_records(arrays):
 """Bind raw public types separately from their numerical float64 identity."""
 import numpy as np
 return {n:dict(shape=list(a.shape),dtype=a.dtype.str,
  sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest(),
  normalized_float64_sha256=hashlib.sha256(np.asarray(a,dtype="<f8").tobytes(order="C")).hexdigest())
  for n,a in arrays.items()}
def prepare(module,spec,output,*,expected_weight_binding,resource_profile=PROFILE):
 import torch
 from pytorch_fused_matrix_normalization import _state
 from pytorch_weight_package import export_state,read,binding
 from pytorch_ordered_capture import prepare_module
 from pytorch_ordered_contract import capture_binding,check_reference,source_identity,reference_version
 from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
 from pytorch_mha_ordered_reference import validate
 require(spec["format"]==SOURCE_FORMAT and
  spec["dtype_policy"]in("strict-float64-v1","convert-float64-v1"),"Explicit staging base contract and float policy")
 float_policy=spec["dtype_policy"]
 source=copy.deepcopy(module).cpu().eval()
 require(all((v.is_floating_point() and (v.dtype is torch.float64 or
  float_policy=="convert-float64-v1" and v.dtype is torch.float32)) or
  v.dtype in (torch.int32,torch.int64,torch.bool) for v in source.state_dict().values()),
  "Draft public source dtype")
 from pytorch_approximation import validate as validate_approximations
 validate_approximations(spec["approximations"])
 before=_state(source)
 check_original(before,spec["weights"],float_policy=float_policy)
 require(expected_weight_binding["declarations_sha256"]==digest(spec["weights"]),"Original weight declarations binding")
 require(set(before)==set(spec["weights"])and
  {n:v["normalized_float64_sha256"]for n,v in before.items()}==expected_weight_binding["normalized_float64_sha256"],
  "Original loaded source weight binding")
 reference_source=copy.deepcopy(source).double().eval()
 converted=_state(reference_source);check_normalization(before,converted,float_policy=float_policy)
 conversion=dict(format="poseidon-public-state-float-normalization-v1",float_policy=float_policy,
  raw_state_sha256=digest(before),normalized_state_sha256=digest(converted),
  converted_names=sorted(n for n,v in before.items()if v["dtype"]=="<f4"),
  compute_dtype="float64",source_precision_execution_compared=False)
 original_trace=trace(reference_source,spec["inputs"],resource_profile=resource_profile,**TRACE_OPTIONS)
 require(_state(source)==before and _state(reference_source)==converted,"Original tracing changed public state")
 traced_state=_state(original_trace)
 check_trace(converted,traced_state,sorted(set(converted)-set(traced_state)))
 from pytorch_fused_matrix_normalization import inventory
 declared=direct_divisors(inventory(original_trace),traced_state)
 normalized,division=normalize(original_trace,spec["inputs"],declared,resource_profile=resource_profile)
 normalized,boolean_masks=normalize_masks(normalized,resource_profile=resource_profile)
 # Bind the complete transformed state, then project a new module from its actual graph.
 # Original bool bytes remain authenticated separately; no input model or buffer is changed.
 full_derived_state=_state(normalized);full_graph_text=str(normalized.graph)
 original_meta={n.name:copy.deepcopy(n.meta)for n in normalized.graph.nodes}
 effective=torch.fx.GraphModule(normalized,copy.deepcopy(normalized.graph))
 for node in effective.graph.nodes:node.meta=original_meta[node.name]
 effective_state=_state(effective)
 unreferenced_derived_names=sorted(set(full_derived_state)-set(effective_state))
 check_trace(full_derived_state,effective_state,unreferenced_derived_names)
 require(str(effective.graph)==full_graph_text and _state(normalized)==full_derived_state,
  "Effective derived graph must preserve source computation")
 source_projection=dict(format="poseidon-effective-derived-public-state-v1",
  full_state_sha256=digest(full_derived_state),effective_state=effective_state,
  unreferenced_state_names=unreferenced_derived_names,source_fx_sha256=boolean_masks["normalized_fx_sha256"],
  effective_fx_sha256=hashlib.sha256(str(effective.graph).encode()).hexdigest(),original_state_preserved=True)
 normalized=effective
 derived_path=Path(output)/"internal-weight-snapshot";derived_path.mkdir()
 internal_weights=export_state(normalized,derived_path,resource_profile=resource_profile)
 arrays,_=read(derived_path,internal_weights,resource_profile=resource_profile)
 internal_binding=binding(arrays,internal_weights)
 internal=copy.deepcopy(spec);internal["format"]=BASE_FORMAT;internal["weights"]=internal_weights
 prepared=prepare_module(normalized,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
  resource_profile=resource_profile,dtype_policy=spec["dtype_policy"],functional_policy=ADAPTIVE_FUNCTION_POLICY,**CAPTURE_OPTIONS)
 high,lowered,captured,context=prepared;metadata=captured["pytorch_capture"];metadata["weight_binding"]=internal_binding
 recaptured=metadata["ordered_alias_capture"]["capture_adapter"]["exact_arithmetic_normalization"]["source_inventory"]
 require(structural(recaptured)==structural(boolean_masks["normalized_inventory"]),"Division/retrace complete structural bridge")
 lowered,bound=capture_binding(high,internal,resource_profile=resource_profile,capture_metadata=metadata,
  expected_weight_binding=internal_binding)
 base_context=(high,lowered,bound,context)
 finite=validate(base_context,spec["inputs"],spec["outputs"],resource_profile=resource_profile,count=16)
 source_context=dict(context);source_context["original_module"]=reference_source
 original_finite=validate((high,lowered,bound,source_context),spec["inputs"],spec["outputs"],
  resource_profile=resource_profile,count=16)
 require(_state(source)==before and _state(reference_source)==converted,"Original reference changed public state")
 ref=dict(format=reference_version(internal),mode="isolated_trusted_pytorch_reference",original_sha256=digest(high),
  model_sha256=digest(lowered),manifest_sha256=digest(internal),source_hashes=source_identity(),
  runtime=dict(python=list(__import__("sys").version_info[:3]),numpy=__import__("numpy").__version__,torch=torch.__version__,
   cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None)),
  probes=16,resource_profile=__import__("model_resources").identity(resource_profile),reference_status="passed",
  reference_dtype="float64",finite_reference=finite,capture_metadata=metadata,
  input_mutation_policy="ordered-nonoverlapping-registered-effects",source_precision_execution_compared=False,
  approximation_accuracy_certified=False,runtime_layout_validated=False,encrypted_execution=False,agent_generated=False,paid_calls=0)
 check_reference(ref,high,lowered,internal,resource_profile=resource_profile,expected_weight_binding=internal_binding)
 record=dict(format=FORMAT,original_manifest_sha256=digest(spec),internal_manifest=internal,
  original_weight_binding=expected_weight_binding,internal_weight_binding=internal_binding,
  division_capture=division,boolean_mask_capture=boolean_masks,source_projection=source_projection,original_public_state=before,normalized_source_state=converted,
  float_normalization=conversion,traced_public_state=traced_state,
  unreferenced_state_names=sorted(set(converted)-set(traced_state)),original_reference=original_finite,
  base_reference=ref,base_binding=bound,high_graph=high,lowered_graph=lowered,
  production_intake_integrated=True,source_intake_version=SOURCE_FORMAT,provider_calls=0,
  compiler_calls=0,encrypted_calls=0,source_dtype="float64",resource_profile=resource_profile)
 record["binding"]=digest(record)
 return record
def check_record(record,spec,original_arrays,internal_arrays,*,expected_record_binding,resource_profile=PROFILE):
 import numpy as np
 from pytorch_ordered_contract import capture_binding,check_reference
 from pytorch_weight_package import binding
 require(type(expected_record_binding)is str and len(expected_record_binding)==64 and
  record.get("binding")==expected_record_binding==digest({k:v for k,v in record.items()if k!="binding"}),
  "External source capture authority")
 require(record["format"]==FORMAT and record["original_manifest_sha256"]==digest(spec)and
  record["resource_profile"]==resource_profile and record["production_intake_integrated"] is True,
  "Staging original source contract")
 expected=binding(original_arrays,spec["weights"])
 require(record["original_weight_binding"]==expected,"Original source public weights changed")
 states=state_records(original_arrays)
 require(record["original_public_state"]==states,"Original source state")
 check_original(states,spec["weights"],float_policy=spec["dtype_policy"])
 converted_arrays={n:np.asarray(a,dtype="<f8")if a.dtype.str=="<f4"else a for n,a in original_arrays.items()}
 converted=state_records(converted_arrays)
 check_normalization(states,converted,float_policy=spec["dtype_policy"])
 require(record["normalized_source_state"]==converted,"Original normalized source state")
 expected_conversion=dict(format="poseidon-public-state-float-normalization-v1",float_policy=spec["dtype_policy"],
  raw_state_sha256=digest(states),normalized_state_sha256=digest(converted),
  converted_names=sorted(n for n,v in states.items()if v["dtype"]=="<f4"),
  compute_dtype="float64",source_precision_execution_compared=False)
 require(record["float_normalization"]==expected_conversion,"Original float conversion policy binding")
 division=record["division_capture"];traced=record["traced_public_state"]
 check_trace(converted,traced,record["unreferenced_state_names"])
 require(division["public_state"]==traced and record["source_intake_version"]==SOURCE_FORMAT,"Declared public divisor source")
 declared=direct_divisors(division["source_inventory"],traced)
 used=set(declared)
 require(division["declared_divisors"]==declared,"Exact direct divisor inventory")
 payloads={n:converted_arrays[n].tobytes(order="C")for n in used}
 check_division(division,spec["inputs"],declared,payloads,
  expected_source_identity=division["source_identity"],resource_profile=resource_profile)
 internal=record["internal_manifest"]
 require(internal["format"]==BASE_FORMAT and {k:v for k,v in internal.items()if k not in ("weights","format")}=={k:v for k,v in spec.items()if k not in ("weights","format")},
  "Internal source changed beyond trusted derived weights")
 actual_states=state_records(internal_arrays)
 boolean_masks=record["boolean_mask_capture"]
 require(boolean_masks["public_state"]==division["normalized_state"]and
  structural(boolean_masks["source_inventory"])==structural(division["normalized_inventory"]),"Division/boolean graph bridge")
 mask_payloads={n:converted_arrays[n].tobytes(order="C")for n,v in traced.items()if v["dtype"]=="|b1"}
 check_masks(boolean_masks,mask_payloads,expected_source_identity=boolean_masks["source_identity"],resource_profile=resource_profile)
 projection=record["source_projection"]
 require(set(projection)=={"format","full_state_sha256","effective_state","unreferenced_state_names",
  "source_fx_sha256","effective_fx_sha256","original_state_preserved"}and
  projection["format"]=="poseidon-effective-derived-public-state-v1"and
  projection["full_state_sha256"]==digest(boolean_masks["normalized_state"])and
  projection["source_fx_sha256"]==boolean_masks["normalized_fx_sha256"]==projection["effective_fx_sha256"]and
  projection["original_state_preserved"]is True,"Effective source projection identity")
 check_trace(boolean_masks["normalized_state"],projection["effective_state"],projection["unreferenced_state_names"])
 require(actual_states==projection["effective_state"],"Derived public weights changed")
 internal_bound=binding(internal_arrays,internal["weights"])
 require(record["internal_weight_binding"]==internal_bound,"Derived weight payload binding")
 metadata=record["base_reference"]["capture_metadata"]
 recaptured=metadata["ordered_alias_capture"]["capture_adapter"]["exact_arithmetic_normalization"]["source_inventory"]
 require(structural(recaptured)==structural(boolean_masks["normalized_inventory"]),"Complete source/derived graph bridge")
 lowered,bound=capture_binding(record["high_graph"],internal,resource_profile=resource_profile,
  capture_metadata=metadata,expected_weight_binding=internal_bound)
 require(lowered==record["lowered_graph"]and bound==record["base_binding"],"Derived graph and capture identity")
 check_reference(record["base_reference"],record["high_graph"],lowered,internal,
  resource_profile=resource_profile,expected_weight_binding=internal_bound)
 original=record["original_reference"]
 require(original["status"]=="passed"and original["groups"]==16 and
  original["graph_sha256"]==digest(record["high_graph"])and original["lowered_sha256"]==digest(lowered)and
  original["capture_binding_sha256"]==bound["binding_sha256"]and original["max_reference_tolerance_ratio"]<=1 and
  original["max_normalization_tolerance_ratio"]<=1 and original["public_state_immutable"]is True and
  original["input_storage_isolated"]is True,"Original source numerical reference")
 require(record["provider_calls"]==record["compiler_calls"]==record["encrypted_calls"]==0,"Capture only scope")
 return record

def derived_arrays(record,spec,arrays,*,resource_profile):
 """Reconstruct trusted derived bytes; never execute source or FX during reload."""
 import io,numpy as np
 from pytorch_weight_package import declarations
 from pytorch_public_tensor_division import plan as divide_plan
 from pytorch_public_boolean_arithmetic import mask_plan
 require(type(arrays)is dict and set(arrays)==set(spec["weights"]),"Original public arrays required")
 converted={n:np.asarray(v,dtype="<f8")if v.dtype.str=="<f4"else np.array(v,copy=True)for n,v in arrays.items()}
 states=state_records(arrays);check_original(states,spec["weights"],float_policy=spec["dtype_policy"])
 require(states==record["original_public_state"],"Raw original public state changed")
 traced=record["traced_public_state"]
 check_trace(state_records(converted),traced,record["unreferenced_state_names"])
 values={n:converted[n]for n in traced}
 for entry in record["division_capture"]["entries"]:
  state=entry["state_name"];require(state in values,"Known public divisor source")
  selected,raw=divide_plan(entry["plan"]["input_shape"],list(values[state].shape),values[state].tobytes(order="C"),
   inplace=entry["plan"]["inplace"],resource_profile=resource_profile)
  require(selected==entry["plan"]and entry["reciprocal_state_name"]not in values,"Derived divisor plan or collision")
  values[entry["reciprocal_state_name"]]=np.frombuffer(raw,dtype="<f8").copy().reshape(selected["divisor_shape"])
 for entry in record["boolean_mask_capture"]["entries"]:
  state=entry["mask_state_name"];require(state in values and values[state].dtype.str=="|b1","Known bool mask source")
  selected,raw=mask_plan(entry["plan"]["input_shape"],list(values[state].shape),values[state].tobytes(order="C"),
   inplace=entry["inplace"],resource_profile=resource_profile)
  require(selected==entry["plan"]and entry["derived_state_name"]not in values,"Derived mask plan or collision")
  values[entry["derived_state_name"]]=np.frombuffer(raw,dtype="<f8").copy().reshape(selected["mask_shape"])
 names=record["source_projection"]["effective_state"]
 require(set(names)<=set(values),"Effective weights have no unregistered sources")
 values={n:values[n]for n in names}
 # Export is deterministic; bind the exact NPY bytes and file inventory without file I/O.
 expected={}
 for index,name in enumerate(sorted(values)):
  value=np.array(values[name],copy=True,order="C");stream=io.BytesIO()
  np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
  expected[name]=dict(file=f"state_{index:04d}.npy",sha256=hashlib.sha256(raw).hexdigest(),
   bytes=len(raw),shape=list(value.shape),dtype=value.dtype.str)
 require(record["internal_manifest"]["weights"]==expected,"Derived NPY content, type or file inventory changed")
 declarations(expected,resource_profile=resource_profile)
 return values

def wrapped_metadata(record):
 metadata=copy.deepcopy(record["base_reference"]["capture_metadata"])
 metadata["weight_binding"]=copy.deepcopy(record["original_weight_binding"])
 types={"<f4":"float32","<f8":"float64","<i4":"int32","<i8":"int64","|b1":"bool"}
 metadata["source_state_dtypes"]={n:types[v["dtype"]]for n,v in record["original_public_state"].items()}
 metadata["public_tensor_arithmetic"]=copy.deepcopy(record)
 return metadata

def check_capture_metadata(metadata,original,spec,*,resource_profile,expected_weight_binding,expected_weight_arrays):
 require(type(metadata)is dict and "public_tensor_arithmetic"in metadata,"Explicit public tensor capture metadata")
 record=metadata["public_tensor_arithmetic"]
 require(type(spec)is dict and spec.get("format")==SOURCE_FORMAT,"Explicit V22 source contract")
 require(type(record)is dict and digest({k:v for k,v in record.items()if k!="binding"})==record.get("binding"),
  "Public source capture digest")
 require(type(expected_weight_arrays)is dict,"Authenticated original public array payloads required")
 from pytorch_weight_package import binding
 actual=binding(expected_weight_arrays,spec["weights"])
 require(actual==expected_weight_binding==record["original_weight_binding"],"Public source expected weights differ")
 internal_arrays=derived_arrays(record,spec,expected_weight_arrays,resource_profile=resource_profile)
 check_record(record,spec,expected_weight_arrays,internal_arrays,expected_record_binding=record["binding"],
  resource_profile=resource_profile)
 require(original==record["high_graph"]and metadata==wrapped_metadata(record),"Public source metadata or graph changed")
 # Both references use the retained complete V21 checker. They differ only in the
 # already executed nn.Module: original source versus normalized source.
 from pytorch_ordered_contract import check_reference
 source_ref=copy.deepcopy(record["base_reference"])
 source_ref["finite_reference"]=copy.deepcopy(record["original_reference"])
 check_reference(source_ref,record["high_graph"],record["lowered_graph"],record["internal_manifest"],
  resource_profile=resource_profile,expected_weight_binding=record["internal_weight_binding"])
 return record["lowered_graph"]

def check_reference(report,original,lowered,spec,*,resource_profile,expected_weight_binding,expected_weight_arrays):
 from pytorch_ordered_contract import source_identity,reference_version,capture_binding
 metadata=report.get("capture_metadata")
 checked=check_capture_metadata(metadata,original,spec,resource_profile=resource_profile,
  expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
 require(checked==lowered,"Public source decomposition differs")
 record=metadata["public_tensor_arithmetic"]
 expected=copy.deepcopy(record["base_reference"])
 expected.update(format=reference_version(spec),manifest_sha256=digest(spec),source_hashes=source_identity(),
  capture_metadata=copy.deepcopy(metadata))
 finite=copy.deepcopy(record["original_reference"])
 _,bound=capture_binding(original,spec,resource_profile=resource_profile,capture_metadata=metadata,
  expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
 finite["capture_binding_sha256"]=bound["binding_sha256"];finite.pop("binding",None);finite["binding"]=digest(finite)
 expected["finite_reference"]=finite
 require(report==expected,"Public tensor reference identity, metrics or validation scope changed")
 return report

def capture_request(request,model_root,output_root):
 """Called solely inside the existing isolated ordered-intake worker."""
 import importlib,sys,numpy as np,torch
 from pytorch_source_contract import factory_parameters
 from pytorch_weight_package import read,load_state
 from pytorch_ordered_contract import validate_package,shim,source_identity,capture_binding,reference_version
 from platform_config import require_python_packages
 require_python_packages(torch,np)
 spec=request["manifest"];profile=request["resource_profile"]
 require(spec.get("format")==SOURCE_FORMAT and request["source_hashes"]==source_identity(),"Public worker source identity")
 runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
  cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
 require(runtime==request["runtime"],"Pinned public capture runtime")
 files={n:Path(model_root,n).read_text()for n in spec["files"]};validate_package(spec,files)
 parameters=factory_parameters(shim(spec));module_name,entry=spec["entry"].split(".")
 torch.set_num_threads(1)
 module=getattr(importlib.import_module(module_name),entry)(*parameters["args"],**parameters["kwargs"])
 require(isinstance(module,torch.nn.Module),"Trusted factory must return nn.Module")
 arrays,_=read(model_root,spec["weights"],resource_profile=profile)
 loaded=load_state(module,arrays,spec["weights"])
 record=prepare(module,spec,output_root,expected_weight_binding=loaded,resource_profile=profile)
 metadata=wrapped_metadata(record)
 high=record["high_graph"];lowered,bound=capture_binding(high,spec,resource_profile=profile,
  capture_metadata=metadata,expected_weight_binding=loaded,expected_weight_arrays=arrays)
 report=copy.deepcopy(record["base_reference"])
 report.update(format=reference_version(spec),manifest_sha256=digest(spec),source_hashes=source_identity(),capture_metadata=metadata)
 finite=copy.deepcopy(record["original_reference"])
 finite["capture_binding_sha256"]=bound["binding_sha256"];finite.pop("binding",None);finite["binding"]=digest(finite)
 report["finite_reference"]=finite
 check_reference(report,high,lowered,spec,resource_profile=profile,expected_weight_binding=loaded,expected_weight_arrays=arrays)
 Path(output_root,"original.json").write_text(json.dumps(high,allow_nan=False))
 Path(output_root,"capture-reference.json").write_text(json.dumps(report,allow_nan=False))
 Path(output_root,"fx-source.txt").write_text(record["division_capture"]["source_fx_text"])
 Path(output_root,"fx-normalized.txt").write_text(record["boolean_mask_capture"]["normalized_fx_text"])
