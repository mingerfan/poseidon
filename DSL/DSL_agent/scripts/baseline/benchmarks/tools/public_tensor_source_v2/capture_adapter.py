"""Agent-owned staging capture with an explicit original-source/derived-weight boundary."""
import copy,hashlib,json
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from public_tensor_division import POLICY,normalize,check as check_division,state_payload
from public_state_binding import check_original,check_trace,direct_divisors,float64_state,check_normalization
from staged_static_trace import trace
from pytorch_static_trace import ALIAS_POLICY
FORMAT="poseidon-public-tensor-source-capture-draft-v2"
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
 require(spec["format"]=="poseidon-trusted-pytorch-v21"and
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
 require(division["entries"],"Public tensor division task requires a declared divisor")
 derived_path=Path(output)/"internal-weight-snapshot";derived_path.mkdir()
 internal_weights=export_state(normalized,derived_path,resource_profile=resource_profile)
 arrays,_=read(derived_path,internal_weights,resource_profile=resource_profile)
 internal_binding=binding(arrays,internal_weights)
 internal=copy.deepcopy(spec);internal["weights"]=internal_weights
 prepared=prepare_module(normalized,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
  resource_profile=resource_profile,dtype_policy=spec["dtype_policy"],functional_policy=ADAPTIVE_FUNCTION_POLICY,**CAPTURE_OPTIONS)
 high,lowered,captured,context=prepared;metadata=captured["pytorch_capture"];metadata["weight_binding"]=internal_binding
 recaptured=metadata["ordered_alias_capture"]["capture_adapter"]["exact_arithmetic_normalization"]["source_inventory"]
 require(structural(recaptured)==structural(division["normalized_inventory"]),"Division/retrace complete structural bridge")
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
  division_capture=division,original_public_state=before,normalized_source_state=converted,
  float_normalization=conversion,traced_public_state=traced_state,
  unreferenced_state_names=sorted(set(converted)-set(traced_state)),original_reference=original_finite,
  base_reference=ref,base_binding=bound,high_graph=high,lowered_graph=lowered,
  production_intake_integrated=False,source_intake_version="draft-only",provider_calls=0,
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
  record["resource_profile"]==resource_profile and record["production_intake_integrated"] is False,
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
 require(division["public_state"]==traced and division["entries"],"Declared public divisor source")
 declared=direct_divisors(division["source_inventory"],traced)
 used=set(declared)
 require(division["declared_divisors"]==declared,"Exact direct divisor inventory")
 payloads={n:converted_arrays[n].tobytes(order="C")for n in used}
 check_division(division,spec["inputs"],declared,payloads,
  expected_source_identity=division["source_identity"],resource_profile=resource_profile)
 internal=record["internal_manifest"]
 require({k:v for k,v in internal.items()if k!="weights"}=={k:v for k,v in spec.items()if k!="weights"},
  "Internal source changed beyond trusted derived weights")
 actual_states=state_records(internal_arrays)
 require(actual_states==division["normalized_state"],"Derived public weights changed")
 internal_bound=binding(internal_arrays,internal["weights"])
 require(record["internal_weight_binding"]==internal_bound,"Derived weight payload binding")
 metadata=record["base_reference"]["capture_metadata"]
 recaptured=metadata["ordered_alias_capture"]["capture_adapter"]["exact_arithmetic_normalization"]["source_inventory"]
 require(structural(recaptured)==structural(division["normalized_inventory"]),"Complete source/derived graph bridge")
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
