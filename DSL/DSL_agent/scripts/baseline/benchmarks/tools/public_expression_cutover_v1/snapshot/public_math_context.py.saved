"""Draft v2 public mathematical roles; stdlib only, never executes a candidate."""
import copy,hashlib,json,math,re,struct
VERSION="poseidon-public-math-roles-v1"
POLICY="descriptors-math-roles-v2"
FORMAT="poseidon-public-weight-descriptors-v2"
MAX_MATH_ELEMENTS=256
MAX_ROLE_BYTES=49152
MAX_CONTEXT_BYTES=131072
ELEMENTWISE={"add","subtract","multiply","divide","maximum","minimum"}
TOP_KEYS={"schema","task","model","fx_graph","public_constants","constant_origins","layout","rules","response_schema","semantic_guidance","compiler_profile_sha256","privacy","request_id","public_weight_binding","construction_profile","construction_exercise","compiler_configuration","upstream_helpers","upstream_exercise","generation_guidance","capability_composition"}
RULES="""\nPublic weight descriptor contract v2:
Large public tensors retain shape and content-hash descriptors.
math_roles supplies checked public mathematical values and exact bindings for
scalar operands and polynomial coefficients. Other public tensors retain existing
constant_origins block/layout roles and descriptors; no scalar Plain is required
for each tensor element.
Coefficient index is the declared ascending degree/basis index. pN numbering
has no mathematical meaning. Use the listed immutable Plain names and retain
every nonzero coefficient. exact_zero denotes mathematical zero, not a request
to produce a transparent ciphertext. A zero result requires the existing trusted
encrypted-zero capability. Tensor/scalar equality does not change logical shape,
layout, matrix semantics, rotation keys or backend capability. These are public
mathematical roles, not a DSL implementation or private numerical answers.
\n"""
class RoleContextRejected(ValueError):pass
def require(ok,code):
 if not ok:raise RoleContextRejected(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def numeric(v):
 require(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1024,"public_numeric_value")
 return v
def binary64(v):return struct.pack(">d",float(numeric(v)))
def shape(v):
 if type(v) in (int,float):numeric(v);return []
 require(type(v) is list and len(v)>0,"public_rectangular_tensor")
 child=shape(v[0])
 require(all(shape(x)==child for x in v),"public_rectangular_tensor")
 dims=[len(v),*child]
 require(len(dims)<=32 and math.prod(dims)<=1000000,"public_tensor_resource")
 return dims
def descriptor(v):
 dims=shape(v)
 return dict(kind="tensor" if dims else "scalar",shape=dims,elements=math.prod(dims),content_sha256=digest(v))
def leaves(v):
 if type(v) is list:
  for x in v:yield from leaves(x)
 else:yield numeric(v)
def graph_projection(request,info):
 m=copy.deepcopy(request["model"]);m["constants"]=copy.deepcopy(info["model_constants"])
 return m
def usage(model):
 out={k:[] for k in model["constants"]}
 for n in model["nodes"]:
  require(type(n) is dict and type(n.get("id")) is str and type(n.get("op")) is str and type(n.get("inputs")) is list,"public_node_shape")
  for i,x in enumerate(n["inputs"]):
   if x in out:
    role="polynomial_coefficients" if n["op"]=="polynomial" and i==1 else "elementwise_operand" if n["op"] in ELEMENTWISE else "operator_operand"
    out[x].append(dict(node=n["id"],operator=n["op"],input_index=i,role=role))
 return out
def _base(request,info):
 require(type(request) is dict and set(request)<=TOP_KEYS,"public_envelope_fields")
 require(request.get("privacy")=={"input":"encrypted","weights":"public"},"public_model_privacy")
 require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu","public_context_backend")
 require(type(request.get("model")) is dict and type(request["model"].get("constants")) is dict
  and type(request["model"].get("nodes")) is list and len(request["model"]["nodes"])<=4096,"public_model_structure")
 require(type(request.get("public_constants")) is dict and len(request["public_constants"])<=256,"public_registry_structure")
 require(type(info) is dict and type(info.get("model_constants")) is dict and type(info.get("public_constants")) is dict,"public_descriptor_registry")
 expected={"model_constants":{k:descriptor(v) for k,v in request["model"]["constants"].items()},
  "public_constants":{k:descriptor(v) for k,v in request["public_constants"].items()}}
 require(info["model_constants"]==expected["model_constants"] and info["public_constants"]==expected["public_constants"],"public_descriptor_content_hash")
 require(len(expected["model_constants"])<=1024 and sum(v["elements"] for v in expected["model_constants"].values())<=1000000,"public_constant_resource")
 for name in expected["public_constants"]:
  require(re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,63}",name) is not None,"prebound_plain_name")
 return expected
def _zero_capability(request):
 return any(type(x) is dict and x.get("kind")=="encrypted_zero" and x.get("source")=="trusted_client"
  and type(x.get("dsl_name")) is str for x in request.get("layout",{}).get("auxiliary_ciphertexts",[]))
def _semantics(model,name,values):
 nodes={n["id"]:n for n in model["nodes"]};result=[]
 for u in usage(model)[name]:
  if u["role"]=="polynomial_coefficients":
   node=nodes[u["node"]];basis=node.get("attrs",{}).get("basis")
   require(basis in ("power","chebyshev"),"explicit_polynomial_basis")
   degree=max((i for i,x in enumerate(values) if x!=0),default=None)
   result.append(dict(node=node["id"],basis=basis,coefficient_order="ascending_basis_index",
    declared_terms=len(values),mathematical_degree=degree,all_zero=degree is None,
    nonzero_indices=[i for i,x in enumerate(values) if x!=0]))
 return result
def build(request,info=None):
 """Checked role record from host-owned numeric registries. No name/family dispatch."""
 info=request["public_weight_binding"] if info is None else info
 expected=_base(request,info);used=usage(request["model"])
 scalar_names={}
 for name,v in request["public_constants"].items():
  if type(v) in (int,float):scalar_names.setdefault(binary64(v),[]).append(name)
 rows=[];opaque=[];matched={};count=0
 for name,value in sorted(request["model"]["constants"].items()):
  desc=expected["model_constants"][name];dims=desc["shape"];uses=used[name]
  polynomial=any(u["role"]=="polynomial_coefficients" for u in uses)
  math_tensor=any(u["role"]=="elementwise_operand" for u in uses)
  if not dims:kind="scalar_operand";vals=[value]
  elif polynomial:
   require(len(dims)==1 and desc["elements"]<=MAX_MATH_ELEMENTS,"polynomial_coefficient_capacity")
   kind="polynomial_coefficients";vals=list(value)
  else:
   opaque.append(dict(logical_name=name,descriptor=desc,usages=uses,
    exact_whole_plain_names=sorted(k for k,v in expected["public_constants"].items() if v==desc)))
   continue
  count+=len(vals);require(count<=MAX_MATH_ELEMENTS,"public_math_metadata_capacity")
  elements=[]
  for index,v in enumerate(vals):
   names=sorted(scalar_names.get(binary64(v),[]));is_zero=v==0
   require(names or is_zero,"missing_required_public_plain_binding")
   for p in names:matched[p]=dict(value=request["public_constants"][p],descriptor=expected["public_constants"][p])
   elements.append(dict(index=index,value=v,exact_zero=is_zero,
    logical_scalar_content_sha256=digest(v),plain_names=names,
    binding_required=not is_zero))
  semantics=_semantics(request["model"],name,vals)
  if any(s["all_zero"] for s in semantics):
   require(_zero_capability(request),"missing_trusted_encrypted_zero_capability")
  rows.append(dict(logical_name=name,descriptor=desc,kind=kind,usages=uses,elements=elements,
   polynomial_semantics=semantics))
 record=dict(format=VERSION,scope="public mathematical values and exact prebound roles; no DSL answer",
  equality="exact-binary64-with-signed-zero-v1",source_registry_sha256=digest(expected),
  projected_graph_sha256=digest(graph_projection(request,info)),math_elements=count,
  required_scalar_polynomial_roles_complete=True,physical_layout_validated=False,
  backend_executable_not_established=True,encrypted_zero_available=_zero_capability(request),existing_layout_roles_sha256=digest(request.get("constant_origins",{})),
  constants=rows,opaque_constants=opaque,prebound_math_scalars={k:matched[k] for k in sorted(matched)})
 record["binding"]=digest(record)
 require(len(canonical(record))<=MAX_ROLE_BYTES,"public_math_metadata_bytes")
 return record
def _nested(vals,dims):
 if not dims:
  require(len(vals)==1,"scalar_math_value_count");return vals[0]
 if len(dims)==1:return list(vals)
 width=math.prod(dims[1:])
 return [_nested(vals[i*width:(i+1)*width],dims[1:]) for i in range(dims[0])]
def verify_projected(context):
 """Check projected mathematical values against logical/Plain hashes and roles."""
 require(type(context) is dict and set(context)<=TOP_KEYS,"public_envelope_fields")
 info=context.get("public_weight_binding",{})
 require(info.get("format")==FORMAT and info.get("policy")==POLICY and "math_roles" in info,"explicit_math_role_policy")
 require(context["model"]["constants"]==info["model_constants"] and context["public_constants"]==info["public_constants"],"projected_descriptor_registry")
 r=info["math_roles"]
 require(type(r) is dict and set(r)=={"format","scope","equality","source_registry_sha256","projected_graph_sha256","math_elements","required_scalar_polynomial_roles_complete","physical_layout_validated","backend_executable_not_established","encrypted_zero_available","existing_layout_roles_sha256","constants","opaque_constants","prebound_math_scalars","binding"} and r.get("format")==VERSION and r.get("binding")==digest({k:v for k,v in r.items() if k!="binding"}),"public_math_role_binding")
 require(r["source_registry_sha256"]==digest({k:info[k] for k in ("model_constants","public_constants")})
  and r["projected_graph_sha256"]==digest(context["model"]),"public_math_role_source_binding")
 require(r["scope"]=="public mathematical values and exact prebound roles; no DSL answer" and r["equality"]=="exact-binary64-with-signed-zero-v1" and r["required_scalar_polynomial_roles_complete"] is True
  and r["physical_layout_validated"] is False and r["backend_executable_not_established"] is True
  and r["encrypted_zero_available"]==_zero_capability(context) and r["existing_layout_roles_sha256"]==digest(context.get("constant_origins",{})),"public_math_role_scope")
 used=usage(context["model"]);seen=set();count=0;required_scalars=set()
 for row in r["constants"]:
  require(type(row) is dict and set(row)=={"logical_name","descriptor","kind","usages","elements","polynomial_semantics"},"logical_role_fields")
  name=row["logical_name"];require(name not in seen and name in info["model_constants"],"logical_role_inventory");seen.add(name)
  desc=info["model_constants"][name]
  require(row["descriptor"]==desc and row["usages"]==used[name],"logical_role_identity")
  entries=row["elements"];require(type(entries) is list and len(entries)==desc["elements"]<=MAX_MATH_ELEMENTS,"math_element_shape")
  vals=[]
  for index,e in enumerate(entries):
   require(type(e) is dict and set(e)=={"index","value","exact_zero","logical_scalar_content_sha256","plain_names","binding_required"},"math_element_fields")
   v=numeric(e["value"]);names=e["plain_names"]
   require(e["index"]==index and e["logical_scalar_content_sha256"]==digest(v) and
    e["exact_zero"] is (v==0) and e["binding_required"] is (v!=0),"math_element_role")
   require(type(names) is list and names==sorted(set(names)) and (names or v==0),"public_math_element_names")
   for p in names:
    require(p in info["public_constants"] and p in r["prebound_math_scalars"],"prebound_math_name")
    plain=r["prebound_math_scalars"][p]
    require(type(plain) is dict and set(plain)=={"value","descriptor"},"prebound_math_fields")
    require(plain["descriptor"]==info["public_constants"][p] and descriptor(plain["value"])==plain["descriptor"]
     and plain["descriptor"]["kind"]=="scalar" and binary64(v)==binary64(plain["value"]),"prebound_math_value")
    required_scalars.add(p)
   vals.append(v)
  require(descriptor(_nested(vals,desc["shape"]))==desc,"logical_math_value_hash")
  require(row["polynomial_semantics"]==_semantics(context["model"],name,vals),"polynomial_math_semantics")
  if any(s["all_zero"] for s in row["polynomial_semantics"]):require(_zero_capability(context),"missing_trusted_encrypted_zero_capability")
  polynomial=any(u["role"]=="polynomial_coefficients" for u in used[name])
  math_tensor=any(u["role"]=="elementwise_operand" for u in used[name])
  expected_kind="scalar_operand" if not desc["shape"] else "polynomial_coefficients" if polynomial else None
  require(row["kind"]==expected_kind and (not polynomial or len(desc["shape"])==1),"logical_math_kind")
  count+=len(vals)
 for row in r["opaque_constants"]:
  require(type(row) is dict and set(row)=={"logical_name","descriptor","usages","exact_whole_plain_names"},"opaque_role_fields")
  name=row["logical_name"];require(name not in seen and name in info["model_constants"],"opaque_role_inventory");seen.add(name)
  desc=info["model_constants"][name]
  require(row["descriptor"]==desc and row["usages"]==used[name] and bool(desc["shape"])
   and not any(u["role"]=="polynomial_coefficients" for u in used[name]),"opaque_role_identity")
  require(row["exact_whole_plain_names"]==sorted(k for k,v in info["public_constants"].items() if v==desc),"opaque_whole_binding")
 require(seen==set(info["model_constants"]) and count==r["math_elements"]<=MAX_MATH_ELEMENTS and
  set(r["prebound_math_scalars"])==required_scalars,"complete_math_role_inventory")
 require(len(canonical(r))<=MAX_ROLE_BYTES and len(canonical(context))<=MAX_CONTEXT_BYTES,"public_context_bytes")
 return copy.deepcopy(r)
def upgrade(request):
 """Draft opt-in transform; never mutates or approves an existing request."""
 info=request["public_weight_binding"]
 require(info.get("policy")=="descriptors-v1" and info.get("format")=="poseidon-public-weight-descriptors-v1","legacy_upgrade_source")
 _base(request,info)
 legacy=copy.deepcopy(request);legacy["model"]["constants"]=copy.deepcopy(info["model_constants"]);legacy["public_constants"]=copy.deepcopy(info["public_constants"])
 require(request["request_id"]==digest({k:v for k,v in legacy.items() if k!="request_id"}),"legacy_request_identity")
 out=copy.deepcopy(request);b=copy.deepcopy(info);b.update(format=FORMAT,policy=POLICY)
 b["math_roles"]=build(request,info);out["public_weight_binding"]=b
 from_rule="""\nPublic weight descriptor contract v1:
Model constants name the logical weights. Only names listed in public_constants
are prebound DSL values; constant_origins describes their model origin.
The generation context gives shapes and content hashes instead of numeric data.
Use public_constants names directly; never substitute weights, open a file or load
a mapping. The trusted frontend supplies their fixed values during tracing.
\n"""
 require(out["rules"].endswith(from_rule),"exact_legacy_rule_suffix")
 out["rules"]=out["rules"][:-len(from_rule)]+RULES
 projected=project(out,verify=False)
 out["request_id"]=digest({k:v for k,v in projected.items() if k!="request_id"})
 project(out);return out
def project(request,*,verify=True):
 info=request["public_weight_binding"]
 require(info.get("format")==FORMAT and info.get("policy")==POLICY,"explicit_math_role_policy")
 if verify:
  require(info["math_roles"]==build(request,info),"trusted_math_role_content")
 context=copy.deepcopy(request);context["model"]["constants"]=copy.deepcopy(info["model_constants"])
 context["public_constants"]=copy.deepcopy(info["public_constants"])
 if verify:
  verify_projected(context)
  require(request["request_id"]==digest({k:v for k,v in context.items() if k!="request_id"}),"new_math_role_request_identity")
 return context
