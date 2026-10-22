"""Public high-level model context from host-verified decomposition provenance.

No source code, private samples, reference answers or deterministic DSL is emitted.
The fixed lowered graph remains authoritative for candidate qualification.
"""
import copy
from benchmark_graph import canonical,digest,require
from public_weight_context import descriptor

FORMAT="poseidon-generation-model-origin-v1"
MAX_BYTES=65536

def build(provenance,request):
    from provenance_json_transport import unpack
    from unified_graph_contract import validate_request
    validate_request(request)
    from candidate_bundle import check_provenance
    value=unpack(provenance,purpose="model-provenance")
    binding=check_provenance(value,request["model"])
    original=value["original"]
    constants={name:descriptor(data)for name,data in original["constants"].items()}
    # Keep mathematical attributes, source locations and logical port order only.
    nodes=[dict(id=node["id"],op=node["op"],inputs=list(node["inputs"]),
                attrs=copy.deepcopy(node["attrs"]),outputs=list(node["outputs"]))
           for node in original["nodes"]]
    result=dict(format=FORMAT,authority="host-verified mathematical source description",
        target_model_sha256=digest(request["model"]),original_model_sha256=digest(original),
        decomposition_binding_sha256=binding["binding_sha256"],
        registry_version=binding["registry_version"],
        inputs=copy.deepcopy(original["inputs"]),outputs=copy.deepcopy(original["outputs"]),
        constants=constants,nodes=nodes,
        source_map=copy.deepcopy(binding["source_map"]),
        original_function_accuracy_certified=False,
        source_code_included=False,reference_answers_included=False,rule_dsl_included=False)
    result["binding"]=digest(result)
    require(len(canonical(result))<=MAX_BYTES,"Public model origin context budget")
    return result

def verify(value,provenance,request):
    """Full source consistency is checked with retained host provenance."""
    require(type(value)is dict and value==build(provenance,request),"Public model origin context changed")
    return value


def checked_public(value,request=None):
    """Closed public shape/hash check; source truth needs host verify()."""
    import math,re
    fields={"format","authority","target_model_sha256","original_model_sha256",
      "decomposition_binding_sha256","registry_version","inputs","outputs","constants","nodes",
      "source_map","original_function_accuracy_certified","source_code_included",
      "reference_answers_included","rule_dsl_included","binding"}
    require(type(value)is dict and set(value)==fields and value["format"]==FORMAT,
            "Public model origin fields")
    require(len(canonical(value))<=MAX_BYTES and
      value["binding"]==digest({k:v for k,v in value.items()if k!="binding"}),"Public model origin binding")
    require(value["authority"]=="host-verified mathematical source description" and
      all(value[k]is False for k in("original_function_accuracy_certified","source_code_included",
        "reference_answers_included","rule_dsl_included")),"Public model origin claims")
    require(all(type(value[k])is str and re.fullmatch("[0-9a-f]{64}",value[k])
        for k in("target_model_sha256","original_model_sha256","decomposition_binding_sha256"))and
      type(value["registry_version"])is str and 0<len(value["registry_version"])<=128,
      "Public model origin identities")
    name=lambda n:type(n)is str and re.fullmatch("[A-Za-z_][A-Za-z0-9_]*",n) is not None and len(n)<=256
    require(type(value["constants"])is dict and len(value["constants"])<=1024,"Public origin constants")
    refs=set()
    for key,item in value["constants"].items():
        require(name(key)and type(item)is dict and set(item)=={"kind","shape","elements","content_sha256"},
          "Public origin constant descriptor")
        dims=item["shape"]
        require(type(dims)is list and len(dims)<=32 and all(type(n)is int and 1<=n<=65536for n in dims)and
          type(item["elements"])is int and item["elements"]==math.prod(dims)and 1<=item["elements"]<=1000000and
          item["kind"]==("tensor"if dims else"scalar")and type(item["content_sha256"])is str and
          re.fullmatch("[0-9a-f]{64}",item["content_sha256"]),"Public origin descriptor shape")
        refs.add(key)
    require(type(value["inputs"])is list and 1<=len(value["inputs"])<=4and type(value["outputs"])is list and
      1<=len(value["outputs"])<=4 and type(value["nodes"])is list and 1<=len(value["nodes"])<=1024,
      "Public origin graph count")
    for row in value["inputs"]:
        require(type(row)is dict and set(row)=={"name","shape"}and name(row["name"])and row["name"]not in refs and
          type(row["shape"])is list and len(row["shape"])<=32and all(type(n)is int and 1<=n<=65536for n in row["shape"]),
          "Public origin input")
        refs.add(row["name"])
    ids=set()
    for node in value["nodes"]:
        require(type(node)is dict and set(node)=={"id","op","inputs","attrs","outputs"}and name(node["id"])and
          node["id"]not in ids and name(node["op"])and type(node["attrs"])is dict and
          type(node["inputs"])is list and node["inputs"]and all(r in refs for r in node["inputs"])and
          type(node["outputs"])is list and node["outputs"]and all(name(n)and n not in refs for n in node["outputs"])and
          len(set(node["outputs"]))==len(node["outputs"]),"Public origin node")
        ids.add(node["id"]);refs.update(node["outputs"])
    require(all(type(row)is dict and set(row)=={"name","value"}and name(row["name"])and row["value"]in refs
      for row in value["outputs"])and len({r["name"]for r in value["outputs"]})==len(value["outputs"]),"Public origin outputs")
    require(type(value["source_map"])is dict and len(value["source_map"])<=4096and
      all(name(k)and type(v)is str and v in ids for k,v in value["source_map"].items()),"Public origin source map")
    if request is not None:
        require(value["target_model_sha256"]==digest(request["model"])and
          set(value["source_map"])=={n["id"]for n in request["model"]["nodes"]},"Public origin target binding")
    return copy.deepcopy(value)
