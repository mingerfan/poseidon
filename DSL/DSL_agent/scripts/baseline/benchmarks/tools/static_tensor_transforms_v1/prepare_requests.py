"""Freeze exact application requests from verified independent capture results."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from component_contract import prepare_task,reconstruct_request
from deepseek_provider import public_request
OPTIONS=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",
             chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open("x")as s:json.dump(v,s,sort_keys=True,indent=2)
rows=[]
for case in read(HERE/"capture-index.json"):
 name=case["name"];folder=Path(case["model_import"]);model=read(folder/"model.json")
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
 options=preparation_options(provenance,model,OPTIONS)
 options=graph_options(provenance["original"],model,options)
 request=prepare_task(model,options);assert reconstruct_request(request)==request
 result=HERE/"prepared-requests"/name;result.mkdir(parents=True)
 write(result/"request.json",request);write(result/"public-request.json",public_request(request))
 reference=provenance["reference_report"]["finite_reference"]
 rows.append(dict(id=name,source_manifest=case["manifest"],source_manifest_sha256=sha(case["manifest"]),
  source_format=read(case["manifest"])["format"],operator=case["operator"],kind=case["kind"],model_import=str(folder),
  model_sha256=digest(model),provenance_digest=digest(provenance),generation_guidance="explicit-v13",validation_level="numerical",
  request_file=str(result/"request.json"),request_file_sha256=sha(result/"request.json"),request_id=request["request_id"],
  public_request_file=str(result/"public-request.json"),public_request_file_sha256=sha(result/"public-request.json"),
  plaintext_groups=16,plaintext_values=reference["compared_values"]))
result=dict(format="poseidon-static-transforms-prepared-requests-v1",cases=rows,paid_calls=0,compiler_calls=0,encrypted_calls=0)
result["binding"]=digest(result);write(HERE/"prepared-requests.json",result)
print(json.dumps(dict(cases=len(rows),groups=sum(r["plaintext_groups"]for r in rows),values=sum(r["plaintext_values"]for r in rows),binding=result["binding"])))
