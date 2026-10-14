"""Freeze three GELU public requests for review; never constructs a Provider."""
import hashlib,json
from pathlib import Path
from hecate_python_env import ROOT,WORK
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from component_contract import prepare_task,reconstruct_request
from public_weight_context import project
from benchmark_graph import canonical,digest
from application_backend import PoseidonBackend
from candidate_contract import request_file_bytes
OUT=ROOT/"docs/baseline/agent-gelu-live-proposal-r1306-cases"
assert not OUT.exists(),"preserve prior proposal"
OUT.mkdir();rows=[]
for name in ["tanh_module","tanh_functional","tanh_input_keyword"]:
 capture=WORK/"results/agent-gelu-source-v15-capture"/name
 model=json.loads((capture/"model.json").read_bytes())
 provenance=load_model_provenance(capture,model,max_bytes=4*1024**2)
 options=preparation_options(provenance,model,dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-v1"))
 request=prepare_task(model,options);reconstruct_request(request)
 context=project(request)
 request_file=OUT/(name+"-request.json");request_file.write_bytes(request_file_bytes(request))
 context_file=OUT/(name+"-context.json");context_file.write_bytes(canonical(context))
 source_manifest=ROOT/"scripts/baseline/cases/trusted-pytorch-gelu-v15"/(name+".json")
 rows.append(dict(id=name,source_manifest=str(source_manifest.relative_to(ROOT)),
  source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
  source_files=json.loads(source_manifest.read_bytes())["files"],
  source_format="poseidon-trusted-pytorch-v15",request_id=request["request_id"],
  full_request_file=str(request_file.relative_to(ROOT)),full_request_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(),
  generation_context_file=str(context_file.relative_to(ROOT)),generation_context_sha256=hashlib.sha256(context_file.read_bytes()).hexdigest(),
  model_sha256=digest(model),provenance_sha256=digest(provenance),
  current_validation_purpose="new source GELU variant generation plus mandatory execution resource precheck",
  raw_source_sent_to_provider=False,private_reference_sent=False,deterministic_dsl_answer_sent=False))
backend=PoseidonBackend()
value=dict(format="poseidon-gelu-live-proposal-v1",status="awaiting_human_approval",cases=rows,
 source_stage_identity=backend.identity_manifest(),backend_identity=backend.identity(),
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True),
 limits=dict(generations=12,http_attempts=48,repairs_per_case=3,transport_retries_per_generation=3,
  api_timeout_seconds=1200,max_tokens=384000,wall_seconds=43200,case_wall_seconds=14400,
  api_concurrency=1,native_concurrency=1,compile_jobs=2,link_jobs=1,
  case_artifact_bytes=4*1024**3,global_artifact_bytes=32*1024**3,minimum_free_bytes=4*1024**3),
 explicit_currency_cap=None,retries_may_repeat_billing=True,
 validation=dict(level="numerical",backend="poseidon_cpu",qualification_groups=4,new_input_groups=3,
  package_reload=True,repeat_execution=True,cache_reuse=True,atol=1e-5,rtol=1e-4,
  source_variant_mathematics="tanh",accuracy_against_original_application="not certified"),
 external_content=["public normalized graph","fixed public weights or descriptors","layout","DSL rules","controlled diagnostics"],
 permissions=dict(api_calls_made=0,installs=0,downloads=0,backend_edits=0,parameters_changed=False,
  commit=False,push=False,pr=False))
value["binding"]=digest(value)
out=ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json"
assert not out.exists()
out.write_bytes(canonical(value))
print("READY_NOT_EXECUTED",value["binding"],[r["request_id"] for r in rows])
