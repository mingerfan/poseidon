
"""Prepare the six new broadcast source contexts without generation or native calls."""
import sys,shlex,json,time,hashlib,copy,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
if "--inside"not in sys.argv:
 from hecate_python_env import enter_nix,VENV
 cmd='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=480,environment="poseidon-cpu"))
import numpy as np
from benchmark_graph import require,digest
from hecate_python_env import WORK
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_backend import PoseidonBackend
from component_contract import reconstruct_request
from deepseek_provider import public_request
from pytorch_ordered_contract import source_identity
from application_execution.context_identity import authority
sys.path.insert(0,str(ROOT/"scripts"))
from prepare_pytorch_source import build
sys.path.insert(0,str(HERE.parent/"normalized_source_live_v1"))
import run as previous_run
previous_run.no_active()
baseline=previous_run.measure(WORK/"results");start=time.monotonic()
def boundary():
 require(time.monotonic()-start<450,"broadcast_preparation_time")
 size=previous_run.measure(WORK/"results")
 require(size<=32*1024**3 and size-baseline<=20*1024**2 and shutil.disk_usage(WORK).free>=4*1024**3,"broadcast_preparation_space")
root=HERE/"requests-v3";require(root.is_dir(),"retained_preparation_sources")
receipts=WORK/"results"/("agent-static-broadcast-prepare-"+str(time.time_ns()));receipts.mkdir()
sdk=PoseidonBackend().identity_manifest()
body='''import torch
class Model(torch.nn.Module):
    def __init__(self,method,context):
        super().__init__();self.method=method;self.context=context
        if context=="scalar_linear":self.linear=torch.nn.Linear(3,2,bias=True,dtype=torch.float64)
    def forward(self,x,y):
        if self.context=="arithmetic":
            source=x
        elif self.context=="transpose_alias":
            source=x.transpose(0,1).clone()
        else:
            source=x.mean()
        z=source.expand_as(other=y) if self.method=="expand_as" else source.broadcast_to(size=y.shape)
        if self.context=="arithmetic":return z+y
        if self.context=="transpose_alias":
            source.add_(0.25)
            return (z*y,z.mean(dim=0))
        return {"result":self.linear(z)+y[:,:2]}
def build(method,context):return Model(method,context).double().eval()
'''
sources=[]
options=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
for method in ("expand_as","broadcast_to"):
 for context in ("arithmetic","transpose_alias","scalar_linear"):
  boundary();name=method+"__"+context;folder=root/name
  if (folder/"qualification.json").exists():
   entry=json.loads((folder/"qualification.json").read_text())
   meta=json.loads(Path(entry["qualification"]["source_task_metadata"]).read_text())
   require(meta["binding"]==digest({k:v for k,v in meta.items()if k!="binding"})and meta["identity"]==sdk,"retained_source_metadata")
   for field in("source_files","declared_weight_files"):
    for path,want in entry[field].items():require(sha(path)==want,"retained_source_bytes")
   require(sha(ROOT/entry["source_manifest"])==entry["source_manifest_sha256"]and
    sha(entry["qualification"]["source_task_metadata"])==entry["qualification"]["source_task_metadata_sha256"],"retained_qualification")
   q=json.loads(Path(entry["request_file"]).read_text())
   require(meta["request"]==q==reconstruct_request(q)and digest(q)==entry["public_request_sha256"],"retained_public_request")
   from candidate_bundle import check_provenance
   check_provenance(meta["model_provenance"],q["model"])
   sources.append(entry);print(json.dumps(dict(context=name,status="retained_capture_revalidated",api_calls=0)),flush=True);continue
  folder.mkdir();source=folder/"source";source.mkdir()
  (source/"broadcast_model.py").write_text(body)
  shapes=([1,2],[2,2])if context=="arithmetic"else([3,1],[2,3])if context=="transpose_alias"else([2,2],[2,3])
  weights=[]
  if context=="scalar_linear":
   np.save(source/"linear_weight.npy",np.array([[.17,-.29,.31],[-.37,.41,.23]],dtype="<f8"),allow_pickle=False)
   np.save(source/"linear_bias.npy",np.array([.11,-.13],dtype="<f8"),allow_pickle=False)
   weights=["linear.weight=linear_weight.npy","linear.bias=linear_bias.npy"]
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name=n)for n in("x","y")],kwargs={})
  outputs=["product","mean"]if context=="transpose_alias"else["result"]
  manifest=source/"manifest.json"
  m,_=build(source/"broadcast_model.py",entry="broadcast_model.build",inputs=[n+":"+",".join(map(str,s))for n,s in zip(("x","y"),shapes)],outputs=outputs,
    weights=weights,factory_args=[method,context],source_format="poseidon-trusted-pytorch-v28",resource_profile="general-tensors-v1",call=call)
  write(manifest,m)
  app=application(WorkerContext.capture(receipts/name/"tasks"))
  task=app.submit_source(dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",manifest=str(manifest),trust_local_code=True),
   resource_profile="general-tensors-v1",options=options,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task)
  write(folder/"preparation-state.json",state)
  require(state["status"]=="queued"and state["intake"]["reference_verified"]is True,"broadcast_source_reference_qualification:"+name)
  require(state["generations"]==state["http_attempts"]==0,"broadcast_no_generation")
  meta=app._checked_task_metadata(task);q=meta["request"];require(reconstruct_request(q)==q and meta["identity"]==sdk,"broadcast_request_identity")
  full=folder/"request.json";public=folder/"public-request.json";write(full,q);write(public,public_request(q))
  capture=meta["model_provenance"]
  entry=dict(context=name,source_manifest=str(manifest.relative_to(ROOT)),source_manifest_sha256=sha(manifest),source_format=m["format"],
    input_signature=m["inputs"],output_names=m["outputs"],dtype_policy=m["dtype_policy"],
    source_files={str(source/n):h for n,h in m["files"].items()},
    declared_weight_files={str(source/i["file"]):i["sha256"]for i in m["weights"].values()},
    request_id=q["request_id"],model_sha256=digest(q["model"]),public_request_sha256=digest(q),
    request_file=str(full),request_file_sha256=sha(full),public_request_file=str(public),public_payload_sha256=sha(public),
    qualification=dict(current_source_capture="passed",current_independent_reference="passed",compilations=0,encrypted_calls=0,
      source_task_directory=str(app._dir(task)),source_task_metadata=str(app._dir(task)/"task.json"),source_task_metadata_sha256=sha(app._dir(task)/"task.json"),
      not_current_agent_generation=True))
  sources.append(entry);write(folder/"qualification.json",entry);boundary()
  print(json.dumps(dict(context=name,status="capture_reference_passed",api_calls=0)),flush=True)
groups={}
for item in sources:groups.setdefault(item["public_request_sha256"],[]).append(item)
cases=[];request_groups=[]
for entries in groups.values():
 representative=entries[0];case=copy.deepcopy(representative);case.pop("context")
 case["public_request_sha256"]=sha(case["public_request_file"])
 case.update(id=representative["context"],resource_profile="general-tensors-v1",generation_guidance="explicit-v13",slot_period=4,
  validation_level="numerical",track="free_generation",max_repairs=3,max_generations=4,max_http_attempts=16,approximation_accuracy_certified=False)
 cases.append(case)
 request_groups.append(dict(representative_context=representative["context"],representative_case=case["id"],contexts=[s["context"]for s in entries],
  request_id=representative["request_id"],public_request_sha256=representative["public_request_sha256"]))
scope=dict(sources=sources,cases=cases,request_groups=request_groups,sdk=sdk,source_capture_implementation=dict(source_identity(),**authority(ROOT)["sources"]),
  preparation_result=str(receipts),preparation_seconds=time.monotonic()-start,preparation_baseline_bytes=baseline,preparation_final_bytes=previous_run.measure(WORK/"results"))
scope["binding"]=digest(scope);write(HERE/"prepared-source-scope.json",scope)
require(PoseidonBackend().identity_manifest()==sdk,"broadcast_preparation_stable_sdk")
print(json.dumps(dict(status="prepared",contexts=len(sources),distinct_requests=len(cases),api_calls=0,compiler_calls=0,encrypted_calls=0,sdk=sdk["manifest_sha256"])),flush=True)
