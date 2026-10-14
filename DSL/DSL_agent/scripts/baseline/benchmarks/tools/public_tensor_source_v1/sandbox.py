"""Bounded, network-isolated capture for the inactive public-divisor source draft."""
import hashlib,importlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from benchmark_graph import require,digest
from model_resources import resolve
from model_weight_package import regular_bytes
from pytorch_ordered_contract import prepare as prepare_spec,implementation_paths,source_identity,validate_package,shim
HERE=Path(__file__).resolve().parent
DRAFT=("capture_adapter.py","public_tensor_division.py","staged_static_trace.py","sandbox.py")
PROFILE="general-tensors-v1"
def draft_identity():
 return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in DRAFT}
def worker():
 import numpy as np,torch
 from capture_adapter import prepare
 from pytorch_source_contract import factory_parameters
 from pytorch_weight_package import read,load_state
 from platform_config import require_python_packages
 import trusted_pytorch_import as legacy
 require_python_packages(torch,np);torch.set_num_threads(1)
 request=json.loads(Path("/request.json").read_text())
 require(request["source_hashes"]==source_identity()and request["draft_hashes"]==draft_identity(),"Frozen staging implementation")
 require(request["runtime"]==legacy.runtime_identity(),"Pinned staging runtime")
 spec=request["manifest"];profile=request["resource_profile"]
 files={name:Path("/model",name).read_text()for name in spec["files"]}
 validate_package(spec,files)
 parameters=factory_parameters(shim(spec));module_name,entry=spec["entry"].split(".")
 module=getattr(importlib.import_module(module_name),entry)(*parameters["args"],**parameters["kwargs"])
 require(isinstance(module,torch.nn.Module),"Trusted source factory module")
 arrays,_=read("/model",spec["weights"],resource_profile=profile)
 bound=load_state(module,arrays,spec["weights"])
 record=prepare(module,spec,"/out",expected_weight_binding=bound,resource_profile=profile)
 record["source_hashes"]=source_identity();record["draft_hashes"]=draft_identity()
 record["binding"]=digest({k:v for k,v in record.items()if k!="binding"})
 with Path("/out/capture.json").open("x")as f:json.dump(record,f,allow_nan=False)
 Path("/out/worker-identity.json").write_text(json.dumps(dict(runtime=request["runtime"],network_isolated=True,
   environment_cleared=True,source_root_readonly=True,model_files_readonly=True,draft_hashes=request["draft_hashes"])))
def run(manifest_path,output,*,resource_profile=PROFILE,trusted=False):
 require(trusted is True,"Explicit trusted-local source")
 from hecate_python_env import VENV,WORK
 from process_resource_limits import limits
 from platform_config import configuration
 from workspace_paths import work_root
 from pytorch_weight_package import read,binding
 from capture_adapter import check_record
 import trusted_pytorch_import as legacy
 spec,paths=prepare_spec(manifest_path);cap=resolve(resource_profile)
 require(spec["format"]=="poseidon-trusted-pytorch-v21","Staging base format")
 arrays,weights=read(Path(manifest_path).parent,spec["weights"],resource_profile=resource_profile)
 frozen=source_identity();draft=draft_identity()
 output=Path(output)
 require(not output.exists()and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh isolated source output")
 output.mkdir(mode=0o700)
 request=dict(manifest=spec,resource_profile=resource_profile,runtime=legacy.runtime_identity(),source_hashes=frozen,draft_hashes=draft)
 req=output/"request.json";req.write_text(json.dumps(request,allow_nan=False))
 (output/"weights").mkdir()
 for name,raw in weights.items():(output/"weights"/name).write_bytes(raw)
 library=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
 require(all(Path(p).is_relative_to("/nix/store")for p in library.split(":")),"Pinned source libraries")
 cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
  "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
  "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu","--proc","/proc","--dev","/dev",
  "--tmpfs","/tmp","--dir","/app","--dir","/model","--ro-bind",str(req),"/request.json","--bind",str(output),"/out"]
 for name,path in implementation_paths().items():cmd+=["--ro-bind",str(path),"/app/"+name]
 for name in DRAFT:cmd+=["--ro-bind",str(HERE/name),"/app/"+name]
 for name,path in paths.items():cmd+=["--ro-bind",str(path),"/model/"+name]
 for name in weights:cmd+=["--ro-bind",str(output/"weights"/name),"/model/"+name]
 env=dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",LD_LIBRARY_PATH=library,
  HOME="/nonexistent",POSEIDON_PLATFORM=configuration()["id"],POSEIDON_WORK_ROOT=str(work_root()),
  OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8")
 for k,v in env.items():cmd+=["--setenv",k,v]
 cmd+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/sandbox.py","--worker"]
 started=time.monotonic()
 with(output/"capture.log").open("x")as log:
  proc=subprocess.Popen(cmd,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits,start_new_session=True)
  try:proc.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise ValueError("Staging source capture timeout")
 require(proc.returncode==0,"Staging source worker failed; inspect capture.log")
 require(source_identity()==frozen and draft_identity()==draft,"Capture implementation changed")
 record=json.loads(regular_bytes(output/"capture.json",cap.json_bytes))
 require(record["source_hashes"]==frozen and record["draft_hashes"]==draft,"Source worker identity")
 internal,_=read(output/"internal-weight-snapshot",record["internal_manifest"]["weights"],resource_profile=resource_profile)
 check_record(record,spec,arrays,internal,expected_record_binding=record["binding"],resource_profile=resource_profile)
 for name,path in paths.items():require(hashlib.sha256(regular_bytes(path,65536)).hexdigest()==spec["files"][name],"Source changed during capture")
 for name,raw in weights.items():require(regular_bytes(Path(manifest_path).parent/name,len(raw))==raw,"Source weights changed during capture")
 receipt=dict(format="poseidon-public-tensor-isolated-capture-receipt-v1",status="captured",seconds=time.monotonic()-started,
  source_capture=True,independent_reference=True,probes=16,original_manifest_sha256=digest(spec),capture_binding=record["binding"],
  source_hashes=frozen,draft_hashes=draft,resource_profile=resource_profile,production_activation=False,
  provider_calls=0,compiler_calls=0,encrypted_calls=0,network_configuration_changes=0)
 receipt["binding"]=digest(receipt)
 with(output/"receipt.json").open("x")as f:json.dump(receipt,f,allow_nan=False)
 return receipt
if __name__=="__main__":
 require(sys.argv[1:]==["--worker"],"Internal isolated worker only");worker()
