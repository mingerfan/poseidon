"""Three ordinary PyTorch gather compositions; independent captured references."""
from pathlib import Path
import sys,shlex,json,time,hashlib,shutil
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);TOOLS=BASE/"benchmarks/tools"
sys.path[:0]=[str(BASE),str(TOOLS/"static_broadcast_live_v1"),str(TOOLS/"gather_capability_backend_v1")]
import run as live,qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=900,environment="poseidon-cpu"))
import torch,numpy as np
from platform_config import require_python_packages
from application_backend import PoseidonBackend
from pytorch_weight_package import export_state
from pytorch_model_intake import run as capture
from candidate_bundle import load_model_provenance
require_python_packages(torch,np)
identity=PoseidonBackend().identity_manifest()
q.require(identity["manifest_sha256"]=="605f3b5e330454bfe8ff6b146d54c878e570d1500264f1f9ebd36dd189227a1b","current_source_identity")
q.write(HERE/"identity.json",identity)
parent=q.RESULTS/"agent-gather-composition-source-r1583"
q.require(not parent.exists(),"fresh_capture_parent");parent.mkdir(mode=0o700)
cases=[
("gather_residual",[dict(name="x",shape=[5])],["out"],"""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([4,0,4,2],dtype=torch.int64))
        self.linear=torch.nn.Linear(4,4)
    def forward(self,x):
        selected=torch.gather(x,0,self.indices)
        return self.linear(selected)+selected
def make():
    return Model().double().eval()
"""),
("projected_gather_fusion",[dict(name="x",shape=[2,3]),dict(name="y",shape=[2,3])],["out"],"""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([[3,0,3],[1,2,0]],dtype=torch.int64))
        self.linear=torch.nn.Linear(3,4)
    def forward(self,x,y):
        projected=self.linear(x)
        return projected.gather(dim=1,index=self.indices)+y
def make():
    return Model().double().eval()
"""),
("permuted_gather_outputs",[dict(name="x",shape=[2,3,2])],["selected","row_mean"],"""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([[[1,0]],[[0,1]],[[1,1]]],dtype=torch.int64))
    def forward(self,x):
        viewed=x.transpose(0,1)
        selected=torch.gather(viewed,-2,self.indices)
        return {"selected":selected,"row_mean":selected.mean(dim=-1)}
def make():
    return Model().double().eval()
""")]
prepared=[];source_files={};started=time.monotonic()
for ordinal,(name,inputs,outputs,code)in enumerate(cases):
 namespace={};exec(compile(code,"fixed_developer_fixture","exec"),namespace)
 module=namespace["make"]();rng=np.random.default_rng(2026101300+ordinal)
 with torch.no_grad():
  for parameter in module.parameters():
   parameter.copy_(torch.from_numpy(rng.uniform(-.3,.3,tuple(parameter.shape)).astype(np.float64)))
 folder=HERE/"source-cases"/name;folder.mkdir(parents=True,mode=0o700)
 weights=export_state(module,folder,resource_profile="general-tensors-v1")
 (folder/"model.py").write_text(code)
 manifest=dict(format="poseidon-trusted-pytorch-v37",entry="model.make",
  files={"model.py":hashlib.sha256(code.encode()).hexdigest()},inputs=inputs,outputs=outputs,
  dtype_policy="strict-float64-v1",factory=dict(args=[],kwargs={}),weights=weights,
  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name=x["name"])for x in inputs],kwargs={}))
 q.write(folder/"manifest.json",manifest)
 for path in folder.iterdir():source_files[str(path.relative_to(ROOT))]=q.sha(path)
 prepared.append(dict(id=name,manifest=str(folder/"manifest.json"),inputs=inputs,outputs=outputs))
scope=dict(format="poseidon-gather-composition-source-scope-v1",authorization="human_explicit_current_plan_test_approval",
 sdk=identity["manifest_sha256"],cases=prepared,source_files=source_files,script_sha256=q.sha(Path(__file__)),
 budget=dict(wall_seconds=900,source_captures=3,max_reference_groups=48,api_calls=0,compiler_calls=0,encrypted_calls=0,
  new_artifact_bytes=256*1024**2,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),reference_tolerance=dict(atol=1e-12,rtol=1e-12),
 backend_changed=False,parameters_changed=False,network_changed=False,installations=0,downloads=0)
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope)
q.require(live.measure(q.RESULTS)+scope["budget"]["new_artifact_bytes"]<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+scope["budget"]["new_artifact_bytes"],"source_storage")
rows=[]
for c in prepared:
 try:
  q.require(time.monotonic()-started<850,"source_time")
  for rel,want in source_files.items():q.require(q.sha(ROOT/rel)==want,"fixture_unchanged")
  dest=parent/c["id"];capture(c["manifest"],dest,resource_profile="general-tensors-v1",trusted=True)
  model=q.read(dest/"model.json");pv=load_model_provenance(dest,model,max_bytes=4*1024**2)
  cp=pv["reference_report"]["capture_metadata"]["public_value_expressions"]
  q.require(cp["source_intake_version"]=="poseidon-trusted-pytorch-v37"and
   cp["logical_view_reference"]["groups"]==16,"source_reference_scope")
  q.require(len(cp["tensor_gather_flow"]["entries"])==1,"actual_gather_normalization")
  row=dict(case=c["id"],status="passed",capture=str(dest),model_sha256=q.digest(model),manifest=c["manifest"],
   source_version="poseidon-trusted-pytorch-v37",reference_groups=16,source_provenance_loaded=True)
 except Exception as e:
  import traceback
  (HERE/(c["id"]+"-private-failure.txt")).write_text(traceback.format_exc())
  row=dict(case=c["id"],status="failed",error_type=type(e).__name__,diagnostic=str(e)[:260])
 rows.append(row);print(json.dumps(row),flush=True)
 q.require(PoseidonBackend().identity_manifest()==identity,"source_frozen_sdk")
 q.require(live.measure(parent)<=scope["budget"]["new_artifact_bytes"],"source_artifact_budget")
record=dict(format="poseidon-gather-composition-source-acceptance-v1",scope_binding=scope["binding"],sdk=scope["sdk"],
 cases=rows,passed=sum(x["status"]=="passed"for x in rows),failed=sum(x["status"]!="passed"for x in rows),skipped=0,
 reference_groups=sum(x.get("reference_groups",0)for x in rows),provider_calls=0,compiler_calls=0,encrypted_calls=0,
 source_files=source_files,seconds=time.monotonic()-started)
record["binding"]=q.digest(record);q.write(HERE/"report.json",record)
raise SystemExit(0 if record["failed"]==0 else 1)
