"""Six original v38 PyTorch models, isolated references and provenance reload."""
from pathlib import Path
import sys,shlex,json,time,copy,hashlib,shutil
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];TOOLS=BASE/"benchmarks/tools";ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(BASE),str(TOOLS/"static_broadcast_live_v1"),str(TOOLS/"gather_composition_live_v1")]
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
from pytorch_model_intake import run as capture
from pytorch_weight_package import export_state
from candidate_bundle import load_model_provenance
require_python_packages(torch,np)
controls=q.read(HERE/"source38-controls-r2.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==156,"source_controls")
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-before-source.json",identity)
code="""import torch
class Model(torch.nn.Module):
    def __init__(self,dims,tensor_dimensions,mean_left):
        super().__init__()
        if tensor_dimensions:
            self.register_buffer("dimensions",torch.tensor(dims,dtype=torch.int64))
        else:
            self.dimensions=dims
        self.mean_left=mean_left
    def forward(self,left,right):
        if self.mean_left:
            left=left.mean()
        return torch.tensordot(left,right,dims=self.dimensions)
def make(dims,tensor_dimensions,mean_left):
    return Model(dims,tensor_dimensions,mean_left).double().eval()
"""
cases=[
("dot_scalar",[3],[3],1,False,False),
("matrix",[2,3],[3,2],1,False,False),
("negative_axes",[2,3,2],[2,3],[[-1,1],[0,1]],True,False),
("outer",[2],[2],0,True,False),
("singleton_broadcast",[2,1],[3,2],1,False,False),
("scalar_intermediate",[3],[2],0,False,True)]
namespace={};exec(compile(code,"fixed_developer_fixture","exec"),namespace)
parent=q.RESULTS/"agent-tensor-contraction-source-r1594";q.require(not parent.exists(),"fresh_capture_parent");parent.mkdir(mode=0o700)
prepared=[];files={}
for name,ash,bsh,dims,tensor,mean_left in cases:
    folder=HERE/"source-cases"/name;folder.mkdir(mode=0o700,parents=True)
    module=namespace["make"](dims,tensor,mean_left)
    weights=export_state(module,folder,resource_profile="general-tensors-v1")
    (folder/"model.py").write_text(code)
    spec=dict(format="poseidon-trusted-pytorch-v38",entry="model.make",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
        inputs=[dict(name="left",shape=ash),dict(name="right",shape=bsh)],outputs=["out"],
        dtype_policy="strict-float64-v1",factory=dict(args=[],kwargs=dict(dims=dims,tensor_dimensions=tensor,mean_left=mean_left)),
        approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
        weights=weights,call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name=n)for n in ("left","right")],kwargs={}))
    q.write(folder/"manifest.json",spec)
    for path in folder.iterdir():files[str(path.relative_to(ROOT))]=q.sha(path)
    prepared.append(dict(id=name,manifest=str(folder/"manifest.json")))
scope=dict(format="poseidon-contraction-source-capture-scope-v1",sdk=identity["manifest_sha256"],
    cases=prepared,source_files=files,script_sha256=q.sha(Path(__file__)),
    authorization="human_explicit_current_plan_test_approval",budget=dict(wall_seconds=900,source_captures=6,reference_groups=96,
        provider_calls=0,compiler_calls=0,encrypted_calls=0,new_bytes=256*1024**2,global_bytes=48*1024**3,reserve_bytes=4*1024**3))
scope["binding"]=q.digest(scope);q.write(HERE/"source-scope.json",scope)
q.require(live.measure(q.RESULTS)+scope["budget"]["new_bytes"]<=48*1024**3 and
    shutil.disk_usage(q.RESULTS).free>=4*1024**3+scope["budget"]["new_bytes"],"source_storage")
started=time.monotonic();rows=[]
for c in prepared:
    try:
        q.require(time.monotonic()-started<850,"source_time")
        for relative,want in files.items():q.require(q.sha(ROOT/relative)==want,"frozen_source")
        folder=parent/c["id"];capture(c["manifest"],folder,resource_profile="general-tensors-v1",trusted=True)
        model=q.read(folder/"model.json");pv=load_model_provenance(folder,model,max_bytes=4*1024**2)
        cp=pv["reference_report"]["capture_metadata"]["public_value_expressions"]
        q.require(cp["source_intake_version"]=="poseidon-trusted-pytorch-v38"and
            cp["logical_view_reference"]["groups"]==16 and len(cp["tensor_contraction_flow"]["entries"])==1,"complete_contraction_source")
        row=dict(case=c["id"],status="passed",source_version=cp["source_intake_version"],reference_groups=16,
            provenance_reloaded=True,model_sha256=q.digest(model),capture=str(folder),manifest=c["manifest"])
    except Exception as error:
        import traceback
        (HERE/(c["id"]+"-source-private-failure.txt")).write_text(traceback.format_exc())
        row=dict(case=c["id"],status="failed",error_type=type(error).__name__,diagnostic=str(error)[:300])
    rows.append(row);print(json.dumps(row),flush=True)
    q.require(PoseidonBackend().identity_manifest()==identity,"frozen_source_sdk")
    q.require(live.measure(parent)<=scope["budget"]["new_bytes"],"source_artifact_budget")
record=dict(format="poseidon-contraction-source-acceptance-v1",scope_binding=scope["binding"],sdk=scope["sdk"],cases=rows,
    passed=sum(x["status"]=="passed"for x in rows),failed=sum(x["status"]!="passed"for x in rows),skipped=0,
    reference_groups=sum(x.get("reference_groups",0)for x in rows),seconds=time.monotonic()-started,
    api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=q.digest(record);q.write(HERE/"source-acceptance.json",record)
raise SystemExit(0 if record["failed"]==0 else 1)
