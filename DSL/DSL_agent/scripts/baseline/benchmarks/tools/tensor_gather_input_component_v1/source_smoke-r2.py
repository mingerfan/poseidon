"""Three isolated v37 source captures plus retained v36 compatibility."""
import sys,json,shlex,hashlib,time,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path[:0]=[str(HERE),str(BASE)]
if sys.argv[1:]!=["--inside"]:
 sys.path.insert(0,str(BASE/"benchmarks/tools/static_broadcast_live_v1"))
 import run as live
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
import torch,numpy as np
from platform_config import require_python_packages
require_python_packages(torch,np)
from pytorch_weight_package import export_state
from pytorch_model_intake import run
from candidate_bundle import load_model_provenance
from benchmark_graph import digest
from pytorch_ordered_contract import source_identity
code="""import torch
class Model(torch.nn.Module):
    def __init__(self,index,axis,method):
        super().__init__()
        self.register_buffer("indices",torch.tensor(index,dtype=torch.int64))
        self.axis=axis
        self.method=method
    def forward(self,x):
        if self.method:
            return x.gather(dim=self.axis,index=self.indices)
        return torch.gather(x,self.axis,self.indices)
def make(index,axis,method):
    return Model(index,axis,method)
"""
identity=source_identity();rows=[];started=time.monotonic()
cases=[("vector",[5],[4,0,4,2],0,False),("rows",[2,4],[[3,0,3],[1,2,0]],1,True),
       ("middle",[2,3,2],[[[2],[0]],[[1],[2]]],-2,False)]
for name,shape,index,axis,method in cases:
 folder=HERE/"source-cases-r2"/name;folder.mkdir(parents=True)
 module=__import__("test_fx").Gather(index,axis,method)
 weights=export_state(module,folder,resource_profile="general-tensors-v1")
 (folder/"model.py").write_text(code)
 spec=dict(format="poseidon-trusted-pytorch-v37",entry="model.make",
 files={"model.py":hashlib.sha256(code.encode()).hexdigest()},inputs=[dict(name="x",shape=shape)],
 outputs=["out"],dtype_policy="strict-float64-v1",factory=dict(args=[],kwargs=dict(index=index,axis=axis,method=method)),
 approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
 weights=weights,call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
 path=folder/"manifest.json";path.write_text(json.dumps(spec))
 try:
  output=run(path,folder/"capture",resource_profile="general-tensors-v1",trusted=True)
  model=json.loads((folder/"capture/model.json").read_text())
  pv=load_model_provenance(folder/"capture",model,max_bytes=4*1024**2)
  report=pv["reference_report"]
  row=dict(case=name,status="passed",model_sha256=digest(model),
           capture=output,source_intake_version=spec["format"],provenance_loaded=True,
           reference_groups=report["capture_metadata"]["public_value_expressions"]["logical_view_reference"]["groups"])
 except BaseException as e:
  row=dict(case=name,status="failed",error_type=type(e).__name__,diagnostic=str(e)[:260])
 rows.append(row);print(json.dumps(row),flush=True)
 assert source_identity()==identity
record=dict(format="poseidon-tensor-gather-isolated-source-v1",cases=rows,
 passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),
 source_identity=identity,seconds=time.monotonic()-started,api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=digest(record)
with(HERE/"source-smoke-r2.json").open("x")as f:json.dump(record,f,sort_keys=True,indent=2)
raise SystemExit(0 if record["failed"]==0 else 1)
