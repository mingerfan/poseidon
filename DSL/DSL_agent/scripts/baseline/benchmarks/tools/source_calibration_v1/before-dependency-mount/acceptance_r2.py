"""Three real isolated source forwards for calibration; signature-only ABI fixtures."""
from pathlib import Path
import sys,shlex,json,time,shutil,hashlib
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
import torch,numpy as np
from source_calibration import run,implementation
from source_calibration_contract import FORMAT as SELECTION
from pytorch_weight_package import export_state
from test_model_output_contract import model,tensor
from application_backend import PoseidonBackend
controls=q.read(HERE/"controls-r2.json");q.bound(controls)
q.require(controls["status"]=="passed","controls")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==controls["sdk"],"sdk")
for path,want in controls["source_files"].items():q.require(q.sha(path)==want,"tested_source")
OUT=q.RESULTS/"agent-source-calibration-acceptance-r1621"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-source-calibration-scope-v1",authorization="human_explicit_current_plan_test_approval",
 cases=["gelu_inputs_f32","layer_variance_f64","rms_variance_f64"],sdk=sdk["manifest_sha256"],
 implementation=implementation(),controls_binding=controls["binding"],api_calls=0,compiler_calls=0,encrypted_calls=0,
 native_forward_workers=3,abi_fixture_only=True,source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 budget=dict(wall_seconds=600,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 network_changed=False,backend_changed=False,parameters_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"scope-r2.json",scope);q.write(OUT/"scope.json",scope)
def selector(path,projection,domain=None,argument=0):
 return dict(format=SELECTION,purpose="calibration",selectors=[
  dict(id="domain_values",module=path,argument=argument,projection=projection,domain=domain)])
cases=[]
for name in scope["cases"]:
 folder=OUT/name;folder.mkdir(mode=0o700)
 source_folder=folder/"source";source_folder.mkdir(mode=0o700)
 if name=="gelu_inputs_f32":
  code="import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear=nn.Linear(2,2,dtype=torch.float32)\n        self.activation=nn.GELU()\n    def forward(self,x): return self.activation(self.linear(x))\ndef make(): return Model().eval()\n"
  state=torch.nn.Module();state.linear=torch.nn.Linear(2,2,dtype=torch.float32)
  with torch.no_grad():
   state.linear.weight.copy_(torch.tensor([[.5,-.25],[.75,.375]],dtype=torch.float32))
   state.linear.bias.copy_(torch.tensor([.1,-.2],dtype=torch.float32))
  shape=(2,);samples=[dict(x=[-.7,.3]),dict(x=[2.,-2.]),dict(x=[0.,0.])]
  dtype="float32";selection=selector("activation",dict(kind="tensor"),[-.1,.1])
  def expected(sample):
   x=torch.tensor(sample["x"],dtype=torch.float32);return state.linear(x).detach().numpy()
 elif name=="layer_variance_f64":
  code="import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.norm=nn.LayerNorm(2,eps=1e-4,dtype=torch.float64)\n    def forward(self,x): return self.norm(input=x)\ndef make(): return Model().eval()\n"
  state=torch.nn.Module();state.norm=torch.nn.LayerNorm(2,eps=1e-4,dtype=torch.float64)
  shape=(2,2);samples=[dict(x=[[.1,.6],[-.3,.2]]),dict(x=[[1.,-1.],[.5,.5]]),dict(x=[[0.,0.],[2.,1.]])]
  dtype="float64";selection=selector("norm",dict(kind="layer_norm_variance"),[1e-4,.1],argument="input")
  def expected(sample):
   x=np.asarray(sample["x"],dtype=np.float64)
   return np.mean((x-x.mean(axis=-1,keepdims=True))**2,axis=-1,keepdims=True)+1e-4
 else:
  code="import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.norm=nn.Identity()\n        self.eps=1e-5\n    def forward(self,x):\n        y=self.norm(x)\n        return y*torch.rsqrt((y*y).mean(dim=-1,keepdim=True)+self.eps)\ndef make(): return Model().eval()\n"
  state=torch.nn.Module()
  shape=(2,2);samples=[dict(x=[[.1,.2],[.3,.4]]),dict(x=[[1.,-1.],[.5,.5]]),dict(x=[[0.,0.],[2.,1.]])]
  dtype="float64";selection=selector("norm",dict(kind="rms_mean_square",epsilon=1e-5),[1e-5,.2])
  def expected(sample):
   x=np.asarray(sample["x"],dtype=np.float64);return np.mean(x*x,axis=-1,keepdims=True)+1e-5
 (source_folder/"model.py").write_text(code)
 weights=export_state(state,source_folder/"weight-stage")
 for item in weights.values():
  (source_folder/item["file"]).write_bytes((source_folder/"weight-stage"/item["file"]).read_bytes())
 graph=model(shape=shape,profile="general-tensors-v1")
 spec=dict(format="poseidon-trusted-pytorch-v38",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
  entry="model.make",inputs=graph["inputs"],outputs=["out"],
  factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights=weights,
  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
 manifest=source_folder/"manifest.json";q.write(manifest,spec)
 report=run(manifest,graph,samples,folder/"observation",expected_manifest_sha256=q.digest(spec),
  selection=selection,expected_selection_sha256=q.digest(selection),input_dtypes={"x":dtype},
  output_structure=tensor(),trusted=True)
 obs=report["observations"]["domain_values"]
 arrays=[np.asarray(expected(sample),dtype=np.float64)for sample in samples]
 values=np.concatenate([array.reshape(-1)for array in arrays])
 q.require(obs["values"]==values.size and abs(obs["minimum"]-float(values.min()))<=1e-15 and
  abs(obs["maximum"]-float(values.max()))<=1e-15,"independent_observed_interval")
 q.require(obs["samples"]==3 and obs["domain_status"]=="outside"and report["approximation_modified"]is False and
  report["reference_modified"]is False and report["native_dtype_preserved"]is True,"observe_not_fit_or_change")
 q.require(implementation()==scope["implementation"]and PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 row=dict(case=name,status="passed",samples=3,calls=obs["calls"],values=obs["values"],dtype=dtype,
  domain_status=obs["domain_status"],observer_output_equal_to_original=True,report_binding=report["binding"],
  api_calls=0,compiler_calls=0,encrypted_calls=0)
 cases.append(row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=scope["budget"]["new_bytes"],"artifact_budget")
report=dict(format="poseidon-source-calibration-acceptance-v1",passed=3,failed=0,skipped=0,cases=cases,
 native_forward_workers=3,samples=9,api_calls=0,compiler_calls=0,encrypted_calls=0,sdk=scope["sdk"],
 scope_binding=scope["binding"],abi_fixture_only=True,agent_input_semantic_equivalence_not_assessed=True,
 approximation_not_fitted=True,source_reference_not_modified=True,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance-r2.json",report);q.write(OUT/"report.json",report)
