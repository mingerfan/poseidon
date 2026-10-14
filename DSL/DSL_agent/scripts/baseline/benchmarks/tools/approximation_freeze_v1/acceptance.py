"""Original source -> calibration -> explicit freeze -> existing trusted frontend."""
from pathlib import Path
import sys,shlex,json,time,shutil,hashlib
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=1200,environment="poseidon-cpu"))
import numpy as np,torch
from source_approximation_preparation import inspect_source,observe_source,freeze,PreparationError
from source_calibration import implementation,check_report
from source_calibration_signature import FORMAT as SIGNATURE
from source_calibration_contract import FX_FORMAT,CalibrationError
from pytorch_weight_package import export_state
from pytorch_model_intake import run as capture
from application_backend import PoseidonBackend
from test_model_output_contract import tensor
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==105,"controls")
sdk=PoseidonBackend().identity_manifest();frozen=implementation()
q.require(sdk["manifest_sha256"]==controls["sdk"],"sdk")
OUT=q.RESULTS/"agent-pre-generation-approximation-freeze-r1641"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+256*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+256*1024**2,"storage")
OUT.mkdir(mode=0o700)
cases=[
 ("gelu_base",(2,),"return F.gelu(x)","torch.nn.functional.gelu","gelu"),
 ("gelu_affine",(2,2),"return F.gelu(x*.25+.1)","torch.nn.functional.gelu","gelu"),
 ("gelu_residual",(1,4),"return x+F.gelu(x*.125)","torch.nn.functional.gelu","gelu"),
 ("sigmoid_base",(2,),"return torch.sigmoid(x)","torch.sigmoid","sigmoid"),
 ("sigmoid_square",(2,2),"return torch.sigmoid(x*x*.125)","torch.sigmoid","sigmoid"),
 ("sigmoid_residual",(1,4),"return x+torch.sigmoid(x*.25)","torch.sigmoid","sigmoid"),
 ("rsqrt_base",(2,),"return x*torch.rsqrt(x*x*.125+.5)","torch.rsqrt","rsqrt"),
 ("rsqrt_mean",(2,2),"return x*torch.rsqrt((x*x).mean(dim=-1,keepdim=True)*.125+.5)","torch.rsqrt","rsqrt"),
 ("rsqrt_sum",(1,4),"return x*torch.rsqrt((x*x).sum(dim=-1,keepdim=True)*.03125+.5)","torch.rsqrt","rsqrt")]
profiles={"gelu":dict(method="polynomial",basis="power",domain=[-4.,4.],coefficients=[0.,.5,.25]),
 "sigmoid":dict(method="polynomial",basis="power",domain=[-4.,4.],coefficients=[.5,.25]),
 "rsqrt":dict(method="newton",domain=[.25,4.],iterations=2)}
scope=dict(format="poseidon-pre-generation-approximation-freeze-scope-v1",
 authorization="human_current_plan_all_tests_approved",cases=[dict(id=n,shape=list(shape),function=fn)
  for n,shape,body,target,fn in cases],profiles=profiles,source_implementation=frozen,sdk=sdk["manifest_sha256"],
 controls_binding=controls["binding"],max_native_calibration_workers=18,max_capture_workers=9,
 api_calls=0,compiler_calls=0,encrypted_calls=0,
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve()),
  str(B/"source_approximation_preparation.py"):q.sha(B/"source_approximation_preparation.py")},
 budget=dict(wall_seconds=1200,new_bytes=256*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3))
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
start=time.monotonic();rows=[]
def hashes(root):return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
for name,shape,body,target,fn in cases:
 q.require(time.monotonic()-start<1200,"wall_budget")
 folder=OUT/name;folder.mkdir(mode=0o700);source=folder/"original-source";source.mkdir(mode=0o700)
 code="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def forward(self,x): "+body+"\ndef make(): return Model().eval()\n"
 (source/"model.py").write_text(code);weights=export_state(torch.nn.Module(),source/"weight-stage")
 spec=dict(format="poseidon-trusted-pytorch-v38",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
  entry="model.make",inputs=[dict(name="x",shape=list(shape))],outputs=["out"],
  factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights=weights,
  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
 q.write(source/"manifest.json",spec);original_hashes=hashes(source)
 signature=dict(format=SIGNATURE,outputs=[dict(name="out",shape=list(shape))])
 rng=np.random.default_rng(2026101441);samples=[dict(x=rng.uniform(-.4,.4,shape).tolist())for _ in range(3)]
 q.write(folder/"inputs.json",samples)
 args=dict(expected_manifest_sha256=q.digest(spec),source_signature=signature,
  input_dtypes={"x":"float64"},output_structure=tensor(),resource_profile="general-tensors-v1",trust_local_code=True)
 try:
  inventory=inspect_source(source/"manifest.json",samples,folder/"inventory",**args)
  node=next(n for n in inventory["fx_graph"]["nodes"]if n["target"]==target)
  selection=dict(format=FX_FORMAT,purpose="calibration",mode="observe",
   graph_sha256=inventory["fx_graph"]["graph_sha256"],selectors=[dict(id="input_domain",node=node["name"],
      op=node["op"],target=target,point="input",argument=0,projection=dict(kind="tensor"),domain=None)])
  report=observe_source(source/"manifest.json",samples,folder/"observed",selection=selection,
       expected_selection_sha256=q.digest(selection),**args)
  config=dict(format="poseidon-pytorch-approximations-v1",defaults={fn:profiles[fn]},modules=[],nodes=[])
  result=freeze(source/"manifest.json",folder/"freeze",expected_manifest_sha256=q.digest(spec),
       source_signature=signature,calibration_evidence=[
        dict(directory=str(folder/"observed"),report_binding=report["binding"])],
       configuration=config,trust_local_code=True)
  q.require(result["model_graph_constructed"]is False and result["profile_fitted"]is False and
   result["original_application_accuracy_certified"]is False,"freeze_not_accuracy_or_graph")
  captured=capture(Path(result["configured_manifest"]),folder/"capture",
        resource_profile="general-tensors-v1",trusted=True)
  q.require(captured["status"]=="captured"and captured["reference_verified"]is True,"actual_existing_frontend_capture")
  decomposition=q.read(folder/"capture/decomposition.json")
  model=q.read(folder/"capture/model.json")
  selection_record=decomposition["pytorch_capture"]["approximation_selection"]
  q.require(selection_record["configuration_sha256"]==q.digest(config)and
   all(row["profile_sha256"]==q.digest(profiles[fn])for row in selection_record["selections"]),
   "same_configuration_in_real_frontend")
  q.require(hashes(source)==original_hashes and implementation()==frozen and
   PoseidonBackend().identity_manifest()==sdk,"original_source_and_sdk_unchanged")
  row=dict(case=name,status="passed",calibration_samples=3,calibration_workers=2,capture_workers=1,
   source_function=fn,freeze_binding=result["binding"],captured_model_sha256=q.digest(model),
   original_source_unchanged=True,model_existed_before_calibration=False,
   actual_frontend_selection_matches=True,application_accuracy_certified=False)
 except (PreparationError,CalibrationError)as error:
  row=dict(case=name,status="failed",code=error.code)
 except (ValueError,KeyError,RuntimeError,StopIteration)as error:
  import traceback
  (folder/"private-failure.log").write_text(traceback.format_exc())
  row=dict(case=name,status="failed",code="capture_or_selection_rejected",error_type=type(error).__name__)
 rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=256*1024**2,"space")
report=dict(format="poseidon-pre-generation-freeze-acceptance-v1",scope_binding=scope["binding"],
 cases=rows,passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),skipped=0,
 sdk=sdk["manifest_sha256"],controls_binding=controls["binding"],api_calls=0,compiler_calls=0,encrypted_calls=0,
 seconds=time.monotonic()-start,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
