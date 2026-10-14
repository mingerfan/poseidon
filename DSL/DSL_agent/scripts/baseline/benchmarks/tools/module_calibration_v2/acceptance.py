"""Original nn.Module source -> v2 calibration -> profile freeze -> native capture."""
from pathlib import Path
import sys,shlex,json,time,hashlib,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=1200,environment="poseidon-cpu"))
import numpy as np,torch
from source_preparation_service import SourcePreparation,MODULE_REQUEST
from source_calibration_contract import FX_MODULE_FORMAT
from source_calibration import implementation
from source_calibration_signature import FORMAT as SIGNATURE
from pytorch_weight_package import export_state
from pytorch_model_intake import run as capture
from application_backend import PoseidonBackend
from test_model_output_contract import tensor
current=q.read(HERE/"controls-r3.json");q.bound(current)
q.require(current["failed"]==current["errors"]==current["skipped"]==0,"module_controls")
sdk=PoseidonBackend().identity_manifest();frozen=implementation()
OUT=q.RESULTS/"agent-module-calibration-v2-r1655";q.require(not OUT.exists(),"new_campaign")
baseline=live.measure(q.RESULTS)
q.require(baseline+256*1024**2<=48*1024**3and shutil.disk_usage(q.RESULTS).free>=4*1024**3+256*1024**2,"storage")
OUT.mkdir(mode=0o700)
cases=[
 ("gelu_module_base",(2,),"torch.nn.GELU","return self.activation(x)","gelu","activation"),
 ("gelu_module_affine",(2,2),"torch.nn.GELU","return self.activation(x*.25+.1)","gelu","activation"),
 ("gelu_module_residual",(1,4),"torch.nn.GELU","return x+self.activation(x*.125)","gelu","activation"),
 ("sigmoid_module_base",(2,),"torch.nn.Sigmoid","return self.activation(x)","sigmoid","activation"),
 ("sigmoid_module_square",(2,2),"torch.nn.Sigmoid","return self.activation(x*x*.125)","sigmoid","activation"),
 ("sigmoid_module_residual",(1,4),"torch.nn.Sigmoid","return x+self.activation(x*.25)","sigmoid","activation"),
 ("nested_gelu_base",(2,),None,"return F.gelu(x)","gelu","block"),
 ("nested_gelu_affine",(2,2),None,"return F.gelu(x*.25+.1)","gelu","block"),
 ("nested_gelu_residual",(1,4),None,"return x+F.gelu(x*.125)","gelu","block")]
profiles={"gelu":dict(method="polynomial",basis="power",domain=[-4.,4.],coefficients=[0.,.5,.25]),
 "sigmoid":dict(method="polynomial",basis="power",domain=[-4.,4.],coefficients=[.5,.25])}
scope=dict(format="poseidon-module-calibration-v2-scope",authorization="human_current_plan_all_tests_approved",
 cases=[dict(id=n,shape=list(shape),function=fn,path=path)for n,shape,kind,body,fn,path in cases],
 sdk=sdk["manifest_sha256"],implementation=frozen,source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 max_calibration_workers=18,max_capture_workers=9,api_calls=0,compiler_calls=0,encrypted_calls=0,
 wall_seconds=1200,new_bytes=256*1024**2,total_bytes=48*1024**3,reserve_bytes=4*1024**3)
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
q.write(ROOT/"docs/baseline/agent-module-calibration-offline-scope-r1655.json",scope)
start=time.monotonic();rows=[]
def guard():
 q.require(time.monotonic()-start<1200,"wall")
 q.require(implementation()==frozen and PoseidonBackend().identity_manifest()==sdk,"frozen_implementation")
 total=live.measure(q.RESULTS)
 q.require(total-baseline<=256*1024**2and total<=48*1024**3and shutil.disk_usage(q.RESULTS).free>=4*1024**3,"space")
for name,shape,kind,body,fn,path in cases:
 guard();folder=OUT/name;folder.mkdir();source=folder/"original-source";source.mkdir()
 if kind:
  code="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.activation="+kind+"()\n    def forward(self,x): "+body+"\ndef make(): return Model().eval()\n"
 else:
  code="import torch\nfrom torch.nn import functional as F\nclass Block(torch.nn.Module):\n    def forward(self,x): "+body+"\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.block=Block()\n    def forward(self,x): return self.block(x)\ndef make(): return Model().eval()\n"
 (source/"model.py").write_text(code)
 weights=export_state(torch.nn.Module(),source/"weight-stage")
 spec=dict(format="poseidon-trusted-pytorch-v38",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
 entry="model.make",inputs=[dict(name="x",shape=list(shape))],outputs=["out"],
 factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights=weights,
 approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
 call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
 q.write(source/"manifest.json",spec)
 original_files={str(p.relative_to(source)):q.sha(p)for p in source.rglob("*")if p.is_file()}
 samples=[dict(x=np.random.default_rng(2026101655+i).uniform(-.4,.4,shape).tolist())for i in range(3)]
 signature=dict(format=SIGNATURE,outputs=[dict(name="out",shape=list(shape))])
 base=dict(format=MODULE_REQUEST,action="inspect",manifest=dict(path=str(source/"manifest.json"),sha256=q.digest(spec)),
 source_signature=signature,samples=samples,input_dtypes={"x":"float64"},output_structure=tensor(),
 resource_profile="general-tensors-v1")
 try:
  inventory=SourcePreparation().run(base,folder/"inventory",trust_local_code=True)
  q.write(folder/"inventory-result.json",inventory);q.require(inventory["status"]=="described","inventory")
  graph=inventory["receipt"]["fx_graph"]
  node=next(n for n in graph["nodes"]if n["op"]=="call_module"and n["target"]=="activation")if kind else next(
   n for n in graph["nodes"]if n["target"]=="torch.nn.functional.gelu")
  q.require(node["source"]["module_paths"]==[path],"original_module_path")
  selection=dict(format=FX_MODULE_FORMAT,purpose="calibration",mode="observe",graph_sha256=graph["graph_sha256"],
   selectors=[dict(id="domain",node=node["name"],op=node["op"],target=node["target"],point="input",argument=0,
      projection={"kind":"tensor"},domain=None)])
  observed=SourcePreparation().run(dict(base,action="observe",selection=selection),folder/"observed",trust_local_code=True)
  q.write(folder/"observed-result.json",observed);q.require(observed["status"]=="observed","observation")
  config=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],
      modules=[dict(path=path,function=fn,profile=profiles[fn])])
  freezing={k:v for k,v in base.items()if k in("format","manifest","source_signature")}
  freezing.update(action="freeze",calibration_evidence=[dict(directory=str(folder/"observed"),report_binding=observed["receipt_binding"])],configuration=config)
  result=SourcePreparation().run(freezing,folder/"freeze",trust_local_code=True)
  q.write(folder/"freeze-result.json",result);q.require(result["status"]=="frozen","explicit_freeze")
  captured=capture(Path(result["configured_manifest"]),folder/"capture",resource_profile="general-tensors-v1",trusted=True)
  q.require(captured["status"]=="captured"and captured["reference_verified"]is True,"real_existing_capture")
  decomposition=q.read(folder/"capture/decomposition.json")
  recorded=decomposition["pytorch_capture"]["approximation_selection"]
  expected=result["receipt"]["approximation_selection"]
  q.require(recorded==expected,"exact_original_source_to_frontend_selection")
  q.require(original_files=={str(p.relative_to(source)):q.sha(p)for p in source.rglob("*")if p.is_file()},"original_source_unchanged")
  row=dict(case=name,status="passed",source_module_path=path,calibration_workers=2,capture_workers=1,calibration_samples=3,
     frozen_binding=result["receipt_binding"],captured_model_sha256=q.digest(q.read(folder/"capture/model.json")),
     exact_source_selection_matched=True,application_accuracy_certified=False)
 except Exception as error:
  import traceback
  (folder/"private-failure.log").write_text(traceback.format_exc())
  row=dict(case=name,status="failed",code="source_prepare_or_capture_rejected",error_type=type(error).__name__)
 q.write(folder/"case-report.json",row);rows.append(row);print(json.dumps(row),flush=True);guard()
report=dict(format="poseidon-module-calibration-v2-acceptance",scope_binding=scope["binding"],sdk=sdk["manifest_sha256"],
 cases=rows,planned=9,passed=sum(x["status"]=="passed"for x in rows),failed=sum(x["status"]!="passed"for x in rows),
 skipped=0,not_run=0,api_calls=0,compiler_calls=0,encrypted_calls=0,seconds=time.monotonic()-start,new_bytes=live.measure(q.RESULTS)-baseline)
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
