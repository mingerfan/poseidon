"""Bounded original functional FX observations; no generation or FHE."""
from pathlib import Path
import sys,shlex,json,time,shutil,hashlib
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=1800,environment="poseidon-cpu"))
import numpy as np,torch
from source_calibration import run,implementation
from source_calibration_contract import FX_FORMAT,CalibrationError
from task_source_calibration import TaskSourceCalibration
from task_calibration_dataset import TaskCalibrationDataset,DECLARATION,identity as collector_identity
from assessment_array_dataset import export_dataset
from pytorch_weight_package import export_state
from test_model_output_contract import model,tensor
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_backend import PoseidonBackend
controls=q.read(HERE/"controls-r3.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==62,"controls")
for path,want in controls["source_files"].items():q.require(q.sha(Path(path))==want,"tested_sources")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==controls["sdk"],"sdk")
OUT=q.RESULTS/"agent-fx-calibration-acceptance-r1630"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+256*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+256*1024**2,"storage")
OUT.mkdir(mode=0o700)
bodies=[
 ("gelu_affine",(2,),"float32","z=x*.5+.25\n        return F.gelu(z)","torch.nn.functional.gelu","input",0),
 ("gelu_residual",(2,2),"float64","z=x*x+x*.25+.1\n        return F.gelu(z)+x","torch.nn.functional.gelu","input",0),
 ("gelu_view",(1,4),"float64","z=x.reshape(2,2).transpose(0,1)-.2\n        return F.gelu(z).transpose(0,1).reshape(1,4)","torch.nn.functional.gelu","input",0),
 ("rms_base",(2,),"float64","d=(x*x).mean(dim=-1,keepdim=True)+1e-4\n        return x*torch.rsqrt(d)","torch.rsqrt","input",0),
 ("rms_residual",(2,2),"float64","z=x+.125\n        d=(z*z).mean(dim=-1,keepdim=True)+.01\n        return x+z*torch.rsqrt(d)","torch.rsqrt","input",0),
 ("rms_transpose",(1,4),"float64","z=x.transpose(0,1)\n        d=(z*z).mean(dim=0,keepdim=True)+.001\n        return (z*torch.rsqrt(d)).transpose(0,1)","torch.rsqrt","input",0),
 ("inverse_rms",(2,),"float64","d=(x*x).mean(dim=-1,keepdim=True)+.01\n        return x*d.rsqrt()","rsqrt","output",None),
 ("inverse_variance",(2,2),"float64","d=x.var(dim=-1,unbiased=False,keepdim=True)+.01\n        return (x-x.mean(dim=-1,keepdim=True))*d.rsqrt()","rsqrt","output",None),
 ("inverse_sum",(1,4),"float64","d=(x*x).sum(dim=-1,keepdim=True)/4+.001\n        return x*d.rsqrt()*2","rsqrt","output",None),
]
scope=dict(format="poseidon-fx-calibration-scope-v1",
 authorization="human_current_plan_all_tests_approved",
 standalone_cases=[dict(id=n,shape=list(shape),dtype=dtype,target=target,point=point)
  for n,shape,dtype,body,target,point,arg in bodies],
 task_cases=["dot_scalar","matrix","scalar_intermediate"],
 max_native_workers=27,api_calls=0,compiler_calls=0,encrypted_calls=0,
 sdk=sdk["manifest_sha256"],observer_implementation=implementation(),
 collector_identity=collector_identity(),controls_binding=controls["binding"],
 standalone_abi_fixture_only=True,agent_input_semantic_equivalence_not_assessed=True,
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 budget=dict(wall_seconds=1800,new_bytes=256*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 native_limits=dict(memory_bytes=4*1024**3,cpu_seconds=150))
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
start=time.monotonic();rows=[]
def frozen():
 q.require(time.monotonic()-start<1800,"wall_budget")
 q.require(implementation()==scope["observer_implementation"]and collector_identity()==scope["collector_identity"]and
  PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 q.require(live.measure(OUT)<=256*1024**2,"space")
def make_selection(graph,target,point,argument):
 nodes=[n for n in graph["nodes"]if n["target"]==target and n["op"]in("call_function","call_method")]
 q.require(len(nodes)==1,"unique_requested_node")
 node=nodes[0]
 return dict(format=FX_FORMAT,purpose="calibration",mode="observe",graph_sha256=graph["graph_sha256"],
  selectors=[dict(id="selected_value",node=node["name"],op=node["op"],target=target,point=point,
                 argument=argument,projection=dict(kind="tensor"),domain=[-.1,.1])])
def expected(name,x):
 if name=="gelu_affine":return x*.5+.25
 if name=="gelu_residual":return x*x+x*.25+.1
 if name=="gelu_view":return x.reshape(2,2).T-.2
 if name=="rms_base":return (x*x).mean(-1,keepdims=True)+1e-4
 if name=="rms_residual":
  z=x+.125;return (z*z).mean(-1,keepdims=True)+.01
 if name=="rms_transpose":
  z=x.T;return (z*z).mean(0,keepdims=True)+.001
 if name=="inverse_rms":return 1/np.sqrt((x*x).mean(-1,keepdims=True)+.01)
 if name=="inverse_variance":return 1/np.sqrt(x.var(-1,keepdims=True)+.01)
 if name=="inverse_sum":return 1/np.sqrt((x*x).sum(-1,keepdims=True)/4+.001)
 raise AssertionError("unknown acceptance id")
for name,shape,dtype,body,target,point,argument in bodies:
 frozen();folder=OUT/name;folder.mkdir(mode=0o700);source=folder/"source";source.mkdir(mode=0o700)
 code="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        "+body+"\ndef make(): return Model().eval()\n"
 (source/"model.py").write_text(code)
 weights=export_state(torch.nn.Module(),source/"weight-stage")
 graph=model(shape=shape,profile="general-tensors-v1")
 spec=dict(format="poseidon-trusted-pytorch-v38",files={"model.py":hashlib.sha256(code.encode()).hexdigest()},
  entry="model.make",inputs=graph["inputs"],outputs=["out"],factory=dict(args=[],kwargs={}),
  dtype_policy="convert-float64-v1",weights=weights,
  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
  call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
 q.write(source/"manifest.json",spec)
 rng=np.random.default_rng(2026101430)
 arrays=[rng.uniform(-.4,.4,shape).astype(dtype)for _ in range(3)]
 samples=[dict(x=x.tolist())for x in arrays]
 inventory=dict(format=FX_FORMAT,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])
 try:
  described=run(source/"manifest.json",graph,samples,folder/"inventory",
   expected_manifest_sha256=q.digest(spec),selection=inventory,expected_selection_sha256=q.digest(inventory),
   input_dtypes={"x":dtype},output_structure=tensor(),trusted=True)
  q.require(described["status"]=="described"and described["observations"]=={},"inventory_scope")
  selection=make_selection(described["fx_graph"],target,point,argument)
  report=run(source/"manifest.json",graph,samples,folder/"observation",
   expected_manifest_sha256=q.digest(spec),selection=selection,expected_selection_sha256=q.digest(selection),
   input_dtypes={"x":dtype},output_structure=tensor(),trusted=True)
  stats=report["observations"]["selected_value"]
  values=np.concatenate([np.asarray(expected(name,x),dtype=np.float64).reshape(-1)for x in arrays])
  for actual,want in ((stats["minimum"],float(values.min())),(stats["maximum"],float(values.max())),
                      (stats["mean"],float(values.mean())),(stats["std_population"],float(values.std()))):
   q.require(abs(actual-want)<=1e-12+1e-12*abs(want),"independent_observation_statistics")
  q.require(stats["values"]==values.size and report["fx_graph"]==described["fx_graph"],"count_and_graph")
  row=dict(case=name,status="passed",scope="standalone_observer_abi_only",samples=3,observed_values=int(values.size),
   dtype=dtype,point=point,target=target,graph_sha256=report["fx_graph"]["graph_sha256"],
   inventory_binding=described["binding"],observation_binding=report["binding"],native_workers=2)
 except CalibrationError as error:
  row=dict(case=name,status="unsupported"if error.code=="calibration_fx_trace_rejected"else "failed",
           code=error.code,scope="standalone_observer_abi_only")
 rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True)
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
def hashes(root):
 return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
for name in scope["task_cases"]:
 frozen();folder=OUT/name;folder.mkdir(mode=0o700)
 old=Path(prior["campaign"])/name;task=q.read(old/"task-id.json")["task_id"];store=old/"tasks"
 before=hashes(store/task);app=application(WorkerContext.capture(store));outcome=app.result(task)
 metadata=app._checked_task_metadata(task);request=metadata["request"]
 rng=np.random.default_rng(2026101431);dtype="float32"if name=="matrix"else"float64"
 arrays={item["name"]:rng.uniform(-.4,.4,[17]+item["shape"]).astype(dtype)
         for item in request["model"]["inputs"]}
 data=export_dataset(request,arrays,folder/"dataset")
 dtypes={i["name"]:dtype for i in request["model"]["inputs"]}
 first=[{n:a[i].copy()for n,a in arrays.items()}for i in range(3)]
 described=TaskSourceCalibration(app).describe_fx(task,first,folder/"inventory",
    expected_package_binding=outcome["package"]["binding"],input_dtypes=dtypes,trust_local_code=True)
 q.write(folder/"inventory-frame.json",described)
 if described["status"]!="described":
  row=dict(case=name,status="failed",scope="historical_true_agent_task",failure=described.get("failure"))
  rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True);continue
 selection=make_selection(described["calibration_report"]["fx_graph"],"torch.tensordot","input",0)
 declared=dict(format=DECLARATION,role="calibration",dataset_binding=data["binding"],
               source=dict(name="fixed-seed-fx-interface-acceptance",revision="2026101431"))
 args=dict(expected_dataset_binding=data["binding"],calibration_declaration=declared,
  expected_declaration_sha256=q.digest(declared),expected_package_binding=outcome["package"]["binding"],
  selection=selection,expected_selection_sha256=q.digest(selection),input_dtypes=dtypes,trust_local_code=True)
 collector=TaskCalibrationDataset(app,task,folder/"dataset/dataset.json",**args)
 q.write(folder/"plan.json",collector.plan)
 okay=True
 for index,batch in enumerate(collector.batches()):
  frame=TaskSourceCalibration(app).observe(task,batch["samples"],folder/("observation_"+str(index)),
   expected_package_binding=args["expected_package_binding"],selection=selection,
   expected_selection_sha256=q.digest(selection),input_dtypes=dtypes,trust_local_code=True)
  q.write(folder/("frame_"+str(index)+".json"),frame)
  if frame["status"]!="observed":okay=False;break
  collector.record(batch["id"],frame,expected_binding=frame["binding"])
  if index==0:
   partial=collector.summary();q.require(not partial["complete"],"partial")
   cache=collector.receipts;q.write(folder/"cached-receipts.json",cache)
   collector=TaskCalibrationDataset(app,task,folder/"dataset/dataset.json",cached_receipts=cache,**args)
   q.require(collector.summary()==partial,"cached_fx_receipt_resume")
 if not okay:
  row=dict(case=name,status="failed",scope="historical_true_agent_task",failure=frame.get("failure"))
 else:
  summary=collector.summary();q.write(folder/"summary.json",summary)
  q.require(summary["complete"]and summary["observed_samples"]==17,"complete")
  stats=summary["observations"]["selected_value"]
  left=request["model"]["inputs"][0]["name"]
  raw=arrays[left].astype(np.float64)
  expected_values=raw.mean(axis=tuple(range(1,raw.ndim)))if name=="scalar_intermediate"else raw.reshape(-1)
  q.require(abs(stats["minimum"]-float(expected_values.min()))<=1e-12 and
   abs(stats["maximum"]-float(expected_values.max()))<=1e-12,"real_task_source_domain")
  q.require(app.result(task)==outcome and hashes(store/task)==before,"historical_task_unchanged")
  row=dict(case=name,status="passed",scope="historical_true_agent_task",samples=17,
   inventory_samples=3,observed_values=stats["values"],native_workers=3,first_receipt_reused=True,
   summary_binding=summary["binding"],historical_program_unchanged=True)
 rows.append(row);q.write(folder/"result.json",row);print(json.dumps(row),flush=True)
reports=list(OUT.glob("*/inventory/calibration.json"))+list(OUT.glob("*/observation/calibration.json"))+list(OUT.glob("*/observation_*/calibration.json"))
failures=list(OUT.glob("*/inventory/calibration-failure.json"))+list(OUT.glob("*/observation/calibration-failure.json"))+list(OUT.glob("*/observation_*/calibration-failure.json"))
result=dict(format="poseidon-fx-calibration-acceptance-v1",scope_binding=scope["binding"],sdk=scope["sdk"],
 controls_binding=controls["binding"],passed=sum(r["status"]=="passed"for r in rows),
 failed=sum(r["status"]=="failed"for r in rows),unsupported=sum(r["status"]=="unsupported"for r in rows),skipped=0,
 successful_native_reports=len(reports),native_failure_reports=len(failures),
 cases=rows,api_calls=0,compiler_calls=0,encrypted_calls=0,seconds=time.monotonic()-start,
 bytes=live.measure(OUT),observer_implementation=scope["observer_implementation"])
result["binding"]=q.digest(result);q.write(HERE/"acceptance.json",result);q.write(OUT/"report.json",result)
print(json.dumps({k:v for k,v in result.items()if k not in("cases","observer_implementation")}),flush=True)
