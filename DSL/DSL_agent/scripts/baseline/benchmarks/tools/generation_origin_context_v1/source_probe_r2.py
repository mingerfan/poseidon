from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=360,environment="poseidon-cpu"))
import qualification_scope as q
from generation_model_origin import build,verify
D=q.RESULTS/"agent-layernorm-calibration-live-r1668"
plan=q.read(D/"live-plan.json");rows=[]
out=q.RESULTS/"agent-generation-origin-source-r1676";q.require(not out.exists(),"new_origin_probe");out.mkdir()
for case in plan["cases"]:
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 folder=out/case["id"];folder.mkdir()
 app=application(WorkerContext.capture(folder/"tasks"))
 source=case["source_manifest"]
 task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=source,trust_local_code=True),
    resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v20",public_weight_policy="descriptors-math-roles-v2",
    chunk_period=4,capacity_packing_version="capacity-public-tensors-v10"))
 status=app.prepare_source(task)
 q.require(status["status"]=="queued","fresh_source_admission")
 meta=app._checked_task_metadata(task)
 request=meta["request"];provenance=meta["model_provenance"]
 old=q.read(Path(case["request_file"]))
 q.require(request["model"]==old["model"],"same_original_frozen_model")
 q.write(folder/"task-id.json",dict(task_id=task))
 context=build(provenance,request);verify(context,provenance,request)
 q.write(out/(case["id"]+"-origin.json"),context)
 row=dict(case=case["id"],source_node_count=len(context["nodes"]),
  lowered_node_count=len(request["model"]["nodes"]),source_ops=sorted({n["op"]for n in context["nodes"]}),
  bytes=len(q.canonical(context)),binding=context["binding"],host_source_reverified=True)
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(format="poseidon-generation-model-origin-source-probe-v1",cases=rows,passed=len(rows),
 api_calls=0,compiler_calls=0,encrypted_calls=0,application_wiring_complete=False,fresh_source_captures=3)
report["binding"]=q.digest(report);q.write(HERE/"source-probe-r2.json",report);q.write(out/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}))
