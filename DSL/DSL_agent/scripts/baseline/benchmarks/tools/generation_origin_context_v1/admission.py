"""Normal application origin admission; no provider, compiler or ciphertext."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=600,environment="poseidon-cpu"))
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT
from component_contract import reconstruct_request
from public_weight_context import verify_context
from deepseek_provider import public_request
from generation_model_origin import verify,checked_public
from application_backend import PoseidonBackend
OUT=q.RESULTS/"agent-generation-origin-admission-r1677"
q.require(not OUT.exists(),"new_origin_admission");OUT.mkdir(mode=0o700)
baseline=live.measure(q.RESULTS);start=time.monotonic()
q.require(baseline+128*1024**2<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
sdk=PoseidonBackend().identity_manifest()
prior=q.read(q.RESULTS/"agent-layernorm-calibration-live-r1668/live-plan.json")
requests=[(c["id"],Path(c["source_manifest"]),Path(c["request_file"]))for c in prior["cases"]]
for name,root in [("conv2_c1","agent-source-transparent-modules-repair-r1672"),("bn1_c2","agent-source-transparent-modules-repair-r1672"),("pool2_c2","agent-source-transparent-modules-r1671")]:
 manifest=q.RESULTS/root/name/"freeze/source/manifest.json"
 old=q.RESULTS/"agent-source-transparent-poseidon-r1673"/name/"request.json"
 requests.append((name,manifest,old))
rows=[]
for name,source,old in requests:
 q.require(time.monotonic()-start<600 and live.measure(q.RESULTS)-baseline<=128*1024**2,"bounded_admission")
 folder=OUT/name;folder.mkdir(mode=0o700);app=application(WorkerContext.capture(folder/"tasks"))
 task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=str(source),trust_local_code=True),
   resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v21",
   public_weight_policy="descriptors-math-roles-v2",chunk_period=4,capacity_packing_version="capacity-public-tensors-v10"))
 state=app.prepare_source(task);q.write(folder/"task-id.json",dict(task_id=task));q.write(folder/"prepared-state.json",state)
 row=dict(case=name,status=state["status"],api_calls=0,compiler_calls=0,encrypted_calls=0)
 if state["status"]=="queued":
  meta=app._checked_task_metadata(task);request=meta["request"];origin=request["generation_model_origin"]
  q.require(request["model"]==q.read(old)["model"],"original_model_unchanged")
  checked_public(origin,request);verify(origin,meta["model_provenance"],request)
  q.require(reconstruct_request(request)==request,"full_request_roundtrip")
  projected=public_request(request);verify_context(projected)
  q.write(folder/"request.json",request);q.write(folder/"public-request.json",projected)
  row.update(request_id=request["request_id"],origin_binding=origin["binding"],bytes=len(q.canonical(projected)),
   source_nodes=len(origin["nodes"]),lowered_nodes=len(request["model"]["nodes"]),
   application_wiring=True,model_unchanged=True,source_manifest=str(source),source_manifest_sha256=q.sha(source))
 else:row["failure"]=state.get("failure")
 q.write(folder/"case-report.json",row);rows.append(row)
 print(json.dumps({k:v for k,v in row.items()if k not in("failure","source_manifest")}),flush=True)
 q.require(PoseidonBackend().identity_manifest()==sdk,"sdk_frozen")
report=dict(format="poseidon-generation-origin-admission-v1",sdk=sdk["manifest_sha256"],
 cases=rows,planned=6,passed=sum(r["status"]=="queued"for r in rows),failed=sum(r["status"]!="queued"for r in rows),
 skipped=0,api_calls=0,compiler_calls=0,encrypted_calls=0,
 source_captures=6,seconds=time.monotonic()-start,new_bytes=live.measure(q.RESULTS)-baseline)
report["binding"]=q.digest(report);q.write(HERE/"admission-r1677.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
raise SystemExit(0 if report["passed"]==6else 1)
