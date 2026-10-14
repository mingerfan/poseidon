from pathlib import Path
import sys,shlex,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=120,environment="poseidon-cpu"))
import qualification_scope as q
from generation_model_origin import build,verify
D=q.RESULTS/"agent-layernorm-calibration-live-r1668"
plan=q.read(D/"live-plan.json");rows=[]
out=q.RESULTS/"agent-generation-origin-source-r1675";q.require(not out.exists(),"new_origin_probe");out.mkdir()
for case in plan["cases"]:
 td=D/case["id"]/"tasks"/case["task_id"];meta=q.read(td/"task.json");q.bound(meta)
 request=meta["request"];provenance=meta["model_provenance"]
 context=build(provenance,request);verify(context,provenance,request)
 q.write(out/(case["id"]+"-origin.json"),context)
 row=dict(case=case["id"],source_node_count=len(context["nodes"]),
  lowered_node_count=len(request["model"]["nodes"]),source_ops=sorted({n["op"]for n in context["nodes"]}),
  bytes=len(q.canonical(context)),binding=context["binding"],host_source_reverified=True)
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(format="poseidon-generation-model-origin-source-probe-v1",cases=rows,passed=len(rows),
 api_calls=0,compiler_calls=0,encrypted_calls=0,application_wiring_complete=False)
report["binding"]=q.digest(report);q.write(HERE/"source-probe.json",report);q.write(out/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}))
