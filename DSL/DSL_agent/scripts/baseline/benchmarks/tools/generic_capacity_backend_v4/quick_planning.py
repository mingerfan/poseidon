"""All planned graphs checked independently; no backend or provider calls."""
import sys,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
if "--inside"not in sys.argv:
 import run as live
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
from application_planning.layout import plan_model,check_plan
from deepseek_provider import public_request
from benchmark_graph import canonical,digest
cases=json.loads((HERE/"cases.json").read_text())
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generic-capacity-capture-r1493")
rows=[]
for c in cases:
 m=json.loads((R/c["name"]/"model.json").read_text())
 o=dict(c["options"],public_weight_policy="descriptors-compact-origins-v3")
 if c["name"]=="weights_linear_10k":o["generation_guidance"]="explicit-v14"
 d=plan_model(m,options=o)
 row=dict(case=c["name"],status=d["status"],failure=d["failure"],model_sha256=digest(m),decision=d)
 if d["status"]=="prepared":
  q=check_plan(d,m);v=public_request(q)
  row.update(request_id=q["request_id"],period=q["layout"]["input_slot_period"],public_bytes=len(canonical(v)))
 print(json.dumps({k:v for k,v in row.items()if k!="decision"}),flush=True);rows.append(row)
result=dict(format="poseidon-generic-capacity-planning-probe-v1",cases=rows,planned=len(rows),
 prepared=sum(r["status"]=="prepared"for r in rows),rejected=sum(r["status"]!="prepared"for r in rows),
 paid_calls=0,compiler_calls=0,encrypted_calls=0)
result["binding"]=digest(result)
with (HERE/"quick-planning.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2)
