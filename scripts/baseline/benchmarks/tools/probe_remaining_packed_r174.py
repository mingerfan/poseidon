"""One exact-model balanced-tree probe, without compiler or parameter changes."""
import json,sys,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=240)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_lowering import lower
 from unified_graph_contract import validate_candidate
 from component_backend import qualify
 from benchmark_graph import digest
 out=RESULTS/"stage2-remaining-r174/compiler-probe-packed-v2";out.mkdir()
 old=json.loads((ROOT/"docs/baseline/stage2-agent-repair-result-r173.json").read_text())
 row=next(r for r in old["rows"] if r["id"]=="free_bench_helper_0104")
 q=json.loads((Path(row["evidence"])/"request.json").read_text())
 before=json.loads((RESULTS/"stage2-remaining-r174/before.json").read_text())
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert all(sha(p)==h for p,h in before["compiler"].items())
 sources=runtime_sources();value=dict(id=row["id"],paid_calls=0,new_agent_success=False,model=q["model"],
  original_request_id=q["request_id"],source_hashes=sources,compiler_changed=False)
 try:
  # Diagnostic only: lane-local original graph, no extraction/repacking overhead.
  # Every Chebyshev coefficient is retained; no shape/layout/config changes.
  from unified_graph_lowering import Emitter,Scalar
  from chebyshev_lowering import balanced
  assert q["layout"]["execution_abi"]=="unified-periodic-inputs-v1"
  assert len(q["layout"]["inputs"])==1 and len(q["model"]["outputs"])==1
  em=Emitter(q);values=dict(q["model"]["constants"])
  binding=q["layout"]["inputs"][0]
  values[binding["name"]]=Scalar(em,binding["dsl_name"])
  for node in q["model"]["nodes"]:
   args=[values[k] for k in node["inputs"]]
   assert len(node["outputs"])==1
   if node["op"]=="polynomial":
    assert node["attrs"]=={"basis":"chebyshev"}
    answer=balanced(args[0],args[1],em.zero)
   elif node["op"]=="add":answer=args[0]+args[1]
   elif node["op"]=="multiply":answer=args[0]*args[1]
   else:raise ValueError("Unsupported diagnostic lane-local graph")
   values[node["outputs"][0]]=answer
  answer=values[q["model"]["outputs"][0]["value"]]
  source=chr(10).join(['@hc.func("c,c")', "def golden("+binding["dsl_name"]+", zero_ct):", *em.lines, "    return ["+answer.name+"]", ""])
  value["strategy"]="balanced Chebyshev per packed lane; original model and selected output slots"

  candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
  (out/"job.json").write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
  validate_candidate(candidate,q)
  value["result"]=qualify(q,candidate)
 except (ValueError,TypeError) as exc:value.update(status="preflight_failed",diagnostic=str(exc))
 assert sources==runtime_sources()
 assert all(sha(p)==h for p,h in before["compiler"].items())
 value["binding"]=digest(value)
 (out/"report.json").write_text(json.dumps(value,indent=2))
 print(json.dumps({k:v for k,v in value.items() if k not in ("model","source_hashes")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
