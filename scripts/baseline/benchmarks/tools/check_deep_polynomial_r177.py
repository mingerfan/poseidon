"""Representative exact-graph diagnostics; no compiler/security changes."""
import sys,json,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=900)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_lowering import lower
 from unified_graph_contract import validate_candidate
 from component_backend import qualify
 from benchmark_graph import digest
 sources=runtime_sources();out=RESULTS/"stage2-noncompiler-r177/deep-representatives";out.mkdir()
 compiler=json.loads((out.parent/"before.json").read_text())["compiler"]
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 index=json.loads((ROOT/"docs/baseline/compiler-blocked-models-r159/index.json").read_text())
 targets={"bench_helper_0032","bench_helper_0040","semantic_context_331124fd5981bd48a0c2971f"}
 report=dict(source_hashes=sources,paid_calls=0,new_agent_successes=0,rows=[])
 for row in index["rows"]:
  if row["model_id"] not in targets:continue
  task=next(v for v in json.loads(Path(row["origin_shard"]).read_text())["cases"] if v["id"]==row["task_id"])
  q=task["request"];entry=dict(id=row["task_id"],request_id=q["request_id"])
  try:
   source=lower(q,packed_prefix=True,balanced_chebyshev=True)
   candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
   folder=out/row["task_id"];folder.mkdir()
   (folder/"job.json").write_text(json.dumps(dict(request=q,candidate=candidate),indent=2))
   validate_candidate(candidate,q)
   entry["result"]=qualify(q,candidate)
  except (ValueError,TypeError) as exc:entry.update(status="preflight_failed",diagnostic=str(exc))
  report["rows"].append(entry)
  print(json.dumps(entry),flush=True)
  assert sources==runtime_sources()
  assert all(sha(p)==h for p,h in compiler.items())
 report["binding"]=digest(report)
 (out/"report.json").write_text(json.dumps(report,indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
