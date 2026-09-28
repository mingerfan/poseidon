"""Diagnose two frozen Agent failures; manual fixes never count as Agent passes."""
import hashlib,json,shlex,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1200)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from component_backend import qualify
 from benchmark_graph import digest
 original=json.loads((RESULTS/"stage2-noncompiler-r177/before.json").read_text())
 frozen=runtime_sources();started=time.monotonic()
 report=dict(format="manual-followup-r179",paid_calls=0,new_agent_successes=0,rows=[],source_hashes=frozen)
 target=RESULTS/"stage2-noncompiler-r177/followup-report-r179.json"
 assert not target.exists()
 for name in ("followup-construct-088-1","followup-helper-0113"):
  assert runtime_sources()==frozen
  assert all(sha(p)==v for p,v in original["compiler"].items())
  p=RESULTS/"stage2-noncompiler-r177"/name/"job.json";job=json.loads(p.read_text())
  result=qualify(job["request"],job["candidate"])
  row=dict(name=name,job_sha256=sha(p),request_id=job["request"]["request_id"],result=result,
           report_sha256=sha(Path(result["evidence"])/"report.json"))
  report["rows"].append(row)
  print(json.dumps(row),flush=True)
  if result.get("failure",{}).get("layer") in ("environment","integrity"):raise RuntimeError("Stop on environment/integrity")
  report["seconds"]=time.monotonic()-started
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  target.write_text(json.dumps(report,indent=2))
 assert runtime_sources()==frozen
 assert all(sha(p)==v for p,v in original["compiler"].items())
 return 0
if __name__=="__main__":raise SystemExit(main())
