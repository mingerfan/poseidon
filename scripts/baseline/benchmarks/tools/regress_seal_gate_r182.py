"""Fresh evidence under the new gate; historical evidence is never rewritten."""
import sys,shlex,json,subprocess,time,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800)
 from component_backend import qualify
 from semantic_benchmark_execution import runtime_sources
 from benchmark_graph import digest
 out=WORK/"results/seal-gate-r182";out.mkdir()
 report=dict(paid_calls=0,new_agent_generation=False,source_hashes=runtime_sources(),rows=[])
 before=json.loads((WORK/"results/seal-gate-r181/before.json").read_text())
 def guard():
  assert runtime_sources()==report["source_hashes"]
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before["compiler"].items())
 def save():
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  (out/"report.json").write_text(json.dumps(report,indent=2))
 for name,args in [
  ("linear_self_test",["scripts/baseline/run_candidate.py","--inside","--case","scripts/baseline/cases/linear-example.json","--self-test"]),
  ("mlp4x4x2",["scripts/baseline/seal_cpu_golden.py","--inside","--case","mlp4x4x2"])]:
  guard()
  with (out/(name+".log")).open("w") as log:
   r=subprocess.run([str(VENV/"bin/python"),"-B",*args],stdout=log,stderr=subprocess.STDOUT,timeout=600)
  lines=(out/(name+".log")).read_text().splitlines()
  evidence=next((l.split(": ",1)[1] for l in lines if l.startswith(("Candidate evidence: ","SEAL CPU golden evidence: "))),None)
  report["rows"].append(dict(id=name,exit_code=r.returncode,evidence=evidence));save()
  print(json.dumps(report["rows"][-1]),flush=True)
 fixtures=[("construct_088_1",WORK/"results/stage2-noncompiler-r177/followup-construct-088-1/job.json"),
  ("silu_0113_manual",WORK/"results/stage2-noncompiler-r177/followup-helper-0113/job.json")]
 for name,jobpath in fixtures:
  guard();job=json.loads(jobpath.read_text());r=qualify(job["request"],job["candidate"])
  report["rows"].append(dict(id=name,result=r));save();print(json.dumps(report["rows"][-1]),flush=True)
 previous=json.loads((ROOT/"docs/baseline/stage2-agent-repair-result-r179.json").read_text())
 for name in ("free_bench_helper_0112","free_bench_helper_0117"):
  guard();entry=next(r for r in previous["rows"] if r["id"]==name);folder=Path(entry["evidence"])
  q=json.loads((folder/"request.json").read_text());old=json.loads((folder/"report.json").read_text())
  a=old["attempts"][-1]["index"];source=(folder/("attempt-%02d"%a)/"candidate.py").read_text()
  r=qualify(q,dict(schema=1,request_id=q["request_id"],hecate_source=source))
  report["rows"].append(dict(id=name+"_old_agent_replay",result=r));save();print(json.dumps(report["rows"][-1]),flush=True)
 guard();report["completed"]=True;save()
 return 0 if all(r.get("exit_code",r.get("result",{}).get("exit_code"))==0 for r in report["rows"]) else 1
if __name__=="__main__":raise SystemExit(main())
