"""Explicitly scripted application acceptance; no provider API or Agent claim."""
import sys,shlex,json,hashlib,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
    shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800)
 from application_component import Application
 from application_backend import SealBackend
 from benchmark_suite import Builder
 from benchmark_graph import digest
 from unified_graph_lowering import lower
 from component_backend import qualify
 from candidate_bundle import check_bundle,replay_bundle
 from semantic_benchmark_execution import runtime_sources
 out=WORK/"results/application-component-r212"
 path=out/"acceptance-r214.json"
 if path.exists():raise ValueError("Preserve existing evidence")
 app=Application(out/"tasks-r214",SealBackend())
 before=json.loads((out/"before.json").read_text());sources=runtime_sources()
 report=dict(paid_calls=0,new_agent_successes=0,source_hashes=sources,rows=[])
 def save():
  assert runtime_sources()==sources
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before["compiler"].items())
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  path.write_text(json.dumps(report,indent=2))
 class Provider:
  kind="scripted_replay";agent_calls=0
  def __init__(self):self.calls=0
  def generate(self,q,history):
   self.calls+=1
   return json.dumps(dict(schema=1,request_id=q["request_id"],hecate_source=lower(q)))
 def model():
  b=Builder([(4,)])
  w=b.const([[.125 if i==j else .03125 for j in range(4)] for i in range(4)])
  y=b.node("linear",["input0",w,b.const([.0625]*4)])
  y=b.node("square",[y])
  y=b.node("linear",[y,b.const([[.125,-.0625,.125,.0625],[-.0625,.125,.0625,.125]]),b.const([.03125,-.03125])])
  return b.finish(y)
 for level in ("compiled","numerical"):
  task=app.submit(model(),validation_level=level);provider=Provider();result=app.run(task,provider)
  row=dict(id=level,task_id=task,result=result,scripted_generations=provider.calls)
  report["rows"].append(row);save();print(json.dumps(row),flush=True)
  if result["status"]!="succeeded":return 1
  folder=app._dir(task)/"program"
  if level=="compiled":
   from audit_unified_candidate import verify_candidate
   validation=json.loads((app._dir(task)/"validation-0.json").read_text())
   try:verify_candidate(Path(validation["evidence"]))
   except ValueError:row["numerical_upgrade_rejected"]=True
   else:raise AssertionError("Compiled result promoted to numerical")
  # Existing bundle API reloads and requalifies; no application task or historical report required.
  manifest=check_bundle(folder)
  code=replay_bundle(folder);row.update(reload_binding=manifest["binding"],replay_exit=code);save()
  if code!=0:return 1
  second=app.submit(model(),validation_level=level);unused=Provider();cached=app.run(second,unused)
  row["reuse"]=dict(result=cached,generation_calls=unused.calls);save()
  assert cached["status"]=="succeeded" and unused.calls==0
 # Existing CLI numerical fault/recovery remains a separate compatibility check.
 log=out/"legacy-linear-r214.log"
 with log.open("w") as stream:
  code=subprocess.run([str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside",
    "--case","scripts/baseline/cases/linear-example.json","--self-test"],stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
 evidence=next((l.split(": ",1)[1] for l in log.read_text().splitlines() if l.startswith("Candidate evidence: ")),None)
 report["rows"].append(dict(id="legacy_linear_self_test",exit_code=code,evidence=evidence));save()
 report["complete"]=code==0;save();return code
if __name__=="__main__":raise SystemExit(main())
