"""Run exactly one frozen approved batch then its independent terminal aggregation."""
import argparse,json,subprocess,sys,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(ROOT/"scripts/baseline/benchmarks/tools")]
from workspace_paths import RESULTS
from stage2_agent_repair_plan_r196 import NAME,document
from run_stage2_guidance_retest import stop_child
def main():
 p=argparse.ArgumentParser();p.add_argument("--approve-live-binding",required=True);a=p.parse_args()
 out=RESULTS/NAME;plan=document(out/"plan.json")
 if plan["binding"]!=a.approve_live_binding:raise ValueError("Exact authorized binding")
 status={"binding":plan["binding"],"started":time.time(),"automatic_restart":False}
 cmd=[sys.executable,"-B",str(ROOT/"scripts/baseline/benchmarks/tools/stage2_agent_repair_r196.py"),
      "run","--output",str(out),"--approve-live-binding",a.approve_live_binding]
 child=None
 try:
  child=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
  status["runner_pid"]=child.pid
  (out/"job-status.json").write_text(json.dumps(status))
  status["runner_exit"]=child.wait(timeout=43360)
 finally:
  if child is not None:stop_child(child)
  status["runner_finished"]=time.time()
  (out/"job-status.json").write_text(json.dumps(status))
 # No provider calls in the reporter. Uncertain or unaudited tasks never count as pass.
 report=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/stage2_agent_repair_report_r197.py")],cwd=ROOT,timeout=300)
 status["reporter_exit"]=report.returncode;status["finished"]=time.time()
 (out/"job-status.json").write_text(json.dumps(status))
 return report.returncode or status["runner_exit"]
if __name__=="__main__":raise SystemExit(main())
