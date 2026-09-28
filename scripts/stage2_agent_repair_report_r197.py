"""Independent aggregation of SiLU's remaining authorized quota; no paid calls."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(ROOT/"scripts/baseline/benchmarks/tools")]
from workspace_paths import RESULTS as R
from stage2_agent_repair_plan_r196 import NAME,document,verify
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
def main():
 p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");a=p.parse_args()
 out=R/NAME;plan=document(out/"plan.json");verify(plan,out)
 if a.check:print(json.dumps(dict(preflight=True,paid_calls=0,binding=plan["binding"])));return 0
 master=document(out/"report.json")
 if not master["queue_finished"] or master["failure"]:raise ValueError("Continuation not independently terminal; preserve evidence")
 ref=plan["shards"][0];audit=document(Path(ref["audit"])/"report.json")
 assert sha(Path(ref["audit"])/"report.json")==master["rows"][0]["audit_sha256"]
 assert audit["plan_binding"]==ref["plan"]["binding"] and len(audit["rows"])==1
 for name,h in audit["parents"].items():assert sha(Path(name))==h
 row=audit["rows"][0]
 if row["status"]=="passed":
  details=document(Path(ref["audit"])/(row["id"]+".audit.json"))
  assert sha(Path(ref["audit"])/(row["id"]+".audit.json"))==row["audit_sha256"]
  folder=Path(row["evidence"])
  for name,h in details["files"].items():
   path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()) and sha(path)==h
  c=details["comparison"];assert c["passed"] and c["atol"]==1e-5 and c["rtol"]==1e-4 and all(all(x) for x in c["elementwise_pass"])
  row["numerical_summary"]={k:c[k] for k in ("compared_values","mae","max_absolute_error")}
 result=sealed(dict(format="poseidon-silu-remaining-result-r197",plan_binding=plan["binding"],rows=[row],
  current_generations=audit["generations"],current_http_attempts=audit["http_attempts"],
  cumulative_generations=plan["prior_generations"]+audit["generations"],
  cumulative_http_attempts=plan["prior_http_attempts"]+audit["http_attempts"],
  case_cumulative_generations=2+audit["generations"],old_in_flight_preserved=True,
  source_hashes=plan["source_hashes"],new_session_first_pass_not_original_first_pass=True,
  parents={str(out/"plan.json"):sha(out/"plan.json"),str(out/"report.json"):sha(out/"report.json"),
   str(Path(ref["audit"])/"report.json"):sha(Path(ref["audit"])/"report.json")}))
 assert result["cumulative_generations"]<=60 and result["cumulative_http_attempts"]<=240 and result["case_cumulative_generations"]<=4
 for target in (R/"stage2-agent-repair-result-r197.json",ROOT/"docs/baseline/stage2-agent-repair-result-r197.json"):
  with target.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(dict(status=row["status"],generations=result["current_generations"],total_generations=result["cumulative_generations"])))
 return 0
if __name__=="__main__":raise SystemExit(main())
