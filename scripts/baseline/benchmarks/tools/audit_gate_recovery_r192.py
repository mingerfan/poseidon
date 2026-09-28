"""Read-only reconciliation of interrupted r189; never retry or erase calls."""
import json,sys,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=300)
 from workspace_paths import RESULTS as R
 from campaign_live_state import check,sealed
 from stage2_agent_provenance import provider_accounting
 from audit_stage2_live_candidate import verify_candidate,verify_files
 from campaign_request_files import read_request
 from campaign_budget_outcome import outcome
 from benchmark_graph import digest
 from semantic_benchmark_execution import runtime_sources
 parents={}
 def read(p,sealed_=False):
  if p.is_symlink():raise ValueError("Unsafe evidence")
  parents[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  d=json.loads(p.read_text());return check(d) if sealed_ else d
 plan=read(R/"stage2-agent-repair-r189/plan.json",True)
 master=read(R/"stage2-agent-repair-r189/report.json",True)
 assert plan["source_hashes"]==runtime_sources()
 rows=[];not_started=[];generations=http=0
 for ref in plan["shards"]:
  out=Path(ref["output"]);shard=ref["plan"]
  read(out/"report.json",True)
  for spec in shard["cases"]:
   launch=out/spec["id"]/"launch.json"
   if not launch.exists():
    claim=Path(plan["claim_registry"])/(spec["evaluation_identity"]+".json")
    assert not claim.exists()
    not_started.append(spec["id"]);continue
   mark=read(launch,True);assert mark["evaluation_identity"]==spec["evaluation_identity"]
   log=out/spec["id"]/"run.log"
   parents[str(log)]=hashlib.sha256(log.read_bytes()).hexdigest()
   folder=next(Path(l.split(": ",1)[1]) for l in log.read_text().splitlines() if l.startswith("Candidate evidence: "))
   report=read(folder/"report.json");assert report["status"]!="running"
   read_request(folder/"request.json",spec["request"])
   assert digest(read(folder/"model.json"))==spec["model_sha256"]
   verify_files(BASE,report["source_hashes"]);verify_files(folder,report["frozen_hashes"])
   a=provider_accounting(report,shard["paid_configuration"])
   generations+=a["generations"];http+=a["http_attempts"]
   row=dict(id=spec["id"],evidence=str(folder),status="interrupted" if report["status"]=="cancelled" else "failed",
    provider_status=report["status"],generations=a["generations"],http_attempts=a["http_attempts"],
    in_flight=sum(c["status"]=="in_flight" for c in report["provider_metrics"]["calls"]),
    attempts=[{k:v.get(k) for k in ("index","failure_layer","diagnostic","compiled","executed")} for v in report["attempts"]],
    automatic_retry=False,paid_configuration=shard["paid_configuration"])
   if report["status"]=="passed":
    assert outcome(report,0,shard["paid_configuration"])=="runner_reported_pass_pending_audit"
    details=verify_candidate(folder,shard["paid_configuration"])
    detailpath=R/"stage2-recovery-r192-pass-audit.json"
    with detailpath.open("x") as f:json.dump(sealed(details),f,indent=2)
    parents[str(detailpath)]=hashlib.sha256(detailpath.read_bytes()).hexdigest()
    c=details["comparison"]
    row.update(status="passed",comparison={k:c[k] for k in ("compared_values","max_absolute_error","atol","rtol")})
   rows.append(row)
 assert len(rows)==5 and len(not_started)==23 and generations==http==9
 result=sealed(dict(format="poseidon-r189-recovery-r192",source_hashes=runtime_sources(),rows=rows,
  never_started_ids=not_started,prior_generations=generations,prior_http_attempts=http,
  prior_seconds=master["seconds"],in_flight_requests=3,parents=parents,
  policy="Only 23 never-launched cases can continue; do not replay the 3 interrupted paid tasks.",
  original_maximum_generations=60,original_maximum_http_attempts=240,
  remaining_planned_generations=46,remaining_planned_http_attempts=184))
 p=ROOT/"docs/baseline/stage2-recovery-r192.json"
 with p.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps({k:result[k] for k in ("binding","prior_generations","prior_http_attempts","in_flight_requests","prior_seconds")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
