"""Verify application acceptance and protected backend identities; no model calls."""
import json,hashlib,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/application-component-r212")
DEST=ROOT/"docs/baseline/application-component-acceptance-r216.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
 if "--verify" in sys.argv:
  d=json.loads(DEST.read_text());b=d.pop("binding");assert digest(d)==b
  for group in ("parents","compiler_guard","sources"):
   assert all(sha(p)==h for p,h in d[group].items()),group
  print(json.dumps(dict(verified=True,binding=b,tests=d["unit_tests"])));return
 if DEST.exists():raise ValueError("Preserve prior report")
 parents={}
 def read(p):
  parents[str(p)]=sha(p);return json.loads(Path(p).read_text())
 before=read(R/"before.json")
 unit=read(R/"unit-r215.json");compat=read(R/"compatibility-r215.json");accept=read(R/"acceptance-r215.json")
 assert (unit["run"],unit["failures"],unit["errors"],unit["skipped"])==(249,0,0,0)
 assert compat["old_requests_exact"]==2208 and accept["complete"]
 compiled,numeric,legacy=accept["rows"]
 assert compiled["numerical_upgrade_rejected"]
 for row in (compiled,numeric):
  r=row["result"];assert r["status"]=="succeeded" and row["replay_exit"]==0
  assert row["reuse"]["generation_calls"]==0 and row["reuse"]["result"]["package"]["reused"]
  level=row["id"];a=r["acceptance"];assert a["level"]==level
  assert a["encrypted_execution"]==a["numerically_validated"]==(level=="numerical")
  task=R/"tasks-r215"/row["task_id"]
  validation=read(task/"validation-0.json")
  evidence=Path(validation["evidence"])
  report=read(evidence/"report.json");request=read(evidence/"request.json")
  attempt=read(evidence/"attempt-00/report.json")
  assert report["status"]=="passed" and report["validation_level"]==level and report["agent_calls"]==0
  assert attempt["compiled"] and attempt["trace"]["frontend"]=="real_Hecate"
  assert attempt["executed"]==attempt["numerically_correct"]==(level=="numerical")
  if level=="numerical":
   assert attempt["comparison"]["passed"] and attempt["comparison"]["atol"]==1e-5 and attempt["comparison"]["rtol"]==1e-4
  m=read(task/"program/manifest.json")
  assert m["binding"]==r["package"]["binding"]
  for name,h in m["files"].items():
   f=task/"program"/name;assert sha(f)==h;parents[str(f)]=h
 assert legacy["exit_code"]==0
 legacy_report=read(Path(legacy["evidence"])/"report.json")
 assert legacy_report["status"]=="passed" and legacy_report["agent_calls"]==0
 assert [a.get("failure_layer","complete") for a in legacy_report["attempts"]]==["response_parse","numerical_comparison","complete"]
 assert all(sha(p)==h for p,h in before["compiler"].items())
 sources={str(p):sha(p) for p in (ROOT/"scripts/baseline").glob("*.py")}
 prior={str(ROOT/p):h for p,h in before["source_hashes"].items()}
 changes=dict(modified=[p for p in prior if sources.get(p)!=prior[p]],new=[p for p in sources if p not in prior])
 for p in prior:assert Path(p).exists()
 # Replays are independent of application task execution; preserve their identities.
 log=R/"acceptance-r215.log";parents[str(log)]=sha(log)
 replays=[]
 for line in log.read_text().splitlines():
  if line.startswith("{"):
   v=json.loads(line)
   if "bundle_binding" in v:
    assert v["status"]=="passed" and v["paid_calls"]==0
    replays.append(v)
 assert len(replays)==2
 for f in ("application-agent-component-v1.md","agent-component-v1.md"):
  p=ROOT/"docs/baseline"/f;parents[str(p)]=sha(p)
 d=dict(format="poseidon-application-component-acceptance-r216",
   scope="Application task boundary and explicitly graded source packages; not full deployment SDK",
   unit_tests={k:unit[k] for k in ("run","failures","errors","skipped")},
   old_request_hashes_exact=compat["old_requests_exact"],acceptance=accept["rows"],replays=replays,
   paid_calls=0,new_agent_successes=0,prior_agent_failures_unchanged=26,x86_executed=False,
   compiler_modified=False,security_parameters_modified=False,models_modified=False,reference_modified=False,
   atol=1e-5,rtol=1e-4,no_install_download_commit_push_pr=True,
   local_changes=changes,compiler_guard=before["compiler"],sources=sources,parents=parents)
 d["binding"]=digest(d);DEST.write_text(json.dumps(d,indent=2)+"\n")
 print(json.dumps(dict(binding=d["binding"],tests=d["unit_tests"],old_requests=d["old_request_hashes_exact"],paid_calls=0)))
if __name__=="__main__":main()
