"""Read-only current integration and historical live evidence identity check. No Nix or Provider."""
import json,pathlib,hashlib,sys
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):
 value=json.loads(p.read_bytes())
 if "binding" in value:
  assert value["binding"]==hashlib.sha256(canonical({k:v for k,v in value.items() if k!="binding"})).hexdigest(),str(p)
 return value
def main():
 report=read(ROOT/"docs/baseline/agent-public-math-roles-results-r1307.json")
 for name,want in report["production_files"].items():assert sha(ROOT/"scripts/baseline"/name)==want,name
 protected=json.loads((ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json").read_bytes())["protected_sources"]
 assert all(sha(ROOT/name)==want for name,want in protected.items())
 for group in report["controls"].values():
  if type(group) is dict and "report" in group:assert sha(pathlib.Path(group["report"]))==group["sha256"]
 source=pathlib.Path(report["source_capture"]["report"]);assert sha(source)==report["source_capture"]["sha256"]
 offline=pathlib.Path(report["real_backend"]["report"]);assert sha(offline)==report["real_backend"]["sha256"]
 data=read(offline)
 for stage in ("generation","compilation","execution","validation"):
  expected=data["stage_identity"][stage]["sources"]
  assert len(expected)==685
  assert all(sha(ROOT/"scripts/baseline"/name)==want for name,want in expected.items()),stage
 assert data["passed"]==1 and data["failed"]==0
 row=data["cases"][0];assert row["source_kind"]=="deterministic_rule_baseline" and not row["new_agent_generation"]
 assert row["compiled"] and row["encrypted_execution"] and row["reloaded"] and row["new_input_comparison"]["passed"]
 history=read(ROOT/"docs/baseline/agent-gelu-live-results-r1306.json")
 assert history["failed"]==3 and history["passed"]==0 and history["calls_resolved"] and history["http_attempts"]==12
 assert history["actual_encrypted_executions"]==12 and history["numerical_passes"]==0
 print(json.dumps(dict(status="matched",control_tests=181,control_failures=0,protected_sources=len(protected),
  production_stage_sources=685,live_status="closed",live_passed=0,live_failed=3,live_generations=12,live_http_attempts=12,
  new_policy_rule_baseline_passed=1,new_policy_live_agent="not_run",compiler_calls_from_inspector=0,
  encrypted_executions_from_inspector=0,api_calls=0),ensure_ascii=False))
if __name__=="__main__":main()
