"""Read-only stage receipt/source checks; does not enter Nix or run candidates."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
STAGE=Path(__file__).resolve().parent
REPORT=ROOT/"docs/baseline/agent-mha-ordered-intake-results-r1308.json"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 phase=json.loads(REPORT.read_bytes())
 canonical=lambda o:json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 assert phase["binding"]==hashlib.sha256(canonical({k:v for k,v in phase.items() if k!="binding"})).hexdigest(),"Phase binding changed"
 for record in phase["evidence_reports"]:
  assert sha(Path(record["path"]))==record["sha256"],"Historical report changed:"+record["path"]
 for name,expected in phase["source_capture_implementation"].items():
  local=STAGE/"overlay"/name
  assert sha(local if local.is_file() else BASE/name)==expected,"Stage source changed:"+name
 for name,expected in phase["stage_tool_files"].items():
  assert sha(STAGE/name)==expected,"Stage tool changed:"+name
 current=subprocess.run(["python3","-B",str(BASE/"benchmarks/tools/public_constant_role_context_v1/inspect_current.py")],
 capture_output=True,text=True,timeout=30)
 assert current.returncode==0,current.stderr
 production=json.loads(current.stdout)
 assert production["status"]=="matched","Production source changed"
 print(json.dumps(dict(status="matched",phase_binding=phase["binding"],
  production_mha_active=False,source_capture_passed=phase["source_capture"]["passed"],
  component_controls_passed=phase["component_controls"]["passed"],
  backend=phase["backend_summary"],production_sources=production["production_stage_sources"],
  protected_sources=production["protected_sources"],new_paid_api_calls=0,
  compiler_calls_from_inspector=0,encrypted_executions_from_inspector=0),ensure_ascii=False))
 return 0
if __name__=="__main__":raise SystemExit(main())
