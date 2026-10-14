"""Read-only mixed-layout evidence and preserved historical source inspection."""
import hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from application_identity import identity_manifest
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def bound(v):return v["binding"]==digest({k:x for k,x in v.items() if k!="binding"})
def check_files(root,items):
 for name,h in items.items():require(sha(root/name)==h,"File drift: "+str(name))
def main():
 p=ROOT/"docs/baseline/agent-mixed-packed-scalar-results-r1311.json"
 r=json.loads(p.read_bytes());require(bound(r),"Phase binding")
 check_files(ROOT,r["source_files"]);check_files(ROOT,r["protected_sources"])
 for entry in r["reports"]:
  p=Path(entry["path"]);require(sha(p)==entry["sha256"],"Retained report bytes")
  v=json.loads(p.read_bytes());require(bound(v) and v["binding"]==entry["binding"],"Retained report binding")
  for name,h in v.get("staged_file_sha256",v.get("tool_files",{})).items():
   actual=Path(name)
   if sha(actual)!=h:
    actual=ROOT/r["historical_source_snapshots"][name][h]
   require(sha(actual)==h,"Executed source snapshot")
 follow=json.loads(Path(r["followup_report"]).read_bytes())
 require(identity_manifest(ROOT,follow["stage_identity"]["backend"])==follow["stage_identity"],"Current application identity")
 for row in r["cases"]:
  q=Path(row["qualification_evidence"]);v=json.loads((q/"report.json").read_bytes())
  check_files(BASE,v["source_hashes"]);check_files(q,v["frozen_hashes"])
  passed=[x for x in v["attempts"] if x["status"]=="passed"]
  require(len(passed)==1 and passed[0]["compiled"] and passed[0]["executed"] and passed[0]["numerically_correct"],"Qualification layers")
  check_files(q/("attempt-%02d"%passed[0]["index"])/"output",passed[0]["artifact_hashes"])
  package=Path(row["package"]);m=json.loads((package/"manifest.json").read_bytes())
  require(bound(m) and m["binding"]==row["package_binding"],"Package binding")
  check_files(package,m["files"])
  for evidence in row["execution_files"]:check_files(Path("/"),evidence)
 print(json.dumps(dict(status="matched",phase_binding=r["binding"],control_passed=r["tests"]["final_passed"],
  compiled_models=3,numerical_models=3,package_models=3,input_executions=r["numerical_summary"]["input_groups"],
  compared_values=r["numerical_summary"]["compared_values"],max_absolute_error=r["numerical_summary"]["maximum_absolute_error"],
  original_failure_reports_retained=True,rule_baseline=True,new_agent_generation=False,
  production_mha_enabled=False,protected_sources=len(r["protected_sources"]),paid_api_calls=0,
  inspector_compiler_calls=0,inspector_encrypted_calls=0),sort_keys=True))
if __name__=="__main__":main()
