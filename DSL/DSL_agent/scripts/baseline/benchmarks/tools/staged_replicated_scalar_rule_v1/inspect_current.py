"""Read-only phase binding inspection; no compiler/runtime/Provider."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
def main():
 phase=ROOT/"docs/baseline/agent-replicated-scalar-results-r1309.json"
 r=json.loads(phase.read_bytes())
 require(digest({k:v for k,v in r.items() if k!="binding"})==r["binding"],"Phase binding")
 for relative,expected in r["stage_files"].items():
  p=ROOT/relative
  require(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,"Staged source drift")
 for entry in r["evidence_reports"]:
  p=Path(entry["path"])
  require(hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"],"Historical evidence drift")
  v=json.loads(p.read_bytes())
  require(v.get("binding")==entry["binding"] and digest({k:x for k,x in v.items() if k!="binding"})==v["binding"],"Evidence binding")
 for entry in r["cases"]:
  attempt=Path(entry["qualification_evidence"])/"attempt-00"
  audit=json.loads((attempt/"report.json").read_bytes())
  for name,expected in audit["artifact_hashes"].items():
   p=attempt/"output"/name
   require(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,"Runtime artifact drift")
 package=Path(r["program_package"]["path"])
 manifest=json.loads((package/"manifest.json").read_bytes())
 require(manifest["binding"]==r["program_package"]["binding"] and digest({k:v for k,v in manifest.items() if k!="binding"})==manifest["binding"],"Program manifest drift")
 for name,expected in manifest["files"].items():
  p=package/name
  require(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,"Program byte drift")
 previous=ROOT/"scripts/baseline/benchmarks/tools/staged_mha_ordered_graph_v1/inspect_current.py"
 checked=subprocess.run([sys.executable,"-B",str(previous)],capture_output=True,text=True,timeout=30)
 require(checked.returncode==0,"Production/protected source drift")
 snapshot=json.loads(checked.stdout)
 print(json.dumps(dict(status="matched",phase_binding=r["binding"],controls_passed=r["tests"]["final_controls_passed"],
  traced=r["backend_summary"]["tracing_passed"],compiled=r["backend_summary"]["compiled_passed"],
  numerical_models_passed=r["backend_summary"]["numerical_models_passed"],package_models_passed=r["backend_summary"]["package_models_passed"],
  resource_failed=r["backend_summary"]["runtime_resource_failed"],production_sources=snapshot["production_sources"],
  protected_sources=snapshot["protected_sources"],production_mha_active=False,paid_api_calls=0,
  compiler_calls_from_inspector=0,encrypted_executions_from_inspector=0),sort_keys=True))
if __name__=="__main__":main()
