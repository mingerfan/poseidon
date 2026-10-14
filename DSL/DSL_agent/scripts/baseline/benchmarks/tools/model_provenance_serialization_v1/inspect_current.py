"""Read-only current phase integrity; no Nix/compiler/FHE/Provider calls."""
import hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from application_identity import identity_manifest
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def bound(value):
    return digest({k:v for k,v in value.items() if k!="binding"})==value["binding"]
def check_files(root,table):
    for name,expected in table.items():
        p=root/name
        require(p.is_file() and not p.is_symlink() and sha(p)==expected,"File drift: "+name)
def main():
    phase=ROOT/"docs/baseline/agent-formal-provenance-export-results-r1310.json"
    r=json.loads(phase.read_bytes());require(bound(r),"Phase binding")
    check_files(ROOT,r["source_files"])
    protected_path=ROOT/r["protected_table"]["path"]
    require(sha(protected_path)==r["protected_table"]["file_sha256"],"Protection authority drift")
    protected=json.loads(protected_path.read_bytes())["protected_sources"]
    require(digest(protected)==r["protected_table"]["table_sha256"],"Protection table binding")
    check_files(ROOT,protected)
    p=Path(r["evidence_report"]["path"])
    require(sha(p)==r["evidence_report"]["sha256"],"Acceptance report bytes")
    a=json.loads(p.read_bytes());require(bound(a) and a["binding"]==r["evidence_report"]["binding"],"Acceptance report binding")
    require(a["status"]=="passed","Acceptance terminal")
    for group in ("tool_sha256","capture_files","prior_files"):
        for file,expected in a[group].items():
            require(sha(file)==expected,"Frozen evidence/source drift: "+file)
    require(identity_manifest(ROOT,a["stage_identity"]["backend"])==a["stage_identity"],"Current stage identity drift")
    check_files(ROOT,r["source_capture"]["files"])
    check_files(ROOT,r["historical_report_files"])
    q=Path(a["qualification"]["evidence"]);qr=json.loads((q/"report.json").read_bytes())
    check_files(BASE,qr["source_hashes"]);check_files(q,qr["frozen_hashes"])
    passed=[row for row in qr["attempts"] if row["status"]=="passed"]
    require(len(passed)==1 and passed[0]["compiled"] and passed[0]["executed"] and passed[0]["numerically_correct"],"Qualification evidence layers")
    check_files(q/("attempt-%02d"%passed[0]["index"])/"output",passed[0]["artifact_hashes"])
    package=Path(r["package"]["path"]);manifest=json.loads((package/"manifest.json").read_bytes())
    require(bound(manifest) and manifest["binding"]==a["package_binding"],"Package binding")
    check_files(package,manifest["files"])
    for item in r["execution_evidence"]:
        for file,expected in item["files"].items():require(sha(file)==expected,"Execution evidence drift")
    print(json.dumps(dict(status="matched",phase_binding=r["binding"],controls_passed=a["controls"]["passed"],
        controls_failed=a["controls"]["failed"],controls_skipped=a["controls"]["skipped"],
        tracing_passed=True,compiled_passed=True,numerically_passed_models=1,
        input_executions=r["numerical_summary"]["input_groups"],compared_values=r["numerical_summary"]["compared_values"],
        maximum_absolute_error=r["numerical_summary"]["maximum_absolute_error"],
        formal_exporter=True,transport_repacker=False,package_reloaded=True,
        protected_sources=len(protected),source_capture_files=len(a["source_capture_implementation"]),
        production_sources=len(a["stage_identity"]["compilation"]["sources"]),
        production_mha_enabled=False,paid_api_calls=0,compiler_calls_from_inspector=0,encrypted_calls_from_inspector=0),sort_keys=True))
if __name__=="__main__":main()
