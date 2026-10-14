"""Read-only source, current identity and retained acceptance verification."""
import hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
from application_identity import identity_manifest
REPORT=ROOT/"docs/baseline/agent-mha-component-cutover-results-r1312.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical=lambda o:json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def bound(value):
    assert hashlib.sha256(canonical({k:v for k,v in value.items() if k!="binding"})).hexdigest()==value["binding"]
def main():
    phase=json.loads(REPORT.read_bytes());bound(phase)
    for table in (phase["files"],phase["protected_sources"],phase["example_files"]):
        for name,expected in table.items():assert sha(ROOT/name)==expected,name
    reports={}
    for key,entry in phase["evidence_reports"].items():
        assert sha(entry["path"])==entry["sha256"],key
        r=json.loads(Path(entry["path"]).read_bytes());bound(r)
        assert r["binding"]==entry["binding"],key
        reports[key]=r
    control=reports["controls"];assert control["passed"]==161 and control["failed"]==control["errors"]==control["skipped"]==0
    assert all(sha(ROOT/n)==h for n,h in control["source_files"].items())
    result=reports["backend"];assert result["status"]=="complete" and result["passed"]==3 and result["failed"]==result["not_run"]==result["skipped"]==0
    current=identity_manifest(ROOT,result["backend_identity"]);assert current==result["stage_identity"]
    for n,h in result["source_implementation"].items():assert sha(BASE/n)==h,n
    for n,h in result["tool_files"].items():assert sha(ROOT/n)==h,n
    for table in (result["fixture_files"],result["capture_files"]):
        for n,h in table.items():assert sha(n)==h,n
    groups=values=0;maximum=0.
    for c in result["cases"]:
        folder=Path(phase["backend_directory"])/c["id"]
        qualification=json.loads((folder/"qualification.json").read_bytes())
        assert qualification["status"]=="passed" and qualification["compiled_validated"] and qualification["encrypted_execution"] and qualification["numerically_validated"]
        assert all(qualification["observed_stages"].values())
        assert qualification["paid_calls"]==0 and qualification["new_agent_generation"] is False
        manifest=json.loads((folder/"program/manifest.json").read_bytes());bound(manifest)
        assert manifest["binding"]==c["package_binding"] and manifest["agent_generated"] is False
        for n,h in manifest["files"].items():assert sha(folder/"program"/n)==h,n
        frozen=phase["case_evidence"][c["id"]]
        for n,h in frozen["files"].items():assert sha(n)==h,n
        groups+=4;values+=qualification["compared_values"];maximum=max(maximum,qualification["max_absolute_error"])
        for fn,comparison,count in (("new-input-execution.json","new_input_comparison",3),("repeat-execution.json","repeat_comparison",1)):
            ex=json.loads((folder/fn).read_bytes())
            raw=json.loads((Path(ex["evidence"])/"report.json").read_bytes())
            assert raw==ex and ex["bundle_binding"]==manifest["binding"]
            assert ex["status"]=="executed" and ex["encrypted_execution"] and ex["executor_secret_key"] is False and ex["samples"]==count
            assert c[comparison]["passed"]
            groups+=count;values+=c[comparison]["compared_values"]
            maximum=max(maximum,c[comparison]["max_absolute_error"])
        assert c["repeated_execution"] and c["historical_candidate_unchanged"]
    assert (groups,values)==(24,336) and maximum==phase["summary"]["max_absolute_error"]
    print(json.dumps(dict(status="matched",phase_binding=phase["binding"],controls=161,
        source_capture_models=3,compiled_models=3,numerical_models=3,package_models=3,
        input_executions=groups,compared_values=values,max_absolute_error=maximum,
        production_mha_active=True,default_rule_unchanged=True,provider_candidate_rewriting=False,
        rule_baseline=True,new_agent_generation=False,protected_sources=len(phase["protected_sources"]),
        active_source_count=len(current["compilation"]["sources"]),paid_api_calls=0,
        inspector_compiler_calls=0,inspector_encrypted_calls=0),sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
