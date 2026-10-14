"""Verify the node-qualified adaptive downsampling refusal without rerunning positives."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_operator_shape_capture_prototype import checked,sha,validate_prototype,sandbox
def run(previous,prototype,proposal_path,output):
    from hecate_python_env import VENV,WORK
    from pytorch_model_intake import prepare,source_identity
    from benchmarks.tools.audit_pytorch_operator_shape_contexts import frozen
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh refusal continuation")
    proposal=checked(json.loads(proposal_path.read_text()));bound=frozen(proposal)
    staged,current,expected=validate_prototype(prototype)
    old=checked(json.loads(previous.read_text()))
    require(old["prototype_binding"]==staged["binding"] and old["production_source_identity"]==current
        and old["staged_source_identity"]==expected and old["proposal_binding"]==proposal["binding"],"Retained staged identity")
    require(old["original_failures_passed"]==18 and old["controls_passed"]==6 and old["guards_passed"]==6
        and old["completed"]==31 and old["halt"] is None,"Exact original completion")
    tool=ROOT/"scripts/baseline/benchmarks/tools/audit_operator_shape_capture_prototype.py"
    require(sha(tool)==old["tool_sha256"],"Original probe tool retained")
    guard_files=ROOT/"scripts/baseline/benchmarks/tools/fixtures/operator-shape-rejections-v1"
    inventory=json.loads((guard_files/"inventory.json").read_text())
    require(all(sha(guard_files/p)==h for p,h in inventory["files"].items()),"Original negative fixture files retained")
    retained={str(p):sha(p) for p in previous.parent.rglob("*") if p.is_file()}
    require(all(sha(p)==h for p,h in old["preserved_inventory"].items()),"Original source/rejection evidence preserved")
    failed=[r for r in old["guards"] if r["status"]=="guard_failed"]
    require(len(failed)==1 and failed[0]["id"]=="adaptive_upsample" and failed[0]["exit"]==1,"Only test diagnostic mismatch")
    previous_log=previous.parent/"captures/guard_adaptive_upsample/capture.log"
    require(sha(previous_log)==failed[0]["log_sha256"],"Original rejected execution retained")
    original_expected="ValueError: Adaptive pool N,C,spatial downsampling shape"
    corrected="ValueError: node fx_adaptive_avg_pool1d (adaptive_avg_pool): Adaptive pool N,C,spatial downsampling shape"
    case=next(c for c in inventory["cases"] if c["id"]=="adaptive_upsample")
    require(case["expected_terminal"]==original_expected and failed[0]["terminal"]==corrected,"Specific node-qualified correction")
    manifest=Path(case["manifest"]);spec,paths=prepare(manifest)
    require(spec["weights"]=={},"No guard weight changes")
    output.mkdir(mode=0o700);started=time.monotonic()
    code=sandbox(spec,paths,{},output/"capture",expected,prototype,60)
    log=(output/"capture/capture.log").read_text();terminal=log.splitlines()[-1]
    require(code==1 and terminal==corrected and "/app/model_decomposition.py" in log,"Actual downsampling refusal with location")
    require(frozen(proposal)==bound and source_identity()==current and validate_prototype(prototype)[2]==expected
        and all(sha(p)==h for p,h in retained.items()),"Frozen scope and previous report unchanged")
    for row in old["guards"]:
        require(sha(previous.parent/"captures"/("guard_"+row["id"])/"capture.log")==row["log_sha256"],"Retained guard evidence")
    report=dict(format="poseidon-operator-shape-guard-continuation-v1",previous_report_binding=old["binding"],
        prototype_binding=staged["binding"],proposal_binding=proposal["binding"],tool_sha256=sha(__file__),
        original_test_mismatch=dict(id="adaptive_upsample",expected=original_expected,actual=failed[0]["terminal"]),
        corrected_expected=corrected,rechecked=1,rechecked_passed=1,retained_guards_passed=6,total_guard_evidence=7,
        worker_exit=code,worker_log_sha256=sha(output/"capture/capture.log"),seconds=time.monotonic()-started,
        previous_file_hashes=retained,production_integrated=False,production_changes=0,backend_changes=0,
        reference_changes=0,hecate_tracing=0,compiled=0,encrypted_execution=0,decrypted=0,paid_calls=0,
        downloads=0,installs=0,agent_generated=0,scope="Independent refusal probe; test expectation fix only")
    report["binding"]=digest(report);(output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:report[k] for k in ("binding","rechecked_passed","total_guard_evidence","seconds")}))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("previous","prototype","proposal","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.previous,a.prototype,a.proposal,a.output))
