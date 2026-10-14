"""Read-only report and source verification, without Nix or backend execution."""
import hashlib,json
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def main():
    path=ROOT/"docs/baseline/agent-resource-gelu-progress-r1305.json"
    report=json.loads(path.read_bytes())
    assert report["binding"]==hashlib.sha256(canonical({k:v for k,v in report.items() if k!="binding"})).hexdigest(),"progress binding"
    for name,row in report["source_reports"].items():
        assert sha(Path(row["path"]))==row["sha256"],"report changed:"+name
    for item in report["following_work"]["changed_agent_files"]:
        assert sha(BASE/item["path"])==item["sha256"],"agent source changed:"+item["path"]
    counts={}
    stage=report["following_work"]["final_integrity_guard_cpu"]["stage_identity"]
    for name in ["generation","compilation","validation"]:
        sources=stage[name]["sources"]
        assert all(sha(BASE/p)==h for p,h in sources.items()),"stage source changed:"+name
        counts[name]=len(sources)
    protected=json.loads((BASE/"benchmarks/tools/source_live_rebind_v2/proposal-ready.json").read_bytes())["protected_sources"]
    assert all(sha(ROOT/p)==h for p,h in protected.items()),"protected source changed"
    proposal=json.loads((ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json").read_bytes())
    assert proposal["binding"]==hashlib.sha256(canonical({k:v for k,v in proposal.items() if k!="binding"})).hexdigest(),"proposal binding"
    for row in proposal["cases"]:
        assert sha(ROOT/row["source_manifest"])==row["source_manifest_sha256"]
        assert sha(ROOT/row["full_request_file"])==row["full_request_sha256"]
        assert sha(ROOT/row["generation_context_file"])==row["generation_context_sha256"]
    print(json.dumps(dict(status="verified",report_binding=report["binding"],stage_source_counts=counts,
        protected_sources=len(protected),gelu_live_proposal=proposal["status"],api=0,compiler=0,encrypted_execution=0)))
if __name__=="__main__":main()
