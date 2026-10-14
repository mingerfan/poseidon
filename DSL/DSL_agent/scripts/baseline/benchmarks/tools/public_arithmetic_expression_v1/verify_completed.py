"""Read-only verification of frozen acceptance identities and retained evidence."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from benchmark_graph import digest,require
from offline_cpu_continue import read,sha,verify,PROPOSAL
from hecate_python_env import ROOT
from candidate_bundle import check_bundle
def main():
    plan=read(PROPOSAL);identity=verify(plan)
    result=read(ROOT/"docs/baseline/agent-public-arithmetic-offline-results-r1386.json")
    require(result["binding"]==digest({k:v for k,v in result.items() if k!="binding"}),"completed_report_binding")
    require(result["sdk_identity"]==identity["manifest_sha256"],"completed_report_identity")
    report=read(result["batch_report"])
    require(report["binding"]==result["batch_report_binding"] and report["status"]=="complete","completed_batch_binding")
    for name,expected in result["inspection_files"].items():
        require(sha(name)==expected,"completed_evidence_changed")
    for row in result["completed"]:
        require(check_bundle(row["package"])["binding"]==row["package_binding"],"completed_package_changed")
    print(json.dumps(dict(status="verified",files=len(result["inspection_files"]),
        package_reloaded=len(result["completed"]),binding=result["binding"],
        new_compiler_calls=0,new_encrypted_calls=0,provider_calls=0),sort_keys=True))
    return 0
if __name__=="__main__":sys.exit(main())
