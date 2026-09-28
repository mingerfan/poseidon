"""Read-only current capability view. Historical scores never become current passes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest, require
from benchmark_runner import dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true")
    a=p.parse_args()
    require(not a.output.exists(),"Preserve existing report")
    from hecate_python_env import enter_nix,VENV
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output.resolve())]),seconds=600)
    from workspace_paths import RESULTS
    from component_contract import validate_program, reconstruct_request
    from candidate_contract import strict_json
    from semantic_benchmark_execution import runtime_sources
    sources=runtime_sources()
    parents={}
    def read(path,bound=False):
        path=Path(path);parents[str(path)]=sha(path);value=json.loads(path.read_text())
        if bound:require(value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}),"Report binding mismatch")
        return value
    suite=BASE/"benchmarks/semantic-v2-ledger-r38"
    index=read(suite/"index.json");ledger=read(suite/"coverage.json")
    require(index["coverage_sha256"]==sha(suite/"coverage.json"),"Frozen ledger hash")
    offline=read(RESULTS/"stage2-offline-revalidation-r152.json",True)
    agents=read(RESULTS/"stage2-distinct-contexts-r145.json",True)
    prior=read(ROOT/"docs/baseline/stage2-noncompiler-repairs-r160.json")
    old_by={r["id"]:r for r in offline["requirements"]}
    agent_by={r["id"]:r for r in agents["records"]}
    require(len(old_by)==len(ledger["requirements"])==401,"Semantic denominator")
    partitions=[]
    for row in ledger["requirements"]:
        old=old_by[row["id"]];generated=agent_by.get(row["id"],{})
        partitions.append(dict(id=row["id"],layer=row["layer"],scope=row["claim_scope"],
            contracts=row["applicable_contracts"],blocker=old["blocker"],
            historical_offline=dict(status=old["status"],report_binding=offline["binding"],
                evidence_source_bindings=sorted({c.get("source_binding","unspecified") for c in old["contexts"]})),
            historical_agent=dict(status=generated.get("status","not_in_agent_context_denominator"),
                report_binding=agents["binding"],runtime_source_sha256=agents["runtime_source_sha256"],
                passed_distinct_topologies=generated.get("passed_distinct_topologies"),
                minimum_three_topologies_verified=generated.get("minimum_three_topologies_verified",False)),
            current_full_partition_execution="not_revalidated",
            current_three_contexts_verified=False))
    checked=[]
    for row in prior["remaining_static_cases"]:
        raw=Path(row["original_response"])
        require(sha(raw)==row["original_response_sha256"],"Retained response changed")
        request=read(raw.parent.parent/"request.json")
        reconstruct_request(request)
        candidate=strict_json(raw.read_text())
        require(hashlib.sha256(candidate["hecate_source"].encode()).hexdigest()==row["source_sha256"],"Retained source changed")
        require(request["request_id"]==row["request_id"],"Retained request changed")
        result=validate_program(candidate,request)
        checked.append(dict(id=row["id"],request_id=request["request_id"],source_sha256=row["source_sha256"],
                            result=result,execution="not_run",new_agent_generation=False))
    require(runtime_sources()==sources,"Current checker changed")
    counts=Counter((r["result"]["semantic_validation"],r["result"]["construction_coverage"]) for r in checked)
    report=dict(format="poseidon-component-capability-report-v1",source_hashes=sources,parents=parents,
        semantic_partitions=401,upstream_api_classifications=len(ledger["upstream_api"]),
        partitions=partitions,retained_candidate_static_checks=checked,
        static_counts=[dict(semantic_validation=k[0],construction_coverage=k[1],count=v) for k,v in sorted(counts.items())],
        compiler_models_retained=prior["compiler_models"],new_compilations=0,new_encrypted_executions=0,new_paid_calls=0,
        current_full_coverage_rate=None,
        scope="Current static checks of retained failed answers; historical partition scores stay separate. No new numerical or Agent success.")
    report["binding"]=digest(report);dump(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("partitions","retained_candidate_static_checks","parents","source_hashes")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
