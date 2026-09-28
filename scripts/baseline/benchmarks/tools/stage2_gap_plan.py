"""Produce a hash-bound, read-only deficit plan from historical verified evidence.

Never credits old results to a new runtime, removes failed tasks, or auto-executes.
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(audit,ledger):
    body={k:v for k,v in audit.items() if k!="binding"}
    if digest(body)!=audit["binding"]:raise ValueError("Historical closure audit changed")
    deficits=[]
    for partition in audit["math_partitions"]:
        if partition["verified_contexts"]>=3:continue
        deficits.append(dict(kind="mathematical_context",requirement=partition["id"],
            verified=partition["verified_contexts"],required=3,
            unresolved_tasks=[t for t in partition["tasks"] if t["state"]!="verified"],
            next_action="Inspect same model's existing helper evidence or repair lowering; preserve original failed task"))
    # Classification labels are not proof of execution or per-branch acceptance.
    noncandidate={"trusted_framework_io","frontend_metaclass_and_operator_factory",
        "defined_but_unregistered_inplace","trusted_framework_introspection","frontend_expression_handle",
        "plaintext_diagnostic_utility","mutable_maximum_hook"}
    api=[]
    for row in ledger["upstream_api"]:
        if row["backend_blocker"]:
            state="backend_blocked";action="Retain actual unsupported implementation; never substitute decrypt/re-encrypt"
        elif row["role"] in noncandidate:
            state="outside_candidate_computation";action=row["source_review"]
        else:
            state="execution_partition_mapping_pending";action="Map all relevant behavior partitions to real three-context evidence; add only missing tasks"
        api.append(dict(source=row["source"],symbol=row["symbol"],source_sha256=row["source_sha256"],
                        role=row["role"],disposition=state,reason=action))
    cases=sorted({t["model_id"] for d in deficits for t in d["unresolved_tasks"]})
    report=dict(format="poseidon-stage2-gap-plan-v1",historical_runtime_sha256=audit["historical_runtime_sha256"],
        verified_directed_tasks=audit["directed_verified"],
        verified_directed_partitions=audit["directed_three_context_partitions"],
        verified_math_partitions=audit["math_three_context_partitions"],mathematical_deficits=deficits,
        original_failed_and_blocked_tasks_preserved=True,
        api_dispositions=api,api_counts=dict(Counter(x["disposition"] for x in api)),
        proposed_model_shards=[cases[i:i+48] for i in range(0,len(cases),48)],
        execution_policy=dict(default="plan_only",concurrency=1,max_concurrency=2,max_shard_size=48,
                              replay_only_same_binding=True,compiler_jobs=2,link_jobs=1),
        offline_stage_complete=False,agent_stage_complete=False,paid_calls=0)
    report["binding"]=digest(report);return report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path);a=p.parse_args()
    suite=BASE/"benchmarks/semantic-v2-ledger-r38"
    audit=strict_file(a.audit,4*1024**2);ledger=strict_file(suite/"coverage.json",4*1024**2)
    if sha(suite/"coverage.json")!=audit["ledger_sha256"]:raise ValueError("Different coverage ledger")
    result=build(audit,ledger)
    result["provenance"]=dict(audit_sha256=sha(a.audit),ledger_sha256=sha(suite/"coverage.json"),
                             runner_sha256=sha(Path(__file__)))
    result["binding"]=digest({k:v for k,v in result.items() if k!="binding"})
    if a.output:
        if a.output.exists():p.error("Preserve existing plan")
        a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("mathematical_deficits","api_dispositions")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
