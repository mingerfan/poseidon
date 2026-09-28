"""Finish the frozen r29 evidence audit without rerunning encrypted cases."""
import argparse, hashlib, json, os, shlex, sys, time
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from semantic_benchmark_execution import runtime_sources
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_records(folder):
    report=strict_file(folder/"report.json",4*1024**2)
    for name,h in report["record_hashes"].items():
        if Path(name).name!=name or sha(folder/name)!=h:raise ValueError("Audit record integrity")
    if report["remaining"]!=0:raise ValueError("Incomplete independent audit")
    return report
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    from workspace_paths import RESULTS
    from hecate_python_env import enter_nix,VENV
    if not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform result directory required")
    if a.output.exists():p.error("Preserve prior audit")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=930)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment required")
    import numpy,torch
    from platform_config import require_python_packages
    require_python_packages(torch,numpy);torch.set_num_threads(1)
    from audit_unified_candidate import verify_candidate
    sources=runtime_sources();start=time.monotonic()
    base=RESULTS/"benchmark-r41-independent-audit"
    directed=RESULTS/"benchmark-r43-directed-independent"
    br=verify_records(base);dr=verify_records(directed)
    for folder in (base,directed):
        if strict_file(folder/"plan.json",8*1024**2)["runtime_sources"]!=sources:
            raise ValueError("Runtime changed before historical audit")
    suite=BASE/"benchmarks/semantic-v2-ledger-r38"
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location("enrich_semantic_ledger",Path(__file__).with_name("enrich_semantic_ledger.py"))
    mod=module_from_spec(spec);sys.path.insert(0,str(Path(__file__).parent));spec.loader.exec_module(mod)
    mod.verify_release(suite)
    ledger=strict_file(suite/"coverage.json",4*1024**2)
    contexts=strict_file(suite/"semantic-component-math_contexts.json",4*1024**2)
    passed={Path(n).stem:strict_file(base/n,8*1024**2)["model_sha256"] for n in br["record_hashes"]}
    a.output.mkdir(parents=True)
    supplement={};blocked={};input_hashes={str(base/"report.json"):sha(base/"report.json"),str(directed/"report.json"):sha(directed/"report.json")}
    for label in ("model-contexts-baseline-000","model-contexts-baseline-001"):
        folder=RESULTS/label;report=strict_file(folder/"report.json",4*1024**2)
        input_hashes[str(folder/"report.json")]=sha(folder/"report.json")
        if report["not_run_selected"]!=0:raise ValueError("Incomplete supplement")
        for b in report["blocked_in_window"]:blocked[b["id"]]=b["reason"]
        for row in report["reports"]:
            if time.monotonic()-start>900:raise TimeoutError("Audit wall budget")
            if runtime_sources()!=sources:raise ValueError("Runtime drift")
            evidence=Path(row["evidence"])
            if sha(evidence/"report.json")!=row["sha256"]:raise ValueError("Supplement report drift")
            checked=verify_candidate(evidence)
            if row["id"] in supplement:raise ValueError("Duplicate supplement")
            dump(a.output/(row["id"]+".json"),checked)
            supplement[row["id"]]=checked["model_sha256"]
    passed.update(supplement)
    by={}
    for task in contexts["tasks"]:
        good=passed.get(task["model_id"])==task["model_sha256"]
        by.setdefault(task["requirement"],[]).append(dict(task_id=task["id"],model_id=task["model_id"],
            topology=task["topology"],state="verified" if good else "missing_execution_evidence",
            reason=blocked.get(task["model_id"])))
    math_rows=[dict(id=k,verified_contexts=len({x["topology"] for x in v if x["state"]=="verified"}),
                    tasks=v) for k,v in by.items()]
    # API source classification is deliberately not promoted to execution coverage.
    api=[]
    for r in ledger["upstream_api"]:
        state=("backend_blocked" if r["backend_blocker"] else
               "source_classified_execution_mapping_pending")
        api.append(dict(source=r["source"],symbol=r["symbol"],role=r["role"],
                        state=state,reason=r["backend_blocker"] or r["source_review"]))
    result=dict(schema=1,kind="stage2_evidence_gap_audit",historical_runtime_sha256=digest(sources),
        runner_sha256=sha(Path(__file__)),ledger_sha256=sha(suite/"coverage.json"),
        input_reports=input_hashes,baseline_verified=br["independently_verified"],
        directed_verified=dr["independently_verified"],directed_three_context_partitions=len(dr["requirements_three_verified_contexts"]),
        directed_numeric=dr["numeric_counts"],directed_mixed=dr["mixed_counts"],directed_structural=dr["structural_counts"],
        supplemental_verified=len(supplement),supplemental_blocked=blocked,
        math_partitions=math_rows,math_three_context_partitions=sum(x["verified_contexts"]>=3 for x in math_rows),
        upstream_api=api,upstream_api_counts=dict(Counter(x["state"] for x in api)),
        offline_stage_complete=False,agent_evaluation_complete=False,paid_calls=0,new_encrypted_executions=0,
        seconds=time.monotonic()-start)
    result["binding"]=digest(result);dump(a.output/"report.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("math_partitions","upstream_api","input_reports")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
