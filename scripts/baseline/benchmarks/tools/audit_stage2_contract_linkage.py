"""Current candidate-contract linkage and immutable legacy evidence; not fresh FHE."""
import argparse,hashlib,json,os,shlex,sys,time
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file,load
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):
    d=strict_file(p,8*1024**2)
    if digest({k:v for k,v in d.items() if k!="binding"})!=d["binding"]:raise ValueError("Evidence binding")
    return d
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    if a.output.exists():p.error("Preserve evidence")
    from hecate_python_env import VENV,enter_nix
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned environment")
    from workspace_paths import RESULTS
    from semantic_benchmark_execution import runtime_sources,preflight
    from unified_graph_contract import validate_request,validate_candidate
    from platform_config import require_python_packages
    import numpy as np,torch
    require_python_packages(torch,np);torch.set_num_threads(1)
    start=time.monotonic();sources=runtime_sources()
    suite=BASE/"benchmarks/semantic-v2-ledger-r38"
    models,index=load(suite,check_sources=False) # frozen bytes/signatures/split checks remain mandatory
    ledger=strict_file(suite/"coverage.json",4*1024**2)
    for component in ledger["components"].values():
        if sha(suite/component["file"])!=component["sha256"]:raise ValueError("Ledger component drift")
    requests=preflight(models,need_rule=False)
    parents={};rows=[]
    directed=RESULTS/"benchmark-r43-directed-independent";report=strict_file(directed/"report.json",8*1024**2)
    legacy_plan=strict_file(directed/"plan.json",8*1024**2)
    if report["binding"]!=legacy_plan["binding"] or report["remaining"]!=0:raise ValueError("Incomplete legacy audit plan")
    parents[str(directed/"plan.json")]=sha(directed/"plan.json")
    parents[str(directed/"report.json")]=sha(directed/"report.json")
    for name,hsh in report["record_hashes"].items():
        if Path(name).name!=name or sha(directed/name)!=hsh:raise ValueError("Directed audit drift")
        record=strict_file(directed/name,8*1024**2)
        rows.append(dict(id=record["task_id"],partition=record["requirement"],evidence=record["evidence"],
                         acceptance_kind=record["acceptance_kind"],historical_audit_sha256=hsh,kind="construction",
                         record=record))
    helperpath=ROOT/"docs/baseline/stage2-helper-mapping-r49.json";helpers=bound(helperpath)
    parents[str(helperpath)]=sha(helperpath)
    for task,record in helpers["records"].items():
        rows.append(dict(id=task,evidence=record["evidence"],kind="helper",record=record))
    for row in rows:
        if time.monotonic()-start>240:raise TimeoutError("Contract audit budget")
        folder=Path(row["evidence"]);record=row.pop("record")
        if folder.parent!=RESULTS:raise ValueError("Evidence location")
        hashes=record.get("files",{})
        for name,hsh in hashes.items():
            rel=Path(name);p=folder/rel
            if rel.is_absolute() or ".." in rel.parts or p.is_symlink() or sha(p)!=hsh:raise ValueError("Historical artifact changed")
        if "report_sha256" in record and sha(folder/"report.json")!=record["report_sha256"]:raise ValueError("Historical helper report changed")
        request=strict_file(folder/"request.json",8*1024**2)
        current=strict_file(folder/"report.json",8*1024**2)
        if current["status"]!="passed":raise ValueError("Nonpassed legacy evidence")
        candidate=(folder/"attempt-00/candidate.py").read_text()
        row.update(request_sha256=sha(folder/"request.json"),candidate_sha256=sha(folder/"attempt-00/candidate.py"),
                   original_request_id=request["request_id"],fresh_encrypted_execution=False)
        try:
            validate_request(request)
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=candidate),request)
            row["current_contract"]="accepted"
        except ValueError as error:row.update(current_contract="rejected",diagnostic=str(error))
    if runtime_sources()!=sources:raise ValueError("Source changed")
    before=strict_file(RESULTS/"stage2-polynomial-contract-r87/before-sources.json",8*1024**2)
    changes={name:dict(before=before.get(name),after=hsh) for name,hsh in sources.items() if before.get(name)!=hsh}
    expected={"scripts/baseline/"+n for n in ("run_candidate.py","candidate_sandbox.py","unified_graph_contract.py",
                                             "upstream_candidate_helpers.py","upstream_helper_coverage.py","upstream_adapters/fixed_polynomial.py")}
    if set(changes)!=expected or set(before)-set(sources):raise ValueError("Unexpected production scope")
    result=dict(format="poseidon-stage2-contract-linkage-v1",parents=parents,
                corpus=dict(models=len(models),unique_signatures=index["unique_signatures"],
                            topology_groups=index["topology_groups"],small_models=index["small_models"],split_counts=index["split_counts"]),
                free_request_preflight=dict(counts=dict(Counter(r["status"] for r in requests)),
                                            blocked=[r for r in requests if r["status"]!="ready"],requires_deterministic_DSL=False),
                legacy_candidates=rows,legacy_counts=dict(Counter((r["kind"]+":"+r["current_contract"]) for r in rows)),
                changes=changes,source_hashes=sources,current_source_sha256=digest(sources),runner_sha256=sha(Path(__file__)),
                all_legacy_contracts_accepted=all(r["current_contract"]=="accepted" for r in rows),
                paid_calls=0,new_encrypted_executions=0,seconds=time.monotonic()-start,
                offline_stage_complete=False,agent_stage_complete=False,
                limitation="Current contract acceptance plus unchanged retained evidence; not a rerun of historical FHE or full public API closure")
    result["binding"]=digest(result);dump(a.output,result)
    print(json.dumps({k:result[k] for k in ("corpus","free_request_preflight","legacy_counts","all_legacy_contracts_accepted","current_source_sha256","seconds","binding")}))
    return int(not result["all_legacy_contracts_accepted"])
if __name__=="__main__":raise SystemExit(main())
