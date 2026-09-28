"""Synthetic trace-evidence gate counterexamples, never real frontend results."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest

def fixtures(ledger):
    groups={}
    for task in ledger["directed_tasks"]:groups.setdefault(task["requirement"],[]).append(task)
    out=[]
    for req in ledger["requirements"]:
        if req["layer"]!="construction":continue
        tasks=groups.get(req["id"],[])
        if len(tasks)!=3:raise ValueError("Construction must have three bound contexts")
        task=tasks[0];required=task["required_features"]
        structural=task["structural_features"];numeric=[f for f in required if f not in structural]
        checked=dict(id=task["exercise"],numeric_features=numeric,structural_features=structural,witnesses={})
        profile=task["profile"]
        if profile=="hecate-unified-public-v1":
            events=[]
            for i,f in enumerate(required):
                span=[10+i,0,10+i,8]
                checked["witnesses"][f]=dict(span=span)
                events.append(dict(features=[f],span=span))
            checked.update(normalized_sha256="a"*64,events_sha256=digest(events))
            record=dict(normalized_sha256=checked["normalized_sha256"],candidate_python_executed=False,events=events)
            changed=copy.deepcopy(checked);changed["witnesses"][required[0]]["span"]=[1000000,0,1000000,8]
            reason="Missing isolated public event: "+required[0]
            checker="unified_public_coverage.verify_trace_coverage"
        elif profile=="hecate-unified-native-v1":
            record=dict(storage=[],star=[],calls=[],augmented=[],mutation=[])
            for i,f in enumerate(required):
                trace=dict(kind="call",caller="golden",callee="fixture"+str(i),span=[10+i,0,10+i,8])
                witness=dict(trace=trace,influencing_cells=[0] if f in numeric else [])
                if f=="call.repeated":
                    witness["additional_traces"]=[copy.deepcopy(trace)]
                    record["calls"].append({k:v for k,v in trace.items() if k!="kind"})
                checked["witnesses"][f]=witness
                record["calls"].append({k:v for k,v in trace.items() if k!="kind"})
            changed=copy.deepcopy(checked)
            changed["witnesses"][required[0]]["trace"]["span"]=[1000000,0,1000000,8]
            reason="Missing real native trace witness: "+required[0]
            checker="unified_native_coverage.verify_trace_coverage"
        else:raise ValueError("Unbound construction profile")
        case=dict(id="trace_gate_"+digest(req["id"])[:16],requirement=req["id"],profile=profile,
                  applicable_contract=profile,related_positive_task_ids=[t["id"] for t in tasks],
                  checker=checker,positive_checked=checked,negative_checked=changed,record=record,
                  expected_rejection=reason,
                  evidence_scope="Synthetic trace-to-witness consistency only; no AST contribution, real tracing or encrypted execution",
                  actual_frontend_executed=False,encrypted_execution=False)
        case["fixture_sha256"]=digest(case);out.append(case)
    return out

def verify(case):
    if case["profile"]=="hecate-unified-public-v1":
        from unified_public_coverage import verify_trace_coverage
    else:
        from unified_native_coverage import verify_trace_coverage
    # The function's positive return contains a frontend-checked flag intended
    # for its trusted caller. NEVER propagate it as this synthetic test's claim.
    verify_trace_coverage(copy.deepcopy(case["positive_checked"]),copy.deepcopy(case["record"]))
    try:verify_trace_coverage(copy.deepcopy(case["negative_checked"]),copy.deepcopy(case["record"]))
    except ValueError as error:
        if str(error)!=case["expected_rejection"]:raise ValueError("Rejected for unrelated reason: "+str(error))
    else:raise ValueError("Missing synthetic trace witness was accepted")
    return dict(id=case["id"],requirement=case["requirement"],fixture_sha256=case["fixture_sha256"],
                positive_gate_passed=True,negative_gate_rejected=True,
                observed_rejection=case["expected_rejection"],
                actual_frontend_executed=False,encrypted_execution=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    ledger_path=a.suite/"coverage.json"
    index=json.loads((a.suite/"index.json").read_text())
    if hashlib.sha256(ledger_path.read_bytes()).hexdigest()!=index["coverage_sha256"]:
        raise ValueError("Changed source ledger")
    ledger=json.loads(ledger_path.read_text());cases=fixtures(ledger)
    from semantic_benchmark_execution import runtime_sources
    sources=runtime_sources()
    plan=dict(kind="synthetic_construction_trace_gate_tests",coverage_sha256=index["coverage_sha256"],
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes=sources,
              fixture_hashes={c["id"]:c["fixture_sha256"] for c in cases},
              planned=len(cases),max_wall_seconds=120,paid_calls=0,actual_frontend_executed=False,encrypted_execution=False)
    plan["binding"]=digest(plan)
    if not a.execute:
        print(json.dumps({k:v for k,v in plan.items() if k not in ("source_hashes","fixture_hashes")}))
        return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output is None or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform results output required")
    if a.output.exists():p.error("Preserve existing test evidence")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=120)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    a.output.mkdir(parents=True)
    def dump(name,value):(a.output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    dump("plan.json",plan);dump("fixtures.json",cases)
    (a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    results=[]
    for case in cases:
        try:result=dict(verify(case),state="passed")
        except (ValueError,KeyError,TypeError,AssertionError) as error:
            result=dict(id=case["id"],state="failed",error=str(error),actual_frontend_executed=False,encrypted_execution=False)
        results.append(result);dump("results.json",results)
    if runtime_sources()!=sources:raise ValueError("Source changed during gate tests")
    report=dict(binding=plan["binding"],planned=len(cases),passed=sum(r["state"]=="passed" for r in results),
                failed=sum(r["state"]=="failed" for r in results),skipped=0,paid_calls=0,
                actual_frontend_executed=False,encrypted_execution=False,
                evidence_scope="Synthetic checker units, not semantic contribution or real FHE coverage",
                results_sha256=hashlib.sha256((a.output/"results.json").read_bytes()).hexdigest(),
                fixtures_sha256=hashlib.sha256((a.output/"fixtures.json").read_bytes()).hexdigest())
    dump("report.json",report);print(json.dumps(report))
    return int(report["failed"]>0)
if __name__=="__main__":raise SystemExit(main())
