"""Bounded helper call-omission negatives over the frozen directed fixtures.

AST/finite-probe validation only: no candidate Python, Hecate tracing, compilation,
encrypted execution, private test inputs or paid provider calls.
"""
import argparse
import ast
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

def omit_calls(source,callee):
    class Omit(ast.NodeTransformer):
        def __init__(self):self.count=0
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id==callee:
                if not node.args or isinstance(node.args[0],ast.Starred):
                    raise ValueError("Unsupported negative-fixture argument form")
                self.count+=1
                return ast.copy_location(copy.deepcopy(node.args[0]),node)
            return node
    transform=Omit();tree=transform.visit(ast.parse(source))
    if not transform.count:raise ValueError("Required helper absent from positive source")
    result=ast.unparse(ast.fix_missing_locations(tree))+"\n"
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==callee
           for n in ast.walk(ast.parse(result))):raise ValueError("Helper omission incomplete")
    return result,transform.count

def counterexamples(tasks):
    """One negative per immutable helper binding, including every output chunk."""
    out=[]
    for task in tasks:
        for target in sorted(task["required_helpers"]):
            out.append(dict(id=task["id"]+"__omit_"+target,task=task,target=target))
    if len({r["id"] for r in out})!=len(out):raise ValueError("Duplicate helper negative")
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--shard-index",type=int,default=0);p.add_argument("--execute",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    index=json.loads((a.suite/"index.json").read_text())
    raw=(a.suite/"coverage.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=index["coverage_sha256"]:raise ValueError("Frozen ledger hash")
    frozen=json.loads(raw)["helper_directed_tasks"]
    definitions=counterexamples(frozen)
    if not 0<=a.shard_index<(len(definitions)+47)//48:p.error("Invalid helper gate shard")
    selected=definitions[a.shard_index*48:(a.shard_index+1)*48]
    from semantic_benchmark_execution import runtime_sources
    sources=runtime_sources()
    plan=dict(schema=1,kind="helper_AST_call_omission_counterexamples",
              tasks=[dict(id=t["id"],task_id=t["task"]["id"],task_sha256=t["task"]["task_sha256"],omitted_helper=t["target"]) for t in selected],
              total_frozen_tasks=len(frozen),total_counterexamples=len(definitions),shard_index=a.shard_index,
              coverage_sha256=index["coverage_sha256"],source_hashes=sources,
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              max_wall_seconds=180,paid_calls=0,actual_frontend_executed=False,encrypted_execution=False)
    plan["binding"]=digest(plan)
    if not a.execute:
        print(json.dumps({k:v for k,v in plan.items() if k not in ("source_hashes","tasks")}|{"selected":len(selected)}))
        return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output is None or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform result path required")
    if a.output.exists():p.error("Preserve existing evidence")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=180)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    import numpy,torch
    from platform_config import require_python_packages
    require_python_packages(torch,numpy)
    from upstream_helper_directed_cases import cases,tasks
    from upstream_candidate_helpers import BN_PROFILE
    from unified_graph_contract import prepare,validate_candidate
    from compiler_configuration import PROFILE_SHA256,configuration
    if tasks()!=frozen:raise ValueError("Helper generator/frozen task drift")
    by_id={"upstream_"+r["name"]:r for r in cases()}
    a.output.mkdir(parents=True)
    def dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    dump(a.output/"plan.json",plan);(a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    results=[]
    for definition in selected:
        task=definition["task"];target=definition["target"]
        row=by_id[task["id"]];directory=a.output/definition["id"];directory.mkdir()
        result=dict(id=definition["id"],task_id=task["id"],task_sha256=task["task_sha256"],state="not_run",
                    positive_contract_accepted=False,call_omission_rejected=False,
                    actual_frontend_executed=False,encrypted_execution=False)
        try:
            request=prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),
                            helper_profile=row.get("profile",BN_PROFILE),helper_exercise=row["required_helpers"],
                            chunk_period=row.get("chunk_period"))
            candidate=dict(schema=1,request_id=request["request_id"],hecate_source=row["source"])
            validate_candidate(candidate,request)
            result["positive_contract_accepted"]=True
            negative,count=omit_calls(row["source"],target)
            dump(directory/"model.json",row["model"]);dump(directory/"request.json",request)
            (directory/"positive.py").write_text(row["source"]);(directory/"negative.py").write_text(negative)
            expected="No reachable upstream helper: "+target
            try:validate_candidate(dict(candidate,hecate_source=negative),request)
            except ValueError as error:
                if str(error)!=expected:raise ValueError("Unrelated rejection: "+str(error))
            else:raise ValueError("Omitted helper was accepted")
            result.update(state="passed",call_omission_rejected=True,omitted_helper=target,
                          removed_calls=count,observed_rejection=expected,
                          positive_source_sha256=hashlib.sha256(row["source"].encode()).hexdigest(),
                          negative_source_sha256=hashlib.sha256(negative.encode()).hexdigest())
        except (ValueError,TypeError,KeyError,IndexError,AssertionError) as error:
            result.update(state="failed",error=str(error))
        results.append(result);dump(a.output/"results.json",results)
    if runtime_sources()!=sources:raise ValueError("Source drift during helper counterexamples")
    report=dict(binding=plan["binding"],planned=len(selected),passed=sum(r["state"]=="passed" for r in results),
                failed=sum(r["state"]=="failed" for r in results),skipped=0,paid_calls=0,
                actual_frontend_executed=False,encrypted_execution=False,
                scope="Real restricted candidate validator and finite helper probe; not real Hecate or numerical FHE",
                results_sha256=hashlib.sha256((a.output/"results.json").read_bytes()).hexdigest())
    dump(a.output/"report.json",report);print(json.dumps(report))
    return int(report["failed"]>0)
if __name__=="__main__":raise SystemExit(main())
