"""Explain historical evidence reuse by exact source and reviewed AST deltas."""
import argparse,ast,copy,hashlib,json,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
from semantic_benchmark_execution import runtime_sources
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump_ast(node):return ast.dump(node,include_attributes=False)
def need(ok,why):
 if not ok:raise ValueError(why)
def expression(source):return ast.parse(source,mode="eval").body
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Preserve earlier audit")
 from workspace_paths import RESULTS
 parent=RESULTS/"benchmark-r43-directed-independent/plan.json";old_plan=strict_file(parent,8*1024**2)
 old=old_plan["runtime_sources"];now=runtime_sources()
 changed={k for k in old if now.get(k)!=old[k]};added=set(now)-set(old);removed=set(old)-set(now)
 expected_changed={"scripts/baseline/run_candidate.py","scripts/baseline/unified_graph_lowering.py","scripts/baseline/test_packed_prefix_lowering.py"}
 expected_added={"scripts/baseline/"+n+".py" for n in ("test_candidate_bundle","test_model_decomposition","model_import_cli","restricted_model_python","candidate_bundle","model_decomposition","operator_reference","test_nested_nix_launcher")}
 need(changed==expected_changed and added==expected_added and not removed,"Unreviewed runtime change")
 archive=RESULTS/"benchmark-r30-resource-pilot/source.tar.gz";review=[]
 with tarfile.open(archive) as tar:
  for name in sorted(changed-{"scripts/baseline/test_packed_prefix_lowering.py"}):
   members=[m for m in tar.getmembers() if m.isfile() and m.name.lstrip("./")==name and m.size<2*1024**2]
   need(len(members)==1,"Ambiguous archived source")
   previous=tar.extractfile(members[0]).read()
   need(hashlib.sha256(previous).hexdigest()==old[name],"Historical source does not match original binding")
   before=ast.parse(previous);after=ast.parse((ROOT/name).read_text());edits=0
   if name.endswith("run_candidate.py"):
    condition=expression('os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV')
    expected_return=ast.parse("return inside(args)").body[0]
    for node in ast.walk(after):
     for field in ("body","orelse","finalbody"):
      values=getattr(node,field,None)
      if isinstance(values,list):
       keep=[]
       for item in values:
        if isinstance(item,ast.If) and dump_ast(item.test)==dump_ast(condition):
         need(len(item.body)==1 and dump_ast(item.body[0])==dump_ast(expected_return) and not item.orelse,"Unreviewed launcher branch")
         edits+=1
        else:keep.append(item)
       setattr(node,field,keep)
    scope="Only exact pinned pure-shell reuse; same inside(args), credential cleanup remains unchanged. Fake-key branch regression is not paid API evidence."
   else:
    current=expression('str(error) not in {"Expanded ciphertext operation budget exceeded","Deterministic baseline operation budget","AST node limit"} or request["layout"]["execution_abi"]!="unified-periodic-inputs-v1"')
    previous_condition=expression('str(error)!="Expanded ciphertext operation budget exceeded" or request.get("construction_profile")!=PUBLIC or request["layout"]["execution_abi"]!="unified-periodic-inputs-v1"')
    for node in ast.walk(after):
     if isinstance(node,ast.If) and dump_ast(node.test)==dump_ast(current):
      node.test=previous_condition;edits+=1
    for tree in (before,after):
     for node in ast.walk(tree):
      if isinstance(node,ast.FunctionDef) and node.name=="candidate_source":
       need(isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str),"Expected reviewed docstring")
       node.body=node.body[1:]
    scope="Only known rejection-trigger fallback widened; normal successful generation and existing fallback arithmetic unchanged. Newly reachable paths have separate r48/r58/r78 evidence."
   need(edits==1 and dump_ast(before)==dump_ast(after),"Additional executable source changes")
   review.append(dict(path=name,before_sha256=old[name],after_sha256=now[name],reviewed_changes=edits,scope=scope))
 from python_compiler_smoke import BUILD
 binary_paths={"frontend_sha256":BUILD/"lib/libHecateFrontend.so","runtime_sha256":BUILD/"lib/libSEAL_HEVM.so"}
 need(set(old_plan["binaries"])==set(binary_paths),"Unreviewed historical binary key")
 for key,path in binary_paths.items():need(sha(path)==old_plan["binaries"][key],"Historical frontend/runtime changed")
 regression_path=RESULTS/"benchmark-r50-revision-regression/report.json";regression=strict_file(regression_path,4*1024**2)
 need(regression["runtime_sources"]==now and regression["failed"]==0,"Affected regression binding")
 need(sha(regression_path.parent/"unittest.log")==regression["log_sha256"],"Regression log changed")
 report=dict(format="poseidon-stage2-source-impact-v1",current_runtime_sha256=digest(now),
  historical_runtime_sha256=digest(old),historical_plan_sha256=sha(parent),
  historical_source_archive=str(archive),archive_sha256=sha(archive),
  unchanged_runtime_files=len(set(old)-changed),reviewed_changes=review,
  added_files=sorted(added),changed_test_files=["scripts/baseline/test_packed_prefix_lowering.py"],
  unchanged_frontend_contracts=True,unchanged_reference_and_sandbox=True,unchanged_frontend_and_runtime_binaries=True,
  regression=dict(path=str(regression_path),sha256=sha(regression_path),passed=regression["passed"],failed=0,skipped=regression["skipped"]),
  old_passes_remain_historical=True,current_runtime_fhe_reexecution_claimed=False,
  new_encrypted_executions=0,paid_calls=0,runner_sha256=sha(Path(__file__)))
 report["binding"]=digest(report);dump(a.output,report)
 print(json.dumps(dict(unchanged_runtime_files=report["unchanged_runtime_files"],reviewed_code_deltas=len(review),added_files=len(added),old_passes_rebound=False)))
 return 0
if __name__=="__main__":raise SystemExit(main())
