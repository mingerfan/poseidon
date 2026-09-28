"""Map internal helper return paths to immutable, contributed parent executions.

This maps only recorded bounded adapter contexts, not all branches/parameters.
"""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
LINKS={
 "shapeClosure.MultParBN":("HE_BN","BN"),
 "shapeClosure.MultParConv":("HE_Conv","MPC"),
 "shapeClosure.MultParConvBN":("HE_ConvBN","MPCB"),
 "shapeClosure.DwConvBN":("HE_DwConv","DW"),
 "shapeClosure.Downsamp":("HE_DS","DS"),
 "shapeClosure.AvgPool":("HE_Pool","AP"),
 "shapeClosure.AvgMidPool":("HE_Avg","MA"),
 "shapeClosure.Concat":("HE_Concat","CC"),
 "Linear":("HE_Linear",None),
 "BN":("HE_MPBN",None),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):
 r=strict_file(p,8*1024**2)
 if digest({k:v for k,v in r.items() if k!="binding"})!=r["binding"]:raise ValueError("Report binding")
 return r
def returned_call(node):
 last=node.body[-1]
 if not isinstance(last,ast.Return):raise ValueError("Missing unconditional return")
 value=last.value
 if isinstance(value,ast.Name):
  previous=node.body[-2]
  if not isinstance(previous,ast.Assign) or len(previous.targets)!=1 or not isinstance(previous.targets[0],ast.Name) or previous.targets[0].id!=value.id:
   raise ValueError("Returned value is not direct preceding call")
  value=previous.value
 if not isinstance(value,ast.Call):raise ValueError("Not a directly returned call")
 return value
def prove(function,closure,symbol,parent,key):
 call=returned_call(function)
 if key is None:
  good=isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=="MPCB" and call.func.attr==symbol
 else:
  good=isinstance(call.func,ast.Subscript) and isinstance(call.func.value,ast.Name) and call.func.value.id=="close" and isinstance(call.func.slice,ast.Constant) and call.func.slice.value==key
  last=closure.body[-1]
  if not isinstance(last,ast.Return) or not isinstance(last.value,ast.Dict):raise ValueError("Closure return not explicit dictionary")
  table={k.value:v.id for k,v in zip(last.value.keys,last.value.values) if isinstance(k,ast.Constant) and isinstance(v,ast.Name)}
  if table.get(key)!=symbol.split(".")[-1]:raise ValueError("Closure dispatch mismatch")
 if not good:raise ValueError("Parent return/callee mismatch")
 return dict(parent=parent,callee=symbol,closure_key=key,
  parent_return_line=function.body[-1].lineno,
  parent_ast_sha256=digest(ast.dump(function,include_attributes=False)),
  callee_result_is_parent_result=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Preserve prior audit")
 source=ROOT/"third_party/dacapo/python/poly/poly"
 func={n.name:n for n in ast.parse((source/"Func.py").read_text()).body if isinstance(n,ast.FunctionDef)}
 mpcb={n.name:n for n in ast.parse((source/"MPCB.py").read_text()).body if isinstance(n,ast.FunctionDef)}
 closure=mpcb["shapeClosure"]
 ledger=strict_file(BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json",4*1024**2)
 for f in ("Func.py","MPCB.py"):
  path=source/f
  if sha(path)!=ledger["upstream_source_hashes"][str(path.relative_to(ROOT))]:raise ValueError("Pinned helper drift")
 parent_report=ROOT/"docs/baseline/stage2-helper-mapping-r49.json";report=bound(parent_report)
 requirements={r["id"]:r for r in report["helper_partitions"]};rows=[];negative=0
 for symbol,(parent,key) in LINKS.items():
  proof=prove(func[parent],closure,symbol,parent,key)
  # A changed return or dispatch must never inherit the parent's numerical pass.
  for mutation in ("constant_return","wrong_dispatch","postprocess"):
   changed=copy.deepcopy(func[parent])
   if mutation=="constant_return":changed.body[-1].value=ast.Constant(0)
   elif mutation=="postprocess":changed.body[-1].value=ast.BinOp(changed.body[-1].value,ast.Add(),ast.Constant(1))
   else:
    call=returned_call(changed)
    call.func=ast.Name("different_helper",ast.Load())
   try:prove(changed,closure,symbol,parent,key)
   except ValueError:negative+=1
   else:raise ValueError("Invalid return-path proof accepted")
  required=requirements["helper."+parent];contexts=[];seen=set()
  if required["state"]!="three_contexts_verified":raise ValueError("Unverified parent")
  for task in required["task_ids"]:
   record=report["records"][task]
   if record["topology"] in seen:continue
   evidence=Path(record["evidence"])
   if sha(evidence/"report.json")!=record["report_sha256"]:raise ValueError("Historical candidate changed")
   original=strict_file(evidence/"report.json",8*1024**2)
   attempt=next(x for x in original["attempts"] if x.get("status")=="passed")
   out=evidence/("attempt-%02d"%attempt["index"])/"output"
   for name,h in attempt["artifact_hashes"].items():
    if Path(name).name!=name or sha(out/name)!=h:raise ValueError("Artifact drift")
   events_path=out/"upstream-calls.json"
   events=strict_file(events_path,8*1024**2)
   relevant=[r for r in events["bound_calls"] if r["actual"].get("helper")==parent or r["actual"].get("closure")=="MPCB.shapeClosure."+str(key)]
   if not relevant:raise ValueError("No actual bound parent trace")
   witnesses=attempt["upstream_helper_coverage"]["witnesses"]
   if not any(r["callee"] in witnesses for r in relevant):raise ValueError("No parent return contribution")
   contexts.append(dict(task=task,**record,parent_call_records_sha256=sha(events_path),
       accepted_scope="Inherited exact return value under recorded adapter parameters and projections only"))
   seen.add(record["topology"])
   if len(contexts)==3:break
  if len(contexts)!=3:raise ValueError("Missing three distinct parent contexts")
  rows.append(dict(symbol="MPCB."+symbol,state="internal_direct_return_mapped",proof=proof,contexts=contexts,
      branches_exhausted=False,direct_candidate_access=False,new_encrypted_execution=False,
      scope="Internal function return is the witnessed parent result; neither arbitrary arguments nor each inner operation is certified"))
 result=dict(format="poseidon-upstream-return-path-audit-v1",status="passed",mappings=rows,
   mapped_internal_symbols=len(rows),negative_proof_controls=negative,
   sources={str((source/f).relative_to(ROOT)):sha(source/f) for f in ("Func.py","MPCB.py")},
   parent_audit_sha256=sha(parent_report),parent_audit_binding=report["binding"],
   actual_compile=False,actual_ciphertext_execution=False,paid_calls=0,
   runner_sha256=sha(Path(__file__)))
 result["binding"]=digest(result);a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("mappings","sources")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
