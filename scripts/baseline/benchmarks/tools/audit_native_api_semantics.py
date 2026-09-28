"""Bind actual native-call partitions to pinned Func APIs and real rejections.

Existing FHE results retain their historical identity. New isolated checks test
actual frontend rejection, never mocks or claimed new encrypted execution.
"""
import argparse,hashlib,json,os,shlex,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_runner import strict_file,dump
from benchmark_graph import digest
PARTITIONS=("call.forward","call.identity","call.nested","call.nested_array",
 "call.plain_return","call.public_argument","call.repeated","call.scalar_cipher",
 "call.tuple_multi","call.two_cipher_arguments","call.zero_arguments","call.empty_return",
 "native.call","native.return","return.empty","return.matrix","return.mixed",
 "return.rank4","return.zero","star.arguments","star.call_result")
SYMBOLS=("func","func.generateFunc","Func","Func.__init__","Func.eval","Func.__call__")
KINDS=("signature","argument_count","outside_trace","recursive_trace","return_type","failed_context")
WORKER=r'''
import json
from pathlib import Path
from candidate_trace import load_frontend
hc=load_frontend()
payload=json.loads(Path("/payload.json").read_text())
kind=payload["kind"];mode=payload["context"]
if type(mode) is not int or mode not in (0,1,2):raise ValueError("Trusted context")
events=[]
def rejected(label,fn,types):
 try:fn()
 except types as error:
  events.append(dict(label=label,exception=type(error).__name__,message=str(error)))
 else:raise AssertionError("Expected frontend rejection: "+label)
def identity(x):return x
if kind=="signature":
 signatures=("q",9,"c,c")
 rejected("invalid_signature",lambda:hc.func(signatures[mode])(identity),(ValueError,TypeError))
elif kind=="argument_count":
 good=hc.func("c")(identity)
 rejected("argument_count",lambda:good(*([0]*(0,2,3)[mode])),(TypeError,))
elif kind=="outside_trace":
 good=hc.func("c")(identity)
 rejected("outside_trace",lambda:good((0,1.0,[.25])[mode]),(ValueError,))
elif kind=="recursive_trace":
 @hc.func("c")
 def recursive(x):
  if mode==0:return recursive(x)
  return second(x)
 @hc.func("c")
 def second(x):
  if mode==1:return recursive(x)
  return third(x)
 @hc.func("c")
 def third(x):return recursive(x)
 rejected("recursive_trace",recursive.eval,(ValueError,))
 if hc._active_traces:raise AssertionError("Trace stack leak")
elif kind=="return_type":
 @hc.func("c")
 def invalid(x):
  if mode==0:return 3
  if mode==1:return [x,3]
  return hc.np.array([1.,2.])
 rejected("invalid_return",invalid.eval,(TypeError,))
 if hc._active_traces:raise AssertionError("Trace stack leak")
elif kind=="failed_context":
 @hc.func("c")
 def invalid(x):return (None,{},[x,3])[mode]
 good=hc.func("c")(identity)
 rejected("first_failure",invalid.eval,(TypeError,))
 rejected("poisoned_context",good.eval,(ValueError,))
 if good.evaluation_state!="new" or hc._active_traces:raise AssertionError("Failed context emitted another trace")
else:raise ValueError("Unknown trusted rejection")
Path("/out/native-rejection.json").write_text(json.dumps(dict(kind=kind,context=mode,
 events=events,passed=True,actual_frontend=True,actual_compile=False,actual_ciphertext_execution=False)))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files(folder,hashes):
 for name,hsh in hashes.items():
  rel=Path(name);p=folder/rel
  if rel.is_absolute() or ".." in rel.parts or p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha(p)!=hsh:
   raise ValueError("Changed or indirect evidence: "+name)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV
 if not a.execute:print(json.dumps(dict(historical_partitions=len(PARTITIONS),new_rejection_tasks=18,paid_calls=0)));return 0
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform result directory")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=930)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 from semantic_benchmark_execution import runtime_sources
 import candidate_sandbox as sandbox
 ledger_path=BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json"
 ledger=strict_file(ledger_path,4*1024**2)
 source=ROOT/"third_party/dacapo/python/hecate/hecate/expr.py"
 if sha(source)!=ledger["upstream_source_hashes"][str(source.relative_to(ROOT))]:raise ValueError("Pinned frontend changed")
 previous=RESULTS/"benchmark-r43-directed-independent"
 summary=strict_file(previous/"report.json",4*1024**2)
 oldplan=strict_file(previous/"plan.json",8*1024**2)
 if summary["binding"]!=oldplan["binding"] or oldplan["runtime_sources"][str(source.relative_to(ROOT))]!=sha(source):raise ValueError("Historical frontend identity")
 tasks=ledger["directed_tasks"];contexts=[]
 for partition in PARTITIONS:
  selected=[t for t in tasks if t["requirement"]==partition]
  if len(selected)!=3:raise ValueError("Partition context count")
  topologies=set()
  for task in selected:
   path=previous/(task["id"]+".json")
   if sha(path)!=summary["record_hashes"][path.name]:raise ValueError("Directed audit changed")
   audit=strict_file(path,8*1024**2);evidence=Path(audit["evidence"])
   files(evidence,audit["files"])
   if audit["task_sha256"]!=task["task_sha256"] or audit["requirement"]!=partition:raise ValueError("Task identity")
   coverage=audit["coverage"]
   if not coverage["actual_frontend_checked"]:raise ValueError("Missing actual frontend audit")
   if audit["acceptance_kind"]!=task["acceptance_kind"]:raise ValueError("Acceptance scope changed")
   if task["acceptance_kind"]=="structure_only":
    if coverage["numeric_features"] or not coverage["structural_only"] or partition not in coverage["structural_features"]:
     raise ValueError("Structural evidence misclassified as numeric")
   elif not coverage["numeric_features"] or not coverage["finite_influence_checked"]:
    raise ValueError("Missing numeric contribution audit")
   from audit_semantic_benchmark import verify_directed_coverage
   from unified_graph_exercises import spec
   verify_directed_coverage(coverage,task,spec(task["exercise"]))
   report=strict_file(evidence/"report.json",8*1024**2)
   passed=next(x for x in report["attempts"] if x["status"]=="passed")
   if not all(passed[k] for k in ("compiled","executed","numerically_correct")):raise ValueError("Missing historical FHE")
   if report["agent_calls"]!=0 or report["llm_generation_validated"]:raise ValueError("Not a manual FHE record")
   topology=task["topology"] if "topology" in task else None
   if topology is None:
    from benchmark_graph import signature
    topology=signature(strict_file(evidence/"model.json",131072),True)
   topologies.add(topology)
   contexts.append(dict(partition=partition,task=task["id"],task_sha256=task["task_sha256"],
    topology=topology,evidence=str(evidence),audit_sha256=sha(path),report_sha256=sha(evidence/"report.json"),
    acceptance_kind=audit["acceptance_kind"],numeric_features=coverage["numeric_features"],
    structural_features=coverage["structural_features"],historical_runtime_sha256=digest(oldplan["runtime_sources"])))
  if len(topologies)!=3:raise ValueError("Topology leakage")
 sources=runtime_sources();a.output.mkdir(parents=True)
 plan=dict(runtime_sources=sources,runner_sha256=sha(Path(__file__)),worker_sha256=digest(WORKER),
  ledger_sha256=sha(ledger_path),historical_report_sha256=sha(previous/"report.json"),
  contexts=contexts,symbols=list(SYMBOLS),kinds=list(KINDS),new_rejections=18,max_wall_seconds=900,
  source_sha256=sha(source),paid_calls=0)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan);rows=[];start=time.monotonic()
 for kind in KINDS:
  for context in range(3):
   if runtime_sources()!=sources or time.monotonic()-start>=900:raise ValueError("Source/time boundary")
   folder=a.output/(kind+"_"+str(context));folder.mkdir();out=folder/"output";out.mkdir()
   dump(folder/"payload.json",dict(kind=kind,context=context))
   code=sandbox.run(folder/"payload.json",out,[str(VENV/"bin/python"),"-B","-c",WORKER],folder/"trace.log",seconds=45)
   row=dict(kind=kind,context=context,exit_code=code,status="failed")
   if code==0:
    check=strict_file(out/"native-rejection.json",65536)
    if check["kind"]!=kind or check["context"]!=context or not check["passed"]:raise ValueError("Rejection identity")
    row.update(status="passed",check=check)
   row["files"]={str(f.relative_to(folder)):sha(f) for f in folder.rglob("*") if f.is_file()}
   rows.append(row);dump(a.output/"progress.json",rows)
 if runtime_sources()!=sources:raise ValueError("Source drift")
 report=dict(format="poseidon-native-api-semantic-audit-v1",plan_binding=plan["binding"],
  status="passed" if all(r["status"]=="passed" for r in rows) else "failed",
  symbols=list(SYMBOLS),historical_contexts=contexts,partitions=list(PARTITIONS),
  rejections=rows,seconds=time.monotonic()-start,new_encrypted_executions=0,paid_calls=0,
  current_frontend_rejections=True,all_program_equivalence=False,
  scope="Bounded native signatures/calls/return partitions; structural empty returns score separately; historical FHE binding retained")
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps(dict(status=report["status"],historical_contexts=len(contexts),
  rejected_passed=sum(r["status"]=="passed" for r in rows),failed=[(r["kind"],r["context"]) for r in rows if r["status"]!="passed"],new_encrypted_executions=0,paid_calls=0)))
 return int(report["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
