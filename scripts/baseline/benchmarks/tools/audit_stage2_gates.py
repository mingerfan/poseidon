"""Current static gates plus explicitly historical compiler artifact verification."""
import argparse,hashlib,io,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from semantic_benchmark_execution import runtime_sources
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform result directory")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 from rejection_benchmark import tasks,run_task
 from bind_compiler_context_examples import build
 modules=["test_compiler_context_bindings","test_model_partition_rejections"]
 sources=runtime_sources();a.output.mkdir(parents=True);start=time.monotonic()
 with (a.output/"checks.log").open("w") as stream:
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
 rows=[];catalog=tasks()
 if len(catalog)!=48:raise ValueError("Changed rejection denominator")
 for task in catalog:
  try:row=run_task(task,catalog)
  except Exception as error:row=dict(id=task["id"],status="failed",reason=str(error))
  rows.append(row)
 artifact=build(RESULTS/"upstream-chunks-r29-compiler-evidence.json",RESULTS)
 # build() reports the audit process's source identity, not historical execution.
 artifact["auditor_runtime_sha256"]=artifact.pop("runtime_source_sha256")
 for group in artifact["examples"].values():
  for item in group:
   evidence=RESULTS/item["evidence_relative_to_results"]
   original=strict_file(evidence/"report.json",8*1024**2)
   item["historical_candidate_source_hashes"]=original["source_hashes"]
   item["historical_candidate_source_sha256"]=digest(original["source_hashes"])
   item["historical_runtime_binary_sha256"]=original["runtime_sha256"]
 artifact["execution_scope"]="Previously executed immutable artifacts; no current-runtime FHE credit"
 dump(a.output/"compiler-contexts.json",artifact);dump(a.output/"rejections.json",rows)
 if runtime_sources()!=sources:raise ValueError("Audit source drift")
 report=dict(format="poseidon-stage2-gates-v1",runtime_sources=sources,runtime_sha256=digest(sources),
    unit_tests=result.testsRun,unit_failed=len(result.failures)+len(result.errors),unit_skipped=len(result.skipped),
    static_rejection_tasks=len(rows),static_rejection_passed=sum(r["status"]=="passed" for r in rows),
    static_rejection_partitions=16,static_rejection_contexts=3,
    mathematical_positive_controls=153 if result.wasSuccessful() else None,
    mathematical_rejection_controls=153 if result.wasSuccessful() else None,
    compiler_partitions=artifact["requirements"],compiler_contexts_per_partition=artifact["contexts_per_requirement"],
    compiler_scope=artifact["execution_scope"],actual_compile=False,actual_ciphertext_execution=False,
    paid_calls=0,seconds=time.monotonic()-start,
    files={n:sha(a.output/n) for n in ("compiler-contexts.json","rejections.json","checks.log")},
    checker_sources={str(p.relative_to(BASE.parents[1])):sha(p) for p in
       [Path(__file__),HERE/"test_compiler_context_bindings.py",HERE/"bind_compiler_context_examples.py",
        HERE/"test_model_partition_rejections.py",HERE/"build_model_partition_rejections.py"]})
 report["status"]="passed" if result.wasSuccessful() and report["unit_skipped"]==0 and report["static_rejection_passed"]==48 else "failed"
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps({k:v for k,v in report.items() if k not in ("runtime_sources","checker_sources","files")}))
 return int(report["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
