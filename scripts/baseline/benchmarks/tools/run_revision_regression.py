"""Bounded, source-bound regression for the revised input and coverage changes."""
import argparse,hashlib,json,os,re,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
MODULES=["test_model_decomposition","test_nested_nix_launcher","test_candidate_bundle",
 "test_unified_graph","test_packed_prefix_lowering","test_model_graph","test_schema3_graph",
 "test_chunked_model","test_packed_model","test_portability"]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import enter_nix,VENV
 from workspace_paths import RESULTS
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform results directory")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=330)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 a.output.mkdir(parents=True);sources=runtime_sources();start=time.monotonic()
 command=[str(VENV/"bin/python"),"-B","-m","unittest","-v",*MODULES]
 with (a.output/"unittest.log").open("w") as log:
  run=subprocess.run(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 text=(a.output/"unittest.log").read_text();m=re.search(r"Ran (\d+) tests? in",text)
 if not m or runtime_sources()!=sources:raise ValueError("Incomplete tests or source drift")
 total=int(m[1]);skips=len(re.findall(r" \.\.\. skipped ",text))
 failures=len(re.findall(r" \.\.\. (?:FAIL|ERROR)\n",text))
 report=dict(format="poseidon-revision-regression-v1",command=command,cwd=str(BASE),
  runtime_sources=sources,runtime_sha256=digest(sources),tests=total,passed=total-skips-failures,
  skipped=skips,failed=failures,exit_code=run.returncode,paid_calls=0,actual_compile=False,
  actual_ciphertext_execution=False,seconds=time.monotonic()-start,
  log_sha256=hashlib.sha256((a.output/"unittest.log").read_bytes()).hexdigest(),
  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps({k:v for k,v in report.items() if k not in ("runtime_sources","command")}))
 return run.returncode
if __name__=="__main__":raise SystemExit(main())
