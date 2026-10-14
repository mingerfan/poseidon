"""Bounded capacity acceptance; no Provider calls."""
import argparse,contextlib,json,shlex,sys,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]
def main():
 from hecate_python_env import WORK,VENV,enter_nix
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if not a.inside:
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
    [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output)])
  return enter_nix(cmd,seconds=600,environment="poseidon-cpu")
 from benchmark_graph import require,canonical,digest
 from semantic_benchmark_execution import runtime_sources
 require(not a.output.exists(),"Fresh capacity check");a.output.mkdir();frozen=runtime_sources()
 names=["test_unified_capacity_spatial","test_unified_capacity_tensor","test_unified_capacity_tiles","test_unified_capacity_diagonal","test_unified_capacity_linear","test_unified_capacity","test_agent_component","test_application_component","test_unified_chunk_layout","test_model_decomposition"]
 suite=unittest.TestSuite();loaded=[]
 for name in names:
  if (ROOT/"scripts/baseline"/(name+".py")).is_file():
   suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name));loaded.append(name)
 with (a.output/"tests.log").open("w") as log:
  result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 require(runtime_sources()==frozen,"Capacity test source drift")
 report=dict(format="poseidon-capacity-static-v1",loaded=loaded,tests=result.testsRun,
   passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
   failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),source_hashes=frozen,
   actual_compile=False,encrypted_execution=False,paid_calls=0)
 report["binding"]=digest(report);(a.output/"report.json").write_bytes(canonical(report))
 print({k:v for k,v in report.items() if k!="source_hashes"},flush=True)
 return int(not result.wasSuccessful())
if __name__=="__main__":raise SystemExit(main())
