"""Bounded offline tests and exact historical request reconstruction."""
import sys,json,shlex,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=900)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from component_contract import reconstruct_request
 from benchmark_graph import digest
 out=RESULTS/"stage2-noncompiler-r177";sources=runtime_sources()
 names=["test_noncompiler_repairs","test_chebyshev_lowering","test_packed_prefix_lowering",
  "test_unified_typed_guidance","test_unified_generation_guidance","test_unified_directed_guidance",
  "test_provider_retries","test_deepseek_transport","test_transport_diagnostics",
  "test_multi_input_protocol","test_loopback_tls_route"]
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
 value=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  failures_detail=[(str(t),message) for t,message in result.failures+result.errors],source_hashes=sources,paid_calls=0)
 (out/"unit.json").write_text(json.dumps(value,indent=2))
 if not result.wasSuccessful():return 1
 indexpath=RESULTS/"stage2-agent-campaign-v7-r130/index.json"
 idx=json.loads(indexpath.read_text());counts={}
 for ref in idx["shards"]:
  for row in json.loads((indexpath.parent/ref["file"]).read_text())["cases"]:
   q=row["request"];assert reconstruct_request(q)==q;counts["original2030"]=counts.get("original2030",0)+1
 for tag in ("r173","r176"):
  report=json.loads((ROOT/("docs/baseline/stage2-agent-repair-result-"+tag+".json")).read_text())
  for row in report["rows"]:
   q=json.loads((Path(row["evidence"])/"request.json").read_text())
   assert reconstruct_request(q)==q;counts[tag]=counts.get(tag,0)+1
 assert sources==runtime_sources()
 report=dict(counts=counts,source_hashes=sources,paid_calls=0)
 report["binding"]=digest(report)
 (out/"compatibility.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(dict(tests=value["run"],compatibility=counts)))
 return 0
if __name__=="__main__":raise SystemExit(main())
