"""Final frozen offline gates and request compatibility, without paid calls."""
import sys,shlex,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1200)
 from semantic_benchmark_execution import runtime_sources
 from component_contract import reconstruct_request
 from benchmark_graph import digest
 out=WORK/"results/seal-gate-r186";out.mkdir(exist_ok=True)
 names=["test_seal_scale_gate","test_seal_cpu_golden.SealGateTests","test_failure_repairs","test_validation_adapter",
  "test_noncompiler_repairs","test_chebyshev_lowering","test_packed_prefix_lowering",
  "test_unified_typed_guidance","test_unified_generation_guidance","test_unified_directed_guidance",
  "test_provider_retries","test_deepseek_transport","test_transport_diagnostics","test_multi_input_protocol","test_loopback_tls_route"]
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
 value=dict(run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
  details=[(str(t),s) for t,s in r.failures+r.errors],source_hashes=runtime_sources())
 (out/"unit.json").write_text(json.dumps(value,indent=2))
 if not r.wasSuccessful():return 1
 n=0
 indexpath=WORK/"results/stage2-agent-campaign-v7-r130/index.json"
 idx=json.loads(indexpath.read_text())
 for ref in idx["shards"]:
  for row in json.loads((indexpath.parent/ref["file"]).read_text())["cases"]:
   assert reconstruct_request(row["request"])==row["request"];n+=1
 for version in ("r173","r176","r179"):
  report=json.loads((ROOT/("docs/baseline/stage2-agent-repair-result-"+version+".json")).read_text())
  for row in report["rows"]:
   q=json.loads((Path(row["evidence"])/"request.json").read_text())
   assert reconstruct_request(q)==q;n+=1
 report=dict(old_requests_exact=n,paid_calls=0,source_hashes=runtime_sources())
 report["binding"]=digest(report)
 (out/"compatibility.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(dict(tests=r.testsRun,old_requests=n,paid_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
