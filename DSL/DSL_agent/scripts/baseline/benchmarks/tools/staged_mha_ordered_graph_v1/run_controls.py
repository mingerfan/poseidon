"""Single bounded offline pinned-Python control window; never call Provider/backend."""
import hashlib,importlib.util,json,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
for path in [BASE,HERE/"reference_tests",HERE/"contract_tests",HERE/"overlay"]:sys.path.insert(0,str(path))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
 files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob("*.py")}
 modules=["test_output_approximation_contracts.ContractControls","test_ordered_mha_dataflow.OrderedMhaDataflow","test_fixture_math.FixtureMath","test_mha_provenance.ProvenanceControls"]
 suite=unittest.TestLoader().loadTestsFromNames(modules)
 start=time.monotonic()
 with (out/"controls.log").open("x") as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 after={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob("*.py")}
 report=dict(status="passed" if result.wasSuccessful() and after==files else "failed",
    tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    implementations=files,source_frozen=after==files,elapsed_seconds=time.monotonic()-start,
    api_calls=0,compiler_calls=0,encrypted_executions=0,production_applied=False)
 for key in ["failures","errors"]:report[key+"_details"]=[dict(test=str(t),traceback=e) for t,e in getattr(result,key)]
 (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k not in ["implementations","failures_details","errors_details"]}))
 if not result.wasSuccessful():
  print((out/"controls.log").read_text()[-15000:])
 return 0 if report["status"]=="passed" else 1
if __name__=="__main__":raise SystemExit(main())
