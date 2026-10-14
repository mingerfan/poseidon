"""Bounded component and existing-interface controls in the pinned environment."""
import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,canonical
from pytorch_ordered_contract import source_identity
from application_identity import build
MODULES=("test_unified_mixed_layout","test_pytorch_mha_component","test_pytorch_source_contract",
         "test_application_capabilities","test_pytorch_model_intake","test_application_model_source_context",
         "test_model_source_provenance","test_pytorch_gelu_source_capture","test_candidate_bundle",
         "test_application_identity","test_agent_component","test_unified_capacity_linear")
def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    before=source_identity()
    stage=Path(__file__).resolve().parent
    cutover=json.loads((stage/"cutover.json").read_bytes())
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [*(BASE/n for n in cutover["after"]),*(BASE/n for n in cutover["tests"]),Path(__file__)]}
    modules=[m for m in MODULES if (BASE/(m+".py")).is_file()]
    missing=[m for m in MODULES if m not in modules]
    assert not missing,missing
    start=time.monotonic()
    with (out/"controls.log").open("x") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(
              unittest.TestLoader().loadTestsFromNames(modules))
    stable=source_identity()==before and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s for n,s in files.items())
    report=dict(format="poseidon-mha-layout-component-controls-v1",status="passed" if result.wasSuccessful() and stable else "failed",
                tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                source_files=files,capture_implementation=before,source_frozen=stable,
                elapsed_seconds=time.monotonic()-start,production_mha_format="poseidon-trusted-pytorch-v16",
                paid_api_calls=0,compiler_calls=0,encrypted_calls=0,
                modules=modules,failures_details=[dict(test=str(t),traceback=e) for t,e in result.failures],
                errors_details=[dict(test=str(t),traceback=e) for t,e in result.errors])
    report["binding"]=digest(report);(out/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_files","capture_implementation","modules","failures_details","errors_details")}))
    if not result.wasSuccessful():
        print((out/"controls.log").read_text()[-16000:])
    return 0 if report["status"]=="passed" else 1
if __name__=="__main__":raise SystemExit(main())
