import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from hecate_python_env import WORK
if __name__=="__main__":
    directory=Path(__file__).resolve().parent
    attempt=0
    if len(sys.argv)==3 and sys.argv[1]=="--attempt":
        attempt=int(sys.argv[2]);assert 1<=attempt<=3
    else:assert len(sys.argv)==1
    suffix="" if attempt==0 else ".retry"+str(attempt)
    started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromName("test_source_precision_reference")
    with (directory/("controls"+suffix+".stdout")).open("x") as output:
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    files=[ROOT/"scripts/baseline/source_precision_reference.py",ROOT/"scripts/baseline/test_source_precision_reference.py",Path(__file__)]
    report=dict(format="poseidon-source-precision-controls-v1",status="passed" if result.wasSuccessful() else "failed",
        tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
        elapsed_seconds=time.monotonic()-started,api_calls=0,compiler_calls=0,encrypted_execution=False,
        successful_native_reference_reports=len(sys.modules["test_source_precision_reference"].FORWARD_REPORTS),
        source_forward_executed=bool(sys.modules["test_source_precision_reference"].FORWARD_REPORTS),
        source_models_imported_in_host=False,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    report["binding"]=digest(report)
    with (directory/("report"+suffix+".json")).open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
