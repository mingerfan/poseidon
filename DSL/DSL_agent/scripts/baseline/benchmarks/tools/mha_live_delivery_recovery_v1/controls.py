import os,sys,time,json,unittest
from pathlib import Path
BASE=Path("/home/lhohy/Code/Poseidon/scripts/baseline")
sys.path.insert(0,str(BASE))
modules=("test_application_provenance_transport","test_application_component","test_application_model_source",
 "test_application_generation","test_generation_receipt","test_candidate_bundle","test_application_cache_delivery",
 "test_application_execution_context","test_pytorch_mha_component")
available=[name for name in modules if (BASE/(name+".py")).is_file()]
started=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromNames(available)
result=unittest.TextTestRunner(verbosity=2).run(suite)
out=dict(format="poseidon-mha-delivery-controls-v1",modules=available,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),elapsed_seconds=time.monotonic()-started,
 provider_calls=0,compiler_calls=0,encrypted_calls=0)
Path(__file__).with_name("controls-report.json").write_text(json.dumps(out,sort_keys=True,indent=2))
print(json.dumps(out),flush=True)
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
