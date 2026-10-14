import json,time,unittest
from pathlib import Path
start=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(("test_scope","test_run"))
r=unittest.TextTestRunner(verbosity=2).run(suite)
value=dict(format="poseidon-mha-feedback-runner-controls-v1",tests=r.testsRun,
passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
elapsed_seconds=time.monotonic()-start,provider_calls=0,compiler_calls=0,encrypted_calls=0,
synthetic_authorization_fixtures_only=True,production_authorization_created=False)
Path(__file__).with_name("controls-report.json").write_text(json.dumps(value,sort_keys=True,indent=2))
print(json.dumps(value),flush=True)
raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)
