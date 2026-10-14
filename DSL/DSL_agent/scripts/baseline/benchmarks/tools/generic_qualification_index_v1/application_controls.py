"""Current application controls: injected fake backends and readonly identity."""
import hashlib
import json
import sys
import time
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from application_identity import current_identity_manifest
names=("test_application_component","test_application_diagnostics","test_staged_application_reuse")
start=time.monotonic()
identity=current_identity_manifest()
with (HERE/"application-controls.log").open("x") as out:
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    count=suite.countTestCases()
    result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
report=dict(format="poseidon-generic-application-controls-v1",tests=count,
 passed=count-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-start,status="passed" if result.wasSuccessful() else "failed",
 scope="Injected Fake Backend/Provider; WorkerContext and backend identity read installed files only",
 new_hecate_capture=0,new_compiler_calls=0,new_encrypted_calls=0,new_provider_calls=0,
 sdk_identity=identity["manifest_sha256"],
 stage_identities={s:identity[s]["sha256"]for s in("generation","compilation","execution","validation")},
 test_source_hashes={n:hashlib.sha256((ROOT/"scripts/baseline"/(n+".py")).read_bytes()).hexdigest()for n in names},
 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 log_sha256=hashlib.sha256((HERE/"application-controls.log").read_bytes()).hexdigest())
report["binding"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
with (HERE/"application-controls.json").open("x") as f:
    json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(report,sort_keys=True))
sys.exit(0 if result.wasSuccessful() else 1)
