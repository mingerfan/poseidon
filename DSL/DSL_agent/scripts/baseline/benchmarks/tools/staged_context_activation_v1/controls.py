"""Bounded component controls; no generation, compilation or FHE calls."""
import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from test_staged_application_reuse import CacheControls,StagedContextControls
from test_application_component import ApplicationTests
from test_application_identity import IdentityTests
from application_execution.tests.test_context import ContextTests
from test_application_model_source_context import SourceContextTests
from test_backend_parameters_boundary import ParameterBoundaryTests
suite=unittest.TestSuite()
for klass in (CacheControls,StagedContextControls,ApplicationTests,IdentityTests,ContextTests,SourceContextTests):
 suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(klass))
names=["application_execution/context.py","application_execution/context_identity.py",
       "application_component.py","application_reuse.py","test_staged_application_reuse.py"]
before={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()for n in names}
started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
after={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()for n in names}
record=dict(format="poseidon-staged-worker-reuse-controls-v1",tests=result.testsRun,
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 source_sha256=before,source_unchanged=before==after,elapsed_seconds=time.monotonic()-started,
 injected_results=True,compiler_calls=0,encrypted_calls=0,provider_calls=0,
 failures=[dict(test=str(t),detail=s)for t,s in result.failures],
 error_details=[dict(test=str(t),detail=s)for t,s in result.errors])
raw=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
record["binding"]=hashlib.sha256(raw(record)).hexdigest()
name="controls.initial.json";p=HERE/name
if p.exists():raise SystemExit("Preserve prior control report")
p.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:record[k]for k in ("tests","passed","failed","errors","skipped","elapsed_seconds","source_unchanged")}))
raise SystemExit(0 if result.wasSuccessful()and before==after else 1)
