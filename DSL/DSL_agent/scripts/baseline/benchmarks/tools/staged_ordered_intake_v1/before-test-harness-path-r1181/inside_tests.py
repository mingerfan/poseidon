import hashlib,io,json,os,sys,time,unittest
from pathlib import Path
sys.path[:0]=["/tests","/app"]
import numpy as np
import torch
from platform_config import require_python_packages,loaded_libraries
require_python_packages(torch,np)
assert sys.version_info[:3]==(3,10,14)
assert "POSEIDON_PROBE_SECRET" not in os.environ
torch.set_num_threads(1)
import test_capture_compatibility as compatibility
import test_intake as intake
import test_facade_package as packages
sys.path.append("/home/lhohy/Code/Poseidon/scripts/baseline")
import test_pytorch_model_intake as retained_intake
import test_pytorch_common_intake as retained_common
suite=unittest.TestSuite([
 unittest.defaultTestLoader.loadTestsFromModule(compatibility),
 unittest.defaultTestLoader.loadTestsFromModule(intake),
 unittest.defaultTestLoader.loadTestsFromModule(packages),
 unittest.defaultTestLoader.loadTestsFromModule(retained_intake),
 unittest.defaultTestLoader.loadTestsFromModule(retained_common)])
stream=io.StringIO();started=time.monotonic()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
Path("/out/test.log").write_text(stream.getvalue())
report=dict(format="poseidon-staged-ordered-intake-unit-v1",
 status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,
 runtime=dict(python=list(sys.version_info[:3]),torch=torch.__version__,numpy=np.__version__,
              cuda_available=torch.cuda.is_available()),
 capture_contexts=compatibility.MEASUREMENTS,reference_contexts=compatibility.REFERENCE_MEASUREMENTS,
 source_workers=intake.MEASUREMENTS,source_packages=packages.PACKAGES,
 network_namespace=Path("/proc/self/ns/net").readlink().as_posix(),
 probe_secret_present="POSEIDON_PROBE_SECRET" in os.environ,mapped_libraries=loaded_libraries(),
 production_intake_integrated=False,hecate_traces=0,compilations=0,encrypted_executions=0,paid_calls=0)
Path("/out/report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","runtime")}))
if not result.wasSuccessful():print(stream.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)
