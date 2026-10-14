"""Joint staging tests; no API, compiler or encrypted execution."""
import hashlib,importlib,importlib.util,io,json,os,sys,time,unittest
from pathlib import Path
sys.path[:0]=["/tests","/app"]
import numpy as np,torch
from platform_config import require_python_packages,loaded_libraries
require_python_packages(torch,np)
assert sys.version_info[:3]==(3,10,14)
assert "POSEIDON_PROBE_SECRET" not in os.environ
torch.set_num_threads(1)
def native_test(name,directory="/home/lhohy/Code/Poseidon/scripts/baseline"):
    path=Path(directory)/(name+".py")
    spec=importlib.util.spec_from_file_location("joint_native_"+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
import test_capture_compatibility as compatibility
import test_intake as intake
import test_facade_package as packages
import test_joint_component as joint
ordered=[compatibility,intake,packages,native_test("test_pytorch_model_intake"),native_test("test_pytorch_common_intake")]
feedback=native_test("test_staged","/home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/staged_binding_feedback_v1")
component=[feedback,*[native_test(name) for name in ("test_application_diagnostics","test_decorated_functions",
          "test_native_array_core","test_unified_poseidon_guidance","test_application_component")]]
directed=native_test("test_binding_feedback")
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module) for module in (*ordered,*component,directed,joint)])
stream=io.StringIO();started=time.monotonic()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
Path("/out/test.log").write_text(stream.getvalue())
report=dict(format="poseidon-joint-component-staging-tests-v1",status="passed" if result.wasSuccessful() else "failed",
    tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    seconds=time.monotonic()-started,source_models=intake.MEASUREMENTS,source_packages=packages.PACKAGES,
    original_capture_contexts=compatibility.MEASUREMENTS,original_reference_contexts=compatibility.REFERENCE_MEASUREMENTS,
    application_submissions=joint.MEASUREMENTS,runtime=dict(python=list(sys.version_info[:3]),torch=torch.__version__,numpy=np.__version__),
    mapped_libraries=loaded_libraries(),network_namespace=str(Path("/proc/self/ns/net").readlink()),
    production_applied=False,provider_calls=0,compilations=0,encrypted_executions=0,
    verification_scope="Real isolated source capture/reference and inert submit; remaining component/backend/provider cases use test doubles.")
Path("/out/report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","runtime")}))
if not result.wasSuccessful():print(stream.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)
