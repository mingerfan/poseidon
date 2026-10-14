import io,json,os,sys,unittest,hashlib,time
from pathlib import Path
import numpy,torch
import test_composition as cases
import test_source_package as source_cases
start=time.monotonic()
suite=unittest.TestSuite()
selected=sys.argv[1] if len(sys.argv)==2 else "reference"
groups={"reference":("test_composition","test_pytorch_common_intake"),
 "source":("test_source_package",),"retained":("test_intake","test_facade_package"),
 "component":("test_application_component","test_application_capabilities","test_candidate_bundle",
              "test_binding_feedback","test_cli_feedback_policy") }
if selected not in groups:raise SystemExit("Explicit test suite")
for name in groups[selected]:
 suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
Path("/out/tests.log").write_text(stream.getvalue())
refs=cases.REFERENCES
import test_intake as retained
import test_facade_package as retained_packages
all_references=[r["reference"] for r in refs+source_cases.MEASUREMENTS+retained.MEASUREMENTS]

report=dict(format="poseidon-ordered-common-reference-tests-v1",
 status="passed" if result.wasSuccessful() else "failed",suite=selected,tests=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 contexts=len(all_references),input_groups=sum(r["groups"] for r in all_references),
 compared_values=sum(r["compared_values"] for r in all_references),rows=refs,
 context_count_scope="reference checks; overlapping contexts are not unique model count",
 source_capture_rows=source_cases.MEASUREMENTS,source_package_rows=source_cases.PACKAGE_CHECKS,
 retained_v12_capture_rows=retained.MEASUREMENTS,retained_v12_package_rows=retained_packages.PACKAGES,
 max_decomposition_error=max((r["max_decomposition_error"] for r in all_references),default=0.),
 max_source_normalization_error=max((r["max_original_vs_normalized_torch"] for r in all_references),default=0.),
 max_approximation_error=max((r["max_approximation_error"] for r in all_references),default=0.),
 approximation_accuracy_certified=False,production_applied=False,source_manifest_exposed=False,
 runtime=dict(python=list(sys.version_info[:3]),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda),
 environment_probe_removed="POSEIDON_PROBE_SECRET" not in os.environ,
 namespace=str(Path("/proc/self/ns/net").readlink()),seconds=time.monotonic()-start,
 compiler_calls=0,encrypted_executions=0,provider_calls=0,paid_calls=0)
Path("/out/report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(stream.getvalue()[-13000:] if not result.wasSuccessful() else stream.getvalue()[-350:])
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","contexts","input_groups","compared_values")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
