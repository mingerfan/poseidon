import io,json,os,sys,time,unittest
from pathlib import Path
import numpy,torch
import test_source_intake as cases
start=time.monotonic();stream=io.StringIO()
selected=sys.argv[1] if len(sys.argv)==2 else "source"
suite=unittest.TestSuite()
if selected=="source":suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(cases))
elif selected=="legacy_rejection":
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(
        "test_source_intake.RetainedSourceTests.test_old_source_workers_do_not_silently_enable_fused_calls"))
elif selected=="affected":
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(
        "test_source_intake.SourceTests.test_missing_file_and_reserved_implementation_name_rejected"))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cases.RetainedSourceTests))
    import test_fused
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_fused))
else:raise SystemExit("Explicit test selection required")
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
refs=cases.MEASUREMENTS
regression_refs=test_fused.REFERENCES if selected=="affected" else []
legacy_refs=cases.LEGACY_MEASUREMENTS
report=dict(format="poseidon-fused-matrix-source-tests-v1",status="passed" if result.wasSuccessful() else "failed",
 tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 suite=selected,reference_regression_rows=regression_refs,retained_source_rows=legacy_refs,
 source_models=len(refs),input_groups=sum(r["reference"]["groups"] for r in refs),
 compared_values=sum(r["reference"]["compared_values"] for r in refs),rows=refs,
 package_reloads=cases.PACKAGES,application_submissions=cases.SUBMISSIONS,
 max_normalization_error=max((r["reference"]["max_original_vs_normalized_torch"] for r in refs),default=0.),
 max_decomposition_error=max((r["reference"]["max_decomposition_error"] for r in refs),default=0.),
 runtime=dict(python=list(sys.version_info[:3]),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda),
 namespace=str(Path("/proc/self/ns/net").readlink()),secret_probe_removed="POSEIDON_PROBE_SECRET" not in os.environ,
 seconds=time.monotonic()-start,production_applied=False,source_manifest_exposed=False,
 compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0)
Path("/out/tests.log").write_text(stream.getvalue())
Path("/out/report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(stream.getvalue()[-22000:] if not result.wasSuccessful() else stream.getvalue()[-250:])
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models","input_groups","compared_values")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
