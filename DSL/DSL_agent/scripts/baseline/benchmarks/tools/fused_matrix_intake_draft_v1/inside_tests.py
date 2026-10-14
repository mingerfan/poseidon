import hashlib,io,json,os,sys,time,unittest
from pathlib import Path
import numpy,torch
import test_fused as cases
start=time.monotonic()
result=unittest.TextTestRunner(stream:=(io.StringIO()),verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(cases))
Path("/out/tests.log").write_text(stream.getvalue())
refs=cases.REFERENCES
report=dict(format="poseidon-fused-matrix-reference-tests-v1",
 status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 contexts=len(refs),input_groups=sum(r["reference"]["groups"] for r in refs),
 compared_values=sum(r["reference"]["compared_values"] for r in refs),rows=refs,
 operations=sorted({r["operation"] for r in refs}),
 max_source_normalization_error=max((r["reference"]["max_original_vs_normalized_torch"] for r in refs),default=0.),
 max_decomposition_error=max((r["reference"]["max_decomposition_error"] for r in refs),default=0.),
 runtime=dict(python=list(sys.version_info[:3]),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda),
 secret_probe_removed="POSEIDON_PROBE_SECRET" not in os.environ,namespace=str(Path("/proc/self/ns/net").readlink()),
 seconds=time.monotonic()-start,production_applied=False,source_manifest_exposed=False,
 compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0)
Path("/out/report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(stream.getvalue()[-18000:] if not result.wasSuccessful() else stream.getvalue()[-250:])
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","contexts","input_groups","compared_values")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
