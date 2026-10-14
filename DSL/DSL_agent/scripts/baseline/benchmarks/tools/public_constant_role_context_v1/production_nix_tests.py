"""Existing production contract regression in the pinned, offline Python environment."""
import sys,os,pathlib,unittest,json,hashlib,time
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
os.chdir(BASE)
import numpy,torch
assert numpy.__version__=="1.25.2" and torch.__version__=="2.0.1"
assert os.environ.get("IN_NIX_SHELL")=="pure"
names=["test_public_weight_context","test_application_component","test_application_generation",
 "test_application_model_source","test_application_model_source_context","test_application_identity",
 "test_capacity_binding_storage","test_unified_graph","test_unified_generation_guidance"]
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
out=pathlib.Path(__file__).resolve().parent/"production-compat-pinned-r1307"
out.mkdir(exist_ok=False)
start=time.monotonic()
with (out/"tests.log").open("w") as f:
 result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
report=dict(format="poseidon-public-math-context-production-compat-v1",modules=names,tests=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 seconds=time.monotonic()-start,actual_http=0,compilations=0,encrypted_execution=0,
 uses_mocked_service_and_backend_fixtures=True,python=sys.version.split()[0],
 numpy=numpy.__version__,torch=torch.__version__,
 failed_names=[str(t) for t,_ in result.failures],error_names=[str(t) for t,_ in result.errors],
 production_sources={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()
 for n in ["public_weight_context.py","public_math_context.py","unified_graph_contract.py",
 "run_candidate.py","application_component.py","application_model_source.py"]})
report["binding"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(out/"report.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
print(json.dumps(report,ensure_ascii=False))
if not result.wasSuccessful():print((out/"tests.log").read_text()[-12000:])
sys.exit(0 if result.wasSuccessful() else 1)
