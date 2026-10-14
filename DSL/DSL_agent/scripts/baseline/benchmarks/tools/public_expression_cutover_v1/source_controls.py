"""Old interface controls and new source workers. No compiler, FHE or Provider."""
import os,sys,shlex,unittest,json,hashlib,time,importlib,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,WORK,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
if(HERE/"source-controls-v2.json").exists():raise SystemExit("preserve previous source controls")
start=time.monotonic();tests=unittest.TestSuite()
names=("test_application_capabilities","test_model_output_contract","test_pytorch_dependency_syntax",
       "test_pytorch_public_boolean_arithmetic","test_pytorch_public_tensor_division",
       "test_pytorch_static_capture","test_source_precision_reference","test_public_none_policy")
for name in names:tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
with(HERE/"source-controls-v2.log").open("x")as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(tests)
data=dict(format="poseidon-public-expression-source-controls-v1",tests=result.testsRun,
          failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
          seconds=time.monotonic()-start,provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
          units=names,source_capture_rows=[],native_precision_extension_workers=0)
import test_public_none_policy
data["native_precision_extension_workers"]=test_public_none_policy.COUNTERS["actual_source_precision_workers"]
if result.wasSuccessful():
    from pytorch_model_intake import run
    from pytorch_ordered_contract import source_identity
    from candidate_bundle import check_provenance
    OUT=WORK/"results/agent-public-expression-source-v23";OUT.mkdir(mode=0o700)
    bound=source_identity();data["source_identity"]=bound
    cases=json.loads((HERE/"capture-models-v3.json").read_text())
    selected=("reshape_0","permute_1","slice_2","bool_cast_1","float32","named")
    for case in cases:
        if case["case"]not in selected:continue
        row=dict(case=case["case"])
        try:
            if time.monotonic()-start>=170:raise RuntimeError("source control window exhausted")
            output=OUT/case["case"];answer=run(case["manifest_path"],output,resource_profile="general-tensors-v1",trusted=True)
            high=json.loads((output/"original.json").read_text());model=json.loads((output/"model.json").read_text())
            bound_graph=json.loads((output/"decomposition.json").read_text());reference=json.loads((output/"reference-report.json").read_text())
            original_package=json.loads((HERE/"capture-evidence-v3"/case["case"]/"provenance.json").read_text())["trusted_pytorch"]
            provenance=dict(original=high,decomposition=bound_graph,reference_report=reference,trusted_pytorch=original_package)
            check_provenance(provenance,model)
            assert source_identity()==bound
            row.update(status="passed",source_capture=answer,reference_groups=reference["probes"],
                       output=str(output),model_sha256=hashlib.sha256((output/"model.json").read_bytes()).hexdigest(),
                       sandbox_capture_log_sha256=hashlib.sha256((output/"capture.log").read_bytes()).hexdigest())
        except Exception as exc:
            row.update(status="failed",error_type=type(exc).__name__,error=str(exc));row["traceback"]=traceback.format_exc()
        data["source_capture_rows"].append(row)
    data["source_capture_passed"]=sum(row["status"]=="passed"for row in data["source_capture_rows"])
    data["source_capture_failed"]=sum(row["status"]=="failed"for row in data["source_capture_rows"])
data["seconds"]=time.monotonic()-start
data["binding"]=__import__("benchmark_graph").digest(data)
with(HERE/"source-controls-v2.json").open("x")as f:json.dump(data,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in data.items()if k not in("source_identity",)}))
raise SystemExit(0 if result.wasSuccessful()and data.get("source_capture_failed")==0 else 1)
