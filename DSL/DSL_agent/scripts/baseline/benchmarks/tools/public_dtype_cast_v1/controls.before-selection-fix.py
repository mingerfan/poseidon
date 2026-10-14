"""Current Agent regression tests only: no compiler, runtime or provider."""
import hashlib,json,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
modules=["test_pytorch_public_dtype_cast","test_program_output_tree_contract","test_pytorch_public_tensor_division",
 "test_pytorch_public_boolean_arithmetic","test_source_precision_reference","test_pytorch_model_intake",
 "test_pytorch_weight_package","test_pytorch_source_contract","test_candidate_bundle","test_application_capabilities"]
suite=unittest.TestSuite()
for name in modules:suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
from benchmark_graph import digest
data=dict(format="poseidon-public-dtype-cast-controls-v1",modules=modules,tests=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-start,
 compiler_calls=0,encrypted_calls=0,provider_calls=0)
data["binding"]=digest(data)
name="controls.json"if result.wasSuccessful()else"controls-failed.json"
with(HERE/name).open("x")as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(data))
raise SystemExit(0 if result.wasSuccessful()else 1)
