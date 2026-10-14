from pathlib import Path
import sys,shlex,json,time,unittest
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=360,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from source_precision_reference import source_identity
sdk=PoseidonBackend().identity_manifest()
names=["test_source_forward_call","test_task_source_reference","test_source_fx_observer",
 "test_source_fx_calibration_protocol","test_source_calibration","test_task_source_calibration",
 "test_task_calibration_dataset"]
names += ["test_source_precision_reference.SourcePrecisionTests."+n for n in [
 "test_float32_linear_preserves_source_weight_and_input_precision",
 "test_exact_float64_and_no_input_rounding","test_nested_dictionary_tuple_list_and_none",
 "test_input_alias_mutation_uses_owned_storage","test_registered_integer_and_boolean_buffers_preserve_dtype"]]
frozen=source_identity()
files=["source_forward_call.py","source_precision_reference.py","source_calibration.py","test_source_forward_call.py"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in names)
start=time.monotonic()
with (HERE/"controls.log").open("x")as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(format="poseidon-native-call-controls-v1",tests=result.testsRun,failed=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),status="passed"if result.wasSuccessful()and not result.skipped else "failed",
 source_files={str(B/n):q.sha(B/n)for n in files},source_identity=frozen,
 source_identity_unchanged=source_identity()==frozen,sdk=sdk["manifest_sha256"],sdk_unchanged=PoseidonBackend().identity_manifest()==sdk,
 seconds=time.monotonic()-start,api_calls=0,compiler_calls=0,encrypted_calls=0,
 native_fixture_workers=5,log_sha256=q.sha(HERE/"controls.log"))
report["binding"]=q.digest(report);q.write(HERE/"controls.json",report)
print(json.dumps({k:v for k,v in report.items()if k not in("source_files","source_identity")}),flush=True)
if not result.wasSuccessful():
 for _,error in (result.failures+result.errors)[:4]:print(error[-2500:],flush=True)
raise SystemExit(0 if report["status"]=="passed"else 1)
