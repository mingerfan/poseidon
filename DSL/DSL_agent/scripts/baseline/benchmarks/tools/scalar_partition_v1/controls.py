import hashlib,io,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
def main():
 if sys.argv[1:2]!=["--inside"]:
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*sys.argv[1:]])
  return enter_nix(command,seconds=240,keep_env=(),environment="poseidon-cpu")
 require=os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
 if not require:raise ValueError("locked_test_worker_required")
 round_name=sys.argv[2]if len(sys.argv)>2 else"round1"
 if not round_name.isalnum():raise ValueError("test_round")
 modules=sys.argv[3:]or["test_pytorch_scalar_partitions"]
 tests=unittest.defaultTestLoader.loadTestsFromNames(modules)
 started=time.monotonic();buffer=io.StringIO()
 result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(tests)
 raw=buffer.getvalue();print(raw,flush=True)
 with(HERE/("controls-"+round_name+".log")).open("x")as f:f.write(raw)
 report=dict(format="poseidon-scalar-partition-controls-v1",status="passed"if result.wasSuccessful()else"failed",
  modules=modules,tests_run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  provider_calls=0,compiler_calls=0,encrypted_calls=0,seconds=time.monotonic()-started,
  tested_sources={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()for n in [
   "pytorch_einsum_extension.py","pytorch_einsum_normalization.py","pytorch_partition_normalization.py","pytorch_static_trace.py","pytorch_ordered_capture.py",
   "pytorch_ordered_contract.py","pytorch_ordered_intake.py","pytorch_model_intake.py",
   "source_precision_reference.py","application_capabilities.py","pytorch_alias_capture_bridge.py","pytorch_alias_composition.py","pytorch_model_capture.py","test_pytorch_einsum_views.py","test_pytorch_einsum.py","test_pytorch_partitions.py","test_pytorch_scalar_partitions.py"]})
 with(HERE/("controls-"+round_name+".json")).open("x")as f:json.dump(report,f,sort_keys=True)
 return 0 if result.wasSuccessful()else 1
if __name__=="__main__":raise SystemExit(main())
