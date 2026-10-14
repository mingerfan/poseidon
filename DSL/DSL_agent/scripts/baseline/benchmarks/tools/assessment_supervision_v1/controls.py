"""Scoped entry, process ownership and controller compatibility controls."""
import hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def inside():
 from application_identity import current_identity_manifest
 from benchmark_graph import digest
 proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 assert current_identity_manifest()==proposal["source_stage_identity"],"pending_paid_identity_changed"
 start=time.monotonic()
 suite=unittest.defaultTestLoader.loadTestsFromNames((
  "application_assessment_entry.test_supervision.SupervisionTests",
  "application_assessment_entry.test_cli.CliTests",
  "test_application_assessment.ApplicationAssessmentTests"))
 with (HERE/"controls.log").open("x")as out:
  r=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
 value=dict(format="poseidon-assessment-supervision-controls-v1",tests=r.testsRun,
  passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),
  skipped=len(r.skipped),seconds=time.monotonic()-start,api_calls=0,compiler_calls=0,encrypted_calls=0,
  real_process_controls=True,application_backend_injected=True,pending_live_identity_unchanged=True)
 value["binding"]=digest(value)
 (HERE/"controls.json").write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
 print(json.dumps(value),flush=True)
 return 0 if r.wasSuccessful()and not r.skipped else 1
def main():
 if sys.argv[1:]==["--inside"]:return inside()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(main())
