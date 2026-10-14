"""Finish untouched compatibility cases and qualify affected process/launch paths."""
import json,os,re,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(command,seconds=90,keep_env=(),environment="poseidon-cpu")
 from application_identity import current_identity_manifest
 from benchmark_graph import digest
 from application_assessment_entry.test_cli import CliTests
 from test_application_assessment import ApplicationAssessmentTests
 proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 assert current_identity_manifest()==proposal["source_stage_identity"],"pending_paid_identity_changed"
 completed=set()
 for line in(HERE/"controls.log.first-window").read_text().splitlines():
  found=re.fullmatch(r"(test_[a-zA-Z0-9_]+) \(([^)]+)\) \.\.\. ok",line)
  if found:completed.add(found.group(2)+"."+found.group(1))
 selectors=["application_assessment_entry.test_supervision.SupervisionTests"]
 affected=[]
 for name in unittest.defaultTestLoader.getTestCaseNames(CliTests):
  if name.startswith("test_launcher_")or name.startswith("test_deadline_"):
   affected.append("application_assessment_entry.test_cli.CliTests."+name)
 remaining=["test_application_assessment.ApplicationAssessmentTests."+name
  for name in unittest.defaultTestLoader.getTestCaseNames(ApplicationAssessmentTests)
  if "test_application_assessment.ApplicationAssessmentTests."+name not in completed]
 selectors+=affected+remaining
 start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromNames(selectors)
 with(HERE/"controls-affected.log").open("x")as out:
  r=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
 assert current_identity_manifest()==proposal["source_stage_identity"],"pending_paid_identity_changed"
 value=dict(format="poseidon-assessment-supervision-affected-controls-v1",tests=r.testsRun,
  passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),
  skipped=len(r.skipped),seconds=time.monotonic()-start,selectors=selectors,
  affected_launcher_tests=len(affected),previously_unfinished_controller_tests=len(remaining),
  api_calls=0,compiler_calls=0,encrypted_calls=0,pending_live_identity_unchanged=True,
  real_process_controls=True,application_backend_injected=True)
 value["binding"]=digest(value);(HERE/"controls-affected.json").write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
 print(json.dumps({k:v for k,v in value.items()if k not in("selectors",)}),flush=True)
 return 0 if r.wasSuccessful()and not r.skipped else 1
if __name__=="__main__":raise SystemExit(main())
