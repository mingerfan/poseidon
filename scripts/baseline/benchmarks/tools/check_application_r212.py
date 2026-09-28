import sys,shlex,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=900)
 names=["test_application_component","test_validation_adapter","test_candidate_bundle","test_candidate_pipeline","test_provider_retries","test_unified_expression_guidance"]
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
 out=WORK/"results/application-component-r212"
 value=dict(run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),details=[(str(t),s) for t,s in r.failures+r.errors])
 path=out/"unit-initial.json"
 if path.exists():raise ValueError("Preserve existing unit evidence")
 path.write_text(json.dumps(value,indent=2))
 return 0 if r.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
