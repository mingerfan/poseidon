"""Bounded checker-only build and static tests in the existing pinned Nix SDK."""
import sys,shlex,subprocess,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,ROOT,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
 import os
 from seal_cpu_golden import KEY_BUILD
 out=WORK/"results/seal-gate-r181"
 with (out/"checker-build.log").open("w") as log:
  subprocess.run(["cmake","-S",str(ROOT/"scripts/baseline/seal_keys"),"-B",str(KEY_BUILD),"-G","Ninja",
    "-DCMAKE_BUILD_TYPE=Release","-DSEAL_DIR="+os.environ["SEAL_DIR"]],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
  subprocess.run(["cmake","--build",str(KEY_BUILD),"--target","seal_artifact_parameters","-j2"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 suite=unittest.defaultTestLoader.loadTestsFromNames(["test_seal_scale_gate","test_seal_cpu_golden.SealGateTests","test_failure_repairs"])
 r=unittest.TextTestRunner(verbosity=2).run(suite)
 (out/"unit.json").write_text(json.dumps(dict(run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
  details=[(str(t),s) for t,s in r.failures+r.errors]),indent=2))
 return 0 if r.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
