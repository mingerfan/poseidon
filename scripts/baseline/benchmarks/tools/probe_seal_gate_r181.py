"""Fixed SEAL cross-check and actual public-parameter binding, no paid API."""
import sys,shlex,subprocess,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV,ROOT,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
 from seal_cpu_golden import KEY_BUILD
 from seal_artifact_gate import verify_parameter_file,require
 from benchmark_graph import digest
 out=WORK/"results/seal-gate-r181"
 with (out/"probe-build.log").open("w") as log:
  subprocess.run(["cmake","-S",str(ROOT/"scripts/baseline/seal_keys"),"-B",str(KEY_BUILD)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
  subprocess.run(["cmake","--build",str(KEY_BUILD),"--target","seal_artifact_scale_probe","-j2"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 run=subprocess.run([str(KEY_BUILD/"seal_artifact_scale_probe"),str(out)],capture_output=True,text=True,timeout=150)
 (out/"probe.log").write_text(run.stdout+run.stderr)
 require(run.returncode==0,"SEAL probe failed; see probe.log")
 rows=[json.loads(s) for s in run.stdout.splitlines()]
 binding=verify_parameter_file(out/"probe-parm.seal",KEY_BUILD/"libseal_artifact_parameters.so")
 require(all(r.get("passed",True) for r in rows),"Boundary disagreement")
 rejected=out/"probe-parm-trailing.seal";rejected.write_bytes((out/"probe-parm.seal").read_bytes()+b"X")
 try:verify_parameter_file(rejected,KEY_BUILD/"libseal_artifact_parameters.so")
 except ValueError:pass
 else:raise ValueError("Trailing parameter data accepted")
 report=dict(rows=rows,parameter_binding=binding,paid_calls=0,executed_fixed_seal=True,
  hevm_runtime_changed=False,source_sha256=hashlib.sha256((ROOT/"scripts/baseline/seal_keys/artifact_scale_probe.cpp").read_bytes()).hexdigest())
 report["binding"]=digest(report)
 (out/"probe.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report))
 return 0
if __name__=="__main__":raise SystemExit(main())
