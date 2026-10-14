"""Finite controls and compatibility checks, with actual owned child processes."""
import hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if not sys.argv[1:] or sys.argv[1:]==["--affected"]:
  from hecate_python_env import VENV,enter_nix
  mode=["--inside-affected"]if sys.argv[1:] else["--inside"]
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*mode])
  return enter_nix(cmd,seconds=120,keep_env=(),environment="poseidon-cpu")
 from application_identity import current_identity_manifest
 from benchmark_graph import digest
 start=time.monotonic();identity=current_identity_manifest()
 names=["test_assessment_time_budget","application_assessment_entry.test_total_budget",
  "test_application_assessment.ApplicationAssessmentTests","application_assessment_entry.test_cli.CliTests",
  "application_assessment_entry.test_supervision"]
 if sys.argv[1:]==["--inside-affected"]:names=names[:2]
 else:assert sys.argv[1:]==["--inside"]
 suite=unittest.defaultTestLoader.loadTestsFromNames(names)
 def flat(v):
  for item in v:
   if isinstance(item,unittest.TestSuite):yield from flat(item)
   else:yield item
 unique={case.id():case for case in flat(suite)}
 stem="controls-affected"if sys.argv[1:]==["--inside-affected"]else"controls"
 with (HERE/(stem+".log")).open("x")as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite(unique.values()))
 baseline=json.loads((HERE/"baseline.json").read_text())
 protected=all(sha(Path(n)if Path(n).is_absolute()else ROOT/n)==h for n,h in baseline["protected_sources"].items())
 history=all(sha(n)==h for n,h in baseline["historical_request_files"].items())
 stable=current_identity_manifest()==identity
 r=dict(format="poseidon-assessment-deadline-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped and protected and history and stable else"failed",
  tests_run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  unique_test_ids=list(unique),test_modules=names,new_test_scope=27,new_real_process_tree_tests=2,
  numerical_backend="injected_test_backend",compiler_calls=0,encrypted_calls=0,provider_calls=0,
  protected_sources_checked=len(baseline["protected_sources"]),protected_sources_preserved=protected,
  historical_request_files_preserved=history,active_source_identity_preserved=stable,identity=identity,
  seconds=time.monotonic()-start)
 r["binding"]=digest(r)
 with (HERE/(stem+".json")).open("x")as f:json.dump(r,f,sort_keys=True,indent=2)
 print(json.dumps({k:v for k,v in r.items()if k not in ("identity","unique_test_ids")}))
 return 0 if r["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(main())
