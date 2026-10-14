"""Continue only tests absent from the terminal partial log, in finite slices."""
import hashlib,json,os,re,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
NAMES=["test_assessment_time_budget","application_assessment_entry.test_total_budget",
 "test_application_assessment.ApplicationAssessmentTests","application_assessment_entry.test_cli.CliTests",
 "application_assessment_entry.test_supervision"]
IMPLEMENTATIONS=("application_assessment.py","assessment_time_budget.py","application_assessment_entry/cli.py",
 "test_assessment_time_budget.py","application_assessment_entry/test_total_budget.py",
 "test_application_assessment.py","application_assessment_entry/test_cli.py","application_assessment_entry/supervision.py",
 "application_assessment_entry/test_supervision.py")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_suite():
 def flat(v):
  for t in v:
   if isinstance(t,unittest.TestSuite):yield from flat(t)
   else:yield t
 return {c.id():c for c in flat(unittest.defaultTestLoader.loadTestsFromNames(NAMES))}
def completed():
 result=[]
 for line in (HERE/"controls.log").read_text().splitlines():
  m=re.fullmatch(r"(test_[A-Za-z0-9_]+) \(([A-Za-z0-9_.]+)\) \.\.\. ok",line)
  if m:result.append(m.group(2)+"."+m.group(1))
 assert len(result)==len(set(result))==50
 return result
def main():
 if len(sys.argv)==2:
  assert sys.argv[1].isdigit();part=int(sys.argv[1])
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",str(part)])
  return enter_nix(cmd,seconds=120,keep_env=(),environment="poseidon-cpu")
 assert len(sys.argv)==3 and sys.argv[1]=="--inside" and sys.argv[2].isdigit()
 part=int(sys.argv[2]);all_cases=load_suite();done=completed();assert set(done)<=set(all_cases)
 remaining=[k for k in all_cases if k not in done];pieces=[remaining[i:i+20]for i in range(0,len(remaining),20)]
 assert 0<=part<len(pieces)
 from application_identity import current_identity_manifest
 from benchmark_graph import digest
 identity=current_identity_manifest();sources={n:sha(BASE/n)for n in IMPLEMENTATIONS}
 start=time.monotonic()
 with (HERE/f"controls-part-{part}.log").open("x")as f:
  r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite(all_cases[n]for n in pieces[part]))
 baseline=json.loads((HERE/"baseline.json").read_text())
 protected=all(sha(Path(n)if Path(n).is_absolute()else ROOT/n)==h for n,h in baseline["protected_sources"].items())
 history=all(sha(n)==h for n,h in baseline["historical_request_files"].items())
 stable=identity==current_identity_manifest()and all(sha(BASE/n)==h for n,h in sources.items())
 value=dict(format="poseidon-assessment-deadline-control-continuation-v1",
  status="passed"if r.wasSuccessful()and not r.skipped and protected and history and stable else"failed",
  part=part,total_parts=len(pieces),scope=pieces[part],tests_run=r.testsRun,failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
  observed_prior_passes=done,original_log_sha256=sha(HERE/"controls.log"),original_timeout_exit=124,
  source_hashes=sources,identity=identity,all_planned_ids=list(all_cases),protected_sources_preserved=protected,
  historical_requests_preserved=history,identity_preserved=stable,seconds=time.monotonic()-start,
  provider_calls=0,compiler_calls=0,encrypted_calls=0)
 value["binding"]=digest(value)
 with (HERE/f"controls-part-{part}.json").open("x")as f:json.dump(value,f,sort_keys=True,indent=2)
 print(json.dumps({k:v for k,v in value.items()if k not in("source_hashes","identity","all_planned_ids","observed_prior_passes","scope")}))
 return 0 if value["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(main())
