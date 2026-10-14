"""CLI controls and application compatibility; injected backend is not FHE."""
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
from application_assessment_entry.test_cli import CliTests
from test_application_assessment import ApplicationAssessmentTests
class Results(unittest.TextTestResult):
 def startTest(self,test):
  self.ids=getattr(self,"ids",[]);self.ids.append(test.id());super().startTest(test)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--report",required=True);args=ap.parse_args()
 out=Path(args.report);assert not out.exists()
 names=("scripts/agent_assessment.py","scripts/baseline/application_assessment_entry/__init__.py",
  "scripts/baseline/application_assessment_entry/cli.py","scripts/baseline/application_assessment_entry/test_cli.py",
  "scripts/baseline/application_assessment.py","scripts/baseline/test_application_assessment.py")
 pins={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in names}
 suite=unittest.TestSuite()
 for cls in (CliTests,ApplicationAssessmentTests):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
 start=time.monotonic();r=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
 current={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in names}
 report=dict(format="poseidon-assessment-cli-controls-v1",tests=r.testsRun,
  passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),
  errors=len(r.errors),skipped=len(r.skipped),failed_ids=[t.id()for t,_ in r.failures],
  error_ids=[t.id()for t,_ in r.errors],test_ids=r.ids,source_sha256=pins,source_unchanged=pins==current,
  injected_outputs=True,actual_encrypted_calls=0,actual_source_reference_workers=0,api_calls=0,
  local_deadline_test="One actual isolated sleep child; unrelated to FHE.",elapsed_seconds=time.monotonic()-start)
 report["binding"]=digest(report)
 with out.open("xb")as f:f.write(canonical(report))
 print(json.dumps({k:report[k]for k in ("tests","passed","failed","errors","skipped","binding")}))
 return 0 if r.wasSuccessful()and pins==current else 1
if __name__=="__main__":raise SystemExit(main())
