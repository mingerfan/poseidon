"""Application control tests; injected outputs are not real FHE evidence."""
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
from test_application_assessment import ApplicationAssessmentTests
from test_task_dataset_evaluation import DatasetEvaluationTests
from test_assessment_array_dataset import ArrayDatasetTests
class Results(unittest.TextTestResult):
    def startTest(self,test):
        self.ids=getattr(self,"ids",[]);self.ids.append(test.id());super().startTest(test)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--report",required=True);args=ap.parse_args()
    out=Path(args.report);assert not out.exists()
    names=("application_assessment.py","test_application_assessment.py","task_dataset_evaluation.py",
           "test_task_dataset_evaluation.py","assessment_array_dataset.py","test_assessment_array_dataset.py")
    pins={n:hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest() for n in names}
    suite=unittest.TestSuite()
    for cls in (ApplicationAssessmentTests,DatasetEvaluationTests,ArrayDatasetTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    start=time.monotonic();r=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
    current={n:hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest() for n in names}
    report=dict(format="poseidon-application-assessment-controls-v1",tests=r.testsRun,
        passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
        failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
        failed_ids=[t.id()for t,_ in r.failures],error_ids=[t.id()for t,_ in r.errors],
        test_ids=r.ids,source_sha256=pins,source_unchanged=pins==current,
        injected_outputs=True,actual_encrypted_calls=0,actual_source_forward_calls=0,api_calls=0,
        elapsed_seconds=time.monotonic()-start)
    report["binding"]=digest(report)
    with out.open("xb")as file:file.write(canonical(report))
    print(json.dumps({k:report[k]for k in ("tests","passed","failed","errors","skipped","binding")}))
    return 0 if r.wasSuccessful()and pins==current else 1
if __name__=="__main__":raise SystemExit(main())
