"""Bounded control test entry; injected backend outputs are not FHE evidence."""
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest,canonical
import test_assessment_array_dataset as arrays
import test_task_dataset_evaluation as tasks
import test_task_dataset_assessment as receipts
import test_task_numerical_assessment as math
import test_task_native_assessment as native
class Results(unittest.TextTestResult):
    def startTest(self,test):
        self.ids=getattr(self,"ids",[]);self.ids.append(test.id())
        super().startTest(test)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--report",required=True)
    args=parser.parse_args();out=Path(args.report)
    if out.exists():raise SystemExit("Report already exists")
    classes=(arrays.ArrayDatasetTests,tasks.DatasetEvaluationTests,receipts.DatasetTests,math.AssessmentTests,native.NativeAssessmentTests)
    suite=unittest.TestSuite()
    for cls in classes:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    names=("assessment_array_dataset.py","task_dataset_evaluation.py","test_assessment_array_dataset.py",
        "test_task_dataset_evaluation.py","task_dataset_assessment.py","test_task_dataset_assessment.py",
        "task_numerical_assessment.py","test_task_numerical_assessment.py","test_task_native_assessment.py")
    pins={n:hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest() for n in names}
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
    current={n:hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest() for n in names}
    report=dict(format="poseidon-dataset-io-controls-v1",tests=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        failed_ids=[t.id() for t,_ in result.failures],error_ids=[t.id() for t,_ in result.errors],
        test_ids=result.ids,elapsed_seconds=time.monotonic()-start,source_sha256=pins,
        source_unchanged=pins==current,injected_execution_outputs=True,
        actual_encrypted_execution_calls=0,actual_source_forward_calls=0,compiler_calls=0,paid_api_calls=0)
    report["binding"]=digest(report)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb")as file:file.write(canonical(report))
    print(json.dumps({k:report[k] for k in ("tests","passed","failed","errors","skipped","elapsed_seconds","binding")}))
    return 0 if result.wasSuccessful() and pins==current else 1
if __name__=="__main__":raise SystemExit(main())
