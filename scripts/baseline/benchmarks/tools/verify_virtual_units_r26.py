"""Final-source offline evidence checks; bounded, serial, no provider calls."""
import unittest,subprocess,json,sys,time,hashlib
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
base=WORK/'results';out=base/'upstream-virtual-r26-unit-final.json'
require(not out.exists(),'Preserve verification')
sources=runtime_sources();start=time.monotonic()
names=sorted({p.stem for pattern in ('test_unified*.py','test_upstream*.py') for p in (ROOT/'scripts/baseline').glob(pattern)}|
    {'test_semantic_benchmark','test_benchmark_execution','test_compiler_artifact_evidence','test_rejection_benchmark','test_portability','test_candidate_pipeline'})
class CountingResult(unittest.TextTestResult):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.passed=0
    def addSuccess(self,test):self.passed+=1;super().addSuccess(test)
suite=unittest.defaultTestLoader.loadTestsFromNames(names);discovered=suite.countTestCases()
result=unittest.TextTestRunner(verbosity=2,resultclass=CountingResult).run(suite)
dump(out,dict(modules=names,discovered=discovered,tests_run=result.testsRun,passed=result.passed,
    failures=len(result.failures),errors=len(result.errors),skip_records=[dict(test=str(t),reason=reason) for t,reason in result.skipped],
    skipped_record_count=len(result.skipped),skip_count_scope='includes class-level skips; do not subtract from tests_run',
    source_sha256=digest(sources),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    success=result.wasSuccessful(),agent_calls=0))
require(result.wasSuccessful(),'Offline tests failed; preserved results')