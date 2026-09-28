"""Final-source offline evidence checks; bounded, serial, no provider calls."""
import unittest,subprocess,json,sys,time,hashlib
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
base=WORK/'results';out=base/'upstream-spatial-r22-unit-final.json'
require(not out.exists(),'Preserve verification')
sources=runtime_sources();start=time.monotonic()
names=sorted({p.stem for pattern in ('test_unified*.py','test_upstream*.py') for p in (ROOT/'scripts/baseline').glob(pattern)}|
    {'test_semantic_benchmark','test_benchmark_execution','test_compiler_artifact_evidence','test_rejection_benchmark','test_portability'})
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
def run(args,seconds):
    require(runtime_sources()==sources,'Frozen source changed')
    subprocess.run([str(VENV/'bin/python'),'-B',*args],check=True,timeout=seconds)
run(['scripts/baseline/benchmarks/tools/audit_spatial_frozen_mappings.py'],120)
run(['scripts/benchmark.py','plaintext','--inside','--output',str(base/'upstream-spatial-r22-plaintext')],1800)
run(['scripts/benchmark.py','rejections','--execute','--output',str(base/'upstream-spatial-r22-rejections.json')],120)
for label,folders in [('native',['upstream-spatial-r22-native-initial']),
                       ('regression',['upstream-spatial-r22-regression','upstream-spatial-r22-helper-directed','upstream-spatial-r22-chunk-regression'])]:
    args=['scripts/benchmark.py','compiler-evidence','--inside','--output',str(base/('upstream-spatial-r22-'+label+'-compiler-evidence.json'))]
    for folder in folders:args+=['--batch-report',str(base/folder/'report.json')]
    run(args,300)
require(runtime_sources()==sources,'Frozen source changed')
dump(base/'upstream-spatial-r22-verification-complete.json',dict(status='passed',source_sha256=digest(sources),
    seconds=time.monotonic()-start,unit_passed=result.passed,unit_skip_records=len(result.skipped),agent_calls=0))
