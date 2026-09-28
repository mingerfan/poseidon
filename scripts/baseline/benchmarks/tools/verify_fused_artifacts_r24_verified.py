import subprocess,json,sys,time
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
base=WORK/'results';sources=runtime_sources();start=time.monotonic()
unit=json.loads((base/'upstream-fused-r24-verified-unit-final.json').read_text())
require(unit['success'] and unit['source_sha256']==digest(sources),'Unit source/acceptance mismatch')

def run(args,seconds):
    require(runtime_sources()==sources,'Frozen source changed')
    subprocess.run([str(VENV/'bin/python'),'-B',*args],check=True,timeout=seconds)
run(['scripts/baseline/benchmarks/tools/audit_fused_original_corpus_r24_verified.py'],120)
run(['scripts/benchmark.py','plaintext','--inside','--output',str(base/'upstream-fused-r24-verified-plaintext')],1800)
run(['scripts/benchmark.py','rejections','--execute','--output',str(base/'upstream-fused-r24-verified-rejections.json')],120)
for label,folders in [('native',['upstream-fused-r24-verified-native']),
                       ('helper_regression',['upstream-fused-r24-verified-regression','upstream-fused-r24-verified-helper-directed']),
                       ('chunk_regression',['upstream-fused-r24-verified-chunk-regression'])]:
    args=['scripts/benchmark.py','compiler-evidence','--inside','--output',str(base/('upstream-fused-r24-verified-'+label+'-compiler-evidence.json'))]
    for folder in folders:args+=['--batch-report',str(base/folder/'report.json')]
    run(args,300)
require(runtime_sources()==sources,'Frozen source changed')
dump(base/'upstream-fused-r24-verified-verification-complete.json',dict(status='passed',source_sha256=digest(sources),
    seconds=time.monotonic()-start,unit_passed=unit["passed"],unit_skip_records=unit["skipped_record_count"],agent_calls=0))
