"""Bounded offline compatibility acceptance; no provider or installs."""
import subprocess,json,sys,hashlib,time
from pathlib import Path
from hecate_python_env import WORK,VENV,ROOT
from benchmark_runner import dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate
out=WORK/'results/upstream-virtual-r26-regression';require(not out.exists(),'Preserve results');out.mkdir()
start=time.monotonic();sources=runtime_sources()
dump(out/'plan.json',dict(source_sha256=digest(sources),sources=sources,
    max_wall_seconds=1800,max_result_bytes=512*1024**2,agent_calls=0,native_concurrency=1,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
subprocess.run([str(VENV/'bin/python'),'-B','scripts/baseline/benchmarks/tools/audit_virtual_compatibility_r26.py'],
               check=True,timeout=120)
records=[];rows=[]
for profile in ('native','public-v1'):
    require(runtime_sources()==sources,'Frozen source changed')
    cmd=[str(VENV/'bin/python'),'-B','scripts/baseline/run_candidate.py','--inside','--self-test',
         '--case','scripts/baseline/cases/unified-two-input-two-output.json','--max-repairs','0',
         '--compiler-configuration','seal-cpu-eva-w45-v1','--unified-profile',profile]
    log=out/(profile+'.log')
    with log.open('w') as stream:
        proc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    text=log.read_text();matches=[x.split('Candidate evidence: ',1)[1] for x in text.splitlines() if x.startswith('Candidate evidence: ')]
    row=dict(id=profile,exit_code=proc.returncode,status='runner_failed')
    if matches:
        folder=Path(matches[-1]);r=json.loads((folder/'report.json').read_text())
        row.update(evidence=str(folder),status=r['status'])
        if r['status']=='passed':records.append(verify_candidate(folder))
    rows.append(row);dump(out/'progress.json',dict(rows=rows))
require(time.monotonic()-start<1800,'Bounded follow-up time')
dump(out/'report.json',dict(rows=rows,records=records,passed=len(records),failed=len(rows)-len(records),
    skipped=0,seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0))
cmd=[str(VENV/'bin/python'),'-B','scripts/benchmark.py','helper-directed','--inside','--execute',
     '--output',str(WORK/'results/upstream-virtual-r26-helper-directed')]
subprocess.run(cmd,check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Frozen source changed')
require(all(r['status']=='passed' for r in rows),'Legacy regression failed; preserve evidence')
dump(out/'complete.json',dict(status='passed',seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0))

command=[str(VENV/'bin/python'),'-B','scripts/baseline/benchmarks/tools/run_unified_chunk_batch.py',
         '--output',str(WORK/'results/upstream-virtual-r26-chunk-regression')]
subprocess.run(command,check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Frozen source changed')
dump(out/'all-batches-complete.json',dict(status='passed',seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0))
