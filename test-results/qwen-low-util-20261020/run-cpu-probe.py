#!/usr/bin/env python3
"""Short diagnostic runs of the same graph, intentionally stopped after 85s."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'poseidon-qwen-prefix-20261019'
sys.path.insert(0,str(SOURCE))
from nvml_sampler import Observer

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def process_sample(pid):
    root=Path('/proc')/str(pid)
    try:
        io={line.split(':')[0]:int(line.split()[1]) for line in (root/'io').read_text().splitlines()}
        threads=[]
        for task in (root/'task').iterdir():
            try:
                stat=(task/'stat').read_text().rsplit(') ',1)[1].split()
                threads.append(dict(tid=int(task.name),state=stat[0],utime_ticks=int(stat[11]),
                    stime_ticks=int(stat[12]),wchan=(task/'wchan').read_text().strip()))
            except FileNotFoundError:pass  # A thread exited during the snapshot.
        return dict(io=io,threads=threads)
    except FileNotFoundError:return None  # The child exited during the snapshot.

def run_case(observer,name,probe=None,delay=60,duration=20,executable=None):
    root=ROOT/name;root.mkdir(exist_ok=False)
    assert not observer.applications(), 'Other compute processes are active'
    command=json.loads((SOURCE/'runs-nvml/updated/command.json').read_text())
    command[4]=str(root/'report.json');command[5]=str(root/'decoded.json')
    env=dict(os.environ,**json.loads((SOURCE/'runs-nvml/updated/environment.json').read_text()))
    for key in ('POSEIDON_RUNTIME_TRACE','POSEIDON_TRACE'):env.pop(key,None)
    if executable:
        command[0]=str(executable)
        env['POSEIDON_RUNTIME_TRACE']=str(root/'boot')
    if probe is None:probe=name=='api-probe'
    if probe:
        env.update(LD_PRELOAD=str(ROOT/'cuda_api_probe.so'),CUDA_PROBE_DELAY_SECONDS=str(delay),
            CUDA_PROBE_DURATION_SECONDS=str(duration),CUDA_PROBE_OUTPUT=str(root/'cuda-api.jsonl'))
    save(root/'command.json',command)
    save(root/'environment.json',{k:v for k,v in env.items() if k.startswith(('CUDA_','POSEIDON_','OMP_','OPENBLAS_')) or k=='LD_PRELOAD'})
    started=time.monotonic();markers=[];next_gpu=0
    with (root/'run.log').open('w') as log,(root/'samples.jsonl').open('w') as samples:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(root/'launch.json',dict(pid=process.pid,monotonic=started,
            timestamp=datetime.datetime.now().astimezone().isoformat()))
        try:
            while process.poll() is None and time.monotonic()-started<delay+duration+5:
                tick=time.monotonic();elapsed=tick-started
                sample=dict(monotonic=tick,elapsed=elapsed,process=process_sample(process.pid))
                if elapsed>=next_gpu:
                    sample['nvml']=observer.sample();next_gpu=elapsed+2
                samples.write(json.dumps(sample)+'\n');samples.flush()
                text=(root/'run.log').read_text()
                for marker in ('FORWARD begin','PREFIX_EXECUTION_BEGIN','PREFIX_EXECUTION_END'):
                    if marker in text and not any(m['text']==marker for m in markers):
                        markers.append(dict(text=marker,monotonic=time.monotonic(),elapsed=elapsed))
                time.sleep(max(0,.1-(time.monotonic()-tick)))
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited unexpectedly: {process.returncode}; inspect run.log')
        finally:
            if process.poll() is None:process.terminate()
            code=process.wait(timeout=30)
        save(root/'status.json',dict(pid=process.pid,exit_code=code,
            wall_seconds=time.monotonic()-started,intentionally_terminated=True,
            scope='CPU/proc/NVML diagnostics; not full output validation',
            measured_process_seconds=[delay,delay+duration],
            markers=markers,proc_ticks_per_second=os.sysconf('SC_CLK_TCK')))
    if probe:
        rows=[json.loads(line) for line in (root/'cuda-api.jsonl').read_text().splitlines()]
        assert rows and any(r['api'].startswith('cudaLaunchKernel') for t in rows for r in t['rows'])
    print('Finished',name,flush=True)

def main():
    observer=Observer()
    try:
        for name in ('control','api-probe'):run_case(observer,name)
    finally:observer.close()

if __name__=='__main__':main()
